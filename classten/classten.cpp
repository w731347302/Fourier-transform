#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Point points;
vector<Point> mousePoints;

void on_mouse(int EVENT, int x, int y, int flags, void * userdata)
{
	Mat hh;
	hh = *(Mat*)userdata;
	Point p(x, y);
	switch (EVENT)
	{
	case EVENT_LBUTTONDOWN:
		{
			points.x = x;
			points.y = y;
			mousePoints.push_back(points);
			circle(hh, points, 4, Scalar(255, 255, 255), -1);
			imshow("mouseCallback", hh);
		}
		break;
	}
}

int selectPolygon(Mat srcMat, Mat &dstMat)
{
	vector<vector<Point>> contours;
	Mat sclectMat;
	Mat m = Mat::zeros(srcMat.size(), CV_32F);
	m = 1;
	if (!srcMat.empty())
	{
		srcMat.copyTo(sclectMat);
		srcMat.copyTo(dstMat);
	}
	else
	{
		cout << "failed to read image" << endl;
		return -1;
	}
	namedWindow("mouseCallback");
	imshow("mouseCallback", sclectMat);
	setMouseCallback("mouseCallback", on_mouse, &sclectMat);
	waitKey(0);
	destroyAllWindows();

	contours.push_back(mousePoints);
	if (contours[0].size() < 3)
	{
		cout << "failed to read image" << endl;
		return -1;
	}
	drawContours(m, contours, 0, Scalar(0), -1);
	m.copyTo(dstMat);
	return 0;
}


int dftDemo(Mat src, Mat &magMat,double &normaVal, Mat &ph,Mat &mag)//原图像、输出整型图、矩阵最大值、相位图、输出浮点图
{
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);
	Mat padded;
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	//扩充至最佳边界

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat planes_true = Mat_<float>(padded);

	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);

	magnitude(planes[0], planes[1], planes_true);
	phase(planes[0], planes[1], ph);

	mag = planes_true;
	mag += Scalar::all(1);
	log(mag, mag);

	minMaxLoc(mag, 0, &normaVal, 0, 0);
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	Mat _magI = mag.clone();

	normalize(_magI, _magI, 0, 1, NORM_MINMAX);
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(mag, mag, 0, 1, NORM_MINMAX);
	mag = mag * 255;
	mag.convertTo(magMat, CV_8UC1);
	return 0;
}

int ifftDemo(Mat src, Mat &dspMat)
{
	Mat dst;

	int m = getOptimalDFTSize(src.rows); 
	int n = getOptimalDFTSize(src.cols);
	Mat padded;
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat planes_true = Mat_<float>(padded);
	Mat ph = Mat_<float>(padded);
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes_true);
	phase(planes[0], planes[1], ph);
	Mat A = planes[0];
	Mat B = planes[1];
	Mat mag = planes_true;
	mag += Scalar::all(1);
	log(mag, mag);
	double maxVal;
	minMaxLoc(mag, 0, &maxVal, 0, 0);
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	Mat _magI = mag.clone();
	normalize(_magI, _magI, 0, 1, NORM_MINMAX);
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(mag, mag, 0, 1, NORM_MINMAX);
	mag = mag * 255;
	imwrite("原频谱.jpg", mag);


	Mat mask;
	Mat proceMag;
	Mat res;
	mag.convertTo(res, CV_8UC1);
	selectPolygon(res, mask);
	mag = mag / 255;
	mag = mag.mul(mask);
	proceMag = mag*255;
	imwrite("处理后频谱.jpg", proceMag);


	Mat q00(mag, Rect(0, 0, cx, cy));
	Mat q10(mag, Rect(cx, 0, cx, cy));
	Mat q20(mag, Rect(0, cy, cx, cy));
	Mat q30(mag, Rect(cx, cy, cx, cy));
	q00.copyTo(tmp);
	q30.copyTo(q00);
	tmp.copyTo(q30);
	q10.copyTo(tmp);
	q20.copyTo(q10);
	tmp.copyTo(q20);
	mag = mag * maxVal;
	exp(mag, mag);
	mag = mag - Scalar::all(1);
	polarToCart(mag, ph, planes[0], planes[1]);
	merge(planes, 2, complexImg);
	Mat ifft(Size(src.cols, src.rows), CV_8UC1);
	idft(complexImg, ifft, DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, NORM_MINMAX);
	Rect rect(0, 0, src.cols, src.rows);
	dst = ifft(rect);
	dst = dst * 255;
	dst.convertTo(dspMat, CV_8UC1);
	imshow("dst", dspMat);
	imshow("src", src);
	waitKey(0);

	return 0;
}

int ifftDemo(Mat mag, Mat ph,double normVal, Mat &dstMat,Mat &dspMat)
{
	Mat planes[] = { Mat_<float>(mag), Mat::zeros(mag.size(), CV_32F) };

	Mat complexImg;

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	Mat q00(mag, Rect(0, 0, cx, cy));
	Mat q10(mag, Rect(cx, 0, cx, cy));
	Mat q20(mag, Rect(0, cy, cx, cy));
	Mat q30(mag, Rect(cx, cy, cx, cy));

	Mat tmp;
	q00.copyTo(tmp);
	q30.copyTo(q00);
	tmp.copyTo(q30);
	q10.copyTo(tmp);
	q20.copyTo(q10);
	tmp.copyTo(q20);

	mag = mag * normVal;
	exp(mag, mag);		
	mag = mag - Scalar::all(1);
	polarToCart(mag, ph, planes[0], planes[1]);
	merge(planes, 2, complexImg);

	Mat ifft(Size(mag.cols, mag.rows), CV_8UC1);
	idft(complexImg, ifft, DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, NORM_MINMAX);

	Rect rect(0, 0, mag.cols, mag.rows);
	dstMat = ifft(rect);
	dstMat = dstMat * 255;
	dstMat.convertTo(dspMat, CV_8UC1);

	return 0;
}



int main()
{
	Mat src = imread("test1.png",0);
	Mat src1 = imread("test2.jpg", 0);
	resize(src, src, Size(400, 600));
	resize(src1, src1, Size(400, 600));
	Mat dspMat, dspMat2;
	//Mat dst, dstMat;
	//selectPolygon(src, dst);
	//dst = dst * 255;
	//dst.convertTo(dstMat, CV_8UC1);
	//imshow("src", src);
	//imshow("dst", dstMat);
	//waitKey(0);
	//ifftDemo(src, dspMat);
	double normaVal1, normaVal2;
	Mat ph1,ph2;
	//dftDemo(src);
	Mat dst1, dst2;
	Mat mask1, mask2;
	Mat mag, mag2;
	Mat dstMat, dstMat2;
	dftDemo(src, dst1,normaVal1,ph1,mag);
	//selectPolygon(dst1, mask1);
	mask1 = Mat::zeros(src.size(), CV_32F);
	mask1 = 1;
	circle(mask1,Point(200, 300), 100, Scalar(0), -1, 8, 0);
	mag = mag / 255;
	mag = mag.mul(mask1);
	ifftDemo(mag, ph1, normaVal1, dstMat,dspMat);

	mask2 = Mat::zeros(src1.size(), CV_32F);
	dftDemo(src1, dst2, normaVal2, ph2, mag2);
	circle(mask2, Point(200, 300), 100, Scalar(1), -1, 8, 0);
	mag2 = mag2 / 255;
	mag2 = mag2.mul(mask2);
	//ifftDemo(mag2, ph2, normaVal2, dstMat2, dspMat2, A2, B2);

	Mat res_mag = mag + mag2;
	ph1 = ph1.mul(mask1);
	ph2 = ph2.mul(mask2);
	Mat res_ph = ph1 + ph2;
	double normaVal =(normaVal1+normaVal2)/2;
	Mat res;
	ifftDemo(res_mag, res_ph, normaVal, dstMat, res);

	imshow("res", res);
	imwrite("合成图.jpg", res);
	waitKey(0);
	return 0;
}