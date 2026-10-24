package com.nousresearch.hermes;

import android.app.Application;
import android.content.Context;
import android.util.Log;

public class HermesApplication extends Application {
    private static final String TAG = "HermesApplication";
    private static Context appContext;

    @Override
    public void onCreate() {
        super.onCreate();
        appContext = getApplicationContext();
        Log.i(TAG, "Hermes Application started");
    }

    public static Context getAppContext() {
        return appContext;
    }
}
