package com.nousresearch.hermes;

import android.content.Context;
import android.util.Log;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class PythonManager {
    private static final String TAG = "PythonManager";
    private static Python pythonInstance;
    private static boolean isInitialized = false;

    public static void initialize(Context context) {
        if (isInitialized) {
            Log.i(TAG, "Python already initialized");
            return;
        }

        try {
            Log.i(TAG, "Initializing Python runtime...");
            
            // Initialize Python runtime
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(context));
            }
            
            pythonInstance = Python.getInstance();
            
            // Setup Python environment
            setupPythonEnvironment(context);
            
            isInitialized = true;
            Log.i(TAG, "Python runtime initialized successfully");
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize Python: " + e.getMessage(), e);
            throw new RuntimeException("Python initialization failed", e);
        }
    }

    private static void setupPythonEnvironment(Context context) {
        try {
            // Create hermes directory if it doesn't exist
            File hermesDir = new File(context.getFilesDir(), "hermes");
            if (!hermesDir.exists()) {
                hermesDir.mkdirs();
            }
            
            // Set PYTHONPATH to include our app directory
            String pythonPath = hermesDir.getAbsolutePath();
            System.setProperty("python.path", pythonPath);
            
            // Note: Python packages are handled by Chaquopy's pip configuration in build.gradle
            // No need to copy from assets - Chaquopy bundles Python and packages automatically
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to setup Python environment: " + e.getMessage(), e);
        }
    }

    public static Python getPython() {
        if (!isInitialized) {
            throw new IllegalStateException("Python not initialized. Call initialize() first.");
        }
        return pythonInstance;
    }

    public static boolean isPythonInitialized() {
        return isInitialized;
    }

    public static void shutdown() {
        try {
            if (pythonInstance != null) {
                // Clean up Python resources
                pythonInstance.close();
            }
            isInitialized = false;
            Log.i(TAG, "Python runtime shutdown");
        } catch (Exception e) {
            Log.e(TAG, "Failed to shutdown Python: " + e.getMessage(), e);
        }
    }
}
