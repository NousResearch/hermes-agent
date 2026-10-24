package com.nousresearch.hermes;

import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    
    private ProgressBar progressBar;
    private TextView statusText;
    private Button startButton;
    
    private String initializationError = "Failed to initialize Hermes Agent. Check the Android logs for details.";
    private File hermesDir;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        startButton = findViewById(R.id.startButton);
        
        hermesDir = new File(getFilesDir(), "hermes");
        
        // Check and request permissions
        checkAndRequestPermissions();
        
        startButton.setOnClickListener(v -> startHermes());
    }

    private void checkAndRequestPermissions() {
        // The app stores files in its private internal directory. It does not
        // need legacy external-storage permissions to initialize. INTERNET is
        // a normal install-time permission and must not be requested at runtime.
        initializeHermes();
    }

    private void initializeHermes() {
        new Handler(Looper.getMainLooper()).post(() -> {
            statusText.setText("Initializing Hermes Agent...");
            progressBar.setVisibility(View.VISIBLE);
        });
        
        new AsyncTask<Void, String, Boolean>() {
            @Override
            protected Boolean doInBackground(Void... voids) {
                try {
                    publishProgress("Checking Python runtime...");
                    
                    // Initialize Python
                    PythonManager.initialize(MainActivity.this);
                    
                    publishProgress("Checking Hermes installation...");
                    
                    // Check if Hermes is already installed
                    if (!isHermesInstalled()) {
                        initializationError = "Hermes Python modules are not bundled in this APK yet. The Android package needs an Android-compatible Hermes dependency build before the agent can start.";
                        publishProgress("Hermes modules are missing from this APK.");
                        return false;
                    }
                    
                    publishProgress("Hermes is ready!");
                    return true;
                    
                } catch (Exception e) {
                    initializationError = "Initialization failed: " + e.getMessage();
                    Log.e(TAG, initializationError, e);
                    return false;
                }
            }
            
            @Override
            protected void onProgressUpdate(String... values) {
                statusText.setText(values[0]);
            }
            
            @Override
            protected void onPostExecute(Boolean success) {
                progressBar.setVisibility(View.GONE);
                
                if (success) {
                    startButton.setVisibility(View.VISIBLE);
                    startButton.setEnabled(true);
                } else {
                    showErrorAndExit(initializationError);
                }
            }
        }.execute();
    }

    private boolean isHermesInstalled() {
        // With Chaquopy, hermes-agent is bundled as a Python package
        // Check if we can import it via Python
        try {
            Python python = PythonManager.getPython();
            python.getModule("hermes_cli");
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private void startHermes() {
        new AsyncTask<Void, Void, Boolean>() {
            @Override
            protected Boolean doInBackground(Void... voids) {
                try {
                    // Start Hermes using Python
                    Python python = PythonManager.getPython();
                    
                    // Change to hermes directory
                    File hermesDir = new File(getFilesDir(), "hermes");
                    System.setProperty("HERMES_HOME", hermesDir.getAbsolutePath());
                    
                    // Execute hermes
                    python.getModule("hermes_cli.main").callAttr("main");
                    
                    return true;
                    
                } catch (Exception e) {
                    Log.e(TAG, "Failed to start Hermes: " + e.getMessage(), e);
                    return false;
                }
            }
            
            @Override
            protected void onPostExecute(Boolean success) {
                if (!success) {
                    showError("Failed to start Hermes. Check logs for details.");
                }
            }
        }.execute();
    }

    private void showErrorAndExit(String message) {
        new AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK", (dialog, which) -> finish())
                .setCancelable(false)
                .show();
    }

    private void showError(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    // Asset copying methods removed - not needed with Chaquopy
    // Python packages are bundled automatically by Chaquopy
    // Hermes is installed via pip in build.gradle configuration

    @Override
    protected void onDestroy() {
        super.onDestroy();
        PythonManager.shutdown();
    }
}
