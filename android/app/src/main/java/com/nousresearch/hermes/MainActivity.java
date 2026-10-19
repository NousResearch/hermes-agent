package com.nousresearch.hermes;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.text.Html;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

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
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 1001;
    private static final int REQUEST_CODE_DOWNLOAD = 1002;
    
    private ProgressBar progressBar;
    private TextView statusText;
    private Button startButton;
    
    private String hermesVersion = "latest";
    private File hermesDir;
    private File pythonDir;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        startButton = findViewById(R.id.startButton);
        
        hermesDir = new File(getFilesDir(), "hermes");
        pythonDir = new File(hermesDir, "python");
        
        // Check and request permissions
        checkAndRequestPermissions();
        
        startButton.setOnClickListener(v -> startHermes());
    }

    private void checkAndRequestPermissions() {
        List<String> permissionsNeeded = new ArrayList<>();
        
        // Check storage permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) 
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) 
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            }
        }
        
        // Check network permission (usually normal)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.INTERNET) 
                != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.INTERNET);
        }
        
        if (!permissionsNeeded.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this,
                    permissionsNeeded.toArray(new String[0]),
                    REQUEST_CODE_PERMISSIONS
            );
        } else {
            initializeHermes();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, 
            @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            
            if (allGranted) {
                initializeHermes();
            } else {
                showErrorAndExit("Permissions are required to run Hermes Agent");
            }
        }
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
                        publishProgress("Downloading Hermes...");
                        return downloadAndInstallHermes();
                    }
                    
                    publishProgress("Hermes is ready!");
                    return true;
                    
                } catch (Exception e) {
                    Log.e(TAG, "Initialization failed: " + e.getMessage(), e);
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
                    showErrorAndExit("Failed to initialize Hermes Agent");
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

    private boolean downloadAndInstallHermes() {
        // With Chaquopy, hermes-agent is installed via pip during build
        // No need to download at runtime - it's already bundled in the APK
        Log.i(TAG, "Hermes is pre-installed via Chaquopy pip configuration");
        return true;
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
