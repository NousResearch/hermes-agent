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
        File hermesExec = new File(hermesDir, "bin/hermes");
        return hermesExec.exists();
    }

    private boolean downloadAndInstallHermes() {
        try {
            // Create directories
            hermesDir.mkdirs();
            pythonDir.mkdirs();
            
            // Download the latest hermes-agent wheel or package
            // For now, we'll use a simple approach: download from GitHub releases
            String releaseUrl = "https://api.github.com/repos/NousResearch/hermes-agent/releases/latest";
            String downloadUrl = getLatestReleaseAssetUrl(releaseUrl);
            
            if (downloadUrl == null || downloadUrl.isEmpty()) {
                Log.e(TAG, "Could not find download URL");
                return false;
            }
            
            File downloadFile = new File(getCacheDir(), "hermes-latest.tar.gz");
            
            // Download the file
            downloadFile(downloadUrl, downloadFile);
            
            // Extract and install
            return installHermes(downloadFile);
            
        } catch (Exception e) {
            Log.e(TAG, "Download and install failed: " + e.getMessage(), e);
            return false;
        }
    }

    private String getLatestReleaseAssetUrl(String apiUrl) {
        try {
            URL url = new URL(apiUrl);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setRequestProperty("Accept", "application/vnd.github.v3+json");
            
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(
                        new InputStreamReader(connection.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();
                
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();
                
                // Parse JSON to find the APK or Android asset
                String json = response.toString();
                if (json.contains("android") || json.contains("apk") || json.contains("termux")) {
                    // For now, return a placeholder - in production this would parse the JSON
                    // and find the appropriate Android asset
                    return "https://github.com/NousResearch/hermes-agent/releases/latest/download/hermes-agent.apk";
                }
            }
            connection.disconnect();
        } catch (Exception e) {
            Log.e(TAG, "Failed to get release info: " + e.getMessage(), e);
        }
        
        // Fallback: use a direct URL for testing
        return "https://hermes-assets.nousresearch.com/android/hermes-agent-latest.apk";
    }

    private void downloadFile(String fileUrl, File outputFile) throws IOException {
        URL url = new URL(fileUrl);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.connect();
        
        int fileLength = connection.getContentLength();
        
        try (InputStream input = connection.getInputStream();
             OutputStream output = new FileOutputStream(outputFile)) {
            
            byte[] buffer = new byte[4096];
            int bytesRead;
            long totalRead = 0;
            
            while ((bytesRead = input.read(buffer)) != -1) {
                output.write(buffer, 0, bytesRead);
                totalRead += bytesRead;
                
                // Update progress
                final long current = totalRead;
                final long total = fileLength;
                runOnUiThread(() -> {
                    if (total > 0) {
                        int progress = (int) ((current * 100) / total);
                        progressBar.setProgress(progress);
                    }
                });
            }
        }
        
        connection.disconnect();
    }

    private boolean installHermes(File archiveFile) {
        try {
            // For APK builds, we'll bundle the hermes executable directly
            // For now, we'll copy from assets
            copyAssetFolder(this, "hermes", hermesDir);
            
            // Make sure the hermes script is executable
            File hermesScript = new File(hermesDir, "bin/hermes");
            if (hermesScript.exists()) {
                hermesScript.setExecutable(true);
            }
            
            return true;
            
        } catch (Exception e) {
            Log.e(TAG, "Install failed: " + e.getMessage(), e);
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

    private void copyAssetFolder(Context context, String assetPath, File destination) {
        try {
            String[] files = context.getAssets().list(assetPath);
            if (files != null && files.length > 0) {
                for (String file : files) {
                    File destFile = new File(destination, file);
                    if (file.contains(".")) {
                        copyAssetFile(context, assetPath + "/" + file, destFile);
                    } else {
                        destFile.mkdirs();
                        copyAssetFolder(context, assetPath + "/" + file, destFile);
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to copy asset folder: " + assetPath, e);
        }
    }

    private void copyAssetFile(Context context, String assetPath, File destination) {
        try (InputStream in = context.getAssets().open(assetPath)) {
            FileOutputStream out = new FileOutputStream(destination);
            byte[] buf = new byte[1024];
            int len;
            while ((len = in.read(buf)) > 0) {
                out.write(buf, 0, len);
            }
            out.close();
        } catch (IOException e) {
            Log.e(TAG, "Failed to copy asset file: " + assetPath, e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        PythonManager.shutdown();
    }
}
