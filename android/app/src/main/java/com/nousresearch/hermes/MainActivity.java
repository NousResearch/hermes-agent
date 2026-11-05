package com.nousresearch.hermes;

import android.app.AlertDialog;
import android.content.Intent;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.chaquo.python.Python;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int OVERLAY_PERMISSION_REQUEST = 4108;
    private ProgressBar progressBar;
    private TextView statusText;
    private Button startButton;
    private Button petToggleButton;
    private Spinner mascotSpinner;
    private TextView petStatusText;
    private String initializationError = "Hermes Agent no pudo iniciarse.";
    private File hermesDir;
    private boolean petEnabled = false;

    private static final String[] MASCOTS = {"🐈 Gato", "🤖 Robot", "🦊 Zorro", "🐼 Panda", "🐧 Pingüino", "🐕 Perro"};
    private static final String[] MASCOT_EMOJI = {"🐈", "🤖", "🦊", "🐼", "🐧", "🐕"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        startButton = findViewById(R.id.startButton);
        petToggleButton = findViewById(R.id.petToggleButton);
        mascotSpinner = findViewById(R.id.mascotSpinner);
        petStatusText = findViewById(R.id.petStatusText);
        hermesDir = new File(getFilesDir(), "hermes");

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, MASCOTS);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mascotSpinner.setAdapter(adapter);
        selectSavedMascot();

        petToggleButton.setOnClickListener(v -> togglePet());
        mascotSpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                PetPreferences.setMascot(MainActivity.this, MASCOT_EMOJI[position]);
                if (petEnabled) startPetService();
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) { }
        });

        petEnabled = PetPreferences.isEnabled(this);
        refreshPetUi();
        // Keep Hermes Pet usable even if the experimental APK does not yet
        // contain the full Hermes Python dependency set.
        initializeHermes();
    }

    private void selectSavedMascot() {
        String saved = PetPreferences.getMascot(this);
        for (int i = 0; i < MASCOT_EMOJI.length; i++) {
            if (MASCOT_EMOJI[i].equals(saved)) {
                mascotSpinner.setSelection(i);
                return;
            }
        }
    }

    private void togglePet() {
        if (petEnabled) {
            Intent stop = new Intent(this, HermesPetOverlayService.class);
            stop.setAction("STOP");
            startService(stop);
            petEnabled = false;
            PetPreferences.setEnabled(this, false);
            refreshPetUi();
            return;
        }
        if (!Settings.canDrawOverlays(this)) {
            new AlertDialog.Builder(this)
                    .setTitle("Permitir mascota flotante")
                    .setMessage("Android necesita tu permiso para mostrar Hermes Pet encima de otras aplicaciones. La mascota no se activa hasta que tú lo permitas.")
                    .setNegativeButton("Ahora no", null)
                    .setPositiveButton("Abrir ajustes", (dialog, which) -> {
                        Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                                Uri.parse("package:" + getPackageName()));
                        startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST);
                    })
                    .show();
            return;
        }
        startPetService();
    }

    private void startPetService() {
        if (!Settings.canDrawOverlays(this)) return;
        try {
            Intent intent = new Intent(this, HermesPetOverlayService.class);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ContextCompat.startForegroundService(this, intent);
            } else {
                startService(intent);
            }
            petEnabled = true;
            PetPreferences.setEnabled(this, true);
            refreshPetUi();
        } catch (RuntimeException error) {
            Log.e(TAG, "Could not start Hermes Pet overlay", error);
            Toast.makeText(this, "No se pudo activar Hermes Pet: " + error.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == OVERLAY_PERMISSION_REQUEST) {
            if (Settings.canDrawOverlays(this)) startPetService();
            else Toast.makeText(this, "Permiso no concedido; Hermes Pet sigue apagado.", Toast.LENGTH_LONG).show();
        }
    }

    private void refreshPetUi() {
        petToggleButton.setText(petEnabled ? "Apagar Hermes Pet" : "Activar Hermes Pet");
        petStatusText.setText(petEnabled
                ? "Hermes Pet está activo. Arrástralo para moverlo; tócalo para volver aquí."
                : "La mascota está apagada.");
    }

    private void initializeHermes() {
        statusText.setText("Comprobando Hermes Agent…");
        progressBar.setVisibility(View.VISIBLE);
        new AsyncTask<Void, String, Boolean>() {
            @Override
            protected Boolean doInBackground(Void... voids) {
                try {
                    publishProgress("Comprobando entorno Python…");
                    PythonManager.initialize(MainActivity.this);
                    publishProgress("Comprobando módulos Hermes…");
                    Python python = PythonManager.getPython();
                    python.getModule("hermes_cli");
                    return true;
                } catch (Exception e) {
                    initializationError = "Hermes Agent todavía no está disponible en este APK: " + e.getMessage();
                    Log.w(TAG, initializationError, e);
                    return false;
                }
            }
            @Override protected void onProgressUpdate(String... values) { statusText.setText(values[0]); }
            @Override protected void onPostExecute(Boolean success) {
                progressBar.setVisibility(View.GONE);
                if (success) {
                    statusText.setText("Hermes Agent listo.");
                    startButton.setVisibility(View.VISIBLE);
                    startButton.setEnabled(true);
                } else {
                    // Do not close the whole app: pet customization is independent.
                    statusText.setText("Hermes Agent aún no está listo en este APK. Hermes Pet puede probarse por separado.");
                    startButton.setVisibility(View.GONE);
                    Log.i(TAG, initializationError);
                }
            }
        }.execute();
    }

    private void startHermes() {
        new AsyncTask<Void, Void, Boolean>() {
            @Override
            protected Boolean doInBackground(Void... voids) {
                try {
                    Python python = PythonManager.getPython();
                    System.setProperty("HERMES_HOME", hermesDir.getAbsolutePath());
                    python.getModule("hermes_cli.main").callAttr("main");
                    return true;
                } catch (Exception e) {
                    Log.e(TAG, "Failed to start Hermes", e);
                    return false;
                }
            }
            @Override protected void onPostExecute(Boolean success) {
                if (!success) Toast.makeText(MainActivity.this,
                        "No se pudo iniciar Hermes. Consulta los registros.", Toast.LENGTH_LONG).show();
            }
        }.execute();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        PythonManager.shutdown();
    }
}
