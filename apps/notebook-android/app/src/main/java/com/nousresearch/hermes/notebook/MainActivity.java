package com.nousresearch.hermes.notebook;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private InkCanvasView canvas; private NotebookStore store; private TextView status; private EditText prompt;
    private InkTranscriber transcriber;
    private String chatId;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state); store = new NotebookStore(this); transcriber = new InkTranscriber(this);
        SharedPreferences prefs = getSharedPreferences("notebook", MODE_PRIVATE);
        chatId = prefs.getString("chat_id", ""); if (chatId.isEmpty()) { chatId = UUID.randomUUID().toString(); prefs.edit().putString("chat_id", chatId).apply(); }
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setBackgroundColor(0xfffbfaf4);
        LinearLayout bar = new LinearLayout(this); bar.setGravity(android.view.Gravity.CENTER_VERTICAL);
        status = new TextView(this); status.setText("Hermes Notebook • offline safe"); status.setTextSize(16); bar.addView(status, new LinearLayout.LayoutParams(0, 64, 1));
        bar.addView(button("Undo", v -> canvas.undo())); bar.addView(button("New", v -> confirmClear())); bar.addView(button("Settings", v -> startActivity(new Intent(this, SettingsActivity.class))));
        root.addView(bar); canvas = new InkCanvasView(this); root.addView(canvas, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        LinearLayout composer = new LinearLayout(this); prompt = new EditText(this); prompt.setHint("Optional note for Hermes…"); prompt.setMinHeight(64); composer.addView(prompt, new LinearLayout.LayoutParams(0, 72, 1)); composer.addView(button("Send", v -> send())); root.addView(composer);
        setContentView(root); canvas.setListener(this::persist); restore();
    }

    private Button button(String text, View.OnClickListener action) { Button value = new Button(this); value.setText(text); value.setMinWidth(72); value.setMinHeight(56); value.setOnClickListener(action); return value; }
    private void persist() { status.setText("Saving…"); io.execute(() -> { try { store.save(canvas.snapshot()); main.post(() -> status.setText("Saved offline")); } catch (Exception error) { main.post(() -> status.setText("Save failed: " + error.getMessage())); } }); }
    private void restore() { io.execute(() -> { try { var strokes = store.load(); main.post(() -> { canvas.replace(strokes); status.setText(strokes.isEmpty() ? "Ready" : "Restored offline page"); }); } catch (Exception error) { main.post(() -> status.setText("Notebook recovery failed")); } }); }
    private void confirmClear() { new AlertDialog.Builder(this).setTitle("Start a new page?").setMessage("The current page remains saved until you confirm.").setNegativeButton("Cancel", null).setPositiveButton("New page", (d, w) -> { canvas.clearInk(); prompt.setText(""); chatId = UUID.randomUUID().toString(); getSharedPreferences("notebook", MODE_PRIVATE).edit().putString("chat_id", chatId).apply(); }).show(); }
    private void send() {
        SharedPreferences prefs = getSharedPreferences("notebook", MODE_PRIVATE); String endpoint = prefs.getString("endpoint", ""); String token = prefs.getString("token", "");
        if (endpoint.isEmpty() || token.isEmpty()) { startActivity(new Intent(this, SettingsActivity.class)); return; }
        String note = prompt.getText().toString().trim(); if (note.isEmpty()) note = "Please interpret the attached handwritten notebook page.";
        String typedNote = note; var ink = canvas.snapshot(); status.setText("Reading handwriting…");
        transcriber.recognize(ink, canvas.getWidth(), canvas.getHeight(), (handwriting, recognitionError) -> {
            String finalNote = typedNote;
            if (!handwriting.isEmpty()) finalNote = typedNote.isEmpty() ? handwriting : typedNote + "\n\nHandwriting:\n" + handwriting;
            if (finalNote.isEmpty()) { status.setText("Could not read handwriting; add a short note and retry"); return; }
            String requestText = finalNote; status.setText("Hermes is working…");
            io.execute(() -> { try { String reply = HermesClient.send(endpoint, token, chatId, requestText, ink); main.post(() -> { status.setText("Hermes replied"); new AlertDialog.Builder(this).setTitle("Hermes").setMessage(reply).setPositiveButton("Close", null).show(); }); } catch (Exception error) { main.post(() -> status.setText("Send failed: " + error.getMessage())); } });
        });
    }
    @Override protected void onDestroy() { super.onDestroy(); transcriber.close(); io.shutdown(); }
}
