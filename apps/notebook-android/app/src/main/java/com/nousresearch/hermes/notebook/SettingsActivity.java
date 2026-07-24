package com.nousresearch.hermes.notebook;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.InputType;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class SettingsActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state); SharedPreferences prefs = getSharedPreferences("notebook", MODE_PRIVATE);
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setPadding(32, 32, 32, 32);
        TextView title = new TextView(this); title.setText("Hermes connection"); title.setTextSize(24); root.addView(title);
        EditText endpoint = field("https://your-hermes-host", prefs.getString("endpoint", "")); root.addView(endpoint);
        EditText token = field("Notebook token", prefs.getString("token", "")); token.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD); root.addView(token);
        Button save = new Button(this); save.setText("Save on this device"); root.addView(save, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 64));
        save.setOnClickListener(v -> { prefs.edit().putString("endpoint", endpoint.getText().toString().trim()).putString("token", token.getText().toString()).apply(); finish(); });
        setContentView(root);
    }
    private EditText field(String hint, String value) { EditText view = new EditText(this); view.setHint(hint); view.setText(value); view.setSingleLine(true); view.setMinHeight(64); return view; }
}
