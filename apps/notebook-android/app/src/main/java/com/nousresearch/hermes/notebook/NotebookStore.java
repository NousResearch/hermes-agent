package com.nousresearch.hermes.notebook;

import android.content.Context;
import android.util.AtomicFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

final class NotebookStore {
    private final AtomicFile file;
    NotebookStore(Context context) { file = new AtomicFile(new File(context.getFilesDir(), "current-notebook.json")); }

    synchronized void save(List<InkStroke> strokes) throws Exception {
        JSONArray array = new JSONArray();
        for (InkStroke stroke : strokes) array.put(stroke.toJson());
        byte[] bytes = new JSONObject().put("version", 1).put("strokes", array)
            .toString().getBytes(StandardCharsets.UTF_8);
        FileOutputStream output = file.startWrite();
        try { output.write(bytes); file.finishWrite(output); }
        catch (Exception error) { file.failWrite(output); throw error; }
    }

    synchronized List<InkStroke> load() throws Exception {
        List<InkStroke> result = new ArrayList<>();
        if (!file.getBaseFile().exists()) return result;
        byte[] bytes;
        try (FileInputStream input = file.openRead(); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192]; int read;
            while ((read = input.read(buffer)) != -1) output.write(buffer, 0, read);
            bytes = output.toByteArray();
        }
        JSONArray strokes = new JSONObject(new String(bytes, StandardCharsets.UTF_8)).optJSONArray("strokes");
        if (strokes != null) for (int i = 0; i < strokes.length(); i++) {
            JSONObject value = strokes.optJSONObject(i); if (value != null) result.add(InkStroke.fromJson(value));
        }
        return result;
    }
}
