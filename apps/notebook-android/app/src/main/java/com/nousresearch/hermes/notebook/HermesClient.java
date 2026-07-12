package com.nousresearch.hermes.notebook;

import android.os.Build;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

final class HermesClient {
    static String send(String endpoint, String token, String chatId, String text, List<InkStroke> strokes) throws Exception {
        if (!endpoint.startsWith("https://")) throw new IllegalArgumentException("Hermes endpoint must use HTTPS");
        JSONArray ink = new JSONArray(); for (InkStroke stroke : strokes) ink.put(stroke.toJson());
        JSONObject payload = new JSONObject().put("user", "notebook-user").put("chat_id", chatId)
            .put("message_id", java.util.UUID.randomUUID().toString()).put("text", text)
            .put("source", "notebook").put("ink", ink)
            .put("client", new JSONObject().put("platform", isBoox() ? "boox" : "android")
                .put("stylus", "android-stylus").put("capabilities", new JSONArray().put("pressure").put("tilt").put("eraser").put("offline")));
        HttpURLConnection connection = (HttpURLConnection) new URL(trimSlash(endpoint) + "/ingest").openConnection();
        connection.setRequestMethod("POST"); connection.setConnectTimeout(15_000); connection.setReadTimeout(380_000);
        connection.setDoOutput(true); connection.setRequestProperty("Content-Type", "application/json");
        connection.setRequestProperty("X-Notebook-Token", token);
        try (OutputStream output = connection.getOutputStream()) { output.write(payload.toString().getBytes(StandardCharsets.UTF_8)); }
        int status = connection.getResponseCode(); InputStream stream = status < 400 ? connection.getInputStream() : connection.getErrorStream();
        StringBuilder body = new StringBuilder(); try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) { String line; while ((line = reader.readLine()) != null) body.append(line); }
        JSONObject response = new JSONObject(body.toString());
        if (status >= 400) throw new IllegalStateException(response.optString("error", "Hermes request failed (" + status + ")"));
        return response.optString("reply", "");
    }

    private static String trimSlash(String value) { return value.endsWith("/") ? value.substring(0, value.length() - 1) : value; }
    private static boolean isBoox() { String maker = (Build.MANUFACTURER + " " + Build.BRAND).toLowerCase(); return maker.contains("onyx") || maker.contains("boox"); }
}
