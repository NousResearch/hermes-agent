package com.nousresearch.hermes.notebook;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

final class InkStroke {
    final String id;
    final boolean eraser;
    final List<InkPoint> points = new ArrayList<>();

    InkStroke(boolean eraser) { this(UUID.randomUUID().toString(), eraser); }
    InkStroke(String id, boolean eraser) { this.id = id; this.eraser = eraser; }

    JSONObject toJson() throws JSONException {
        JSONArray values = new JSONArray();
        for (InkPoint point : points) values.put(point.toJson());
        return new JSONObject().put("id", id).put("eraser", eraser).put("points", values);
    }

    static InkStroke fromJson(JSONObject value) {
        InkStroke stroke = new InkStroke(value.optString("id"), value.optBoolean("eraser"));
        JSONArray values = value.optJSONArray("points");
        if (values != null) for (int i = 0; i < values.length(); i++) {
            JSONObject point = values.optJSONObject(i);
            if (point != null) stroke.points.add(InkPoint.fromJson(point));
        }
        return stroke;
    }
}
