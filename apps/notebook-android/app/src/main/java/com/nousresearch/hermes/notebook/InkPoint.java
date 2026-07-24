package com.nousresearch.hermes.notebook;

import org.json.JSONException;
import org.json.JSONObject;

final class InkPoint {
    final float x, y, pressure, tilt, orientation;
    final long timeMillis;

    InkPoint(float x, float y, float pressure, float tilt, float orientation, long timeMillis) {
        this.x = x; this.y = y; this.pressure = pressure; this.tilt = tilt;
        this.orientation = orientation; this.timeMillis = timeMillis;
    }

    JSONObject toJson() throws JSONException {
        return new JSONObject().put("x", x).put("y", y).put("pressure", pressure)
            .put("tilt", tilt).put("orientation", orientation).put("t", timeMillis);
    }

    static InkPoint fromJson(JSONObject value) {
        return new InkPoint((float) value.optDouble("x"), (float) value.optDouble("y"),
            (float) value.optDouble("pressure", 1), (float) value.optDouble("tilt"),
            (float) value.optDouble("orientation"), value.optLong("t"));
    }
}
