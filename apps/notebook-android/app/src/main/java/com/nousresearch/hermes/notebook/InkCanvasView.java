package com.nousresearch.hermes.notebook;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.MotionEvent;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

final class InkCanvasView extends View {
    interface Listener { void onInkChanged(); }
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final List<InkStroke> strokes = new ArrayList<>();
    private InkStroke active;
    private boolean erasing;
    private Listener listener;
    private boolean fingerDrawing;

    InkCanvasView(Context context) {
        super(context); setBackgroundColor(Color.rgb(251, 250, 244));
        paint.setColor(Color.BLACK); paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeCap(Paint.Cap.ROUND); paint.setStrokeJoin(Paint.Join.ROUND);
        setLayerType(View.LAYER_TYPE_HARDWARE, null);
    }

    void setListener(Listener value) { listener = value; }
    void setFingerDrawing(boolean value) { fingerDrawing = value; }
    List<InkStroke> snapshot() { return new ArrayList<>(strokes); }
    void replace(List<InkStroke> values) { strokes.clear(); strokes.addAll(values); invalidate(); }
    void clearInk() { strokes.clear(); active = null; changed(); }
    void undo() { if (!strokes.isEmpty()) { strokes.remove(strokes.size() - 1); changed(); } }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (InkStroke stroke : strokes) drawStroke(canvas, stroke);
        if (active != null) drawStroke(canvas, active);
    }

    private void drawStroke(Canvas canvas, InkStroke stroke) {
        if (stroke.points.isEmpty()) return;
        Path path = new Path(); InkPoint first = stroke.points.get(0); path.moveTo(first.x, first.y);
        for (int i = 1; i < stroke.points.size(); i++) { InkPoint p = stroke.points.get(i); path.lineTo(p.x, p.y); }
        float pressure = stroke.points.get(stroke.points.size() - 1).pressure;
        paint.setStrokeWidth(stroke.eraser ? 28f : 2.2f + Math.max(0, pressure) * 2.8f);
        paint.setColor(stroke.eraser ? Color.rgb(251, 250, 244) : Color.BLACK);
        canvas.drawPath(path, paint);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        int action = event.getActionMasked(); int index = stylusPointerIndex(event);
        if (index < 0) return active != null || erasing;
        int tool = event.getToolType(index);
        int actionTool = event.getToolType(event.getActionIndex());
        boolean actionIsStylus = actionTool == MotionEvent.TOOL_TYPE_STYLUS || actionTool == MotionEvent.TOOL_TYPE_ERASER;
        boolean stylus = tool == MotionEvent.TOOL_TYPE_STYLUS || tool == MotionEvent.TOOL_TYPE_ERASER;
        if (!stylus && !fingerDrawing) return false;
        if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_POINTER_DOWN) {
            if (!actionIsStylus || active != null || erasing) return true;
            erasing = tool == MotionEvent.TOOL_TYPE_ERASER;
            if (erasing) { eraseSamples(event, index); changed(); }
            else { active = new InkStroke(false); addSamples(event, index); invalidate(); }
            return true;
        }
        if (erasing) {
            if (action == MotionEvent.ACTION_MOVE) { eraseSamples(event, index); changed(); }
            if (action == MotionEvent.ACTION_CANCEL || action == MotionEvent.ACTION_UP || (action == MotionEvent.ACTION_POINTER_UP && actionIsStylus)) erasing = false;
            return true;
        }
        if (active == null) return false;
        if (action == MotionEvent.ACTION_MOVE) { addSamples(event, index); invalidate(); return true; }
        if (action == MotionEvent.ACTION_CANCEL || ((event.getFlags() & MotionEvent.FLAG_CANCELED) != 0)) {
            active = null; invalidate(); return true;
        }
        if (action == MotionEvent.ACTION_UP || (action == MotionEvent.ACTION_POINTER_UP && actionIsStylus)) {
            addSamples(event, index); if (!active.points.isEmpty()) strokes.add(active); active = null; changed(); return true;
        }
        return true;
    }

    private int stylusPointerIndex(MotionEvent event) {
        for (int i = 0; i < event.getPointerCount(); i++) {
            int tool = event.getToolType(i);
            if (tool == MotionEvent.TOOL_TYPE_STYLUS || tool == MotionEvent.TOOL_TYPE_ERASER) return i;
        }
        return fingerDrawing && event.getPointerCount() > 0 ? 0 : -1;
    }

    private void eraseSamples(MotionEvent event, int pointerIndex) {
        for (int h = 0; h < event.getHistorySize(); h++) eraseAt(event.getHistoricalX(pointerIndex, h), event.getHistoricalY(pointerIndex, h));
        eraseAt(event.getX(pointerIndex), event.getY(pointerIndex));
    }

    private void eraseAt(float x, float y) {
        final float radiusSquared = 24f * 24f;
        strokes.removeIf(stroke -> {
            for (InkPoint point : stroke.points) {
                float dx = point.x - x, dy = point.y - y;
                if (dx * dx + dy * dy <= radiusSquared) return true;
            }
            return false;
        });
    }

    private void addSamples(MotionEvent event, int pointerIndex) {
        for (int h = 0; h < event.getHistorySize(); h++) addPoint(event.getHistoricalX(pointerIndex, h), event.getHistoricalY(pointerIndex, h), event.getHistoricalPressure(pointerIndex, h), event.getHistoricalAxisValue(MotionEvent.AXIS_TILT, pointerIndex, h), event.getHistoricalOrientation(pointerIndex, h), event.getHistoricalEventTime(h));
        addPoint(event.getX(pointerIndex), event.getY(pointerIndex), event.getPressure(pointerIndex), event.getAxisValue(MotionEvent.AXIS_TILT, pointerIndex), event.getOrientation(pointerIndex), event.getEventTime());
    }

    private void addPoint(float x, float y, float pressure, float tilt, float orientation, long time) { active.points.add(new InkPoint(x, y, pressure, tilt, orientation, time)); }
    private void changed() { invalidate(); if (listener != null) listener.onInkChanged(); }
}
