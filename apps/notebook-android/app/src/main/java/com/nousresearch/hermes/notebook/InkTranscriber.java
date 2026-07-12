package com.nousresearch.hermes.notebook;

import android.content.Context;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.common.model.RemoteModelManager;
import com.google.mlkit.common.MlKitException;
import com.google.mlkit.vision.digitalink.common.DigitalInkRecognitionModelIdentifier;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognition;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognitionModel;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognizer;
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognizerOptions;
import com.google.mlkit.vision.digitalink.recognition.Ink;
import com.google.mlkit.vision.digitalink.recognition.RecognitionContext;
import com.google.mlkit.vision.digitalink.recognition.WritingArea;
import java.util.List;

final class InkTranscriber {
    interface Callback { void complete(String text, Exception error); }
    private final DigitalInkRecognitionModel model;
    private final DigitalInkRecognizer recognizer;

    InkTranscriber(Context context) {
        DigitalInkRecognitionModelIdentifier identifier;
        try { identifier = DigitalInkRecognitionModelIdentifier.fromLanguageTag("en-US"); }
        catch (MlKitException error) { throw new IllegalStateException("Invalid handwriting language", error); }
        if (identifier == null) throw new IllegalStateException("English handwriting model unavailable");
        model = DigitalInkRecognitionModel.builder(identifier).build();
        recognizer = DigitalInkRecognition.getClient(DigitalInkRecognizerOptions.builder(model).build());
        RemoteModelManager.getInstance().download(model, new DownloadConditions.Builder().build());
    }

    void recognize(List<InkStroke> strokes, float width, float height, Callback callback) {
        Ink.Builder ink = Ink.builder();
        for (InkStroke source : strokes) {
            if (source.eraser || source.points.isEmpty()) continue;
            Ink.Stroke.Builder stroke = Ink.Stroke.builder();
            for (InkPoint point : source.points) stroke.addPoint(Ink.Point.create(point.x, point.y, point.timeMillis));
            ink.addStroke(stroke.build());
        }
        RecognitionContext context = RecognitionContext.builder()
            .setWritingArea(new WritingArea(Math.max(1, width), Math.max(1, height))).build();
        recognizer.recognize(ink.build(), context)
            .addOnSuccessListener(result -> callback.complete(
                result.getCandidates().isEmpty() ? "" : result.getCandidates().get(0).getText(), null))
            .addOnFailureListener(error -> callback.complete("", error));
    }

    void close() { recognizer.close(); }
}
