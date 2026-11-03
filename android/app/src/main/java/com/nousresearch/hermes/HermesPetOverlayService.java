package com.nousresearch.hermes;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.IBinder;
import android.provider.Settings;
import android.view.Gravity;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

/**
 * User-controlled, movable Hermes Pet overlay.
 *
 * This first slice deliberately draws only the mascot. It does not inspect
 * screen contents or click other apps; those require a separately explained,
 * user-enabled accessibility/screen-capture integration.
 */
public class HermesPetOverlayService extends Service {
    private static final String CHANNEL_ID = "hermes_pet_overlay";
    private static final int NOTIFICATION_ID = 4107;
    private WindowManager windowManager;
    private TextView petView;
    private WindowManager.LayoutParams layoutParams;
    private float downRawX, downRawY;
    private int downX, downY;
    private boolean moved;
    private ObjectAnimator idleAnimation;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        startForeground(NOTIFICATION_ID, buildNotification());
        showPet();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "STOP".equals(intent.getAction())) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (petView != null) {
            petView.setText(PetPreferences.getMascot(this));
        } else {
            showPet();
        }
        return START_STICKY;
    }

    private void showPet() {
        if (!Settings.canDrawOverlays(this) || petView != null) return;

        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        petView = new TextView(this);
        petView.setText(PetPreferences.getMascot(this));
        petView.setTextSize(42);
        petView.setGravity(Gravity.CENTER);
        petView.setContentDescription("Hermes Pet. Arrastra para mover; toca para abrir Hermes.");
        petView.setBackgroundResource(android.R.drawable.dialog_holo_light_frame);
        petView.setPadding(8, 4, 8, 4);

        int overlayType = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                : WindowManager.LayoutParams.TYPE_PHONE;
        layoutParams = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                overlayType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT);
        layoutParams.gravity = Gravity.TOP | Gravity.START;
        layoutParams.x = 24;
        layoutParams.y = 180;

        petView.setOnTouchListener((view, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    downRawX = event.getRawX();
                    downRawY = event.getRawY();
                    downX = layoutParams.x;
                    downY = layoutParams.y;
                    moved = false;
                    return true;
                case MotionEvent.ACTION_MOVE:
                    int dx = (int) (event.getRawX() - downRawX);
                    int dy = (int) (event.getRawY() - downRawY);
                    if (Math.abs(dx) > 5 || Math.abs(dy) > 5) moved = true;
                    layoutParams.x = Math.max(0, downX + dx);
                    layoutParams.y = Math.max(0, downY + dy);
                    try {
                        windowManager.updateViewLayout(petView, layoutParams);
                    } catch (IllegalArgumentException ignored) {
                        // The overlay was removed while the user was dragging.
                    }
                    return true;
                case MotionEvent.ACTION_UP:
                    if (!moved) {
                        Intent open = new Intent(HermesPetOverlayService.this, MainActivity.class);
                        open.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
                        open.putExtra("open_pet_settings", true);
                        startActivity(open);
                    }
                    return true;
                default:
                    return false;
            }
        });

        try {
            windowManager.addView(petView, layoutParams);
            idleAnimation = ObjectAnimator.ofFloat(petView, View.TRANSLATION_Y, 0f, -7f, 0f);
            idleAnimation.setDuration(1400);
            idleAnimation.setRepeatCount(ValueAnimator.INFINITE);
            idleAnimation.setRepeatMode(ValueAnimator.RESTART);
            idleAnimation.start();
        } catch (RuntimeException error) {
            Toast.makeText(this, "No se pudo mostrar Hermes Pet: " + error.getMessage(),
                    Toast.LENGTH_LONG).show();
            petView = null;
            stopSelf();
        }
    }

    private Notification buildNotification() {
        Intent openIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, openIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | (Build.VERSION.SDK_INT >= 23 ? PendingIntent.FLAG_IMMUTABLE : 0));
        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return builder
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle("Hermes Pet está activo")
                .setContentText("Toca la mascota para abrir sus ajustes. Arrástrala para moverla.")
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "Hermes Pet", NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("Mantiene activa la mascota flotante mientras tú la uses.");
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) manager.createNotificationChannel(channel);
        }
    }

    @Override
    public void onDestroy() {
        if (windowManager != null && petView != null) {
            try {
                windowManager.removeView(petView);
            } catch (IllegalArgumentException ignored) {
                // Already detached.
            }
        }
        if (idleAnimation != null) {
            idleAnimation.cancel();
            idleAnimation = null;
        }
        petView = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
