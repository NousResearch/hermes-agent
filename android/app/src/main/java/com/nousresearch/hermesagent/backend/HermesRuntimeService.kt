package com.nousresearch.hermesagent.backend

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.nousresearch.hermesagent.MainActivity
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.device.DeviceStateWriter

class HermesRuntimeService : Service() {
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        promoteToForeground(runtime = null)
        running = true
        DeviceStateWriter.write(applicationContext)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startOrRefreshForeground()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        running = false
        DeviceStateWriter.write(applicationContext)
        super.onDestroy()
    }

    private fun startOrRefreshForeground() {
        promoteToForeground(runtime = null)
        running = true
        DeviceStateWriter.write(applicationContext)

        val runtime = HermesRuntimeManager.ensureStarted(applicationContext)
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification(runtime))
        running = true
        DeviceStateWriter.write(applicationContext)
    }

    private fun promoteToForeground(runtime: HermesRuntimeManager.RuntimeState?) {
        val notification = buildNotification(runtime)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(runtime: HermesRuntimeManager.RuntimeState?): Notification {
        val contentTitle = when {
            runtime == null -> "Hermes runtime starting"
            runtime.started -> "Hermes runtime active"
            else -> "Hermes runtime waiting for attention"
        }
        val contentText = when {
            runtime == null -> "Preparing the local Hermes backend"
            !runtime.error.isNullOrBlank() -> runtime.error
            !runtime.modelName.isNullOrBlank() -> "Serving ${runtime.modelName} locally"
            !runtime.baseUrl.isNullOrBlank() -> "Serving local Hermes backend"
            else -> "Keeping Hermes ready in the background"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle(contentTitle)
            .setContentText(contentText)
            .setContentIntent(openAppPendingIntent())
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun openAppPendingIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Hermes runtime",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps the Hermes Android runtime available in the background"
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val CHANNEL_ID = "hermes_runtime"
        private const val NOTIFICATION_ID = 7315

        @Volatile
        private var running: Boolean = false

        fun start(context: Context) {
            val intent = Intent(context, HermesRuntimeService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            running = false
            context.stopService(Intent(context, HermesRuntimeService::class.java))
        }

        fun refresh(context: Context) {
            if (running) {
                start(context)
            }
        }

        fun isRunning(): Boolean = running
    }
}
