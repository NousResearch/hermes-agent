package com.nousresearch.hermesagent.device

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
import org.json.JSONObject

class HermesCalendarWatcherService : Service() {
    override fun onCreate() {
        super.onCreate()
        running = true
        createNotificationChannel()
        promoteToForeground()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        running = true
        val intervalSeconds = intent?.getIntExtra(EXTRA_SCAN_INTERVAL_SECONDS, 0)
            ?.takeIf { it > 0 }
            ?: HermesCalendarWatcherBridge.persistedScanIntervalSeconds(applicationContext)
        val lookaheadMinutes = intent?.getIntExtra(EXTRA_LOOKAHEAD_MINUTES, 0)
            ?.takeIf { it > 0 }
            ?: HermesCalendarWatcherBridge.persistedLookaheadMinutes(applicationContext)
        val lookbackMinutes = intent?.getIntExtra(EXTRA_LOOKBACK_MINUTES, -1)
            ?.takeIf { it >= 0 }
            ?: HermesCalendarWatcherBridge.persistedLookbackMinutes(applicationContext)
        HermesCalendarWatcherBridge.resumePersistedWatcherIfRequested(applicationContext)
        HermesCalendarWatcherBridge.startWorker(applicationContext, intervalSeconds, lookaheadMinutes, lookbackMinutes)
        promoteToForeground()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        running = false
        HermesCalendarWatcherBridge.stopWorker()
        super.onDestroy()
    }

    private fun promoteToForeground() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(): Notification {
        val status = JSONObject(runCatching { HermesCalendarWatcherBridge.statusJson(applicationContext) }.getOrDefault("{}"))
        val enabledCount = status.optInt("enabled_calendar_record_count", 0)
        val contentText = when {
            !status.optBoolean("calendar_permission_granted", false) -> "Waiting for calendar permission"
            enabledCount <= 0 -> "No enabled calendar automations"
            else -> "Watching calendar events for $enabledCount saved automation(s)"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle("Hermes calendar watcher")
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
        if (manager.getNotificationChannel(CHANNEL_ID) != null) {
            return
        }
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Hermes calendar watcher",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps Hermes calendar automations active"
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val CHANNEL_ID = "hermes_calendar_watcher"
        private const val NOTIFICATION_ID = 7318
        private const val EXTRA_SCAN_INTERVAL_SECONDS = "scan_interval_seconds"
        private const val EXTRA_LOOKAHEAD_MINUTES = "lookahead_minutes"
        private const val EXTRA_LOOKBACK_MINUTES = "lookback_minutes"

        @Volatile
        private var running: Boolean = false

        fun start(
            context: Context,
            intervalSeconds: Int,
            lookaheadMinutes: Int,
            lookbackMinutes: Int,
        ): Boolean {
            val appContext = context.applicationContext
            val intent = Intent(appContext, HermesCalendarWatcherService::class.java)
                .putExtra(EXTRA_SCAN_INTERVAL_SECONDS, intervalSeconds)
                .putExtra(EXTRA_LOOKAHEAD_MINUTES, lookaheadMinutes)
                .putExtra(EXTRA_LOOKBACK_MINUTES, lookbackMinutes)
            val started = runCatching {
                ContextCompat.startForegroundService(appContext, intent)
            }.isSuccess
            running = started
            return started
        }

        fun startIfDesired(context: Context): Boolean {
            val appContext = context.applicationContext
            if (!HermesCalendarWatcherBridge.isWatcherDesired(appContext)) {
                return false
            }
            return start(
                appContext,
                HermesCalendarWatcherBridge.persistedScanIntervalSeconds(appContext),
                HermesCalendarWatcherBridge.persistedLookaheadMinutes(appContext),
                HermesCalendarWatcherBridge.persistedLookbackMinutes(appContext),
            )
        }

        fun stop(context: Context) {
            running = false
            context.applicationContext.stopService(Intent(context.applicationContext, HermesCalendarWatcherService::class.java))
        }

        fun isRunning(): Boolean = running
    }
}
