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

class HermesLogcatWatcherService : Service() {
    override fun onCreate() {
        super.onCreate()
        running = true
        createNotificationChannel()
        promoteToForeground()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        running = true
        val intervalSeconds = intent?.getLongExtra(EXTRA_SCAN_INTERVAL_SECONDS, 0L)
            ?.takeIf { it > 0 }
            ?: HermesLogcatWatcherBridge.persistedScanIntervalSeconds(applicationContext)
        val maxLines = intent?.getLongExtra(EXTRA_MAX_LINES, 0L)
            ?.takeIf { it > 0 }
            ?: HermesLogcatWatcherBridge.persistedMaxLines(applicationContext)
        HermesLogcatWatcherBridge.resumePersistedWatcherIfRequested(applicationContext)
        HermesLogcatWatcherBridge.startWorker(applicationContext, intervalSeconds, maxLines)
        promoteToForeground()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        running = false
        HermesLogcatWatcherBridge.stopWorker()
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
        val status = HermesPrivilegedAccessBridge.readStatus(applicationContext)
        val enabledCount = JSONObjectSafe {
            org.json.JSONObject(HermesLogcatWatcherBridge.statusJson(applicationContext))
                .optInt("enabled_logcat_record_count", 0)
        }
        val contentText = when {
            !status.shizukuBinderAlive -> "Waiting for Shizuku/Sui to start"
            !status.shizukuPermissionGranted -> "Waiting for Hermes Shizuku permission"
            enabledCount <= 0 -> "No enabled logcat automations"
            else -> "Watching logcat for $enabledCount saved automation(s)"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle("Hermes logcat watcher")
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
            "Hermes logcat watcher",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps Shizuku-backed Hermes logcat automations active"
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val CHANNEL_ID = "hermes_logcat_watcher"
        private const val NOTIFICATION_ID = 7316
        private const val EXTRA_SCAN_INTERVAL_SECONDS = "scan_interval_seconds"
        private const val EXTRA_MAX_LINES = "max_lines"

        @Volatile
        private var running: Boolean = false

        fun start(context: Context, intervalSeconds: Long, maxLines: Long): Boolean {
            val appContext = context.applicationContext
            val intent = Intent(appContext, HermesLogcatWatcherService::class.java)
                .putExtra(EXTRA_SCAN_INTERVAL_SECONDS, intervalSeconds)
                .putExtra(EXTRA_MAX_LINES, maxLines)
            return runCatching {
                ContextCompat.startForegroundService(appContext, intent)
            }.isSuccess
        }

        fun startIfDesired(context: Context): Boolean {
            val appContext = context.applicationContext
            if (!HermesLogcatWatcherBridge.isWatcherDesired(appContext)) {
                return false
            }
            return start(
                appContext,
                HermesLogcatWatcherBridge.persistedScanIntervalSeconds(appContext),
                HermesLogcatWatcherBridge.persistedMaxLines(appContext),
            )
        }

        fun stop(context: Context) {
            running = false
            context.applicationContext.stopService(Intent(context.applicationContext, HermesLogcatWatcherService::class.java))
        }

        fun isRunning(): Boolean = running
    }
}

private inline fun JSONObjectSafe(block: () -> Int): Int {
    return runCatching(block).getOrDefault(0)
}
