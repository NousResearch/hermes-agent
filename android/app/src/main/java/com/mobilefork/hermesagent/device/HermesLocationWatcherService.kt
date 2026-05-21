package com.mobilefork.hermesagent.device

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.mobilefork.hermesagent.MainActivity
import com.mobilefork.hermesagent.R
import org.json.JSONObject

class HermesLocationWatcherService : Service(), LocationListener {
    private var locationManager: LocationManager? = null
    private val registeredProviders = mutableListOf<String>()
    private var minIntervalMs: Long = 30_000L
    private var minDistanceMeters: Double = 25.0

    override fun onCreate() {
        super.onCreate()
        locationManager = getSystemService(Context.LOCATION_SERVICE) as? LocationManager
        HermesLocationWatcherBridge.setServiceState(isRunning = true, providers = emptyList())
        createNotificationChannel()
        promoteToForeground()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        minIntervalMs = intent?.getLongExtra(EXTRA_MIN_INTERVAL_MS, 0L)
            ?.takeIf { it > 0 }
            ?: HermesLocationWatcherBridge.persistedMinIntervalMs(applicationContext)
        minDistanceMeters = intent?.getDoubleExtra(EXTRA_MIN_DISTANCE_METERS, Double.NaN)
            ?.takeIf { !it.isNaN() }
            ?: HermesLocationWatcherBridge.persistedMinDistanceMeters(applicationContext)
        HermesLocationWatcherBridge.resumePersistedWatcherIfRequested(applicationContext)
        registerProviders(
            intent?.getStringArrayListExtra(EXTRA_PROVIDERS)?.filter { it.isNotBlank() }
                ?: HermesLocationWatcherBridge.persistedProviders(applicationContext),
        )
        promoteToForeground()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        unregisterProviders()
        running = false
        HermesLocationWatcherBridge.setServiceState(isRunning = false, providers = emptyList())
        super.onDestroy()
    }

    override fun onLocationChanged(location: Location) {
        HermesLocationWatcherBridge.dispatchLocationJson(applicationContext, location)
    }

    @Deprecated("Deprecated by Android framework; keep for older API callbacks.")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit

    override fun onProviderEnabled(provider: String) = Unit

    override fun onProviderDisabled(provider: String) = Unit

    private fun registerProviders(providers: List<String>) {
        unregisterProviders()
        if (!HermesLocationWatcherBridge.hasLocationPermission(applicationContext)) {
            HermesLocationWatcherBridge.setServiceState(isRunning = true, providers = emptyList())
            return
        }
        val manager = locationManager ?: return
        val activeProviders = mutableListOf<String>()
        providers.distinct().forEach { provider ->
            val registered = runCatching {
                manager.requestLocationUpdates(
                    provider,
                    minIntervalMs,
                    minDistanceMeters.toFloat(),
                    this,
                )
            }.isSuccess
            if (registered) {
                registeredProviders.add(provider)
                activeProviders.add(provider)
            }
        }
        HermesLocationWatcherBridge.setServiceState(isRunning = true, providers = activeProviders)
    }

    private fun unregisterProviders() {
        val manager = locationManager ?: return
        runCatching { manager.removeUpdates(this) }
        registeredProviders.clear()
    }

    private fun promoteToForeground() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(): Notification {
        val status = JSONObject(runCatching { HermesLocationWatcherBridge.statusJson(applicationContext) }.getOrDefault("{}"))
        val registeredCount = status.optJSONArray("registered_providers")?.length() ?: 0
        val enabledCount = status.optInt("enabled_location_record_count", 0)
        val contentText = when {
            !status.optBoolean("location_permission_granted", false) -> "Waiting for location permission"
            enabledCount <= 0 -> "No enabled location automations"
            registeredCount <= 0 -> "Waiting for Android location providers"
            else -> "Watching $registeredCount location provider(s)"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle("Hermes location watcher")
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
            "Hermes location watcher",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps Hermes location automations active"
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val CHANNEL_ID = "hermes_location_watcher"
        private const val NOTIFICATION_ID = 7319
        private const val EXTRA_MIN_INTERVAL_MS = "min_interval_ms"
        private const val EXTRA_MIN_DISTANCE_METERS = "min_distance_meters"
        private const val EXTRA_PROVIDERS = "providers"

        @Volatile
        private var running: Boolean = false

        fun start(context: Context, minIntervalMs: Long, minDistanceMeters: Double, providers: List<String>): Boolean {
            val appContext = context.applicationContext
            val intent = Intent(appContext, HermesLocationWatcherService::class.java)
                .putExtra(EXTRA_MIN_INTERVAL_MS, minIntervalMs)
                .putExtra(EXTRA_MIN_DISTANCE_METERS, minDistanceMeters)
                .putStringArrayListExtra(EXTRA_PROVIDERS, ArrayList(providers))
            val started = runCatching {
                ContextCompat.startForegroundService(appContext, intent)
            }.isSuccess
            running = started
            return started
        }

        fun startIfDesired(context: Context): Boolean {
            val appContext = context.applicationContext
            if (!HermesLocationWatcherBridge.resumePersistedWatcherIfRequested(appContext)) {
                return false
            }
            return start(
                appContext,
                HermesLocationWatcherBridge.persistedMinIntervalMs(appContext),
                HermesLocationWatcherBridge.persistedMinDistanceMeters(appContext),
                HermesLocationWatcherBridge.persistedProviders(appContext),
            )
        }

        fun stop(context: Context) {
            running = false
            context.applicationContext.stopService(Intent(context.applicationContext, HermesLocationWatcherService::class.java))
        }

        fun isRunning(): Boolean = running
    }
}
