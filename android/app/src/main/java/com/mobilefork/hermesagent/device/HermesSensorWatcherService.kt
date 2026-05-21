package com.mobilefork.hermesagent.device

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.mobilefork.hermesagent.MainActivity
import com.mobilefork.hermesagent.R
import org.json.JSONObject

class HermesSensorWatcherService : Service(), SensorEventListener {
    private var sensorManager: SensorManager? = null
    private val registeredSensors = mutableListOf<Sensor>()
    private val accuracyBySensorType = mutableMapOf<Int, String>()
    private val lastDispatchBySensorType = mutableMapOf<String, Long>()
    private var minIntervalMs: Long = 1000L

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        HermesSensorWatcherBridge.setServiceState(isRunning = true, sensors = emptyList())
        createNotificationChannel()
        promoteToForeground()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        minIntervalMs = intent?.getLongExtra(EXTRA_MIN_INTERVAL_MS, 0L)
            ?.takeIf { it > 0 }
            ?: HermesSensorWatcherBridge.persistedMinIntervalMs(applicationContext)
        HermesSensorWatcherBridge.resumePersistedWatcherIfRequested(applicationContext)
        registerEnabledSensors()
        promoteToForeground()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        unregisterSensors()
        running = false
        HermesSensorWatcherBridge.setServiceState(isRunning = false, sensors = emptyList())
        super.onDestroy()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val accuracy = accuracyBySensorType[event.sensor.type].orEmpty()
        val reading = HermesSensorWatcherBridge.readingFromSensorEvent(event, accuracy)
        val now = System.currentTimeMillis()
        val lastDispatch = lastDispatchBySensorType[reading.sensorType] ?: 0L
        if (now - lastDispatch < minIntervalMs) {
            return
        }
        lastDispatchBySensorType[reading.sensorType] = now
        HermesSensorWatcherBridge.dispatchSensorReadingJson(applicationContext, reading)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        sensor ?: return
        accuracyBySensorType[sensor.type] = HermesSensorWatcherBridge.accuracyLabel(accuracy)
    }

    private fun registerEnabledSensors() {
        unregisterSensors()
        val manager = sensorManager ?: return
        val registeredTypes = mutableListOf<String>()
        HermesSensorWatcherBridge.enabledSensorTypes(applicationContext).forEach { sensorType ->
            val androidType = HermesSensorWatcherBridge.androidSensorType(sensorType) ?: return@forEach
            val sensor = manager.getDefaultSensor(androidType) ?: return@forEach
            if (manager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)) {
                registeredSensors.add(sensor)
                registeredTypes.add(sensorType)
            }
        }
        HermesSensorWatcherBridge.setServiceState(isRunning = true, sensors = registeredTypes)
    }

    private fun unregisterSensors() {
        val manager = sensorManager ?: return
        registeredSensors.forEach { sensor ->
            manager.unregisterListener(this, sensor)
        }
        registeredSensors.clear()
        lastDispatchBySensorType.clear()
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
        val status = JSONObject(runCatching { HermesSensorWatcherBridge.statusJson(applicationContext) }.getOrDefault("{}"))
        val registeredCount = status.optJSONArray("registered_sensor_types")?.length() ?: 0
        val enabledCount = status.optInt("enabled_sensor_record_count", 0)
        val contentText = when {
            enabledCount <= 0 -> "No enabled sensor automations"
            registeredCount <= 0 -> "Waiting for supported device sensors"
            else -> "Watching $registeredCount sensor type(s)"
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle("Hermes sensor watcher")
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
            "Hermes sensor watcher",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps Hermes sensor automations active"
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val CHANNEL_ID = "hermes_sensor_watcher"
        private const val NOTIFICATION_ID = 7317
        private const val EXTRA_MIN_INTERVAL_MS = "min_interval_ms"

        @Volatile
        private var running: Boolean = false

        fun start(context: Context, minIntervalMs: Long): Boolean {
            val appContext = context.applicationContext
            val intent = Intent(appContext, HermesSensorWatcherService::class.java)
                .putExtra(EXTRA_MIN_INTERVAL_MS, minIntervalMs)
            val started = runCatching {
                ContextCompat.startForegroundService(appContext, intent)
            }.isSuccess
            running = started
            return started
        }

        fun startIfDesired(context: Context): Boolean {
            val appContext = context.applicationContext
            if (!HermesSensorWatcherBridge.resumePersistedWatcherIfRequested(appContext)) {
                return false
            }
            return start(appContext, HermesSensorWatcherBridge.persistedMinIntervalMs(appContext))
        }

        fun stop(context: Context) {
            running = false
            context.applicationContext.stopService(Intent(context.applicationContext, HermesSensorWatcherService::class.java))
        }

        fun isRunning(): Boolean = running
    }
}
