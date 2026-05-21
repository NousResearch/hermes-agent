package com.mobilefork.hermesagent.device

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorManager
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.sqrt

data class HermesSensorReading(
    val sensorType: String,
    val sensorName: String = sensorType,
    val event: String = SENSOR_EVENT_CHANGED,
    val values: List<Double> = emptyList(),
    val accuracy: String = "",
    val timestampNanos: Long = 0L,
)

object HermesSensorWatcherBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase().ifBlank { "sensor_watcher_status" }) {
            "sensor_watcher_status", "sensor_status", "watch_sensor_status", "watch_sensors_status" -> statusJson(context)
            "start_sensor_watcher", "start_sensor_watch", "watch_sensors", "watch_sensor" -> startJson(context, arguments)
            "stop_sensor_watcher", "stop_sensor_watch" -> stopJson(context)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported sensor watcher action: $action")
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
    }

    fun statusJson(context: Context): String {
        val appContext = context.applicationContext
        val enabledTypes = enabledSensorTypes(appContext)
        return JSONObject()
            .put("success", true)
            .put("running", running.get())
            .put("durable_foreground_service", true)
            .put("foreground_service_running", HermesSensorWatcherService.isRunning())
            .put("watcher_desired", isWatcherDesired(appContext))
            .put("requires_shizuku", false)
            .put("enabled_sensor_record_count", enabledSensorRecordCount(appContext))
            .put("enabled_watched_sensor_types", JSONArray(enabledTypes))
            .put("registered_sensor_types", JSONArray(registeredSensorTypes.get()))
            .put("available_sensor_types", JSONArray(availableSensorTypes(appContext)))
            .put("supported_sensor_types", JSONArray(SUPPORTED_SENSOR_TYPES))
            .put("min_interval_ms", persistedMinIntervalMs(appContext))
            .put("started_at_epoch_ms", startedAtEpochMs.get().takeIf { it > 0 } ?: JSONObject.NULL)
            .put("last_event_epoch_ms", persistedLastEventEpochMs(appContext)?.let { it } ?: JSONObject.NULL)
            .put("dispatched_event_count", persistedDispatchCount(appContext))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    fun startJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val enabledCount = enabledSensorRecordCount(appContext)
        val watchedTypes = enabledSensorTypes(appContext)
        if (enabledCount <= 0 || watchedTypes.isEmpty()) {
            return JSONObject()
                .put("success", false)
                .put("error", "start_sensor_watcher requires at least one enabled sensor automation with a supported sensor_type")
                .put("enabled_sensor_record_count", enabledCount)
                .put("enabled_watched_sensor_types", JSONArray(watchedTypes))
                .put("supported_sensor_types", JSONArray(SUPPORTED_SENSOR_TYPES))
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val intervalMs = longArgument(arguments, "min_interval_ms", "debounce_ms", "sample_interval_ms")
            ?.coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
            ?: DEFAULT_INTERVAL_MS
        persistWatcherRequest(appContext, intervalMs)
        val foregroundStarted = HermesSensorWatcherService.start(appContext, intervalMs)
        if (foregroundStarted) {
            setServiceState(isRunning = true, sensors = emptyList())
        }
        return JSONObject(statusJson(appContext))
            .put("success", foregroundStarted)
            .put("foreground_service_started", foregroundStarted)
            .put(
                "message",
                if (foregroundStarted) {
                    "Started durable Android sensor watcher"
                } else {
                    "Android did not accept the foreground sensor watcher start request"
                },
            )
            .toString()
    }

    fun stopJson(context: Context): String {
        clearWatcherRequest(context.applicationContext)
        HermesSensorWatcherService.stop(context.applicationContext)
        setServiceState(isRunning = false, sensors = emptyList())
        return JSONObject(statusJson(context))
            .put("success", true)
            .put("message", "Stopped sensor watcher")
            .toString()
    }

    internal fun dispatchSensorReadingJson(context: Context, reading: HermesSensorReading): String {
        val appContext = context.applicationContext
        val normalizedType = canonicalSensorType(reading.sensorType)
        if (normalizedType.isBlank()) {
            return JSONObject()
                .put("success", false)
                .put("error", "sensor reading requires a supported sensor_type")
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val normalizedReading = reading.copy(
            sensorType = normalizedType,
            sensorName = reading.sensorName.ifBlank { normalizedType }.take(MAX_SENSOR_TEXT_CHARS),
            event = reading.event.ifBlank { SENSOR_EVENT_CHANGED }.trim().lowercase(Locale.US),
        )
        val selectedSamples = selectedSamplesForReading(appContext, normalizedReading)
        val events = eventNamesForReading(normalizedReading)
        val results = JSONArray()
        var matchedCount = 0
        var dispatchedCount = 0
        events.forEach { eventName ->
            selectedSamples.forEach { sample ->
                val triggerResult = JSONObject(
                    HermesAutomationBridge.runSensorTriggerJson(
                        appContext,
                        JSONObject()
                            .put("sensor_type", normalizedReading.sensorType)
                            .put("sensor_name", normalizedReading.sensorName)
                            .put("sensor_event", eventName)
                            .put("value_name", sample.name)
                            .put("sensor_value", sample.value)
                            .put("sensor_unit", unitForSensorType(normalizedReading.sensorType))
                            .put("sensor_accuracy", normalizedReading.accuracy),
                    ),
                )
                dispatchedCount += 1
                val matches = triggerResult.optInt("matched_count", 0)
                matchedCount += matches
                if (matches > 0) {
                    results.put(triggerResult)
                }
            }
        }
        val now = System.currentTimeMillis()
        persistDispatch(appContext, now)
        lastEventEpochMs.set(now)
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_SENSOR)
            .put("sensor_type", normalizedReading.sensorType)
            .put("sensor_name", normalizedReading.sensorName)
            .put("sensor_event_names", JSONArray(events))
            .put("sample_count", selectedSamples.size)
            .put("dispatched_event_count", dispatchedCount)
            .put("matched_count", matchedCount)
            .put("last_event_epoch_ms", now)
            .put("results", results)
            .toString()
    }

    internal fun readingFromSensorEvent(event: SensorEvent, accuracy: String): HermesSensorReading {
        val sensorType = canonicalSensorType(event.sensor)
        return HermesSensorReading(
            sensorType = sensorType,
            sensorName = event.sensor.name.orEmpty(),
            event = SENSOR_EVENT_CHANGED,
            values = event.values.map { it.toDouble() },
            accuracy = accuracy,
            timestampNanos = event.timestamp,
        )
    }

    internal fun enabledSensorTypes(context: Context): List<String> {
        return HermesAutomationStore(context.applicationContext)
            .list()
            .asSequence()
            .filter { record -> record.enabled && record.triggerType == TRIGGER_SENSOR }
            .mapNotNull { record -> sensorTypeFromTriggerData(record.triggerData) }
            .filter { sensorType -> androidSensorType(sensorType) != null }
            .distinct()
            .toList()
    }

    internal fun enabledSensorRecordCount(context: Context): Int {
        return HermesAutomationStore(context.applicationContext)
            .list()
            .count { record -> record.enabled && record.triggerType == TRIGGER_SENSOR }
    }

    internal fun androidSensorType(sensorType: String): Int? {
        return when (canonicalSensorType(sensorType)) {
            "accelerometer" -> Sensor.TYPE_ACCELEROMETER
            "gyroscope" -> Sensor.TYPE_GYROSCOPE
            "magnetic_field" -> Sensor.TYPE_MAGNETIC_FIELD
            "light" -> Sensor.TYPE_LIGHT
            "proximity" -> Sensor.TYPE_PROXIMITY
            "gravity" -> Sensor.TYPE_GRAVITY
            "linear_acceleration" -> Sensor.TYPE_LINEAR_ACCELERATION
            "rotation_vector" -> Sensor.TYPE_ROTATION_VECTOR
            "step_counter" -> Sensor.TYPE_STEP_COUNTER
            "pressure" -> Sensor.TYPE_PRESSURE
            "ambient_temperature" -> Sensor.TYPE_AMBIENT_TEMPERATURE
            "relative_humidity" -> Sensor.TYPE_RELATIVE_HUMIDITY
            else -> null
        }
    }

    internal fun canonicalSensorType(sensor: Sensor): String {
        return when (sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> "accelerometer"
            Sensor.TYPE_GYROSCOPE -> "gyroscope"
            Sensor.TYPE_MAGNETIC_FIELD -> "magnetic_field"
            Sensor.TYPE_LIGHT -> "light"
            Sensor.TYPE_PROXIMITY -> "proximity"
            Sensor.TYPE_GRAVITY -> "gravity"
            Sensor.TYPE_LINEAR_ACCELERATION -> "linear_acceleration"
            Sensor.TYPE_ROTATION_VECTOR -> "rotation_vector"
            Sensor.TYPE_STEP_COUNTER -> "step_counter"
            Sensor.TYPE_PRESSURE -> "pressure"
            Sensor.TYPE_AMBIENT_TEMPERATURE -> "ambient_temperature"
            Sensor.TYPE_RELATIVE_HUMIDITY -> "relative_humidity"
            else -> canonicalSensorType(sensor.stringType.orEmpty())
        }
    }

    internal fun accuracyLabel(accuracy: Int): String {
        return when (accuracy) {
            SensorManager.SENSOR_STATUS_ACCURACY_HIGH -> "high"
            SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> "medium"
            SensorManager.SENSOR_STATUS_ACCURACY_LOW -> "low"
            SensorManager.SENSOR_STATUS_UNRELIABLE -> "unreliable"
            else -> accuracy.toString()
        }
    }

    internal fun persistedMinIntervalMs(context: Context): Long {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_MIN_INTERVAL_MS, DEFAULT_INTERVAL_MS)
            .coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
    }

    internal fun isWatcherDesired(context: Context): Boolean {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(PREF_DESIRED, false)
    }

    internal fun resumePersistedWatcherIfRequested(context: Context): Boolean {
        return isWatcherDesired(context.applicationContext) && enabledSensorTypes(context.applicationContext).isNotEmpty()
    }

    internal fun setServiceState(isRunning: Boolean, sensors: List<String>) {
        running.set(isRunning)
        registeredSensorTypes.set(sensors.distinct())
        if (isRunning && startedAtEpochMs.get() == 0L) {
            startedAtEpochMs.set(System.currentTimeMillis())
        }
        if (!isRunning) {
            startedAtEpochMs.set(0)
        }
    }

    internal fun persistedLastEventEpochMs(context: Context): Long? {
        val value = context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_LAST_EVENT_EPOCH_MS, 0L)
        return value.takeIf { it > 0 }
    }

    internal fun persistedDispatchCount(context: Context): Long {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_DISPATCH_COUNT, 0L)
    }

    private fun canonicalSensorType(raw: String): String {
        val normalized = raw.trim()
            .lowercase(Locale.US)
            .replace('-', '_')
            .replace(' ', '_')
        return when (normalized) {
            "accel", "accelerometer", "motion" -> "accelerometer"
            "gyro", "gyroscope" -> "gyroscope"
            "magnetometer", "magnetic", "magnetic_field" -> "magnetic_field"
            "lux", "light" -> "light"
            "proximity" -> "proximity"
            "gravity" -> "gravity"
            "linear_accel", "linear_acceleration" -> "linear_acceleration"
            "rotation", "rotation_vector", "orientation" -> "rotation_vector"
            "step", "steps", "step_counter" -> "step_counter"
            "barometer", "pressure" -> "pressure"
            "temperature", "ambient_temperature" -> "ambient_temperature"
            "humidity", "relative_humidity" -> "relative_humidity"
            else -> normalized
        }
    }

    private fun availableSensorTypes(context: Context): List<String> {
        val manager = context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
            ?: return emptyList()
        return manager.getSensorList(Sensor.TYPE_ALL)
            .map { sensor -> canonicalSensorType(sensor) }
            .filter { sensorType -> androidSensorType(sensorType) != null }
            .distinct()
            .sorted()
    }

    private fun sensorTypeFromTriggerData(triggerData: String): String? {
        val filters = runCatching { JSONObject(triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
        val raw = filters.optString("sensor_type").trim()
        if (raw.isBlank() || raw.startsWith("%") || raw.startsWith("{{")) {
            return null
        }
        return canonicalSensorType(raw)
    }

    private fun selectedSamplesForReading(context: Context, reading: HermesSensorReading): List<SensorSample> {
        val samples = samplesForReading(reading)
        if (samples.isEmpty()) {
            return emptyList()
        }
        val requestedNames = requestedValueNamesForType(context, reading.sensorType)
        if (requestedNames.isNotEmpty()) {
            val selected = samples.filter { sample -> sample.name.lowercase(Locale.US) in requestedNames }
            if (selected.isNotEmpty()) {
                return selected
            }
        }
        return samples.firstOrNull { sample -> sample.name == "magnitude" }
            ?.let { listOf(it) }
            ?: samples.take(1)
    }

    private fun samplesForReading(reading: HermesSensorReading): List<SensorSample> {
        val axisNames = axisNamesForSensor(reading.sensorType, reading.values.size)
        val samples = reading.values.mapIndexed { index, value ->
            SensorSample(axisNames.getOrElse(index) { "value_${index + 1}" }, value)
        }.toMutableList()
        if (reading.values.size >= 2 && reading.sensorType in MAGNITUDE_SENSOR_TYPES) {
            val magnitude = sqrt(reading.values.sumOf { value -> value * value })
            samples.add(SensorSample("magnitude", magnitude))
        }
        return samples
    }

    private fun requestedValueNamesForType(context: Context, sensorType: String): Set<String> {
        val values = linkedSetOf<String>()
        HermesAutomationStore(context.applicationContext).list()
            .asSequence()
            .filter { record -> record.enabled && record.triggerType == TRIGGER_SENSOR }
            .forEach { record ->
                val filters = runCatching { JSONObject(record.triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
                val savedType = filters.optString("sensor_type").trim()
                if (savedType.isNotBlank() && canonicalSensorType(savedType) != sensorType) {
                    return@forEach
                }
                val valueName = filters.optString("value_name").trim().lowercase(Locale.US)
                if (valueName.isNotBlank() && !valueName.startsWith("%") && !valueName.startsWith("{{")) {
                    values.add(valueName)
                }
            }
        return values
    }

    private fun eventNamesForReading(reading: HermesSensorReading): List<String> {
        val events = linkedSetOf(reading.event.ifBlank { SENSOR_EVENT_CHANGED }.lowercase(Locale.US))
        val magnitude = if (reading.values.size >= 2 && reading.sensorType in MAGNITUDE_SENSOR_TYPES) {
            sqrt(reading.values.sumOf { value -> value * value })
        } else {
            null
        }
        if (reading.sensorType == "accelerometer" && magnitude != null && magnitude >= SHAKE_MAGNITUDE_THRESHOLD) {
            events.add("shake")
        }
        return events.toList()
    }

    private fun axisNamesForSensor(sensorType: String, valueCount: Int): List<String> {
        return when (sensorType) {
            "accelerometer", "gyroscope", "magnetic_field", "gravity", "linear_acceleration" -> listOf("x", "y", "z")
            "rotation_vector" -> listOf("x", "y", "z", "cos", "heading_accuracy")
            else -> List(valueCount.coerceAtLeast(1)) { index -> if (index == 0) "value" else "value_${index + 1}" }
        }
    }

    private fun unitForSensorType(sensorType: String): String {
        return when (sensorType) {
            "accelerometer", "gravity", "linear_acceleration" -> "m/s^2"
            "gyroscope" -> "rad/s"
            "magnetic_field" -> "uT"
            "light" -> "lx"
            "proximity" -> "cm"
            "pressure" -> "hPa"
            "ambient_temperature" -> "C"
            "relative_humidity" -> "%"
            "step_counter" -> "steps"
            else -> ""
        }
    }

    private fun persistWatcherRequest(context: Context, intervalMs: Long) {
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, true)
            .putLong(PREF_MIN_INTERVAL_MS, intervalMs.coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS))
            .apply()
    }

    private fun clearWatcherRequest(context: Context) {
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, false)
            .apply()
    }

    private fun persistDispatch(context: Context, epochMs: Long) {
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit()
            .putLong(PREF_LAST_EVENT_EPOCH_MS, epochMs)
            .putLong(PREF_DISPATCH_COUNT, prefs.getLong(PREF_DISPATCH_COUNT, 0L) + 1L)
            .apply()
    }

    private fun longArgument(arguments: JSONObject, vararg keys: String): Long? {
        keys.forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            val value = arguments.opt(key)
            return when (value) {
                is Number -> value.toLong()
                else -> value?.toString()?.trim()?.toLongOrNull()
            }
        }
        return null
    }

    private data class SensorSample(val name: String, val value: Double)

    private val running = AtomicBoolean(false)
    private val registeredSensorTypes = AtomicReference<List<String>>(emptyList())
    private val startedAtEpochMs = AtomicLong(0)
    private val lastEventEpochMs = AtomicLong(0)

    private val ACTIONS = listOf(
        "sensor_watcher_status",
        "start_sensor_watcher",
        "stop_sensor_watcher",
    )
    private val SUPPORTED_SENSOR_TYPES = listOf(
        "accelerometer",
        "gyroscope",
        "magnetic_field",
        "light",
        "proximity",
        "gravity",
        "linear_acceleration",
        "rotation_vector",
        "step_counter",
        "pressure",
        "ambient_temperature",
        "relative_humidity",
    )
    private val MAGNITUDE_SENSOR_TYPES = setOf(
        "accelerometer",
        "gyroscope",
        "magnetic_field",
        "gravity",
        "linear_acceleration",
        "rotation_vector",
    )

    private const val DEFAULT_INTERVAL_MS = 1000L
    private const val MIN_INTERVAL_MS = 250L
    private const val MAX_INTERVAL_MS = 60000L
    private const val SHAKE_MAGNITUDE_THRESHOLD = 12.0
    private const val MAX_SENSOR_TEXT_CHARS = 160
    private const val PREFS_NAME = "hermes_sensor_watcher"
    private const val PREF_DESIRED = "desired"
    private const val PREF_MIN_INTERVAL_MS = "min_interval_ms"
    private const val PREF_LAST_EVENT_EPOCH_MS = "last_event_epoch_ms"
    private const val PREF_DISPATCH_COUNT = "dispatch_count"
}

private const val SENSOR_EVENT_CHANGED = "changed"
