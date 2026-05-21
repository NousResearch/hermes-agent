package com.mobilefork.hermesagent.device

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

data class HermesLocationProviderEvent(
    val latitude: Double,
    val longitude: Double,
    val accuracyMeters: Double? = null,
    val provider: String = "",
    val name: String = "",
    val epochMs: Long = System.currentTimeMillis(),
) {
    fun toTriggerArguments(): JSONObject {
        val json = JSONObject()
            .put("latitude", latitude)
            .put("longitude", longitude)
            .put("location_provider", provider)
            .put("location_name", name)
            .put("location_epoch_ms", epochMs)
        if (accuracyMeters != null) {
            json.put("accuracy_meters", accuracyMeters)
        }
        return json
    }
}

object HermesLocationWatcherBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase().ifBlank { "location_watcher_status" }) {
            "location_watcher_status", "location_status", "watch_location_status", "watch_locations_status" -> statusJson(context)
            "start_location_watcher", "start_location_watch", "watch_location", "watch_locations" -> startJson(context, arguments)
            "stop_location_watcher", "stop_location_watch" -> stopJson(context)
            "scan_location", "scan_locations", "scan_location_once", "run_location_watch_once" -> scanOnceJson(context, arguments)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported location watcher action: $action")
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
    }

    fun statusJson(context: Context): String {
        val appContext = context.applicationContext
        return JSONObject()
            .put("success", true)
            .put("running", running.get())
            .put("durable_foreground_service", true)
            .put("foreground_service_running", HermesLocationWatcherService.isRunning())
            .put("watcher_desired", isWatcherDesired(appContext))
            .put("requires_location_permission", true)
            .put("location_permission_granted", hasLocationPermission(appContext))
            .put("enabled_location_record_count", enabledLocationRecordCount(appContext))
            .put("available_providers", JSONArray(availableProviders(appContext)))
            .put("requested_providers", JSONArray(persistedProviders(appContext)))
            .put("registered_providers", JSONArray(registeredProviders.get()))
            .put("min_interval_ms", persistedMinIntervalMs(appContext))
            .put("min_distance_meters", persistedMinDistanceMeters(appContext))
            .put("started_at_epoch_ms", startedAtEpochMs.get().takeIf { it > 0 } ?: JSONObject.NULL)
            .put("last_event_epoch_ms", persistedLastEventEpochMs(appContext) ?: JSONObject.NULL)
            .put("dispatched_event_count", persistedDispatchCount(appContext))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    fun startJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val enabledCount = enabledLocationRecordCount(appContext)
        if (enabledCount <= 0) {
            return JSONObject()
                .put("success", false)
                .put("error", "start_location_watcher requires at least one enabled location automation")
                .put("enabled_location_record_count", enabledCount)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        if (!hasLocationPermission(appContext)) {
            return JSONObject()
                .put("success", false)
                .put("error", "Location permission is not granted to Hermes Agent. Grant location access before starting the watcher.")
                .put("requires_location_permission", true)
                .put("location_permission_granted", false)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val intervalMs = longArgument(arguments, "min_interval_ms", "debounce_ms", "sample_interval_ms")
            ?.coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
            ?: DEFAULT_INTERVAL_MS
        val minDistanceMeters = doubleArgument(arguments, "min_distance_meters", "distance_meters", "distance")
            ?.coerceIn(MIN_DISTANCE_METERS, MAX_DISTANCE_METERS)
            ?: DEFAULT_MIN_DISTANCE_METERS
        val providers = providersFromArguments(appContext, arguments)
        if (providers.isEmpty()) {
            return JSONObject(statusJson(appContext))
                .put("success", false)
                .put("error", "No Android location providers are available for Hermes to watch")
                .toString()
        }
        persistWatcherRequest(appContext, intervalMs, minDistanceMeters, providers)
        val foregroundStarted = HermesLocationWatcherService.start(appContext, intervalMs, minDistanceMeters, providers)
        if (foregroundStarted) {
            setServiceState(isRunning = true, providers = emptyList())
        }
        return JSONObject(statusJson(appContext))
            .put("success", foregroundStarted)
            .put("foreground_service_started", foregroundStarted)
            .put("message", if (foregroundStarted) "Started durable Android location watcher" else "Android did not accept the foreground location watcher start request")
            .toString()
    }

    fun stopJson(context: Context): String {
        clearWatcherRequest(context.applicationContext)
        HermesLocationWatcherService.stop(context.applicationContext)
        setServiceState(isRunning = false, providers = emptyList())
        return JSONObject(statusJson(context))
            .put("success", true)
            .put("message", "Stopped location watcher")
            .toString()
    }

    fun scanOnceJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val injectedLocations = parseInjectedLocations(arguments)
        if (injectedLocations == null && !hasLocationPermission(appContext)) {
            return JSONObject()
                .put("success", false)
                .put("error", "Location permission is not granted to Hermes Agent.")
                .put("requires_location_permission", true)
                .put("location_permission_granted", false)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val enabledCount = enabledLocationRecordCount(appContext)
        if (enabledCount <= 0) {
            return JSONObject()
                .put("success", false)
                .put("error", "scan_location requires at least one enabled location automation")
                .put("enabled_location_record_count", enabledCount)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val providers = providersFromArguments(appContext, arguments)
        val events = injectedLocations ?: lastKnownLocationEvents(appContext, providers)
        val results = JSONArray()
        var scannedCount = 0
        var matchedCount = 0
        events.forEach { event ->
            scannedCount += 1
            val result = JSONObject(HermesAutomationBridge.runLocationTriggerJson(appContext, event.toTriggerArguments()))
            if (result.optInt("matched_count", 0) > 0) {
                matchedCount += result.optInt("matched_count", 0)
                results.put(result)
            }
            persistDispatch(appContext, event.epochMs)
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_LOCATION)
            .put("scanned_location_count", scannedCount)
            .put("matched_count", matchedCount)
            .put("results", results)
            .put("location_permission_granted", hasLocationPermission(appContext))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    internal fun dispatchLocationJson(context: Context, location: Location): String {
        val event = HermesLocationProviderEvent(
            latitude = location.latitude,
            longitude = location.longitude,
            accuracyMeters = if (location.hasAccuracy()) location.accuracy.toDouble() else null,
            provider = location.provider.orEmpty(),
            epochMs = location.time.takeIf { it > 0 } ?: System.currentTimeMillis(),
        )
        val result = JSONObject(HermesAutomationBridge.runLocationTriggerJson(context.applicationContext, event.toTriggerArguments()))
        persistDispatch(context.applicationContext, event.epochMs)
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_LOCATION)
            .put("matched_count", result.optInt("matched_count", 0))
            .put("result", result)
            .toString()
    }

    internal fun enabledLocationRecordCount(context: Context): Int {
        return HermesAutomationStore(context.applicationContext)
            .list()
            .count { record -> record.enabled && record.triggerType == TRIGGER_LOCATION }
    }

    internal fun hasLocationPermission(context: Context): Boolean {
        val appContext = context.applicationContext
        return ContextCompat.checkSelfPermission(appContext, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(appContext, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
    }

    internal fun persistedMinIntervalMs(context: Context): Long {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_MIN_INTERVAL_MS, DEFAULT_INTERVAL_MS)
            .coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
    }

    internal fun persistedMinDistanceMeters(context: Context): Double {
        return java.lang.Double.longBitsToDouble(
            context.applicationContext
                .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .getLong(PREF_MIN_DISTANCE_METERS, java.lang.Double.doubleToRawLongBits(DEFAULT_MIN_DISTANCE_METERS)),
        ).coerceIn(MIN_DISTANCE_METERS, MAX_DISTANCE_METERS)
    }

    internal fun persistedProviders(context: Context): List<String> {
        val stored = context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(PREF_PROVIDERS, "")
            .orEmpty()
            .split(',')
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .distinct()
        return stored.ifEmpty { defaultProviders(context) }
    }

    internal fun isWatcherDesired(context: Context): Boolean {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(PREF_DESIRED, false)
    }

    internal fun resumePersistedWatcherIfRequested(context: Context): Boolean {
        return isWatcherDesired(context.applicationContext) && enabledLocationRecordCount(context.applicationContext) > 0
    }

    internal fun setServiceState(isRunning: Boolean, providers: List<String>) {
        running.set(isRunning)
        registeredProviders.set(providers.distinct())
        if (isRunning && startedAtEpochMs.get() == 0L) {
            startedAtEpochMs.set(System.currentTimeMillis())
        }
        if (!isRunning) {
            startedAtEpochMs.set(0L)
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

    private fun providersFromArguments(context: Context, arguments: JSONObject): List<String> {
        val raw = arguments.opt("providers") ?: arguments.opt("location_providers") ?: arguments.opt("provider")
        val requested = when (raw) {
            is JSONArray -> (0 until raw.length()).mapNotNull { index -> raw.optString(index).trim().takeIf { it.isNotBlank() } }
            null -> persistedProviders(context)
            else -> raw.toString().split(',', ';', '|', '\n').map { it.trim() }.filter { it.isNotBlank() }
        }
        val available = availableProviders(context).toSet()
        return requested
            .map { canonicalProvider(it) }
            .filter { it.isNotBlank() }
            .filter { available.isEmpty() || it in available }
            .distinct()
            .take(MAX_PROVIDER_COUNT)
    }

    private fun defaultProviders(context: Context): List<String> {
        val available = availableProviders(context)
        val preferred = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER)
            .filter { it in available }
        return preferred.ifEmpty { available.take(MAX_PROVIDER_COUNT) }
    }

    private fun availableProviders(context: Context): List<String> {
        val manager = context.applicationContext.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            ?: return emptyList()
        return runCatching { manager.allProviders.orEmpty().map(::canonicalProvider).filter { it.isNotBlank() }.distinct() }
            .getOrDefault(emptyList())
    }

    private fun lastKnownLocationEvents(context: Context, providers: List<String>): List<HermesLocationProviderEvent> {
        val manager = context.applicationContext.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            ?: return emptyList()
        return providers.mapNotNull { provider ->
            val location = runCatching { manager.getLastKnownLocation(provider) }.getOrNull() ?: return@mapNotNull null
            HermesLocationProviderEvent(
                latitude = location.latitude,
                longitude = location.longitude,
                accuracyMeters = if (location.hasAccuracy()) location.accuracy.toDouble() else null,
                provider = location.provider.orEmpty().ifBlank { provider },
                epochMs = location.time.takeIf { it > 0 } ?: System.currentTimeMillis(),
            )
        }
    }

    private fun parseInjectedLocations(arguments: JSONObject): List<HermesLocationProviderEvent>? {
        val array = when {
            arguments.has("locations") && !arguments.isNull("locations") -> arguments.optJSONArray("locations")
            arguments.has("location_events") && !arguments.isNull("location_events") -> arguments.optJSONArray("location_events")
            arguments.has("latitude") || arguments.has("lat") -> JSONArray().put(arguments)
            else -> null
        } ?: return null
        val events = mutableListOf<HermesLocationProviderEvent>()
        for (index in 0 until array.length()) {
            val raw = array.optJSONObject(index) ?: continue
            val latitude = doubleArgument(raw, "latitude", "lat", "location_latitude") ?: continue
            val longitude = doubleArgument(raw, "longitude", "lon", "lng", "location_longitude") ?: continue
            if (latitude !in -90.0..90.0 || longitude !in -180.0..180.0) {
                continue
            }
            events += HermesLocationProviderEvent(
                latitude = latitude,
                longitude = longitude,
                accuracyMeters = doubleArgument(raw, "accuracy_meters", "accuracy", "location_accuracy_meters")?.takeIf { it >= 0.0 },
                provider = raw.optString("location_provider").ifBlank { raw.optString("provider") }.take(MAX_LOCATION_TEXT_CHARS),
                name = raw.optString("location_name").ifBlank { raw.optString("place_name") }.ifBlank { raw.optString("name") }.take(MAX_LOCATION_TEXT_CHARS),
                epochMs = longArgument(raw, "location_epoch_ms", "epoch_ms", "time_ms", "timestamp_ms") ?: System.currentTimeMillis(),
            )
        }
        return events
    }

    private fun persistWatcherRequest(context: Context, intervalMs: Long, minDistanceMeters: Double, providers: List<String>) {
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, true)
            .putLong(PREF_MIN_INTERVAL_MS, intervalMs.coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS))
            .putLong(PREF_MIN_DISTANCE_METERS, java.lang.Double.doubleToRawLongBits(minDistanceMeters.coerceIn(MIN_DISTANCE_METERS, MAX_DISTANCE_METERS)))
            .putString(PREF_PROVIDERS, providers.joinToString(","))
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

    private fun canonicalProvider(raw: String): String {
        return raw.trim().lowercase().replace(' ', '_').replace('-', '_')
    }

    private fun longArgument(arguments: JSONObject, vararg keys: String): Long? {
        keys.forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            return when (val value = arguments.opt(key)) {
                is Number -> value.toLong()
                else -> value?.toString()?.trim()?.toLongOrNull()
            }
        }
        return null
    }

    private fun doubleArgument(arguments: JSONObject, vararg keys: String): Double? {
        keys.forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            return when (val value = arguments.opt(key)) {
                is Number -> value.toDouble()
                else -> value?.toString()?.trim()?.toDoubleOrNull()
            }
        }
        return null
    }

    private val running = AtomicBoolean(false)
    private val registeredProviders = AtomicReference<List<String>>(emptyList())
    private val startedAtEpochMs = AtomicLong(0)

    private val ACTIONS = listOf(
        "location_watcher_status",
        "start_location_watcher",
        "stop_location_watcher",
        "scan_location",
    )

    private const val DEFAULT_INTERVAL_MS = 30_000L
    private const val MIN_INTERVAL_MS = 1_000L
    private const val MAX_INTERVAL_MS = 3_600_000L
    private const val DEFAULT_MIN_DISTANCE_METERS = 25.0
    private const val MIN_DISTANCE_METERS = 0.0
    private const val MAX_DISTANCE_METERS = 10_000.0
    private const val MAX_PROVIDER_COUNT = 8
    private const val MAX_LOCATION_TEXT_CHARS = 160
    private const val PREFS_NAME = "hermes_location_watcher"
    private const val PREF_DESIRED = "desired"
    private const val PREF_MIN_INTERVAL_MS = "min_interval_ms"
    private const val PREF_MIN_DISTANCE_METERS = "min_distance_meters"
    private const val PREF_PROVIDERS = "providers"
    private const val PREF_LAST_EVENT_EPOCH_MS = "last_event_epoch_ms"
    private const val PREF_DISPATCH_COUNT = "dispatch_count"
}
