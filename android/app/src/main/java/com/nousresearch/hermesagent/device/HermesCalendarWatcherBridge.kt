package com.nousresearch.hermesagent.device

import android.Manifest
import android.content.ContentUris
import android.content.Context
import android.content.pm.PackageManager
import android.provider.CalendarContract
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

data class HermesCalendarProviderEvent(
    val eventId: String,
    val calendarName: String,
    val title: String,
    val description: String,
    val location: String,
    val beginMillis: Long,
    val endMillis: Long,
) {
    fun toTriggerArguments(): JSONObject {
        return JSONObject()
            .put("calendar_name", calendarName)
            .put("calendar_title", title)
            .put("calendar_description", description)
            .put("calendar_location", location)
            .put("calendar_event_id", eventId)
            .put("calendar_begin_epoch_ms", beginMillis)
            .put("calendar_end_epoch_ms", endMillis)
    }

    fun signature(): String = "$eventId:$beginMillis:$endMillis:$title"
}

object HermesCalendarWatcherBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase().ifBlank { "calendar_watcher_status" }) {
            "calendar_watcher_status", "calendar_status", "watch_calendar_status" -> statusJson(context)
            "start_calendar_watcher", "start_calendar_watch", "watch_calendar", "watch_calendar_events" -> startJson(context, arguments)
            "stop_calendar_watcher", "stop_calendar_watch" -> stopJson(context)
            "scan_calendar_events", "scan_calendar", "run_calendar_watch_once" -> scanOnceJson(context, arguments)
            "reset_calendar_watcher_cursor", "reset_calendar_cursor", "clear_calendar_watcher_cursor" -> resetCursorJson(context)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported calendar watcher action: $action")
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
            .put("foreground_service_running", HermesCalendarWatcherService.isRunning())
            .put("watcher_desired", isWatcherDesired(appContext))
            .put("requires_calendar_permission", true)
            .put("calendar_permission_granted", hasCalendarPermission(appContext))
            .put("enabled_calendar_record_count", enabledCalendarRecordCount(appContext))
            .put("scan_interval_seconds", persistedScanIntervalSeconds(appContext))
            .put("lookahead_minutes", persistedLookaheadMinutes(appContext))
            .put("lookback_minutes", persistedLookbackMinutes(appContext))
            .put("started_at_epoch_ms", startedAtEpochMs.get().takeIf { it > 0 } ?: JSONObject.NULL)
            .put("last_event_epoch_ms", persistedLastEventEpochMs(appContext) ?: JSONObject.NULL)
            .put("recent_event_signature_count", recentEventSignatureCount(appContext))
            .put("dispatched_event_count", persistedDispatchCount(appContext))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    fun startJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val enabledCount = enabledCalendarRecordCount(appContext)
        if (enabledCount <= 0) {
            return JSONObject()
                .put("success", false)
                .put("error", "start_calendar_watcher requires at least one enabled calendar_event automation")
                .put("enabled_calendar_record_count", enabledCount)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        if (!hasCalendarPermission(appContext)) {
            return JSONObject()
                .put("success", false)
                .put("error", "Calendar permission is not granted to Hermes Agent. Grant calendar access before starting the watcher.")
                .put("requires_calendar_permission", true)
                .put("calendar_permission_granted", false)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        if (arguments.optBoolean("reset_cursor", false)) {
            resetCursor(appContext)
        }
        val intervalSeconds = intArgument(arguments, "scan_interval_seconds", "poll_interval_seconds")
            ?.coerceIn(MIN_SCAN_INTERVAL_SECONDS, MAX_SCAN_INTERVAL_SECONDS)
            ?: DEFAULT_SCAN_INTERVAL_SECONDS
        val lookaheadMinutes = intArgument(arguments, "lookahead_minutes", "future_minutes")
            ?.coerceIn(MIN_LOOKAHEAD_MINUTES, MAX_LOOKAHEAD_MINUTES)
            ?: DEFAULT_LOOKAHEAD_MINUTES
        val lookbackMinutes = intArgument(arguments, "lookback_minutes", "past_minutes")
            ?.coerceIn(MIN_LOOKBACK_MINUTES, MAX_LOOKBACK_MINUTES)
            ?: DEFAULT_LOOKBACK_MINUTES
        persistWatcherRequest(appContext, intervalSeconds, lookaheadMinutes, lookbackMinutes)
        val wasAlreadyRunning = startWorker(appContext, intervalSeconds, lookaheadMinutes, lookbackMinutes)
        val foregroundStarted = HermesCalendarWatcherService.start(appContext, intervalSeconds, lookaheadMinutes, lookbackMinutes)
        if (wasAlreadyRunning) {
            return JSONObject(statusJson(appContext))
                .put("success", true)
                .put("message", "Calendar watcher is already running")
                .put("foreground_service_started", foregroundStarted)
                .toString()
        }
        return JSONObject(statusJson(appContext))
            .put("success", true)
            .put("foreground_service_started", foregroundStarted)
            .put(
                "message",
                if (foregroundStarted) {
                    "Started durable Android calendar watcher"
                } else {
                    "Started process-lifetime Android calendar watcher; foreground service start was not accepted by Android"
                },
            )
            .toString()
    }

    fun stopJson(context: Context): String {
        clearWatcherRequest(context.applicationContext)
        val wasRunning = stopWorker()
        HermesCalendarWatcherService.stop(context.applicationContext)
        return JSONObject(statusJson(context))
            .put("success", true)
            .put("stopped", wasRunning)
            .put("message", if (wasRunning) "Stopped calendar watcher" else "Calendar watcher was not running")
            .toString()
    }

    fun resetCursorJson(context: Context): String {
        resetCursor(context.applicationContext)
        return JSONObject(statusJson(context))
            .put("success", true)
            .put("message", "Reset calendar watcher scan cursor")
            .toString()
    }

    fun scanOnceJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val injectedEvents = parseInjectedEvents(arguments)
        if (injectedEvents == null && !hasCalendarPermission(appContext)) {
            return JSONObject()
                .put("success", false)
                .put("error", "Calendar permission is not granted to Hermes Agent.")
                .put("requires_calendar_permission", true)
                .put("calendar_permission_granted", false)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val enabledCount = enabledCalendarRecordCount(appContext)
        if (enabledCount <= 0) {
            return JSONObject()
                .put("success", false)
                .put("error", "scan_calendar_events requires at least one enabled calendar_event automation")
                .put("enabled_calendar_record_count", enabledCount)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        if (arguments.optBoolean("reset_cursor", false)) {
            resetCursor(appContext)
        }
        val useCursor = arguments.optBoolean("use_cursor", true)
        val lookaheadMinutes = intArgument(arguments, "lookahead_minutes", "future_minutes")
            ?.coerceIn(MIN_LOOKAHEAD_MINUTES, MAX_LOOKAHEAD_MINUTES)
            ?: persistedLookaheadMinutes(appContext)
        val lookbackMinutes = intArgument(arguments, "lookback_minutes", "past_minutes")
            ?.coerceIn(MIN_LOOKBACK_MINUTES, MAX_LOOKBACK_MINUTES)
            ?: persistedLookbackMinutes(appContext)
        val maxEvents = intArgument(arguments, "max_events", "limit")
            ?.coerceIn(MIN_MAX_EVENTS, MAX_MAX_EVENTS)
            ?: DEFAULT_MAX_EVENTS
        val events = injectedEvents ?: queryCalendarEvents(appContext, lookaheadMinutes, lookbackMinutes, maxEvents)
        val recent = recentEventSignatures(appContext).toMutableSet()
        val dispatched = JSONArray()
        var scannedCount = 0
        var skippedRecentCount = 0
        var matchedCount = 0
        events.take(maxEvents).forEach { event ->
            scannedCount += 1
            val signature = event.signature()
            if (useCursor && recent.contains(signature)) {
                skippedRecentCount += 1
                return@forEach
            }
            recent += signature
            val result = JSONObject(HermesAutomationBridge.runCalendarEventTriggerJson(appContext, event.toTriggerArguments()))
            if (result.optInt("matched_count", 0) > 0) {
                matchedCount += result.optInt("matched_count", 0)
                dispatched.put(result)
            }
            persistLastEvent(appContext, event.beginMillis)
        }
        persistRecentEventSignatures(appContext, recent.toList().takeLast(MAX_RECENT_SIGNATURES))
        persistDispatchCount(appContext, dispatched.length())
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_CALENDAR_EVENT)
            .put("scanned_event_count", scannedCount)
            .put("skipped_recent_event_count", skippedRecentCount)
            .put("matched_count", matchedCount)
            .put("results", dispatched)
            .put("calendar_permission_granted", hasCalendarPermission(appContext))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    internal fun startWorker(
        context: Context,
        intervalSeconds: Int = persistedScanIntervalSeconds(context),
        lookaheadMinutes: Int = persistedLookaheadMinutes(context),
        lookbackMinutes: Int = persistedLookbackMinutes(context),
    ): Boolean {
        val appContext = context.applicationContext
        scanIntervalSeconds.set(intervalSeconds.coerceIn(MIN_SCAN_INTERVAL_SECONDS, MAX_SCAN_INTERVAL_SECONDS).toLong())
        configuredLookaheadMinutes.set(lookaheadMinutes.coerceIn(MIN_LOOKAHEAD_MINUTES, MAX_LOOKAHEAD_MINUTES).toLong())
        configuredLookbackMinutes.set(lookbackMinutes.coerceIn(MIN_LOOKBACK_MINUTES, MAX_LOOKBACK_MINUTES).toLong())
        if (running.getAndSet(true)) {
            return true
        }
        startedAtEpochMs.set(System.currentTimeMillis())
        val worker = Thread {
            while (running.get()) {
                runCatching {
                    scanOnceJson(
                        appContext,
                        JSONObject()
                            .put("lookahead_minutes", configuredLookaheadMinutes.get())
                            .put("lookback_minutes", configuredLookbackMinutes.get())
                            .put("watcher_loop", true),
                    )
                }
                try {
                    Thread.sleep(scanIntervalSeconds.get() * 1000L)
                } catch (_: InterruptedException) {
                    // Re-check running and exit quickly after stop.
                }
            }
        }.apply {
            name = "hermes-calendar-watcher"
            isDaemon = true
            start()
        }
        workerThread.set(worker)
        return false
    }

    internal fun stopWorker(): Boolean {
        val wasRunning = running.getAndSet(false)
        workerThread.getAndSet(null)?.interrupt()
        startedAtEpochMs.set(0)
        return wasRunning
    }

    internal fun enabledCalendarRecordCount(context: Context): Int {
        return HermesAutomationStore(context.applicationContext)
            .list()
            .count { record -> record.enabled && record.triggerType == TRIGGER_CALENDAR_EVENT }
    }

    internal fun hasCalendarPermission(context: Context): Boolean {
        return ContextCompat.checkSelfPermission(
            context.applicationContext,
            Manifest.permission.READ_CALENDAR,
        ) == PackageManager.PERMISSION_GRANTED
    }

    internal fun isWatcherDesired(context: Context): Boolean {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(PREF_DESIRED, false)
    }

    internal fun persistedScanIntervalSeconds(context: Context): Int {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_SCAN_INTERVAL_SECONDS, DEFAULT_SCAN_INTERVAL_SECONDS.toLong())
            .toInt()
            .coerceIn(MIN_SCAN_INTERVAL_SECONDS, MAX_SCAN_INTERVAL_SECONDS)
    }

    internal fun persistedLookaheadMinutes(context: Context): Int {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_LOOKAHEAD_MINUTES, DEFAULT_LOOKAHEAD_MINUTES.toLong())
            .toInt()
            .coerceIn(MIN_LOOKAHEAD_MINUTES, MAX_LOOKAHEAD_MINUTES)
    }

    internal fun persistedLookbackMinutes(context: Context): Int {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_LOOKBACK_MINUTES, DEFAULT_LOOKBACK_MINUTES.toLong())
            .toInt()
            .coerceIn(MIN_LOOKBACK_MINUTES, MAX_LOOKBACK_MINUTES)
    }

    internal fun resumePersistedWatcherIfRequested(context: Context): Boolean {
        val appContext = context.applicationContext
        if (!isWatcherDesired(appContext)) {
            return false
        }
        startWorker(
            appContext,
            persistedScanIntervalSeconds(appContext),
            persistedLookaheadMinutes(appContext),
            persistedLookbackMinutes(appContext),
        )
        return true
    }

    private fun queryCalendarEvents(
        context: Context,
        lookaheadMinutes: Int,
        lookbackMinutes: Int,
        maxEvents: Int,
    ): List<HermesCalendarProviderEvent> {
        val now = System.currentTimeMillis()
        val begin = now - lookbackMinutes * 60_000L
        val end = now + lookaheadMinutes * 60_000L
        val uri = CalendarContract.Instances.CONTENT_URI.buildUpon().also { builder ->
            ContentUris.appendId(builder, begin)
            ContentUris.appendId(builder, end)
        }.build()
        val events = mutableListOf<HermesCalendarProviderEvent>()
        context.contentResolver.query(
            uri,
            CALENDAR_PROJECTION,
            null,
            null,
            "${CalendarContract.Instances.BEGIN} ASC",
        )?.use { cursor ->
            while (cursor.moveToNext() && events.size < maxEvents) {
                events += HermesCalendarProviderEvent(
                    eventId = cursor.getLong(COLUMN_EVENT_ID).toString(),
                    calendarName = cursor.getString(COLUMN_CALENDAR_NAME).orEmpty(),
                    title = cursor.getString(COLUMN_TITLE).orEmpty(),
                    description = cursor.getString(COLUMN_DESCRIPTION).orEmpty(),
                    location = cursor.getString(COLUMN_EVENT_LOCATION).orEmpty(),
                    beginMillis = cursor.getLong(COLUMN_BEGIN),
                    endMillis = cursor.getLong(COLUMN_END),
                )
            }
        }
        return events
    }

    private fun parseInjectedEvents(arguments: JSONObject): List<HermesCalendarProviderEvent>? {
        val raw = arguments.optJSONArray("events")
            ?: arguments.optJSONArray("calendar_events")
            ?: arguments.optJSONArray("records")
            ?: return null
        val events = mutableListOf<HermesCalendarProviderEvent>()
        for (index in 0 until raw.length()) {
            val item = raw.optJSONObject(index) ?: continue
            events += HermesCalendarProviderEvent(
                eventId = item.optString("event_id").ifBlank { item.optString("id").ifBlank { "injected-$index" } },
                calendarName = item.optString("calendar_name", item.optString("calendar")),
                title = item.optString("calendar_title", item.optString("title")),
                description = item.optString("calendar_description", item.optString("description")),
                location = item.optString("calendar_location", item.optString("location")),
                beginMillis = item.optLong("calendar_begin_epoch_ms", item.optLong("begin_epoch_ms", System.currentTimeMillis())),
                endMillis = item.optLong("calendar_end_epoch_ms", item.optLong("end_epoch_ms", System.currentTimeMillis())),
            )
        }
        return events
    }

    private fun persistWatcherRequest(
        context: Context,
        intervalSeconds: Int,
        lookaheadMinutes: Int,
        lookbackMinutes: Int,
    ) {
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, true)
            .putLong(PREF_SCAN_INTERVAL_SECONDS, intervalSeconds.toLong())
            .putLong(PREF_LOOKAHEAD_MINUTES, lookaheadMinutes.toLong())
            .putLong(PREF_LOOKBACK_MINUTES, lookbackMinutes.toLong())
            .apply()
    }

    private fun clearWatcherRequest(context: Context) {
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, false)
            .apply()
    }

    private fun persistLastEvent(context: Context, epochMillis: Long) {
        if (epochMillis <= 0) {
            return
        }
        lastEventEpochMs.set(epochMillis)
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putLong(PREF_LAST_EVENT_EPOCH_MS, epochMillis)
            .apply()
    }

    private fun persistedLastEventEpochMs(context: Context): Long? {
        val value = context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_LAST_EVENT_EPOCH_MS, 0L)
        return value.takeIf { it > 0 }
    }

    private fun persistDispatchCount(context: Context, additionalCount: Int) {
        if (additionalCount <= 0) {
            return
        }
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val next = prefs.getLong(PREF_DISPATCH_COUNT, 0L) + additionalCount
        prefs.edit().putLong(PREF_DISPATCH_COUNT, next).apply()
    }

    private fun persistedDispatchCount(context: Context): Long {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_DISPATCH_COUNT, 0L)
    }

    private fun recentEventSignatures(context: Context): List<String> {
        val raw = context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(PREF_RECENT_SIGNATURES, "[]")
            .orEmpty()
        val array = runCatching { JSONArray(raw) }.getOrDefault(JSONArray())
        return (0 until array.length()).mapNotNull { index -> array.optString(index).takeIf { it.isNotBlank() } }
    }

    private fun persistRecentEventSignatures(context: Context, signatures: List<String>) {
        val array = JSONArray()
        signatures.takeLast(MAX_RECENT_SIGNATURES).forEach { signature -> array.put(signature) }
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_RECENT_SIGNATURES, array.toString())
            .apply()
    }

    private fun resetCursor(context: Context) {
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .remove(PREF_RECENT_SIGNATURES)
            .remove(PREF_LAST_EVENT_EPOCH_MS)
            .apply()
        lastEventEpochMs.set(0)
    }

    private fun recentEventSignatureCount(context: Context): Int = recentEventSignatures(context).size

    private fun intArgument(arguments: JSONObject, vararg keys: String): Int? {
        keys.forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            val value = arguments.opt(key)
            val parsed = when (value) {
                is Number -> value.toInt()
                is String -> value.trim().toIntOrNull()
                else -> null
            }
            if (parsed != null) {
                return parsed
            }
        }
        return null
    }

    private val running = AtomicBoolean(false)
    private val startedAtEpochMs = AtomicLong(0)
    private val lastEventEpochMs = AtomicLong(0)
    private val scanIntervalSeconds = AtomicLong(DEFAULT_SCAN_INTERVAL_SECONDS.toLong())
    private val configuredLookaheadMinutes = AtomicLong(DEFAULT_LOOKAHEAD_MINUTES.toLong())
    private val configuredLookbackMinutes = AtomicLong(DEFAULT_LOOKBACK_MINUTES.toLong())
    private val workerThread = AtomicReference<Thread?>(null)

    private val ACTIONS = listOf(
        "calendar_watcher_status",
        "start_calendar_watcher",
        "stop_calendar_watcher",
        "scan_calendar_events",
        "reset_calendar_watcher_cursor",
    )
    private val CALENDAR_PROJECTION = arrayOf(
        CalendarContract.Instances.EVENT_ID,
        CalendarContract.Instances.CALENDAR_DISPLAY_NAME,
        CalendarContract.Instances.TITLE,
        CalendarContract.Instances.DESCRIPTION,
        CalendarContract.Instances.EVENT_LOCATION,
        CalendarContract.Instances.BEGIN,
        CalendarContract.Instances.END,
    )
    private const val COLUMN_EVENT_ID = 0
    private const val COLUMN_CALENDAR_NAME = 1
    private const val COLUMN_TITLE = 2
    private const val COLUMN_DESCRIPTION = 3
    private const val COLUMN_EVENT_LOCATION = 4
    private const val COLUMN_BEGIN = 5
    private const val COLUMN_END = 6
    private const val PREFS_NAME = "hermes_calendar_watcher"
    private const val PREF_DESIRED = "desired"
    private const val PREF_SCAN_INTERVAL_SECONDS = "scan_interval_seconds"
    private const val PREF_LOOKAHEAD_MINUTES = "lookahead_minutes"
    private const val PREF_LOOKBACK_MINUTES = "lookback_minutes"
    private const val PREF_LAST_EVENT_EPOCH_MS = "last_event_epoch_ms"
    private const val PREF_RECENT_SIGNATURES = "recent_event_signatures"
    private const val PREF_DISPATCH_COUNT = "dispatch_count"
    private const val DEFAULT_SCAN_INTERVAL_SECONDS = 300
    private const val MIN_SCAN_INTERVAL_SECONDS = 60
    private const val MAX_SCAN_INTERVAL_SECONDS = 3600
    private const val DEFAULT_LOOKAHEAD_MINUTES = 1440
    private const val MIN_LOOKAHEAD_MINUTES = 1
    private const val MAX_LOOKAHEAD_MINUTES = 10080
    private const val DEFAULT_LOOKBACK_MINUTES = 15
    private const val MIN_LOOKBACK_MINUTES = 0
    private const val MAX_LOOKBACK_MINUTES = 1440
    private const val DEFAULT_MAX_EVENTS = 50
    private const val MIN_MAX_EVENTS = 1
    private const val MAX_MAX_EVENTS = 200
    private const val MAX_RECENT_SIGNATURES = 200
}
