package com.nousresearch.hermesagent.device

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

data class HermesLogcatEvent(
    val timestamp: String,
    val uid: String,
    val pid: String,
    val level: String,
    val tag: String,
    val message: String,
    val packageName: String = "",
    val packageCandidates: List<String> = emptyList(),
    val packageNameSource: String = "",
) {
    fun toTriggerArguments(): JSONObject {
        return JSONObject()
            .put("logcat_timestamp", timestamp)
            .put("logcat_uid", uid)
            .put("logcat_pid", pid)
            .put("logcat_level", level)
            .put("logcat_tag", tag)
            .put("logcat_message", message)
            .put("logcat_package_name", packageName)
            .put("logcat_package_candidates", packageCandidates.joinToString(","))
            .put("logcat_package_source", packageNameSource)
    }

    fun toJson(): JSONObject {
        return toTriggerArguments()
    }
}

object HermesLogcatWatcherBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase().ifBlank { "logcat_watcher_status" }) {
            "logcat_watcher_status", "logcat_status", "watch_logcat_status" -> statusJson(context)
            "start_logcat_watcher", "start_logcat_watch", "watch_logcat" -> startJson(context, arguments)
            "stop_logcat_watcher", "stop_logcat_watch" -> stopJson(context)
            "scan_logcat_entries", "scan_logcat", "run_logcat_watch_once" -> scanOnceJson(context, arguments)
            "reset_logcat_watcher_cursor", "reset_logcat_cursor", "clear_logcat_watcher_cursor" -> resetCursorJson(context)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported logcat watcher action: $action")
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
    }

    fun statusJson(context: Context): String {
        val appContext = context.applicationContext
        val status = HermesPrivilegedAccessBridge.readStatus(appContext)
        return JSONObject()
            .put("success", true)
            .put("running", running.get())
            .put("durable_foreground_service", true)
            .put("foreground_service_running", HermesLogcatWatcherService.isRunning())
            .put("watcher_desired", isWatcherDesired(appContext))
            .put("started_at_epoch_ms", startedAtEpochMs.get().takeIf { it > 0 } ?: JSONObject.NULL)
            .put("scan_interval_seconds", scanIntervalSeconds.get())
            .put("max_lines", maxLines.get())
            .put("enabled_logcat_record_count", enabledLogcatRecordCount(appContext))
            .put("scan_cursor_enabled", true)
            .put("last_event_timestamp", persistedLastEventTimestamp(appContext) ?: JSONObject.NULL)
            .put("recent_event_signature_count", recentEventSignatureCount(appContext))
            .put("requires_shizuku", true)
            .put("shizuku_status", HermesPrivilegedAccessBridge.statusToJson(status))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    fun startJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val status = HermesPrivilegedAccessBridge.readStatus(appContext)
        if (!status.shizukuBinderAlive) {
            return unavailableJson("Shizuku is not running. Start Shizuku/Sui before starting the logcat watcher.", status)
        }
        if (!status.shizukuPermissionGranted) {
            return unavailableJson("Shizuku permission is not granted to Hermes Agent.", status)
        }
        val enabledCount = enabledLogcatRecordCount(appContext)
        if (enabledCount <= 0) {
            return JSONObject()
                .put("success", false)
                .put("error", "start_logcat_watcher requires at least one enabled logcat_entry automation")
                .put("running", running.get())
                .put("enabled_logcat_record_count", enabledCount)
                .put("shizuku_status", HermesPrivilegedAccessBridge.statusToJson(status))
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        if (arguments.optBoolean("reset_cursor", false)) {
            resetCursor(appContext)
        }
        val intervalSeconds = arguments.optInt("scan_interval_seconds", DEFAULT_SCAN_INTERVAL_SECONDS)
            .coerceIn(MIN_SCAN_INTERVAL_SECONDS, MAX_SCAN_INTERVAL_SECONDS)
        val lineLimit = arguments.optInt("max_lines", DEFAULT_MAX_LINES)
            .coerceIn(MIN_MAX_LINES, MAX_MAX_LINES)
        persistWatcherRequest(appContext, intervalSeconds.toLong(), lineLimit.toLong())
        val wasAlreadyRunning = startWorker(appContext, intervalSeconds.toLong(), lineLimit.toLong())
        val foregroundStarted = HermesLogcatWatcherService.start(appContext, intervalSeconds.toLong(), lineLimit.toLong())
        if (wasAlreadyRunning) {
            return JSONObject(statusJson(appContext))
                .put("success", true)
                .put("message", "Logcat watcher is already running")
                .put("foreground_service_started", foregroundStarted)
                .toString()
        }
        val message = if (foregroundStarted) {
            "Started durable Shizuku-backed logcat watcher"
        } else {
            "Started process-lifetime Shizuku-backed logcat watcher; foreground service start was not accepted by Android"
        }
        return JSONObject(statusJson(appContext))
            .put("success", true)
            .put("foreground_service_started", foregroundStarted)
            .put("message", message)
            .toString()
    }

    internal fun startWorker(
        context: Context,
        intervalSeconds: Long = persistedScanIntervalSeconds(context),
        lineLimit: Long = persistedMaxLines(context),
    ): Boolean {
        val appContext = context.applicationContext
        scanIntervalSeconds.set(intervalSeconds.coerceIn(MIN_SCAN_INTERVAL_SECONDS.toLong(), MAX_SCAN_INTERVAL_SECONDS.toLong()))
        maxLines.set(lineLimit.coerceIn(MIN_MAX_LINES.toLong(), MAX_MAX_LINES.toLong()))
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
                            .put("max_lines", maxLines.get())
                            .put("watcher_loop", true),
                    )
                }
                try {
                    Thread.sleep(scanIntervalSeconds.get().toLong() * 1000L)
                } catch (_: InterruptedException) {
                    // Re-check running and exit quickly after stop.
                }
            }
        }.apply {
            name = "hermes-shizuku-logcat-watcher"
            isDaemon = true
            start()
        }
        workerThread.set(worker)
        return false
    }

    fun stopJson(context: Context): String {
        clearWatcherRequest(context.applicationContext)
        val wasRunning = stopWorker()
        HermesLogcatWatcherService.stop(context.applicationContext)
        return JSONObject(statusJson(context))
            .put("success", true)
            .put("stopped", wasRunning)
            .put("message", if (wasRunning) "Stopped logcat watcher" else "Logcat watcher was not running")
            .toString()
    }

    fun resetCursorJson(context: Context): String {
        resetCursor(context.applicationContext)
        return JSONObject(statusJson(context))
            .put("success", true)
            .put("message", "Reset logcat watcher scan cursor")
            .toString()
    }

    internal fun stopWorker(): Boolean {
        val wasRunning = running.getAndSet(false)
        workerThread.getAndSet(null)?.interrupt()
        startedAtEpochMs.set(0)
        return wasRunning
    }

    internal fun isWatcherDesired(context: Context): Boolean {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(PREF_DESIRED, false)
    }

    internal fun persistedScanIntervalSeconds(context: Context): Long {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_SCAN_INTERVAL_SECONDS, DEFAULT_SCAN_INTERVAL_SECONDS.toLong())
            .coerceIn(MIN_SCAN_INTERVAL_SECONDS.toLong(), MAX_SCAN_INTERVAL_SECONDS.toLong())
    }

    internal fun persistedMaxLines(context: Context): Long {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getLong(PREF_MAX_LINES, DEFAULT_MAX_LINES.toLong())
            .coerceIn(MIN_MAX_LINES.toLong(), MAX_MAX_LINES.toLong())
    }

    internal fun resumePersistedWatcherIfRequested(context: Context): Boolean {
        val appContext = context.applicationContext
        if (!isWatcherDesired(appContext)) {
            return false
        }
        startWorker(appContext, persistedScanIntervalSeconds(appContext), persistedMaxLines(appContext))
        return true
    }

    fun scanOnceJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val status = HermesPrivilegedAccessBridge.readStatus(appContext)
        if (!status.shizukuBinderAlive) {
            return unavailableJson("Shizuku is not running. Start Shizuku/Sui before scanning logcat.", status)
        }
        if (!status.shizukuPermissionGranted) {
            return unavailableJson("Shizuku permission is not granted to Hermes Agent.", status)
        }
        val enabledCount = enabledLogcatRecordCount(appContext)
        if (enabledCount <= 0) {
            return JSONObject()
                .put("success", false)
                .put("error", "scan_logcat_entries requires at least one enabled logcat_entry automation")
                .put("enabled_logcat_record_count", enabledCount)
                .put("shizuku_status", HermesPrivilegedAccessBridge.statusToJson(status))
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val lineLimit = arguments.optInt("max_lines", maxLines.get().toInt()).coerceIn(MIN_MAX_LINES, MAX_MAX_LINES)
        val useCursor = arguments.optBoolean("use_cursor", arguments.optBoolean("watcher_loop", false))
        if (arguments.optBoolean("reset_cursor", false)) {
            resetCursor(appContext)
        }
        val command = "logcat -d -v threadtime,uid -t $lineLimit"
        val shellResult = JSONObject(
            HermesPrivilegedAccessBridge.runShellCommandJson(
                appContext,
                command,
                LOGCAT_SCAN_TIMEOUT_SECONDS,
            ),
        )
        if (!shellResult.optBoolean("success", false)) {
            return shellResult
                .put("success", false)
                .put("error", shellResult.optString("error").ifBlank { "logcat scan failed" })
                .put("adb_shell_command", command)
                .put("enabled_logcat_record_count", enabledCount)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        val events = parseThreadtimeLogcatLines(shellResult.optString("output"))
        val cursorFilteredEvents = filterNewCursorEvents(appContext, events, useCursor)
        val packagesByUid = resolvePackagesByUid(appContext, cursorFilteredEvents)
        val results = JSONArray()
        var matchedCount = 0
        cursorFilteredEvents.forEach { event ->
            val eventWithPackage = attributePackages(event, packagesByUid[event.uid].orEmpty())
            val dispatched = JSONObject(
                HermesAutomationBridge.runLogcatEntryTriggerJson(appContext, eventWithPackage.toTriggerArguments()),
            )
            val matches = dispatched.optInt("matched_count", 0)
            if (matches > 0) {
                matchedCount += matches
                results.put(dispatched)
            }
        }
        return JSONObject()
            .put("success", true)
            .put("adb_shell_command", command)
            .put("enabled_logcat_record_count", enabledCount)
            .put("parsed_event_count", events.size)
            .put("cursor_enabled", useCursor)
            .put("cursor_filtered_event_count", events.size - cursorFilteredEvents.size)
            .put("last_event_timestamp", persistedLastEventTimestamp(appContext) ?: JSONObject.NULL)
            .put("recent_event_signature_count", recentEventSignatureCount(appContext))
            .put("uid_package_count", packagesByUid.size)
            .put("matched_count", matchedCount)
            .put("results", results)
            .put("shizuku_status", HermesPrivilegedAccessBridge.statusToJson(status))
            .toString()
    }

    internal fun parseThreadtimeLogcatLines(output: String): List<HermesLogcatEvent> {
        return output
            .lineSequence()
            .mapNotNull { line ->
                val trimmed = line.trim()
                THREADTIME_WITH_UID_REGEX.matchEntire(trimmed)?.let { match ->
                    HermesLogcatEvent(
                        timestamp = "${match.groupValues[1]} ${match.groupValues[2]}",
                        uid = match.groupValues[3],
                        pid = match.groupValues[4],
                        level = match.groupValues[6],
                        tag = match.groupValues[7].trim(),
                        message = match.groupValues[8],
                    )
                } ?: THREADTIME_REGEX.matchEntire(trimmed)?.let { match ->
                    HermesLogcatEvent(
                        timestamp = "${match.groupValues[1]} ${match.groupValues[2]}",
                        uid = "",
                        pid = match.groupValues[3],
                        level = match.groupValues[5],
                        tag = match.groupValues[6].trim(),
                        message = match.groupValues[7],
                    )
                }
            }
            .take(MAX_PARSED_EVENTS)
            .toList()
    }

    internal fun filterNewCursorEvents(
        context: Context,
        events: List<HermesLogcatEvent>,
        cursorEnabled: Boolean,
    ): List<HermesLogcatEvent> {
        if (!cursorEnabled || events.isEmpty()) {
            return events
        }
        val appContext = context.applicationContext
        val alreadySeen = readRecentEventSignatures(appContext).toSet()
        val freshEvents = events.filter { event -> eventSignature(event) !in alreadySeen }
        persistCursor(appContext, events)
        return freshEvents
    }

    internal fun eventSignature(event: HermesLogcatEvent): String {
        return listOf(
            event.timestamp,
            event.uid,
            event.pid,
            event.level,
            event.tag,
            event.message,
        ).joinToString(SIGNATURE_SEPARATOR) { field ->
            field.take(MAX_SIGNATURE_FIELD_CHARS)
        }
    }

    internal fun attributePackages(event: HermesLogcatEvent, packages: List<String>): HermesLogcatEvent {
        val candidates = packages
            .asSequence()
            .map { packageName -> packageName.trim() }
            .filter { packageName -> PACKAGE_NAME_REGEX.matches(packageName) }
            .distinct()
            .toList()
        if (candidates.isEmpty()) {
            return event
        }
        val packageFromLogLine = candidates.firstOrNull { packageName ->
            event.tag.contains(packageName, ignoreCase = true) ||
                event.message.contains(packageName, ignoreCase = true)
        }
        val selectedPackage = packageFromLogLine
            ?: candidates.singleOrNull()
            ?: candidates.joinToString(",")
        val source = when {
            packageFromLogLine != null -> "message"
            candidates.size == 1 -> "uid"
            else -> "uid_shared"
        }
        return event.copy(
            packageName = selectedPackage,
            packageCandidates = candidates,
            packageNameSource = source,
        )
    }

    fun parseThreadtimeLogcatLinesJson(output: String): JSONArray {
        val events = JSONArray()
        parseThreadtimeLogcatLines(output).forEach { event ->
            events.put(event.toJson())
        }
        return events
    }

    internal fun parseUidPackageLines(output: String): Map<String, List<String>> {
        val packagesByUid = linkedMapOf<String, MutableList<String>>()
        output
            .lineSequence()
            .mapNotNull { line -> UID_PACKAGE_REGEX.matchEntire(line.trim()) }
            .forEach { match ->
                val packageName = match.groupValues[1]
                val uid = match.groupValues[2]
                packagesByUid.getOrPut(uid) { mutableListOf() }.add(packageName)
            }
        return packagesByUid
    }

    private fun resolvePackagesByUid(context: Context, events: List<HermesLogcatEvent>): Map<String, List<String>> {
        val uids = events
            .map { it.uid }
            .filter { uid -> UID_REGEX.matches(uid) }
            .distinct()
            .take(MAX_UID_PACKAGE_LOOKUPS)
        if (uids.isEmpty()) {
            return emptyMap()
        }
        val command = uids.joinToString("; ") { uid -> "cmd package list packages --uid $uid" }
        val shellResult = JSONObject(
            HermesPrivilegedAccessBridge.runShellCommandJson(
                context.applicationContext,
                command,
                UID_PACKAGE_LOOKUP_TIMEOUT_SECONDS,
            ),
        )
        if (!shellResult.optBoolean("success", false)) {
            return emptyMap()
        }
        return parseUidPackageLines(shellResult.optString("output"))
    }

    private fun enabledLogcatRecordCount(context: Context): Int {
        return HermesAutomationStore(context)
            .list()
            .count { record -> record.enabled && record.triggerType == TRIGGER_LOGCAT_ENTRY }
    }

    private fun unavailableJson(message: String, status: HermesPrivilegedAccessStatus): String {
        return JSONObject()
            .put("success", false)
            .put("running", running.get())
            .put("durable_foreground_service", true)
            .put("foreground_service_running", HermesLogcatWatcherService.isRunning())
            .put("error", message)
            .put("requires_shizuku", true)
            .put("shizuku_status", HermesPrivilegedAccessBridge.statusToJson(status))
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    private fun persistWatcherRequest(context: Context, intervalSeconds: Long, lineLimit: Long) {
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, true)
            .putLong(
                PREF_SCAN_INTERVAL_SECONDS,
                intervalSeconds.coerceIn(MIN_SCAN_INTERVAL_SECONDS.toLong(), MAX_SCAN_INTERVAL_SECONDS.toLong()),
            )
            .putLong(PREF_MAX_LINES, lineLimit.coerceIn(MIN_MAX_LINES.toLong(), MAX_MAX_LINES.toLong()))
            .apply()
    }

    private fun clearWatcherRequest(context: Context) {
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_DESIRED, false)
            .apply()
    }

    internal fun resetCursor(context: Context) {
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .remove(PREF_LAST_EVENT_TIMESTAMP)
            .remove(PREF_RECENT_EVENT_SIGNATURES)
            .apply()
    }

    internal fun persistedLastEventTimestamp(context: Context): String? {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(PREF_LAST_EVENT_TIMESTAMP, null)
            ?.takeIf { it.isNotBlank() }
    }

    internal fun recentEventSignatureCount(context: Context): Int {
        return readRecentEventSignatures(context.applicationContext).size
    }

    private fun persistCursor(context: Context, events: List<HermesLogcatEvent>) {
        if (events.isEmpty()) {
            return
        }
        val merged = (readRecentEventSignatures(context) + events.map { event -> eventSignature(event) })
            .distinct()
            .takeLast(MAX_RECENT_EVENT_SIGNATURES)
        context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_RECENT_EVENT_SIGNATURES, merged.joinToString("\n"))
            .putString(PREF_LAST_EVENT_TIMESTAMP, events.last().timestamp)
            .apply()
    }

    private fun readRecentEventSignatures(context: Context): List<String> {
        return context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(PREF_RECENT_EVENT_SIGNATURES, "")
            .orEmpty()
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .toList()
            .takeLast(MAX_RECENT_EVENT_SIGNATURES)
    }

    private val running = AtomicBoolean(false)
    private val workerThread = AtomicReference<Thread?>()
    private val startedAtEpochMs = AtomicLong(0)
    private val scanIntervalSeconds = AtomicLong(DEFAULT_SCAN_INTERVAL_SECONDS.toLong())
    private val maxLines = AtomicLong(DEFAULT_MAX_LINES.toLong())

    private val ACTIONS = listOf(
        "logcat_watcher_status",
        "start_logcat_watcher",
        "stop_logcat_watcher",
        "scan_logcat_entries",
        "reset_logcat_watcher_cursor",
    )

    private val THREADTIME_REGEX =
        Regex("""^(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+(\d+)\s+(\d+)\s+([VDIWEAF])\s+([^:]+):\s?(.*)$""")
    private val THREADTIME_WITH_UID_REGEX =
        Regex("""^(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+(\d+)\s+(\d+)\s+(\d+)\s+([VDIWEAF])\s+([^:]+):\s?(.*)$""")
    private val UID_PACKAGE_REGEX = Regex("""^package:([A-Za-z0-9._-]+)\s+uid:(\d+)$""")
    private val PACKAGE_NAME_REGEX = Regex("""^[A-Za-z0-9._-]+$""")
    private val UID_REGEX = Regex("""^\d{1,10}$""")

    private const val DEFAULT_SCAN_INTERVAL_SECONDS = 30
    private const val MIN_SCAN_INTERVAL_SECONDS = 5
    private const val MAX_SCAN_INTERVAL_SECONDS = 3600
    private const val DEFAULT_MAX_LINES = 250
    private const val MIN_MAX_LINES = 10
    private const val MAX_MAX_LINES = 1000
    private const val MAX_PARSED_EVENTS = 1000
    private const val MAX_RECENT_EVENT_SIGNATURES = 1000
    private const val MAX_SIGNATURE_FIELD_CHARS = 512
    private const val LOGCAT_SCAN_TIMEOUT_SECONDS = 15
    private const val UID_PACKAGE_LOOKUP_TIMEOUT_SECONDS = 10
    private const val MAX_UID_PACKAGE_LOOKUPS = 50
    private const val PREFS_NAME = "hermes_logcat_watcher"
    private const val PREF_DESIRED = "desired"
    private const val PREF_SCAN_INTERVAL_SECONDS = "scan_interval_seconds"
    private const val PREF_MAX_LINES = "max_lines"
    private const val PREF_LAST_EVENT_TIMESTAMP = "last_event_timestamp"
    private const val PREF_RECENT_EVENT_SIGNATURES = "recent_event_signatures"
    private const val SIGNATURE_SEPARATOR = "\u001F"
}
