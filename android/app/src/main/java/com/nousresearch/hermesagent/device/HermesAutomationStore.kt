package com.nousresearch.hermesagent.device

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class HermesAutomationRecord(
    val id: String,
    val label: String,
    val actionType: String,
    val command: String,
    val useShizuku: Boolean,
    val triggerType: String,
    val triggerPackageName: String = "",
    val triggerTimeMinutes: Int? = null,
    val triggerDaysOfWeek: String = "",
    val intervalMinutes: Int?,
    val enabled: Boolean,
    val createdAtEpochMs: Long,
    val updatedAtEpochMs: Long,
    val lastRunEpochMs: Long? = null,
    val lastExitCode: Int? = null,
    val lastSuccess: Boolean? = null,
    val lastResult: String = "",
    val triggerData: String = "",
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("label", label)
            .put("action_type", actionType)
            .put("command", command)
            .put("use_shizuku", useShizuku)
            .put("trigger_type", triggerType)
            .put("trigger_package_name", triggerPackageName)
            .put("trigger_time_minutes", triggerTimeMinutes ?: JSONObject.NULL)
            .put("trigger_days_of_week", triggerDaysOfWeek)
            .put("interval_minutes", intervalMinutes ?: JSONObject.NULL)
            .put("enabled", enabled)
            .put("created_at_epoch_ms", createdAtEpochMs)
            .put("updated_at_epoch_ms", updatedAtEpochMs)
            .put("last_run_epoch_ms", lastRunEpochMs ?: JSONObject.NULL)
            .put("last_exit_code", lastExitCode ?: JSONObject.NULL)
            .put("last_success", lastSuccess ?: JSONObject.NULL)
            .put("last_result", lastResult)
            .put("trigger_data", triggerData)
    }

    companion object {
        fun fromJson(json: JSONObject): HermesAutomationRecord {
            return HermesAutomationRecord(
                id = json.optString("id"),
                label = json.optString("label"),
                actionType = json.optString("action_type").ifBlank { ACTION_TYPE_SHELL },
                command = json.optString("command"),
                useShizuku = json.optBoolean("use_shizuku", false),
                triggerType = json.optString("trigger_type").ifBlank { TRIGGER_MANUAL },
                triggerPackageName = json.optString("trigger_package_name"),
                triggerTimeMinutes = if (json.isNull("trigger_time_minutes")) null else json.optInt("trigger_time_minutes"),
                triggerDaysOfWeek = json.optString("trigger_days_of_week"),
                intervalMinutes = if (json.isNull("interval_minutes")) null else json.optInt("interval_minutes"),
                enabled = json.optBoolean("enabled", true),
                createdAtEpochMs = json.optLong("created_at_epoch_ms", 0L),
                updatedAtEpochMs = json.optLong("updated_at_epoch_ms", 0L),
                lastRunEpochMs = if (json.isNull("last_run_epoch_ms")) null else json.optLong("last_run_epoch_ms"),
                lastExitCode = if (json.isNull("last_exit_code")) null else json.optInt("last_exit_code"),
                lastSuccess = if (json.isNull("last_success")) null else json.optBoolean("last_success"),
                lastResult = json.optString("last_result"),
                triggerData = json.optString("trigger_data"),
            )
        }
    }
}

data class HermesAutomationRunEvent(
    val id: String,
    val automationId: String,
    val automationLabel: String,
    val actionType: String,
    val trigger: String,
    val success: Boolean,
    val exitCode: Int,
    val result: String,
    val startedAtEpochMs: Long,
    val finishedAtEpochMs: Long,
    val dispatchSource: String = "",
    val dispatchChannel: String = "",
    val remoteExecutionId: String = "",
    val remoteTaskId: String = "",
    val remoteTaskName: String = "",
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("automation_id", automationId)
            .put("automation_label", automationLabel)
            .put("action_type", actionType)
            .put("trigger", trigger)
            .put("success", success)
            .put("exit_code", exitCode)
            .put("result", result)
            .put("started_at_epoch_ms", startedAtEpochMs)
            .put("finished_at_epoch_ms", finishedAtEpochMs)
            .put("duration_ms", (finishedAtEpochMs - startedAtEpochMs).coerceAtLeast(0L))
            .put("dispatch_source", dispatchSource)
            .put("dispatch_channel", dispatchChannel)
            .put("remote_execution_id", remoteExecutionId)
            .put("remote_task_id", remoteTaskId)
            .put("remote_task_name", remoteTaskName)
    }

    companion object {
        fun fromJson(json: JSONObject): HermesAutomationRunEvent {
            return HermesAutomationRunEvent(
                id = json.optString("id"),
                automationId = json.optString("automation_id"),
                automationLabel = json.optString("automation_label"),
                actionType = json.optString("action_type"),
                trigger = json.optString("trigger"),
                success = json.optBoolean("success", false),
                exitCode = json.optInt("exit_code", -1),
                result = json.optString("result"),
                startedAtEpochMs = json.optLong("started_at_epoch_ms", 0L),
                finishedAtEpochMs = json.optLong("finished_at_epoch_ms", 0L),
                dispatchSource = json.optString("dispatch_source"),
                dispatchChannel = json.optString("dispatch_channel"),
                remoteExecutionId = json.optString("remote_execution_id"),
                remoteTaskId = json.optString("remote_task_id"),
                remoteTaskName = json.optString("remote_task_name"),
            )
        }
    }
}

class HermesAutomationStore(context: Context) {
    private val preferences = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun list(): List<HermesAutomationRecord> {
        val raw = preferences.getString(KEY_RECORDS, "[]").orEmpty()
        val array = runCatching { JSONArray(raw) }.getOrDefault(JSONArray())
        val records = mutableListOf<HermesAutomationRecord>()
        for (index in 0 until array.length()) {
            val json = array.optJSONObject(index) ?: continue
            val record = runCatching { HermesAutomationRecord.fromJson(json) }.getOrNull() ?: continue
            if (record.id.isNotBlank()) {
                records += record
            }
        }
        return records.sortedBy { it.createdAtEpochMs }
    }

    fun get(id: String): HermesAutomationRecord? {
        return list().firstOrNull { it.id == id }
    }

    fun upsert(record: HermesAutomationRecord) {
        val updated = list()
            .filterNot { it.id == record.id }
            .plus(record)
            .sortedBy { it.createdAtEpochMs }
        saveAll(updated)
    }

    fun replaceAll(records: List<HermesAutomationRecord>) {
        saveAll(records.sortedBy { it.createdAtEpochMs })
    }

    fun remove(id: String): Boolean {
        val existing = list()
        val updated = existing.filterNot { it.id == id }
        saveAll(updated)
        return updated.size != existing.size
    }

    fun clear() {
        saveAll(emptyList())
        saveVariables(JSONObject())
        saveRunEvents(emptyList())
        saveStandbyHeartbeat(JSONObject())
    }

    fun addRunEvent(event: HermesAutomationRunEvent) {
        val updated = listRunEvents(MAX_RUN_EVENTS)
            .filterNot { it.id == event.id }
            .plus(event)
            .sortedByDescending { it.finishedAtEpochMs }
            .take(MAX_RUN_EVENTS)
        saveRunEvents(updated)
    }

    fun listRunEvents(limit: Int = 50): List<HermesAutomationRunEvent> {
        val safeLimit = limit.coerceIn(1, MAX_RUN_EVENTS)
        val raw = preferences.getString(KEY_RUN_EVENTS, "[]").orEmpty()
        val array = runCatching { JSONArray(raw) }.getOrDefault(JSONArray())
        val events = mutableListOf<HermesAutomationRunEvent>()
        for (index in 0 until array.length()) {
            val json = array.optJSONObject(index) ?: continue
            val event = runCatching { HermesAutomationRunEvent.fromJson(json) }.getOrNull() ?: continue
            if (event.id.isNotBlank() && event.automationId.isNotBlank()) {
                events += event
            }
        }
        return events.sortedByDescending { it.finishedAtEpochMs }.take(safeLimit)
    }

    fun saveStandbyHeartbeat(heartbeat: JSONObject) {
        preferences.edit().putString(KEY_STANDBY_HEARTBEAT, heartbeat.toString()).apply()
    }

    fun lastStandbyHeartbeat(): JSONObject {
        val raw = preferences.getString(KEY_STANDBY_HEARTBEAT, "{}").orEmpty()
        return runCatching { JSONObject(raw) }.getOrDefault(JSONObject())
    }

    fun listVariables(): JSONObject {
        val raw = preferences.getString(KEY_VARIABLES, "{}").orEmpty()
        return runCatching { JSONObject(raw) }.getOrDefault(JSONObject())
    }

    fun getVariable(name: String): String? {
        val normalized = normalizeVariableName(name) ?: return null
        val variables = listVariables()
        return if (variables.has(normalized) && !variables.isNull(normalized)) {
            variables.optString(normalized)
        } else {
            null
        }
    }

    fun setVariable(name: String, value: String): Boolean {
        val normalized = normalizeVariableName(name) ?: return false
        val variables = listVariables().put(normalized, value.take(MAX_VARIABLE_VALUE_CHARS))
        saveVariables(variables)
        return true
    }

    fun removeVariable(name: String): Boolean {
        val normalized = normalizeVariableName(name) ?: return false
        val variables = listVariables()
        val existed = variables.has(normalized)
        variables.remove(normalized)
        saveVariables(variables)
        return existed
    }

    fun replaceVariables(variables: JSONObject) {
        saveVariables(normalizeVariables(variables))
    }

    fun mergeVariables(variables: JSONObject) {
        val merged = listVariables()
        val normalized = normalizeVariables(variables)
        normalized.keys().forEach { key ->
            merged.put(key, normalized.optString(key).take(MAX_VARIABLE_VALUE_CHARS))
        }
        saveVariables(merged)
    }

    private fun saveAll(records: List<HermesAutomationRecord>) {
        val array = JSONArray().apply {
            records.forEach { record -> put(record.toJson()) }
        }
        preferences.edit().putString(KEY_RECORDS, array.toString()).apply()
    }

    private fun saveVariables(variables: JSONObject) {
        preferences.edit().putString(KEY_VARIABLES, variables.toString()).apply()
    }

    private fun saveRunEvents(events: List<HermesAutomationRunEvent>) {
        val array = JSONArray().apply {
            events
                .sortedByDescending { it.finishedAtEpochMs }
                .take(MAX_RUN_EVENTS)
                .forEach { event -> put(event.toJson()) }
        }
        preferences.edit().putString(KEY_RUN_EVENTS, array.toString()).apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_automations"
        private const val KEY_RECORDS = "records_json"
        private const val KEY_VARIABLES = "variables_json"
        private const val KEY_RUN_EVENTS = "run_events_json"
        private const val KEY_STANDBY_HEARTBEAT = "standby_heartbeat_json"
        private const val MAX_VARIABLE_VALUE_CHARS = 4_000
        private const val MAX_RUN_EVENTS = 50

        fun normalizeVariableName(name: String): String? {
            val trimmed = name.trim().removePrefix("%")
            if (!VARIABLE_NAME_REGEX.matches(trimmed)) {
                return null
            }
            return trimmed.uppercase()
        }

        private fun normalizeVariables(variables: JSONObject): JSONObject {
            val normalized = JSONObject()
            variables.keys().forEach { key ->
                val name = normalizeVariableName(key) ?: return@forEach
                if (!variables.isNull(key)) {
                    normalized.put(name, variables.optString(key).take(MAX_VARIABLE_VALUE_CHARS))
                }
            }
            return normalized
        }

        private val VARIABLE_NAME_REGEX = Regex("[A-Za-z_][A-Za-z0-9_]{0,63}")
    }
}

const val ACTION_TYPE_SHELL = "shell"
const val ACTION_TYPE_FILE_WRITE = "file_write"
const val ACTION_TYPE_FILE_DELETE = "file_delete"
const val ACTION_TYPE_SYSTEM_ACTION = "system_action"
const val ACTION_TYPE_UI_ACTION = "ui_action"
const val ACTION_TYPE_APP_LAUNCH = "app_launch"
const val ACTION_TYPE_INTENT = "intent"
const val ACTION_TYPE_SHIZUKU_ACTION = "shizuku_action"
const val ACTION_TYPE_SUNRISE_SUNSET = "sunrise_sunset"
const val ACTION_TYPE_NOTIFICATION_ACTION = "notification_action"
const val ACTION_TYPE_VARIABLE_ACTION = "variable_action"
const val ACTION_TYPE_WAIT = "wait"
const val ACTION_TYPE_CLIPBOARD_ACTION = "clipboard_action"
const val ACTION_TYPE_VIBRATION_ACTION = "vibration_action"
const val ACTION_TYPE_AUDIO_ACTION = "audio_action"
const val ACTION_TYPE_HTTP_REQUEST = "http_request"
const val ACTION_TYPE_OVERLAY_SCENE = "overlay_scene"
const val ACTION_TYPE_TOAST_ACTION = "toast_action"
const val TRIGGER_MANUAL = "manual"
const val TRIGGER_INTERVAL = "interval"
const val TRIGGER_BOOT = "boot"
const val TRIGGER_POWER_CONNECTED = "power_connected"
const val TRIGGER_POWER_DISCONNECTED = "power_disconnected"
const val TRIGGER_BATTERY_LOW = "battery_low"
const val TRIGGER_BATTERY_OKAY = "battery_okay"
const val TRIGGER_APP_FOREGROUND = "app_foreground"
const val TRIGGER_NOTIFICATION_POSTED = "notification_posted"
const val TRIGGER_TIME = "time"
const val TRIGGER_CALENDAR_EVENT = "calendar_event"
const val TRIGGER_LOCATION = "location"
const val TRIGGER_SENSOR = "sensor"
const val TRIGGER_LOGCAT_ENTRY = "logcat_entry"
const val TRIGGER_REMOTE_DISPATCH = "remote_dispatch"
const val TRIGGER_SHIZUKU_AVAILABLE = "shizuku_available"
const val TRIGGER_SHIZUKU_UNAVAILABLE = "shizuku_unavailable"
const val TRIGGER_EXTERNAL = "external_trigger"
