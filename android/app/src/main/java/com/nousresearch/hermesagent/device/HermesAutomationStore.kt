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
    val intervalMinutes: Int?,
    val enabled: Boolean,
    val createdAtEpochMs: Long,
    val updatedAtEpochMs: Long,
    val lastRunEpochMs: Long? = null,
    val lastExitCode: Int? = null,
    val lastSuccess: Boolean? = null,
    val lastResult: String = "",
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("label", label)
            .put("action_type", actionType)
            .put("command", command)
            .put("use_shizuku", useShizuku)
            .put("trigger_type", triggerType)
            .put("interval_minutes", intervalMinutes ?: JSONObject.NULL)
            .put("enabled", enabled)
            .put("created_at_epoch_ms", createdAtEpochMs)
            .put("updated_at_epoch_ms", updatedAtEpochMs)
            .put("last_run_epoch_ms", lastRunEpochMs ?: JSONObject.NULL)
            .put("last_exit_code", lastExitCode ?: JSONObject.NULL)
            .put("last_success", lastSuccess ?: JSONObject.NULL)
            .put("last_result", lastResult)
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
                intervalMinutes = if (json.isNull("interval_minutes")) null else json.optInt("interval_minutes"),
                enabled = json.optBoolean("enabled", true),
                createdAtEpochMs = json.optLong("created_at_epoch_ms", 0L),
                updatedAtEpochMs = json.optLong("updated_at_epoch_ms", 0L),
                lastRunEpochMs = if (json.isNull("last_run_epoch_ms")) null else json.optLong("last_run_epoch_ms"),
                lastExitCode = if (json.isNull("last_exit_code")) null else json.optInt("last_exit_code"),
                lastSuccess = if (json.isNull("last_success")) null else json.optBoolean("last_success"),
                lastResult = json.optString("last_result"),
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

    fun remove(id: String): Boolean {
        val existing = list()
        val updated = existing.filterNot { it.id == id }
        saveAll(updated)
        return updated.size != existing.size
    }

    fun clear() {
        saveAll(emptyList())
    }

    private fun saveAll(records: List<HermesAutomationRecord>) {
        val array = JSONArray().apply {
            records.forEach { record -> put(record.toJson()) }
        }
        preferences.edit().putString(KEY_RECORDS, array.toString()).apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_automations"
        private const val KEY_RECORDS = "records_json"
    }
}

const val ACTION_TYPE_SHELL = "shell"
const val TRIGGER_MANUAL = "manual"
const val TRIGGER_INTERVAL = "interval"
