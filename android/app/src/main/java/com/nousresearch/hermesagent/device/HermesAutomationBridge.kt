package com.nousresearch.hermesagent.device

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

object HermesAutomationBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase().ifBlank { "list" }) {
            "list", "list_automations", "status" -> listJson(context)
            "create_shell_task", "create_shell", "create" -> createShellTaskJson(context, arguments)
            "run", "run_now", "trigger" -> runAutomationJson(context, arguments.optString("id"), "manual")
            "delete", "remove" -> deleteJson(context, arguments.optString("id"))
            "enable" -> setEnabledJson(context, arguments.optString("id"), true)
            "disable", "pause" -> setEnabledJson(context, arguments.optString("id"), false)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported Android automation action: $action")
                .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
                .toString()
        }
    }

    fun listJson(context: Context): String {
        return JSONObject()
            .put("success", true)
            .put("automations", recordsToJson(HermesAutomationStore(context).list()))
            .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
            .put("min_interval_minutes", HermesAutomationScheduler.MIN_INTERVAL_MINUTES)
            .toString()
    }

    fun createShellTaskJson(context: Context, arguments: JSONObject): String {
        val command = arguments.optString("command").ifBlank { arguments.optString("cmd") }.trim()
        if (command.isBlank()) {
            return errorJson("create_shell_task requires a command argument")
        }
        if (command.indexOf('\u0000') >= 0) {
            return errorJson("create_shell_task command must not contain NUL bytes")
        }
        val intervalMinutes = optionalPositiveInt(arguments, "interval_minutes")
            ?: optionalPositiveInt(arguments, "every_minutes")
        if (intervalMinutes != null && intervalMinutes < HermesAutomationScheduler.MIN_INTERVAL_MINUTES) {
            return errorJson("interval_minutes must be at least ${HermesAutomationScheduler.MIN_INTERVAL_MINUTES}")
        }
        val now = System.currentTimeMillis()
        val record = HermesAutomationRecord(
            id = arguments.optString("id").ifBlank { "auto_${UUID.randomUUID().toString().replace("-", "").take(16)}" },
            label = arguments.optString("label").ifBlank { "Hermes shell automation" }.take(80),
            actionType = ACTION_TYPE_SHELL,
            command = command,
            useShizuku = arguments.optBoolean("use_shizuku", false),
            triggerType = if (intervalMinutes == null) TRIGGER_MANUAL else TRIGGER_INTERVAL,
            intervalMinutes = intervalMinutes,
            enabled = arguments.optBoolean("enabled", true),
            createdAtEpochMs = now,
            updatedAtEpochMs = now,
        )
        val store = HermesAutomationStore(context)
        store.upsert(record)
        HermesAutomationScheduler.schedule(context, record)
        return JSONObject()
            .put("success", true)
            .put("automation", record.toJson())
            .put(
                "message",
                if (record.triggerType == TRIGGER_INTERVAL && record.enabled) {
                    "Created and scheduled Android automation"
                } else {
                    "Created manual Android automation"
                },
            )
            .toString()
    }

    fun runAutomationJson(context: Context, id: String, trigger: String = "manual"): String {
        if (id.isBlank()) {
            return errorJson("run requires an automation id")
        }
        val store = HermesAutomationStore(context)
        val record = store.get(id) ?: return errorJson("Unknown Android automation id: $id")
        if (record.actionType != ACTION_TYPE_SHELL) {
            return errorJson("Unsupported Android automation action type: ${record.actionType}")
        }
        val rawResult = if (record.useShizuku) {
            JSONObject(HermesPrivilegedAccessBridge.runShellCommandJson(context, record.command, AUTOMATION_TIMEOUT_SECONDS))
        } else {
            NativeAndroidShellTool.run(context, record.command, AUTOMATION_TIMEOUT_SECONDS.toLong())
        }
        val exitCode = rawResult.optInt("exit_code", if (rawResult.optBoolean("success", false)) 0 else -1)
        val success = rawResult.optBoolean("success", exitCode == 0) || exitCode == 0
        val resultText = listOf(rawResult.optString("output"), rawResult.optString("error"))
            .filter { it.isNotBlank() }
            .joinToString("\n")
            .take(MAX_RESULT_CHARS)
        val updated = record.copy(
            updatedAtEpochMs = System.currentTimeMillis(),
            lastRunEpochMs = System.currentTimeMillis(),
            lastExitCode = exitCode,
            lastSuccess = success,
            lastResult = resultText,
        )
        store.upsert(updated)
        return JSONObject()
            .put("success", success)
            .put("trigger", trigger)
            .put("automation", updated.toJson())
            .put("result", rawResult)
            .toString()
    }

    fun deleteJson(context: Context, id: String): String {
        if (id.isBlank()) {
            return errorJson("delete requires an automation id")
        }
        HermesAutomationScheduler.cancel(context, id)
        val removed = HermesAutomationStore(context).remove(id)
        return JSONObject()
            .put("success", removed)
            .put("id", id)
            .put("message", if (removed) "Deleted Android automation" else "Android automation was not found")
            .toString()
    }

    fun setEnabledJson(context: Context, id: String, enabled: Boolean): String {
        if (id.isBlank()) {
            return errorJson("enable/disable requires an automation id")
        }
        val store = HermesAutomationStore(context)
        val record = store.get(id) ?: return errorJson("Unknown Android automation id: $id")
        val updated = record.copy(enabled = enabled, updatedAtEpochMs = System.currentTimeMillis())
        store.upsert(updated)
        if (enabled) {
            HermesAutomationScheduler.schedule(context, updated)
        } else {
            HermesAutomationScheduler.cancel(context, id)
        }
        return JSONObject()
            .put("success", true)
            .put("automation", updated.toJson())
            .put("message", if (enabled) "Enabled Android automation" else "Disabled Android automation")
            .toString()
    }

    private fun optionalPositiveInt(arguments: JSONObject, key: String): Int? {
        if (!arguments.has(key) || arguments.isNull(key)) {
            return null
        }
        val value = arguments.optInt(key, 0)
        return value.takeIf { it > 0 }
    }

    private fun recordsToJson(records: List<HermesAutomationRecord>): JSONArray {
        return JSONArray().apply {
            records.forEach { record -> put(record.toJson()) }
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
            .toString()
    }

    private val AUTOMATION_ACTIONS = listOf(
        "list",
        "create_shell_task",
        "run",
        "delete",
        "enable",
        "disable",
    )
    private const val AUTOMATION_TIMEOUT_SECONDS = 30
    private const val MAX_RESULT_CHARS = 2_000
}
