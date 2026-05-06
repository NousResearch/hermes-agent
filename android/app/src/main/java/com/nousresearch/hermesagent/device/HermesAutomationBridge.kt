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
            "create_file_write_task", "create_file_write", "write_file_task" -> createFileWriteTaskJson(context, arguments)
            "create_file_delete_task", "create_file_delete", "delete_file_task" -> createFileDeleteTaskJson(context, arguments)
            "create_system_action_task", "create_system_action", "system_action_task" -> createSystemActionTaskJson(context, arguments)
            "create_ui_action_task", "create_ui_action", "ui_action_task" -> createUiActionTaskJson(context, arguments)
            "create_app_launch_task", "create_app_launch", "launch_app_task" -> createAppLaunchTaskJson(context, arguments)
            "run", "run_now", "trigger" -> runAutomationJson(context, arguments.optString("id"), "manual")
            "run_trigger", "trigger_event", "run_event" -> runTriggerJson(
                context,
                arguments.optString("trigger").ifBlank { arguments.optString("trigger_type") },
            )
            "run_app_foreground_trigger", "trigger_app_foreground", "app_foreground" -> runAppForegroundTriggerJson(
                context,
                stringArgument(arguments, "trigger_package_name", "package_name", "packageName", "package", "app_package").orEmpty(),
            )
            "run_notification_posted_trigger", "trigger_notification_posted", "notification_posted" -> runNotificationPostedTriggerJson(
                context = context,
                packageName = stringArgument(arguments, "trigger_package_name", "package_name", "packageName", "package", "app_package").orEmpty(),
                title = stringArgument(arguments, "notification_title", "title", allowEmpty = true).orEmpty(),
                text = stringArgument(arguments, "notification_text", "text", "content", allowEmpty = true).orEmpty(),
            )
            "delete", "remove" -> deleteJson(context, arguments.optString("id"))
            "enable" -> setEnabledJson(context, arguments.optString("id"), true)
            "disable", "pause" -> setEnabledJson(context, arguments.optString("id"), false)
            "list_variables", "variables" -> listVariablesJson(context)
            "set_variable", "variable_set" -> setVariableJson(context, arguments)
            "get_variable", "variable_get" -> getVariableJson(context, arguments)
            "delete_variable", "remove_variable", "variable_delete" -> deleteVariableJson(context, arguments)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported Android automation action: $action")
                .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
                .put("available_triggers", JSONArray(AUTOMATION_TRIGGERS))
                .toString()
        }
    }

    fun listJson(context: Context): String {
        val store = HermesAutomationStore(context)
        return JSONObject()
            .put("success", true)
            .put("automations", recordsToJson(store.list()))
            .put("variables", store.listVariables())
            .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
            .put("available_triggers", JSONArray(AUTOMATION_TRIGGERS))
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
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_SHELL,
            payload = command,
            defaultLabel = "Hermes shell automation",
        )
    }

    fun createFileWriteTaskJson(context: Context, arguments: JSONObject): String {
        val path = stringArgument(arguments, "path", "file_path", "filename", "name")?.trim()
            ?: return errorJson("create_file_write_task requires a path argument")
        if (path.indexOf('\u0000') >= 0) {
            return errorJson("create_file_write_task path must not contain NUL bytes")
        }
        val content = stringArgument(arguments, "content", "text", "data", allowEmpty = true)
            ?: return errorJson("create_file_write_task requires a content argument")
        val payload = JSONObject()
            .put("path", path)
            .put("content", content)
            .put("append", arguments.optBoolean("append", false))
            .toString()
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_FILE_WRITE,
            payload = payload,
            defaultLabel = "Hermes file write automation",
        )
    }

    fun createFileDeleteTaskJson(context: Context, arguments: JSONObject): String {
        val path = stringArgument(arguments, "path", "file_path", "filename", "name")?.trim()
            ?: return errorJson("create_file_delete_task requires a path argument")
        if (path.indexOf('\u0000') >= 0) {
            return errorJson("create_file_delete_task path must not contain NUL bytes")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_FILE_DELETE,
            payload = path,
            defaultLabel = "Hermes file delete automation",
        )
    }

    fun createSystemActionTaskJson(context: Context, arguments: JSONObject): String {
        val systemAction = stringArgument(arguments, "system_action", "systemAction", "device_action", "command", "target_action")
            ?.trim()
            ?.lowercase()
            ?: return errorJson("create_system_action_task requires a system_action argument")
        if (systemAction in PRIVILEGED_SHELL_ACTIONS) {
            return errorJson("create_system_action_task does not store privileged shell commands; use create_shell_task with use_shizuku instead")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_SYSTEM_ACTION,
            payload = systemAction,
            defaultLabel = "Hermes Android system automation",
        )
    }

    fun createUiActionTaskJson(context: Context, arguments: JSONObject): String {
        val uiAction = normalizeUiAction(
            stringArgument(arguments, "ui_action", "uiAction", "target_action", "ui_command", "command")
                ?: return errorJson("create_ui_action_task requires a ui_action argument"),
        )
        if (uiAction !in UI_AUTOMATION_ACTIONS) {
            return errorJson("Unsupported saved UI action: $uiAction. Use one of: ${UI_AUTOMATION_ACTIONS.joinToString()}")
        }

        val payload = JSONObject().put("ui_action", uiAction)
        putOptionalExpandedPayloadString(
            payload,
            "text_contains",
            arguments,
            "text_contains",
            "textContains",
            "selector_text",
            "match_text",
        )
        putOptionalExpandedPayloadString(
            payload,
            "content_description_contains",
            arguments,
            "content_description_contains",
            "contentDescriptionContains",
            "content_description",
            "description_contains",
        )
        putOptionalExpandedPayloadString(payload, "view_id", arguments, "view_id", "viewId")
        putOptionalExpandedPayloadString(payload, "package_name", arguments, "package_name", "packageName")
        putOptionalExpandedPayloadString(payload, "value", arguments, "value", "text_value", "content", allowEmpty = true)
        if (arguments.has("index") && !arguments.isNull("index")) {
            payload.put("index", arguments.optInt("index", 0).coerceAtLeast(0))
        }

        if (uiAction in UI_SELECTOR_ACTIONS && !hasUiSelector(payload)) {
            return errorJson("create_ui_action_task selector actions require text_contains, content_description_contains, view_id, or package_name")
        }
        if (uiAction == "set_text" && !payload.has("value")) {
            return errorJson("create_ui_action_task set_text requires a value argument")
        }

        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_UI_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes UI automation",
        )
    }

    fun createAppLaunchTaskJson(context: Context, arguments: JSONObject): String {
        val packageName = stringArgument(
            arguments,
            "package_name",
            "packageName",
            "package",
            "app_package",
            "application_id",
            "command",
        )?.trim() ?: return errorJson("create_app_launch_task requires a package_name argument")
        if (packageName.indexOf('\u0000') >= 0) {
            return errorJson("create_app_launch_task package_name must not contain NUL bytes")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_APP_LAUNCH,
            payload = packageName,
            defaultLabel = "Hermes app launch automation",
        )
    }

    private fun createRecordJson(
        context: Context,
        arguments: JSONObject,
        actionType: String,
        payload: String,
        defaultLabel: String,
    ): String {
        val intervalMinutes = optionalPositiveInt(arguments, "interval_minutes")
            ?: optionalPositiveInt(arguments, "every_minutes")
        if (intervalMinutes != null && intervalMinutes < HermesAutomationScheduler.MIN_INTERVAL_MINUTES) {
            return errorJson("interval_minutes must be at least ${HermesAutomationScheduler.MIN_INTERVAL_MINUTES}")
        }
        val triggerType = resolveTriggerType(arguments, intervalMinutes) ?: return errorJson(
            "Unsupported trigger. Use one of: ${AUTOMATION_TRIGGERS.joinToString()}",
        )
        if (triggerType == TRIGGER_INTERVAL && intervalMinutes == null) {
            return errorJson("interval trigger requires interval_minutes")
        }
        val triggerPackageName = stringArgument(
            arguments,
            "trigger_package_name",
            "triggerPackageName",
            "profile_package_name",
            "context_package_name",
            "app_context_package",
        )?.trim().orEmpty()
        if (triggerPackageName.indexOf('\u0000') >= 0) {
            return errorJson("trigger_package_name must not contain NUL bytes")
        }
        if (triggerType == TRIGGER_APP_FOREGROUND && triggerPackageName.isBlank()) {
            return errorJson("app_foreground trigger requires trigger_package_name")
        }
        if (triggerType == TRIGGER_NOTIFICATION_POSTED && triggerPackageName.isBlank()) {
            return errorJson("notification_posted trigger requires trigger_package_name")
        }
        val now = System.currentTimeMillis()
        val record = HermesAutomationRecord(
            id = arguments.optString("id").ifBlank { "auto_${UUID.randomUUID().toString().replace("-", "").take(16)}" },
            label = arguments.optString("label").ifBlank { defaultLabel }.take(80),
            actionType = actionType,
            command = payload,
            useShizuku = arguments.optBoolean("use_shizuku", false),
            triggerType = triggerType,
            triggerPackageName = triggerPackageName,
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
                when {
                    record.triggerType == TRIGGER_INTERVAL && record.enabled -> "Created and scheduled Android automation"
                    record.triggerType == TRIGGER_MANUAL -> "Created manual Android automation"
                    else -> "Created Android automation for ${record.triggerType}"
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
        return runRecordJson(context, store, record, trigger).toString()
    }

    fun runTriggerJson(context: Context, trigger: String): String {
        val normalizedTrigger = normalizeTrigger(trigger) ?: return errorJson(
            "run_trigger requires one of: ${AUTOMATION_TRIGGERS.joinToString()}",
        )
        if (normalizedTrigger == TRIGGER_APP_FOREGROUND) {
            return errorJson("app_foreground trigger requires run_app_foreground_trigger with trigger_package_name or package_name")
        }
        if (normalizedTrigger == TRIGGER_NOTIFICATION_POSTED) {
            return errorJson("notification_posted trigger requires run_notification_posted_trigger with trigger_package_name or package_name")
        }
        val store = HermesAutomationStore(context)
        val records = store.list()
            .filter { record -> record.enabled && record.triggerType == normalizedTrigger }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, normalizedTrigger))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", normalizedTrigger)
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runAppForegroundTriggerJson(context: Context, packageName: String): String {
        val foregroundPackageName = packageName.trim()
        if (foregroundPackageName.isBlank()) {
            return errorJson("app_foreground trigger requires a package name")
        }
        if (foregroundPackageName.indexOf('\u0000') >= 0) {
            return errorJson("app_foreground package name must not contain NUL bytes")
        }
        val store = HermesAutomationStore(context)
        val variables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_APP_FOREGROUND &&
                    triggerPackageMatches(record.triggerPackageName, foregroundPackageName, variables)
            }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_APP_FOREGROUND))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_APP_FOREGROUND)
            .put("package_name", foregroundPackageName)
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runNotificationPostedTriggerJson(
        context: Context,
        packageName: String,
        title: String = "",
        text: String = "",
    ): String {
        val notificationPackageName = packageName.trim()
        if (notificationPackageName.isBlank()) {
            return errorJson("notification_posted trigger requires a package name")
        }
        if (notificationPackageName.indexOf('\u0000') >= 0) {
            return errorJson("notification_posted package name must not contain NUL bytes")
        }
        val store = HermesAutomationStore(context)
        store.setVariable("NOTIFICATION_PACKAGE", notificationPackageName)
        store.setVariable("NOTIFICATION_TITLE", title)
        store.setVariable("NOTIFICATION_TEXT", text)
        val variables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_NOTIFICATION_POSTED &&
                    triggerPackageMatches(record.triggerPackageName, notificationPackageName, variables)
            }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_NOTIFICATION_POSTED))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_NOTIFICATION_POSTED)
            .put("package_name", notificationPackageName)
            .put("notification_title", title.take(MAX_EVENT_VALUE_CHARS))
            .put("notification_text", text.take(MAX_EVENT_VALUE_CHARS))
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    private fun runRecordJson(
        context: Context,
        store: HermesAutomationStore,
        record: HermesAutomationRecord,
        trigger: String,
    ): JSONObject {
        val variables = store.listVariables()
        val rawResult = when (record.actionType) {
            ACTION_TYPE_SHELL -> runShellRecord(context, record, variables)
            ACTION_TYPE_FILE_WRITE -> runFileWriteRecord(context, record, variables)
            ACTION_TYPE_FILE_DELETE -> HermesWorkspaceFileBridge.deleteJson(context, expandVariables(record.command, variables))
            ACTION_TYPE_SYSTEM_ACTION -> runSystemActionRecord(context, record, variables)
            ACTION_TYPE_UI_ACTION -> runUiActionRecord(record, variables)
            ACTION_TYPE_APP_LAUNCH -> HermesAppControlBridge.launchPackage(context, expandVariables(record.command, variables))
            else -> JSONObject(errorJson("Unsupported Android automation action type: ${record.actionType}"))
        }
        val exitCode = rawResult.optInt("exit_code", if (rawResult.optBoolean("success", false)) 0 else -1)
        val success = rawResult.optBoolean("success", exitCode == 0) || exitCode == 0
        val resultText = listOf(
            rawResult.optString("output"),
            rawResult.optString("message"),
            rawResult.optString("path"),
            rawResult.optString("error"),
        )
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
    }

    private fun runShellRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val command = expandVariables(record.command, variables)
        return if (record.useShizuku) {
            JSONObject(HermesPrivilegedAccessBridge.runShellCommandJson(context, command, AUTOMATION_TIMEOUT_SECONDS))
        } else {
            NativeAndroidShellTool.run(context, command, AUTOMATION_TIMEOUT_SECONDS.toLong())
        }
    }

    private fun runFileWriteRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved file_write automation payload is invalid"))
        val path = expandVariables(payload.optString("path"), variables)
        val content = expandVariables(payload.optString("content"), variables)
        return HermesWorkspaceFileBridge.writeTextJson(
            context = context,
            rawPath = path,
            content = content,
            append = payload.optBoolean("append", false),
        )
    }

    private fun runSystemActionRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val systemAction = expandVariables(record.command, variables).trim().lowercase()
        if (systemAction in PRIVILEGED_SHELL_ACTIONS) {
            return JSONObject(errorJson("Saved system_action automation cannot run privileged shell; use a Shizuku shell task"))
        }
        val result = HermesSystemControlBridge.performAction(context, systemAction)
        return JSONObject()
            .put("exit_code", if (result.success) 0 else 1)
            .put("success", result.success)
            .put("action", result.action)
            .put("message", result.message)
    }

    private fun runUiActionRecord(record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved ui_action automation payload is invalid"))
        val uiAction = normalizeUiAction(expandVariables(payload.optString("ui_action"), variables))
        if (uiAction !in UI_AUTOMATION_ACTIONS) {
            return JSONObject(errorJson("Unsupported saved UI action: $uiAction"))
        }
        if (uiAction in UI_GLOBAL_ACTIONS) {
            return JSONObject(HermesAccessibilityUiBridge.performGlobalActionJson(uiAction))
        }
        return JSONObject(
            HermesAccessibilityUiBridge.performActionJson(
                action = uiAction,
                textContains = expandVariables(payload.optString("text_contains"), variables),
                contentDescriptionContains = expandVariables(payload.optString("content_description_contains"), variables),
                viewId = expandVariables(payload.optString("view_id"), variables),
                packageName = expandVariables(payload.optString("package_name"), variables),
                value = expandVariables(payload.optString("value"), variables),
                index = payload.optInt("index", 0),
            ),
        )
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

    fun listVariablesJson(context: Context): String {
        return JSONObject()
            .put("success", true)
            .put("variables", HermesAutomationStore(context).listVariables())
            .toString()
    }

    fun setVariableJson(context: Context, arguments: JSONObject): String {
        val name = arguments.optString("name").ifBlank { arguments.optString("variable") }
        val normalized = HermesAutomationStore.normalizeVariableName(name)
            ?: return errorJson("set_variable requires a variable name like NAME or %NAME")
        val value = arguments.optString("value")
        HermesAutomationStore(context).setVariable(normalized, value)
        return JSONObject()
            .put("success", true)
            .put("name", normalized)
            .put("value", value.take(MAX_VARIABLE_VALUE_CHARS))
            .toString()
    }

    fun getVariableJson(context: Context, arguments: JSONObject): String {
        val name = arguments.optString("name").ifBlank { arguments.optString("variable") }
        val normalized = HermesAutomationStore.normalizeVariableName(name)
            ?: return errorJson("get_variable requires a variable name like NAME or %NAME")
        val value = HermesAutomationStore(context).getVariable(normalized)
        return JSONObject()
            .put("success", true)
            .put("name", normalized)
            .put("found", value != null)
            .put("value", value ?: JSONObject.NULL)
            .toString()
    }

    fun deleteVariableJson(context: Context, arguments: JSONObject): String {
        val name = arguments.optString("name").ifBlank { arguments.optString("variable") }
        val normalized = HermesAutomationStore.normalizeVariableName(name)
            ?: return errorJson("delete_variable requires a variable name like NAME or %NAME")
        val removed = HermesAutomationStore(context).removeVariable(normalized)
        return JSONObject()
            .put("success", removed)
            .put("name", normalized)
            .put("message", if (removed) "Deleted Android automation variable" else "Android automation variable was not found")
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

    private fun stringArgument(arguments: JSONObject, vararg keys: String, allowEmpty: Boolean = false): String? {
        return keys.firstNotNullOfOrNull { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                null
            } else {
                arguments.optString(key).takeIf { allowEmpty || it.isNotBlank() }
            }
        }
    }

    private fun putOptionalExpandedPayloadString(
        payload: JSONObject,
        targetKey: String,
        arguments: JSONObject,
        vararg keys: String,
        allowEmpty: Boolean = false,
    ) {
        stringArgument(arguments, *keys, allowEmpty = allowEmpty)?.let { value ->
            payload.put(targetKey, value)
        }
    }

    private fun recordsToJson(records: List<HermesAutomationRecord>): JSONArray {
        return JSONArray().apply {
            records.forEach { record -> put(record.toJson()) }
        }
    }

    private fun resolveTriggerType(arguments: JSONObject, intervalMinutes: Int?): String? {
        if (intervalMinutes != null) {
            return TRIGGER_INTERVAL
        }
        val rawTrigger = arguments.optString("trigger_type")
            .ifBlank { arguments.optString("trigger") }
            .ifBlank { TRIGGER_MANUAL }
        return normalizeTrigger(rawTrigger)
    }

    private fun normalizeTrigger(trigger: String): String? {
        val normalized = trigger.trim().lowercase().replace("-", "_").replace(" ", "_")
        return TRIGGER_SYNONYMS[normalized] ?: normalized.takeIf { it in AUTOMATION_TRIGGERS }
    }

    private fun triggerPackageMatches(savedPackageName: String, foregroundPackageName: String, variables: JSONObject): Boolean {
        val expanded = expandVariables(savedPackageName, variables).trim()
        return expanded.isNotBlank() && expanded.equals(foregroundPackageName, ignoreCase = true)
    }

    private fun normalizeUiAction(action: String): String {
        val normalized = action.trim().lowercase().replace("-", "_").replace(" ", "_")
        return UI_ACTION_SYNONYMS[normalized] ?: normalized
    }

    private fun hasUiSelector(payload: JSONObject): Boolean {
        return UI_SELECTOR_KEYS.any { key -> payload.optString(key).isNotBlank() }
    }

    private fun expandVariables(command: String, variables: JSONObject): String {
        val taskerExpanded = TASKER_VARIABLE_PATTERN.replace(command) { match ->
            val variableName = match.groupValues[1].uppercase()
            if (variables.has(variableName)) variables.optString(variableName) else match.value
        }
        return BRACE_VARIABLE_PATTERN.replace(taskerExpanded) { match ->
            val variableName = match.groupValues[1].uppercase()
            if (variables.has(variableName)) variables.optString(variableName) else match.value
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
            .put("available_triggers", JSONArray(AUTOMATION_TRIGGERS))
            .toString()
    }

    private val AUTOMATION_ACTIONS = listOf(
        "list",
        "create_shell_task",
        "create_file_write_task",
        "create_file_delete_task",
        "create_system_action_task",
        "create_ui_action_task",
        "create_app_launch_task",
        "run",
        "run_trigger",
        "run_app_foreground_trigger",
        "run_notification_posted_trigger",
        "delete",
        "enable",
        "disable",
        "list_variables",
        "set_variable",
        "get_variable",
        "delete_variable",
    )
    private val AUTOMATION_TRIGGERS = listOf(
        TRIGGER_MANUAL,
        TRIGGER_INTERVAL,
        TRIGGER_BOOT,
        TRIGGER_POWER_CONNECTED,
        TRIGGER_POWER_DISCONNECTED,
        TRIGGER_BATTERY_LOW,
        TRIGGER_BATTERY_OKAY,
        TRIGGER_APP_FOREGROUND,
        TRIGGER_NOTIFICATION_POSTED,
    )
    private val TRIGGER_SYNONYMS = mapOf(
        "boot_completed" to TRIGGER_BOOT,
        "on_boot" to TRIGGER_BOOT,
        "charging" to TRIGGER_POWER_CONNECTED,
        "charger_connected" to TRIGGER_POWER_CONNECTED,
        "power" to TRIGGER_POWER_CONNECTED,
        "unplugged" to TRIGGER_POWER_DISCONNECTED,
        "charger_disconnected" to TRIGGER_POWER_DISCONNECTED,
        "battery_low_state" to TRIGGER_BATTERY_LOW,
        "battery_ok" to TRIGGER_BATTERY_OKAY,
        "battery_normal" to TRIGGER_BATTERY_OKAY,
        "app" to TRIGGER_APP_FOREGROUND,
        "application" to TRIGGER_APP_FOREGROUND,
        "app_context" to TRIGGER_APP_FOREGROUND,
        "app_changed" to TRIGGER_APP_FOREGROUND,
        "app_launch" to TRIGGER_APP_FOREGROUND,
        "app_opened" to TRIGGER_APP_FOREGROUND,
        "foreground_app" to TRIGGER_APP_FOREGROUND,
        "package_foreground" to TRIGGER_APP_FOREGROUND,
        "notification" to TRIGGER_NOTIFICATION_POSTED,
        "notification_posted" to TRIGGER_NOTIFICATION_POSTED,
        "notification_received" to TRIGGER_NOTIFICATION_POSTED,
        "posted_notification" to TRIGGER_NOTIFICATION_POSTED,
        "notify" to TRIGGER_NOTIFICATION_POSTED,
    )
    private val PRIVILEGED_SHELL_ACTIONS = setOf("run_privileged_shell", "shizuku_shell", "privileged_shell")
    private val UI_GLOBAL_ACTIONS = setOf("back", "home", "recents", "notifications", "quick_settings")
    private val UI_SELECTOR_ACTIONS = setOf("click", "long_click", "focus", "set_text", "scroll_forward", "scroll_backward")
    private val UI_AUTOMATION_ACTIONS = UI_GLOBAL_ACTIONS + UI_SELECTOR_ACTIONS
    private val UI_SELECTOR_KEYS = listOf("text_contains", "content_description_contains", "view_id", "package_name")
    private val UI_ACTION_SYNONYMS = mapOf(
        "global_back" to "back",
        "global_home" to "home",
        "global_recents" to "recents",
        "global_notifications" to "notifications",
        "global_quick_settings" to "quick_settings",
        "type" to "set_text",
        "text" to "set_text",
        "input_text" to "set_text",
        "scroll_down" to "scroll_forward",
        "scroll_up" to "scroll_backward",
    )
    private val TASKER_VARIABLE_PATTERN = Regex("%([A-Za-z_][A-Za-z0-9_]{1,63})")
    private val BRACE_VARIABLE_PATTERN = Regex("\\{\\{([A-Za-z_][A-Za-z0-9_]{0,63})\\}\\}")
    private const val AUTOMATION_TIMEOUT_SECONDS = 30
    private const val MAX_VARIABLE_VALUE_CHARS = 4_000
    private const val MAX_RESULT_CHARS = 2_000
    private const val MAX_EVENT_VALUE_CHARS = 500
}
