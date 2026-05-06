package com.nousresearch.hermesagent.device

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.Calendar
import java.util.Locale
import java.util.UUID
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

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
            "create_intent_task", "create_android_intent_task", "intent_task" -> createIntentTaskJson(context, arguments)
            "create_uri_task", "create_open_uri_task", "open_uri_task" -> createIntentTaskJson(context, arguments, "open_uri")
            "create_broadcast_task", "create_send_broadcast_task", "broadcast_task" -> createIntentTaskJson(context, arguments, "send_broadcast")
            "create_activity_task", "create_start_activity_task", "launch_activity_task" -> createIntentTaskJson(context, arguments, "start_activity")
            "create_shizuku_action_task", "create_shizuku_action", "create_privileged_action_task", "privileged_action_task" -> createShizukuActionTaskJson(context, arguments)
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
            "run_calendar_event_trigger", "trigger_calendar_event", "calendar_event", "calendar" -> runCalendarEventTriggerJson(
                context = context,
                arguments = arguments,
            )
            "run_location_trigger", "trigger_location", "location_event", "location" -> runLocationTriggerJson(
                context = context,
                arguments = arguments,
            )
            "run_shizuku_state_trigger", "trigger_shizuku_state", "check_shizuku_trigger", "shizuku_state" -> runShizukuStateTriggerJson(
                context = context,
                requestedState = stringArgument(arguments, "shizuku_state", "state", "expected_state", "trigger_state").orEmpty(),
            )
            "run_time_trigger", "trigger_time", "time" -> runTriggerJson(context, TRIGGER_TIME)
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

    fun createIntentTaskJson(context: Context, arguments: JSONObject, defaultIntentTaskAction: String? = null): String {
        val rawIntentTaskAction = stringArgument(
            arguments,
            "intent_task_action",
            "intent_action_type",
            "intent_mode",
            "intent_type",
            "task_action",
        )?.trim().orEmpty().ifBlank {
            defaultIntentTaskAction ?: inferIntentTaskAction(arguments)
        }
        val intentTaskAction = HermesIntentBridge.normalizeIntentTaskAction(rawIntentTaskAction)
            ?: return errorJson("Unsupported Android intent task action: $rawIntentTaskAction. Use start_activity, open_uri, or send_broadcast")
        val payload = JSONObject().put("intent_task_action", intentTaskAction)
        putOptionalExpandedPayloadString(
            payload,
            "intent_action",
            arguments,
            "intent_action",
            "android_intent_action",
            "action_name",
            "command",
        )
        putOptionalExpandedPayloadString(payload, "data_uri", arguments, "data_uri", "uri", "url", "data")
        putOptionalExpandedPayloadString(
            payload,
            "package_name",
            arguments,
            "package_name",
            "packageName",
            "package",
            "app_package",
            "application_id",
        )
        putOptionalExpandedPayloadString(payload, "class_name", arguments, "class_name", "className", "activity_class")
        putOptionalExpandedPayloadString(payload, "component", arguments, "component", "component_name", "componentName")
        putOptionalExpandedPayloadString(payload, "category", arguments, "category", "intent_category")
        copyStringArrayPayload(payload, arguments, "categories", "categories", "intent_categories")
        copyExtrasPayload(payload, arguments)

        val validation = HermesIntentBridge.performIntentJson(context, payload.put("__validate_only", true))
        payload.remove("__validate_only")
        if (!validation.optBoolean("success", false) && validation.optInt("exit_code", 1) == 64) {
            return errorJson(validation.optString("error").ifBlank { "Invalid Android intent automation payload" })
        }

        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_INTENT,
            payload = payload.toString(),
            defaultLabel = "Hermes Android intent automation",
        )
    }

    fun createShizukuActionTaskJson(context: Context, arguments: JSONObject): String {
        val rawAction = stringArgument(
            arguments,
            "shizuku_action",
            "privileged_action",
            "system_action",
            "target_action",
            "device_action",
            "action_name",
            "command",
        )?.trim() ?: return errorJson("create_shizuku_action_task requires a shizuku_action argument")
        val shizukuAction = HermesPrivilegedAccessBridge.normalizeStructuredAction(rawAction)
            ?: return errorJson("Unsupported saved Shizuku action: $rawAction. Use one of: ${SHIZUKU_AUTOMATION_ACTIONS.joinToString()}")
        if (shizukuAction.indexOf('\u0000') >= 0) {
            return errorJson("create_shizuku_action_task shizuku_action must not contain NUL bytes")
        }

        val packageName = stringArgument(
            arguments,
            "package_name",
            "packageName",
            "package",
            "app_package",
            "application_id",
        )?.trim() ?: return errorJson("create_shizuku_action_task requires a package_name argument")
        if (packageName.indexOf('\u0000') >= 0) {
            return errorJson("create_shizuku_action_task package_name must not contain NUL bytes")
        }

        val payload = JSONObject()
            .put("shizuku_action", shizukuAction)
            .put("package_name", packageName)
        if (shizukuAction in SHIZUKU_PERMISSION_ACTIONS) {
            val permission = stringArgument(arguments, "permission", "permission_name", "permissionName", "android_permission")
                ?.trim()
                ?: return errorJson("create_shizuku_action_task $shizukuAction requires a permission argument")
            if (permission.indexOf('\u0000') >= 0) {
                return errorJson("create_shizuku_action_task permission must not contain NUL bytes")
            }
            payload.put("permission", permission)
        }
        if (shizukuAction == "set_app_enabled") {
            when {
                arguments.has("target_enabled") && !arguments.isNull("target_enabled") ->
                    payload.put("target_enabled", arguments.optBoolean("target_enabled"))
                arguments.has("app_enabled") && !arguments.isNull("app_enabled") ->
                    payload.put("target_enabled", arguments.optBoolean("app_enabled"))
                arguments.has("desired_enabled") && !arguments.isNull("desired_enabled") ->
                    payload.put("target_enabled", arguments.optBoolean("desired_enabled"))
                arguments.has("enabled") && !arguments.isNull("enabled") ->
                    payload.put("target_enabled", arguments.optBoolean("enabled"))
                else -> stringArgument(arguments, "state", "enabled_state")?.trim()?.let { state ->
                    if (state.indexOf('\u0000') >= 0) {
                        return errorJson("create_shizuku_action_task enabled state must not contain NUL bytes")
                    }
                    payload.put("state", state)
                }
            }
        }
        optionalPositiveInt(arguments, "timeout_seconds")?.let { timeout ->
            payload.put("timeout_seconds", timeout)
        }

        val recordArguments = JSONObject(arguments.toString())
            .put("use_shizuku", true)
        if (shizukuAction == "set_app_enabled" && !recordArguments.has("automation_enabled")) {
            recordArguments.put("automation_enabled", true)
        }
        return createRecordJson(
            context = context,
            arguments = recordArguments,
            actionType = ACTION_TYPE_SHIZUKU_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes Shizuku app automation",
            forceUseShizuku = true,
        )
    }

    private fun createRecordJson(
        context: Context,
        arguments: JSONObject,
        actionType: String,
        payload: String,
        defaultLabel: String,
        forceUseShizuku: Boolean = false,
    ): String {
        val intervalMinutes = optionalPositiveInt(arguments, "interval_minutes")
            ?: optionalPositiveInt(arguments, "every_minutes")
        if (intervalMinutes != null && intervalMinutes < HermesAutomationScheduler.MIN_INTERVAL_MINUTES) {
            return errorJson("interval_minutes must be at least ${HermesAutomationScheduler.MIN_INTERVAL_MINUTES}")
        }
        val triggerTime = parseTimeArgument(arguments)
        if (triggerTime.error != null) {
            return errorJson(triggerTime.error)
        }
        val triggerDays = parseDaysOfWeekArgument(arguments)
        if (triggerDays.error != null) {
            return errorJson(triggerDays.error)
        }
        val triggerType = resolveTriggerType(arguments, intervalMinutes, triggerTime.minutes) ?: return errorJson(
            "Unsupported trigger. Use one of: ${AUTOMATION_TRIGGERS.joinToString()}",
        )
        if (triggerType == TRIGGER_INTERVAL && intervalMinutes == null) {
            return errorJson("interval trigger requires interval_minutes")
        }
        if (triggerType == TRIGGER_TIME && triggerTime.minutes == null) {
            return errorJson("time trigger requires a time argument such as 08:30")
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
        val triggerData = buildTriggerData(arguments, triggerType)
        if (triggerData.error != null) {
            return errorJson(triggerData.error)
        }
        val now = System.currentTimeMillis()
        val record = HermesAutomationRecord(
            id = arguments.optString("id").ifBlank { "auto_${UUID.randomUUID().toString().replace("-", "").take(16)}" },
            label = arguments.optString("label").ifBlank { defaultLabel }.take(80),
            actionType = actionType,
            command = payload,
            useShizuku = forceUseShizuku || arguments.optBoolean("use_shizuku", false),
            triggerType = triggerType,
            triggerPackageName = triggerPackageName,
            triggerTimeMinutes = triggerTime.minutes.takeIf { triggerType == TRIGGER_TIME },
            triggerDaysOfWeek = triggerDays.daysCsv.takeIf { triggerType == TRIGGER_TIME }.orEmpty(),
            intervalMinutes = intervalMinutes,
            enabled = recordEnabled(arguments),
            createdAtEpochMs = now,
            updatedAtEpochMs = now,
            triggerData = triggerData.data,
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
                    record.triggerType == TRIGGER_TIME && record.enabled -> "Created Android automation for ${formatTime(record.triggerTimeMinutes)}"
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
        val rawTrigger = trigger.trim().lowercase().replace("-", "_").replace(" ", "_")
        if (rawTrigger == "shizuku_state" || rawTrigger == "check_shizuku" || rawTrigger == "current_shizuku") {
            return runShizukuStateTriggerJson(context)
        }
        val normalizedTrigger = normalizeTrigger(trigger) ?: return errorJson(
            "run_trigger requires one of: ${AUTOMATION_TRIGGERS.joinToString()}",
        )
        if (normalizedTrigger == TRIGGER_APP_FOREGROUND) {
            return errorJson("app_foreground trigger requires run_app_foreground_trigger with trigger_package_name or package_name")
        }
        if (normalizedTrigger == TRIGGER_NOTIFICATION_POSTED) {
            return errorJson("notification_posted trigger requires run_notification_posted_trigger with trigger_package_name or package_name")
        }
        if (normalizedTrigger == TRIGGER_CALENDAR_EVENT) {
            return runCalendarEventTriggerJson(context, JSONObject())
        }
        if (normalizedTrigger == TRIGGER_LOCATION) {
            return errorJson("location trigger requires run_location_trigger with latitude and longitude")
        }
        if (normalizedTrigger == TRIGGER_SHIZUKU_AVAILABLE || normalizedTrigger == TRIGGER_SHIZUKU_UNAVAILABLE) {
            return runShizukuStateTriggerJson(context, normalizedTrigger)
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

    fun runCalendarEventTriggerJson(context: Context, arguments: JSONObject): String {
        val calendarName = stringArgument(
            arguments,
            "calendar_name",
            "calendarName",
            "calendar",
            "calendar_name_contains",
            allowEmpty = true,
        ).orEmpty().trim()
        val title = stringArgument(
            arguments,
            "calendar_title",
            "event_title",
            "title",
            "title_contains",
            allowEmpty = true,
        ).orEmpty()
        val description = stringArgument(
            arguments,
            "calendar_description",
            "event_description",
            "description",
            "description_contains",
            allowEmpty = true,
        ).orEmpty()
        val location = stringArgument(
            arguments,
            "calendar_location",
            "event_location",
            "location",
            "location_contains",
            allowEmpty = true,
        ).orEmpty()
        calendarEventNulError(calendarName, title, description, location)?.let { error ->
            return errorJson(error)
        }
        val store = HermesAutomationStore(context)
        setCalendarEventVariables(store, calendarName, title, description, location)
        val variables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_CALENDAR_EVENT &&
                    calendarEventMatches(record.triggerData, variables, calendarName, title, description, location)
            }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_CALENDAR_EVENT))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_CALENDAR_EVENT)
            .put("calendar_name", calendarName.take(MAX_EVENT_VALUE_CHARS))
            .put("calendar_title", title.take(MAX_EVENT_VALUE_CHARS))
            .put("calendar_description", description.take(MAX_EVENT_VALUE_CHARS))
            .put("calendar_location", location.take(MAX_EVENT_VALUE_CHARS))
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runLocationTriggerJson(context: Context, arguments: JSONObject): String {
        val latitude = optionalDoubleArgument(
            arguments,
            "latitude",
            "lat",
            "location_latitude",
            "trigger_latitude",
        ) ?: return errorJson("location trigger requires latitude")
        val longitude = optionalDoubleArgument(
            arguments,
            "longitude",
            "lon",
            "lng",
            "location_longitude",
            "trigger_longitude",
        ) ?: return errorJson("location trigger requires longitude")
        if (latitude !in -90.0..90.0) {
            return errorJson("location latitude must be between -90 and 90")
        }
        if (longitude !in -180.0..180.0) {
            return errorJson("location longitude must be between -180 and 180")
        }
        val requestedAccuracyMeters = optionalDoubleArgument(
            arguments,
            "accuracy_meters",
            "accuracy",
            "location_accuracy_meters",
            "trigger_accuracy_meters",
        )
        if (requestedAccuracyMeters != null && requestedAccuracyMeters < 0.0) {
            return errorJson("location accuracy_meters must be zero or greater")
        }
        val accuracyMeters = requestedAccuracyMeters
        val provider = stringArgument(
            arguments,
            "location_provider",
            "provider",
            "source",
            allowEmpty = true,
        ).orEmpty().trim()
        val name = stringArgument(
            arguments,
            "location_name",
            "place_name",
            "location_label",
            "place",
            "location",
            allowEmpty = true,
        ).orEmpty().trim()
        locationEventNulError(provider, name)?.let { error ->
            return errorJson(error)
        }
        val store = HermesAutomationStore(context)
        setLocationEventVariables(store, latitude, longitude, accuracyMeters, provider, name)
        val variables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_LOCATION &&
                    locationEventMatches(record.triggerData, variables, latitude, longitude, accuracyMeters, provider, name)
            }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_LOCATION))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_LOCATION)
            .put("latitude", latitude)
            .put("longitude", longitude)
            .put("accuracy_meters", accuracyMeters ?: JSONObject.NULL)
            .put("location_provider", provider.take(MAX_EVENT_VALUE_CHARS))
            .put("location_name", name.take(MAX_EVENT_VALUE_CHARS))
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runShizukuStateTriggerJson(context: Context, requestedState: String = ""): String {
        val store = HermesAutomationStore(context)
        val status = HermesPrivilegedAccessBridge.readStatus(context)
        val available = status.shizukuBinderAlive && status.shizukuPermissionGranted
        setShizukuEventVariables(store, status, available)
        val trigger = normalizeShizukuTrigger(requestedState, available) ?: return errorJson(
            "shizuku_state trigger requires available, unavailable, shizuku_available, or shizuku_unavailable",
        )
        val records = store.list()
            .filter { record -> record.enabled && record.triggerType == trigger }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, trigger))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", trigger)
            .put("shizuku_available", available)
            .put("shizuku_status", HermesPrivilegedAccessBridge.statusToJson(status))
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
        if (trigger == TRIGGER_TIME) {
            setTimeEventVariables(store)
        }
        val variables = store.listVariables()
        val rawResult = when (record.actionType) {
            ACTION_TYPE_SHELL -> runShellRecord(context, record, variables)
            ACTION_TYPE_FILE_WRITE -> runFileWriteRecord(context, record, variables)
            ACTION_TYPE_FILE_DELETE -> HermesWorkspaceFileBridge.deleteJson(context, expandVariables(record.command, variables))
            ACTION_TYPE_SYSTEM_ACTION -> runSystemActionRecord(context, record, variables)
            ACTION_TYPE_UI_ACTION -> runUiActionRecord(record, variables)
            ACTION_TYPE_APP_LAUNCH -> HermesAppControlBridge.launchPackage(context, expandVariables(record.command, variables))
            ACTION_TYPE_INTENT -> runIntentRecord(context, record, variables)
            ACTION_TYPE_SHIZUKU_ACTION -> runShizukuActionRecord(context, record, variables)
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

    private fun runIntentRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved intent automation payload is invalid"))
        return HermesIntentBridge.performIntentJson(context, expandIntentPayload(payload, variables))
    }

    private fun runShizukuActionRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved shizuku_action automation payload is invalid"))
        val rawAction = expandVariables(payload.optString("shizuku_action"), variables)
        val shizukuAction = HermesPrivilegedAccessBridge.normalizeStructuredAction(rawAction)
            ?: return JSONObject(errorJson("Unsupported saved Shizuku action: $rawAction"))
        val actionArguments = JSONObject()
            .put("package_name", expandVariables(payload.optString("package_name"), variables))
        if (shizukuAction in SHIZUKU_PERMISSION_ACTIONS && payload.has("permission")) {
            actionArguments.put("permission", expandVariables(payload.optString("permission"), variables))
        }
        if (payload.has("target_enabled") && !payload.isNull("target_enabled")) {
            actionArguments.put("enabled", payload.optBoolean("target_enabled"))
        }
        if (payload.has("state") && !payload.isNull("state")) {
            actionArguments.put("state", expandVariables(payload.optString("state"), variables))
        }
        if (payload.has("timeout_seconds") && !payload.isNull("timeout_seconds")) {
            actionArguments.put("timeout_seconds", payload.optInt("timeout_seconds", AUTOMATION_TIMEOUT_SECONDS))
        }
        return JSONObject(HermesPrivilegedAccessBridge.performStructuredActionJson(context, shizukuAction, actionArguments))
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

    private fun optionalDoubleArgument(arguments: JSONObject, vararg keys: String): Double? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        val value = arguments.opt(key) ?: return null
        return when (value) {
            is Number -> value.toDouble()
            else -> value.toString().trim().toDoubleOrNull()
        }
    }

    private fun recordEnabled(arguments: JSONObject): Boolean {
        return when {
            arguments.has("automation_enabled") && !arguments.isNull("automation_enabled") ->
                arguments.optBoolean("automation_enabled", true)
            arguments.has("record_enabled") && !arguments.isNull("record_enabled") ->
                arguments.optBoolean("record_enabled", true)
            else -> arguments.optBoolean("enabled", true)
        }
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

    private fun inferIntentTaskAction(arguments: JSONObject): String {
        if (stringArgument(arguments, "data_uri", "uri", "url", "data") != null &&
            stringArgument(arguments, "class_name", "className", "activity_class", "component", "component_name", "componentName") == null
        ) {
            return "open_uri"
        }
        return "start_activity"
    }

    private fun copyStringArrayPayload(payload: JSONObject, arguments: JSONObject, targetKey: String, vararg sourceKeys: String) {
        val sourceKey = sourceKeys.firstOrNull { key -> arguments.has(key) && !arguments.isNull(key) } ?: return
        val raw = arguments.opt(sourceKey) ?: return
        val values = when (raw) {
            is JSONArray -> (0 until raw.length()).mapNotNull { index -> raw.optString(index).takeIf { it.isNotBlank() } }
            else -> raw.toString().split(',', ';', '|', '\n').map { it.trim() }.filter { it.isNotBlank() }
        }
        if (values.isNotEmpty()) {
            payload.put(targetKey, JSONArray(values))
        }
    }

    private fun copyExtrasPayload(payload: JSONObject, arguments: JSONObject) {
        val extras = arguments.optJSONObject("extras") ?: arguments.optJSONObject("intent_extras") ?: return
        payload.put("extras", JSONObject().apply {
            extras.keys().forEach { key ->
                put(key, extras.opt(key))
            }
        })
    }

    private fun buildTriggerData(arguments: JSONObject, triggerType: String): TriggerDataResult {
        return when (triggerType) {
            TRIGGER_CALENDAR_EVENT -> buildCalendarTriggerData(arguments)
            TRIGGER_LOCATION -> buildLocationTriggerData(arguments)
            else -> TriggerDataResult("")
        }
    }

    private fun buildCalendarTriggerData(arguments: JSONObject): TriggerDataResult {
        val payload = JSONObject()
        listOf(
            copyCalendarTriggerFilter(
                payload,
                "calendar_name",
                arguments,
                "calendar_name",
                "calendarName",
                "calendar",
                "calendar_name_contains",
            ),
            copyCalendarTriggerFilter(
                payload,
                "title_contains",
                arguments,
                "title_contains",
                "event_title",
                "calendar_title",
                "title",
            ),
            copyCalendarTriggerFilter(
                payload,
                "description_contains",
                arguments,
                "description_contains",
                "event_description",
                "calendar_description",
                "description",
            ),
            copyCalendarTriggerFilter(
                payload,
                "location_contains",
                arguments,
                "location_contains",
                "event_location",
                "calendar_location",
                "location",
            ),
        ).firstOrNull { it != null }?.let { error ->
            return TriggerDataResult("", error)
        }
        return TriggerDataResult(payload.toString())
    }

    private fun buildLocationTriggerData(arguments: JSONObject): TriggerDataResult {
        val payload = JSONObject()
        listOf(
            copyLocationNumericTriggerFilter(
                payload,
                "latitude",
                arguments,
                "latitude",
                "lat",
                "location_latitude",
                "trigger_latitude",
            ),
            copyLocationNumericTriggerFilter(
                payload,
                "longitude",
                arguments,
                "longitude",
                "lon",
                "lng",
                "location_longitude",
                "trigger_longitude",
            ),
            copyLocationNumericTriggerFilter(
                payload,
                "radius_meters",
                arguments,
                "radius_meters",
                "radius",
                "trigger_radius_meters",
                "distance_meters",
            ),
            copyLocationNumericTriggerFilter(
                payload,
                "max_accuracy_meters",
                arguments,
                "max_accuracy_meters",
                "accuracy_max_meters",
                "accuracy_meters_max",
                "location_max_accuracy_meters",
            ),
            copyCalendarTriggerFilter(
                payload,
                "provider",
                arguments,
                "location_provider",
                "provider",
                "source",
            ),
            copyCalendarTriggerFilter(
                payload,
                "location_name",
                arguments,
                "location_name",
                "location_name_contains",
                "place_name",
                "place",
                "location_label",
            ),
        ).firstOrNull { it != null }?.let { error ->
            return TriggerDataResult("", error)
        }
        val hasLatitude = payload.has("latitude")
        val hasLongitude = payload.has("longitude")
        if (hasLatitude != hasLongitude) {
            return TriggerDataResult("", "location trigger requires both latitude and longitude when using a coordinate filter")
        }
        if (payload.has("radius_meters") && (!hasLatitude || !hasLongitude)) {
            return TriggerDataResult("", "location trigger radius_meters requires latitude and longitude")
        }
        if (hasLatitude && !payload.has("radius_meters")) {
            payload.put("radius_meters", DEFAULT_LOCATION_RADIUS_METERS.toString())
        }
        validateLocationTriggerNumericPayload(payload)?.let { error ->
            return TriggerDataResult("", error)
        }
        return TriggerDataResult(payload.toString())
    }

    private fun validateLocationTriggerNumericPayload(payload: JSONObject): String? {
        literalDoubleFilter(payload, "latitude")?.let { latitude ->
            if (latitude !in -90.0..90.0) {
                return "location trigger latitude must be between -90 and 90"
            }
        }
        literalDoubleFilter(payload, "longitude")?.let { longitude ->
            if (longitude !in -180.0..180.0) {
                return "location trigger longitude must be between -180 and 180"
            }
        }
        literalDoubleFilter(payload, "radius_meters")?.let { radius ->
            if (radius <= 0.0) {
                return "location trigger radius_meters must be greater than zero"
            }
        }
        literalDoubleFilter(payload, "max_accuracy_meters")?.let { maxAccuracy ->
            if (maxAccuracy < 0.0) {
                return "location trigger max_accuracy_meters must be zero or greater"
            }
        }
        return null
    }

    private fun literalDoubleFilter(payload: JSONObject, key: String): Double? {
        if (!payload.has(key) || payload.isNull(key)) {
            return null
        }
        val value = payload.optString(key).trim()
        return if (looksLikeVariableReference(value)) null else value.toDoubleOrNull()
    }

    private fun copyCalendarTriggerFilter(
        payload: JSONObject,
        targetKey: String,
        arguments: JSONObject,
        vararg sourceKeys: String,
    ): String? {
        val sourceKey = sourceKeys.firstOrNull { key -> arguments.has(key) && !arguments.isNull(key) } ?: return null
        val value = arguments.optString(sourceKey).trim()
        if (value.indexOf('\u0000') >= 0) {
            return "$sourceKey must not contain NUL bytes"
        }
        if (value.isNotBlank()) {
            payload.put(targetKey, value)
        }
        return null
    }

    private fun copyLocationNumericTriggerFilter(
        payload: JSONObject,
        targetKey: String,
        arguments: JSONObject,
        vararg sourceKeys: String,
    ): String? {
        val sourceKey = sourceKeys.firstOrNull { key -> arguments.has(key) && !arguments.isNull(key) } ?: return null
        val raw = arguments.opt(sourceKey) ?: return null
        val value = when (raw) {
            is Number -> raw.toDouble().toString()
            else -> raw.toString().trim()
        }
        if (value.indexOf('\u0000') >= 0) {
            return "$sourceKey must not contain NUL bytes"
        }
        if (value.isBlank()) {
            return null
        }
        if (!looksLikeVariableReference(value) && value.toDoubleOrNull() == null) {
            return "$sourceKey must be a number"
        }
        payload.put(targetKey, value)
        return null
    }

    private fun expandIntentPayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        INTENT_STRING_PAYLOAD_KEYS.forEach { key ->
            if (payload.has(key) && !payload.isNull(key)) {
                expanded.put(key, expandVariables(payload.optString(key), variables))
            }
        }
        payload.optJSONArray("categories")?.let { categories ->
            expanded.put(
                "categories",
                JSONArray().apply {
                    for (index in 0 until categories.length()) {
                        put(expandVariables(categories.optString(index), variables))
                    }
                },
            )
        }
        payload.optJSONObject("extras")?.let { extras ->
            expanded.put(
                "extras",
                JSONObject().apply {
                    extras.keys().forEach { key ->
                        when (val value = extras.opt(key)) {
                            is String -> put(key, expandVariables(value, variables))
                            else -> put(key, value)
                        }
                    }
                },
            )
        }
        return expanded
    }

    private fun recordsToJson(records: List<HermesAutomationRecord>): JSONArray {
        return JSONArray().apply {
            records.forEach { record -> put(record.toJson()) }
        }
    }

    private fun resolveTriggerType(arguments: JSONObject, intervalMinutes: Int?, triggerTimeMinutes: Int?): String? {
        if (intervalMinutes != null) {
            return TRIGGER_INTERVAL
        }
        if (triggerTimeMinutes != null && !arguments.has("trigger_type") && !arguments.has("trigger")) {
            return TRIGGER_TIME
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

    private fun calendarEventMatches(
        triggerData: String,
        variables: JSONObject,
        calendarName: String,
        title: String,
        description: String,
        location: String,
    ): Boolean {
        val filters = runCatching { JSONObject(triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
        return textFilterMatches(filters.optString("calendar_name"), calendarName, variables) &&
            textFilterMatches(filters.optString("title_contains"), title, variables) &&
            textFilterMatches(filters.optString("description_contains"), description, variables) &&
            textFilterMatches(filters.optString("location_contains"), location, variables)
    }

    private fun textFilterMatches(filter: String, value: String, variables: JSONObject): Boolean {
        val expanded = expandVariables(filter, variables).trim()
        if (expanded.isBlank() || expanded == "*") {
            return true
        }
        if (expanded.indexOf('\u0000') >= 0) {
            return false
        }
        return value.contains(expanded, ignoreCase = true)
    }

    private fun locationEventMatches(
        triggerData: String,
        variables: JSONObject,
        latitude: Double,
        longitude: Double,
        accuracyMeters: Double?,
        provider: String,
        name: String,
    ): Boolean {
        val filters = runCatching { JSONObject(triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
        if (!textFilterMatches(filters.optString("provider"), provider, variables)) {
            return false
        }
        if (!textFilterMatches(filters.optString("location_name"), name, variables)) {
            return false
        }
        val maxAccuracy = expandedDoubleFilter(filters, "max_accuracy_meters", variables)
        if (maxAccuracy != null && (accuracyMeters == null || accuracyMeters > maxAccuracy)) {
            return false
        }
        if (!filters.has("latitude") && !filters.has("longitude")) {
            return true
        }
        val savedLatitude = expandedDoubleFilter(filters, "latitude", variables) ?: return false
        val savedLongitude = expandedDoubleFilter(filters, "longitude", variables) ?: return false
        if (savedLatitude !in -90.0..90.0 || savedLongitude !in -180.0..180.0) {
            return false
        }
        val radiusMeters = expandedDoubleFilter(filters, "radius_meters", variables) ?: DEFAULT_LOCATION_RADIUS_METERS
        if (radiusMeters <= 0.0) {
            return false
        }
        return distanceMeters(savedLatitude, savedLongitude, latitude, longitude) <= radiusMeters
    }

    private fun expandedDoubleFilter(filters: JSONObject, key: String, variables: JSONObject): Double? {
        if (!filters.has(key) || filters.isNull(key)) {
            return null
        }
        val expanded = expandVariables(filters.optString(key), variables).trim()
        if (expanded.isBlank() || expanded.indexOf('\u0000') >= 0) {
            return null
        }
        return expanded.toDoubleOrNull()
    }

    private fun distanceMeters(latitudeA: Double, longitudeA: Double, latitudeB: Double, longitudeB: Double): Double {
        val dLat = Math.toRadians(latitudeB - latitudeA)
        val dLon = Math.toRadians(longitudeB - longitudeA)
        val lat1 = Math.toRadians(latitudeA)
        val lat2 = Math.toRadians(latitudeB)
        val haversine = sin(dLat / 2) * sin(dLat / 2) +
            cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2)
        return EARTH_RADIUS_METERS * 2 * atan2(sqrt(haversine), sqrt(1 - haversine))
    }

    private fun calendarEventNulError(
        calendarName: String,
        title: String,
        description: String,
        location: String,
    ): String? {
        return when {
            calendarName.indexOf('\u0000') >= 0 -> "calendar_name must not contain NUL bytes"
            title.indexOf('\u0000') >= 0 -> "calendar_title must not contain NUL bytes"
            description.indexOf('\u0000') >= 0 -> "calendar_description must not contain NUL bytes"
            location.indexOf('\u0000') >= 0 -> "calendar_location must not contain NUL bytes"
            else -> null
        }
    }

    private fun locationEventNulError(provider: String, name: String): String? {
        return when {
            provider.indexOf('\u0000') >= 0 -> "location_provider must not contain NUL bytes"
            name.indexOf('\u0000') >= 0 -> "location_name must not contain NUL bytes"
            else -> null
        }
    }

    private fun parseTimeArgument(arguments: JSONObject): TimeParseResult {
        val key = TIME_ARGUMENT_KEYS.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) }
            ?: return TimeParseResult(null)
        val raw = arguments.opt(key) ?: return TimeParseResult(null)
        val minutes = when (raw) {
            is Number -> raw.toInt()
            else -> parseTimeString(raw.toString())
        }
        if (minutes == null || minutes !in 0..1439) {
            return TimeParseResult(null, "time trigger requires HH:mm, H:mm, HH.mm, or minutes from 0 to 1439")
        }
        return TimeParseResult(minutes)
    }

    private fun parseTimeString(raw: String): Int? {
        val trimmed = raw.trim()
        if (trimmed.isBlank()) {
            return null
        }
        val separator = when {
            ":" in trimmed -> ":"
            "." in trimmed -> "."
            else -> null
        }
        if (separator == null) {
            return trimmed.toIntOrNull()
        }
        val parts = trimmed.split(separator)
        if (parts.size != 2) {
            return null
        }
        val hour = parts[0].toIntOrNull() ?: return null
        val minute = parts[1].toIntOrNull() ?: return null
        if (hour !in 0..23 || minute !in 0..59) {
            return null
        }
        return hour * 60 + minute
    }

    private fun parseDaysOfWeekArgument(arguments: JSONObject): DaysParseResult {
        val key = DAYS_ARGUMENT_KEYS.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) }
            ?: return DaysParseResult("")
        val raw = arguments.opt(key) ?: return DaysParseResult("")
        val tokens = when (raw) {
            is JSONArray -> (0 until raw.length()).mapNotNull { index -> raw.optString(index).takeIf { it.isNotBlank() } }
            else -> raw.toString().split(',', ';', '|', ' ', '\n', '\t')
        }
        val selected = linkedSetOf<String>()
        tokens.forEach { token ->
            val normalized = token.trim().lowercase().replace("-", "_")
            if (normalized.isBlank()) {
                return@forEach
            }
            when (normalized) {
                "any", "all", "daily", "everyday", "every_day" -> return DaysParseResult("")
                "weekday", "weekdays", "workday", "workdays" -> selected.addAll(WEEKDAYS)
                "weekend", "weekends" -> selected.addAll(WEEKEND_DAYS)
                else -> {
                    val day = DAY_SYNONYMS[normalized]
                        ?: return DaysParseResult("", "Unsupported day of week: $token")
                    selected += day
                }
            }
        }
        val canonical = DAY_ORDER.filter { day -> day in selected }.joinToString(",")
        return DaysParseResult(canonical)
    }

    private fun setTimeEventVariables(store: HermesAutomationStore) {
        val calendar = Calendar.getInstance()
        val hour = calendar.get(Calendar.HOUR_OF_DAY)
        val minute = calendar.get(Calendar.MINUTE)
        val day = CALENDAR_DAY_CODES[calendar.get(Calendar.DAY_OF_WEEK)].orEmpty()
        store.setVariable("TIME", String.format(Locale.US, "%02d:%02d", hour, minute))
        store.setVariable("TIME_HOUR", String.format(Locale.US, "%02d", hour))
        store.setVariable("TIME_MINUTE", String.format(Locale.US, "%02d", minute))
        store.setVariable("TIME_DAY", day)
    }

    private fun setCalendarEventVariables(
        store: HermesAutomationStore,
        calendarName: String,
        title: String,
        description: String,
        location: String,
    ) {
        store.setVariable("CALNAME", calendarName.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALENDAR_NAME", calendarName.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALTITLE", title.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALENDAR_TITLE", title.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALDESCR", description.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALENDAR_DESCRIPTION", description.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALLOC", location.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("CALENDAR_LOCATION", location.take(MAX_EVENT_VALUE_CHARS))
    }

    private fun setLocationEventVariables(
        store: HermesAutomationStore,
        latitude: Double,
        longitude: Double,
        accuracyMeters: Double?,
        provider: String,
        name: String,
    ) {
        val latitudeText = formatLocationNumber(latitude)
        val longitudeText = formatLocationNumber(longitude)
        val accuracyText = accuracyMeters?.let { formatLocationNumber(it) }.orEmpty()
        store.setVariable("LAT", latitudeText)
        store.setVariable("LON", longitudeText)
        store.setVariable("LOC", "$latitudeText,$longitudeText")
        store.setVariable("LOCACC", accuracyText)
        store.setVariable("LOCPROVIDER", provider.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOCNAME", name.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOCATION_LATITUDE", latitudeText)
        store.setVariable("LOCATION_LONGITUDE", longitudeText)
        store.setVariable("LOCATION_ACCURACY_METERS", accuracyText)
        store.setVariable("LOCATION_PROVIDER", provider.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOCATION_NAME", name.take(MAX_EVENT_VALUE_CHARS))
    }

    private fun setShizukuEventVariables(
        store: HermesAutomationStore,
        status: HermesPrivilegedAccessStatus,
        available: Boolean,
    ) {
        store.setVariable("SHIZUKU_AVAILABLE", available.toString())
        store.setVariable("SHIZUKU_INSTALLED", status.shizukuInstalled.toString())
        store.setVariable("SUI_INSTALLED", status.suiInstalled.toString())
        store.setVariable("SHIZUKU_RUNNING", status.shizukuBinderAlive.toString())
        store.setVariable("SHIZUKU_PERMISSION_GRANTED", status.shizukuPermissionGranted.toString())
        store.setVariable("SHIZUKU_PRIVILEGE_LABEL", status.shizukuPrivilegeLabel)
        store.setVariable("SHIZUKU_UID", status.shizukuUid?.toString().orEmpty())
    }

    private fun normalizeShizukuTrigger(requestedState: String, available: Boolean): String? {
        val normalized = requestedState.trim().lowercase().replace("-", "_").replace(" ", "_")
        return when (normalized) {
            "", "current", "auto", "detected" -> if (available) TRIGGER_SHIZUKU_AVAILABLE else TRIGGER_SHIZUKU_UNAVAILABLE
            "available", "ready", "usable", "can_use", "permission_granted", TRIGGER_SHIZUKU_AVAILABLE -> TRIGGER_SHIZUKU_AVAILABLE
            "unavailable", "missing", "not_available", "not_ready", "not_running", "permission_missing", TRIGGER_SHIZUKU_UNAVAILABLE -> TRIGGER_SHIZUKU_UNAVAILABLE
            else -> normalizeTrigger(normalized)?.takeIf {
                it == TRIGGER_SHIZUKU_AVAILABLE || it == TRIGGER_SHIZUKU_UNAVAILABLE
            }
        }
    }

    private fun formatTime(minutes: Int?): String {
        if (minutes == null) {
            return "time trigger"
        }
        return String.format(Locale.US, "%02d:%02d", minutes / 60, minutes % 60)
    }

    private fun formatLocationNumber(value: Double): String {
        return String.format(Locale.US, "%.6f", value).trimEnd('0').trimEnd('.')
    }

    private fun looksLikeVariableReference(value: String): Boolean {
        return value.contains('%') || value.contains("{{")
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
        "create_intent_task",
        "create_shizuku_action_task",
        "run",
        "run_trigger",
        "run_app_foreground_trigger",
        "run_notification_posted_trigger",
        "run_calendar_event_trigger",
        "run_location_trigger",
        "run_shizuku_state_trigger",
        "run_time_trigger",
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
        TRIGGER_TIME,
        TRIGGER_CALENDAR_EVENT,
        TRIGGER_LOCATION,
        TRIGGER_SHIZUKU_AVAILABLE,
        TRIGGER_SHIZUKU_UNAVAILABLE,
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
        "calendar" to TRIGGER_CALENDAR_EVENT,
        "calendar_event" to TRIGGER_CALENDAR_EVENT,
        "calendar_profile" to TRIGGER_CALENDAR_EVENT,
        "calendar_trigger" to TRIGGER_CALENDAR_EVENT,
        "event" to TRIGGER_CALENDAR_EVENT,
        "event_calendar" to TRIGGER_CALENDAR_EVENT,
        "location" to TRIGGER_LOCATION,
        "location_event" to TRIGGER_LOCATION,
        "location_profile" to TRIGGER_LOCATION,
        "location_trigger" to TRIGGER_LOCATION,
        "place" to TRIGGER_LOCATION,
        "geofence" to TRIGGER_LOCATION,
        "clock" to TRIGGER_TIME,
        "clock_time" to TRIGGER_TIME,
        "daily_time" to TRIGGER_TIME,
        "scheduled_time" to TRIGGER_TIME,
        "time_context" to TRIGGER_TIME,
        "time_of_day" to TRIGGER_TIME,
        "at_time" to TRIGGER_TIME,
        "shizuku" to TRIGGER_SHIZUKU_AVAILABLE,
        "shizuku_state" to TRIGGER_SHIZUKU_AVAILABLE,
        "shizuku_running" to TRIGGER_SHIZUKU_AVAILABLE,
        "shizuku_ready" to TRIGGER_SHIZUKU_AVAILABLE,
        "shizuku_usable" to TRIGGER_SHIZUKU_AVAILABLE,
        "shizuku_permission_granted" to TRIGGER_SHIZUKU_AVAILABLE,
        "shizuku_not_available" to TRIGGER_SHIZUKU_UNAVAILABLE,
        "shizuku_missing" to TRIGGER_SHIZUKU_UNAVAILABLE,
        "shizuku_not_running" to TRIGGER_SHIZUKU_UNAVAILABLE,
        "shizuku_permission_missing" to TRIGGER_SHIZUKU_UNAVAILABLE,
    )
    private data class TimeParseResult(val minutes: Int?, val error: String? = null)
    private data class DaysParseResult(val daysCsv: String, val error: String? = null)
    private data class TriggerDataResult(val data: String, val error: String? = null)

    private const val DEFAULT_LOCATION_RADIUS_METERS = 100.0
    private const val EARTH_RADIUS_METERS = 6_371_008.8

    private val TIME_ARGUMENT_KEYS = listOf(
        "time",
        "time_of_day",
        "at_time",
        "at",
        "clock_time",
        "trigger_time",
        "trigger_time_minutes",
    )
    private val DAYS_ARGUMENT_KEYS = listOf(
        "days",
        "days_of_week",
        "weekdays",
        "trigger_days",
        "trigger_days_of_week",
    )
    private val DAY_ORDER = listOf("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
    private val WEEKDAYS = listOf("MON", "TUE", "WED", "THU", "FRI")
    private val WEEKEND_DAYS = listOf("SAT", "SUN")
    private val DAY_SYNONYMS = mapOf(
        "mon" to "MON",
        "monday" to "MON",
        "tue" to "TUE",
        "tues" to "TUE",
        "tuesday" to "TUE",
        "wed" to "WED",
        "weds" to "WED",
        "wednesday" to "WED",
        "thu" to "THU",
        "thur" to "THU",
        "thurs" to "THU",
        "thursday" to "THU",
        "fri" to "FRI",
        "friday" to "FRI",
        "sat" to "SAT",
        "saturday" to "SAT",
        "sun" to "SUN",
        "sunday" to "SUN",
    )
    private val CALENDAR_DAY_CODES = mapOf(
        Calendar.MONDAY to "MON",
        Calendar.TUESDAY to "TUE",
        Calendar.WEDNESDAY to "WED",
        Calendar.THURSDAY to "THU",
        Calendar.FRIDAY to "FRI",
        Calendar.SATURDAY to "SAT",
        Calendar.SUNDAY to "SUN",
    )
    private val PRIVILEGED_SHELL_ACTIONS = setOf("run_privileged_shell", "shizuku_shell", "privileged_shell")
    private val SHIZUKU_AUTOMATION_ACTIONS = listOf(
        "grant_runtime_permission",
        "revoke_runtime_permission",
        "force_stop_app",
        "enable_app",
        "disable_app",
        "set_app_enabled",
    )
    private val SHIZUKU_PERMISSION_ACTIONS = setOf("grant_runtime_permission", "revoke_runtime_permission")
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
    private val INTENT_STRING_PAYLOAD_KEYS = listOf(
        "intent_task_action",
        "intent_action",
        "data_uri",
        "package_name",
        "class_name",
        "component",
        "category",
    )
    private val TASKER_VARIABLE_PATTERN = Regex("%([A-Za-z_][A-Za-z0-9_]{1,63})")
    private val BRACE_VARIABLE_PATTERN = Regex("\\{\\{([A-Za-z_][A-Za-z0-9_]{0,63})\\}\\}")
    private const val AUTOMATION_TIMEOUT_SECONDS = 30
    private const val MAX_VARIABLE_VALUE_CHARS = 4_000
    private const val MAX_RESULT_CHARS = 2_000
    private const val MAX_EVENT_VALUE_CHARS = 500
}
