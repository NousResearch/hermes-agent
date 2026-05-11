package com.nousresearch.hermesagent.device

import android.content.Context
import android.os.Build
import android.provider.Settings
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.util.Calendar
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

object HermesAutomationBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase().ifBlank { "list" }) {
            "list", "list_automations", "status" -> listJson(context)
            "run_history", "automation_run_history", "recent_runs", "operator_run_history" -> runHistoryJson(context, arguments)
            "operator_devices", "devices", "standby_devices", "remote_devices", "opengui_devices" -> operatorDevicesJson(context)
            "operator_standby_status", "standby_status", "remote_dispatch_status", "dispatch_status" -> operatorStandbyStatusJson(context)
            "operator_execution_status", "execution_status", "remote_execution_status", "opengui_execution_status" -> operatorExecutionStatusJson(context, arguments)
            "operator_command", "opengui_command", "remote_command", "im_command", "discord_command", "telegram_command", "feishu_command" -> operatorCommandJson(context, arguments)
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
            "open_uri", "open_url", "browse_url", "open_browser", "launch_browser" -> performIntentNowJson(context, arguments, "open_uri")
            "start_activity", "launch_activity", "send_intent" -> performIntentNowJson(context, arguments, "start_activity")
            "send_broadcast", "broadcast_intent" -> performIntentNowJson(context, arguments, "send_broadcast")
            "create_shizuku_action_task", "create_shizuku_action", "create_privileged_action_task", "privileged_action_task" -> createShizukuActionTaskJson(context, arguments)
            "create_sunrise_sunset_task", "create_sun_task", "create_solar_task" -> createSunriseSunsetTaskJson(context, arguments)
            "create_notification_task", "create_notify_task", "create_notify", "notify_task" -> createNotificationTaskJson(context, arguments)
            "create_variable_action_task", "create_variable_task", "create_variable_set_task", "create_variable_clear_task" -> createVariableActionTaskJson(context, arguments)
            "create_wait_task", "create_wait", "wait_task", "delay_task" -> createWaitTaskJson(context, arguments)
            "create_clipboard_task", "create_set_clipboard_task", "set_clipboard_task", "clipboard_task" -> createClipboardTaskJson(context, arguments)
            "create_vibration_task", "create_vibrate_task", "vibrate_task", "vibration_task" -> createVibrationTaskJson(context, arguments)
            "create_audio_action_task", "create_audio_task", "audio_task", "create_volume_task", "volume_task", "create_sound_mode_task" -> createAudioActionTaskJson(context, arguments)
            "create_http_request_task", "create_http_task", "http_request_task", "http_task", "create_http_get_task", "create_http_post_task" -> createHttpRequestTaskJson(context, arguments)
            "create_overlay_scene_task", "create_scene_task", "overlay_scene_task", "scene_task" -> createOverlaySceneTaskJson(context, arguments)
            "create_toast_task", "create_flash_task", "toast_task", "flash_task" -> createToastTaskJson(context, arguments)
            "show_toast", "toast", "flash_message", "flash" -> showToastJson(context, arguments)
            "perform_audio_action", "audio_action", "set_audio_volume", "set_sound_mode", "set_ringer_mode", "set_microphone_mute", "set_speakerphone" -> performAudioActionJson(context, action, arguments)
            "perform_http_request", "http_request", "http_get", "http_post", "http_head" -> performHttpRequestJson(action, arguments)
            "overlay_scene_status", "scene_status", "overlay_status", "show_overlay_scene", "show_scene", "overlay_scene", "hide_overlay_scene", "dismiss_overlay_scene", "clear_overlay_scene", "hide_scene" -> HermesOverlaySceneBridge.performSceneJson(context, action, arguments)
            "create_launcher_shortcut", "create_shortcut", "create_home_screen_shortcut", "pin_automation_shortcut" -> HermesLauncherShortcutBridge.createShortcutJson(context, arguments)
            "list_launcher_shortcuts", "list_shortcuts", "launcher_shortcuts" -> HermesLauncherShortcutBridge.listShortcutsJson(context)
            "remove_launcher_shortcut", "delete_launcher_shortcut", "remove_shortcut", "delete_shortcut" -> HermesLauncherShortcutBridge.removeShortcutJson(context, arguments)
            "set_quick_settings_tile_automation", "set_qs_tile_automation", "configure_quick_settings_tile", "configure_qs_tile" -> HermesQuickSettingsTileBridge.setTileAutomationJson(context, arguments)
            "get_quick_settings_tile_automation", "get_qs_tile_automation", "quick_settings_tile_status", "qs_tile_status" -> HermesQuickSettingsTileBridge.getTileAutomationJson(context)
            "clear_quick_settings_tile_automation", "clear_qs_tile_automation" -> HermesQuickSettingsTileBridge.clearTileAutomationJson(context)
            "run_quick_settings_tile", "run_qs_tile" -> HermesQuickSettingsTileBridge.runConfiguredAutomationJson(context)
            "set_home_screen_widget_automation", "set_widget_automation", "configure_home_screen_widget", "configure_widget", "create_home_screen_widget", "create_automation_widget" -> HermesAutomationWidgetBridge.setWidgetAutomationJson(context, arguments)
            "get_home_screen_widget_automation", "get_widget_automation", "home_screen_widget_status", "widget_status" -> HermesAutomationWidgetBridge.getWidgetAutomationJson(context, arguments)
            "list_home_screen_widgets", "list_automation_widgets", "list_widgets" -> HermesAutomationWidgetBridge.listWidgetsJson(context)
            "clear_home_screen_widget_automation", "clear_widget_automation" -> HermesAutomationWidgetBridge.clearWidgetAutomationJson(context, arguments)
            "run_home_screen_widget", "run_widget" -> HermesAutomationWidgetBridge.runConfiguredAutomationJson(context, arguments)
            "calculate_sunrise_sunset", "sunrise_sunset", "sun_times", "solar_times" -> calculateSunriseSunsetJson(context, arguments)
            "export_automations", "export", "backup_automations", "backup" -> exportAutomationsJson(context)
            "import_automations", "import", "restore_automations", "restore" -> importAutomationsJson(context, arguments)
            "import_tasker_xml", "import_tasker_data_uri", "import_tasker_project", "import_tasker_task" -> importTaskerXmlJson(context, arguments)
            "logcat_watcher_status", "start_logcat_watcher", "stop_logcat_watcher", "scan_logcat_entries", "reset_logcat_watcher_cursor", "reset_logcat_cursor", "clear_logcat_watcher_cursor" -> HermesLogcatWatcherBridge.performActionJson(context, action, arguments)
            "sensor_watcher_status", "sensor_status", "watch_sensor_status", "watch_sensors_status", "start_sensor_watcher", "start_sensor_watch", "watch_sensors", "watch_sensor", "stop_sensor_watcher", "stop_sensor_watch" -> HermesSensorWatcherBridge.performActionJson(context, action, arguments)
            "calendar_watcher_status", "calendar_status", "watch_calendar_status", "start_calendar_watcher", "start_calendar_watch", "watch_calendar", "watch_calendar_events", "stop_calendar_watcher", "stop_calendar_watch", "scan_calendar_events", "scan_calendar", "run_calendar_watch_once", "reset_calendar_watcher_cursor", "reset_calendar_cursor", "clear_calendar_watcher_cursor" -> HermesCalendarWatcherBridge.performActionJson(context, action, arguments)
            "location_watcher_status", "location_status", "watch_location_status", "watch_locations_status", "start_location_watcher", "start_location_watch", "watch_location", "watch_locations", "stop_location_watcher", "stop_location_watch", "scan_location", "scan_locations", "scan_location_once", "run_location_watch_once" -> HermesLocationWatcherBridge.performActionJson(context, action, arguments)
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
            "run_sensor_trigger", "trigger_sensor", "sensor_event", "sensor" -> runSensorTriggerJson(
                context = context,
                arguments = arguments,
            )
            "run_logcat_entry_trigger", "trigger_logcat_entry", "logcat_entry", "logcat" -> runLogcatEntryTriggerJson(
                context = context,
                arguments = arguments,
            )
            "run_external_trigger", "trigger_external", "external_trigger", "extra_trigger", "run_trigger_app" -> runExternalTriggerJson(
                context = context,
                arguments = arguments,
            )
            "run_remote_dispatch", "submit_standby_dispatch", "operator_dispatch", "remote_dispatch", "run_opengui_dispatch" -> runRemoteDispatchJson(
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
        val records = store.list()
        return JSONObject()
            .put("success", true)
            .put("automations", recordsToJson(records))
            .put("variables", store.listVariables())
            .put("standby_dispatch", standbySummaryJson(context, records, store))
            .put("recent_run_events", runEventsToJson(store.listRunEvents(10)))
            .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
            .put("available_triggers", JSONArray(AUTOMATION_TRIGGERS))
            .put("min_interval_minutes", HermesAutomationScheduler.MIN_INTERVAL_MINUTES)
            .toString()
    }

    fun runHistoryJson(context: Context, arguments: JSONObject): String {
        val limit = arguments.optInt("limit", 20).coerceIn(1, 50)
        val events = HermesAutomationStore(context).listRunEvents(limit)
        return JSONObject()
            .put("success", true)
            .put("run_count", events.size)
            .put("runs", runEventsToJson(events))
            .toString()
    }

    fun operatorStandbyStatusJson(context: Context): String {
        val store = HermesAutomationStore(context)
        return JSONObject()
            .put("success", true)
            .put("standby_dispatch", standbySummaryJson(context, store.list(), store))
            .toString()
    }

    fun operatorDevicesJson(context: Context): String {
        val store = HermesAutomationStore(context)
        val standby = standbySummaryJson(context, store.list(), store)
        val device = JSONObject()
            .put("device_id", standby.optString("device_id"))
            .put("device_name", standby.optString("device_name"))
            .put("online", true)
            .put("standby_ready", standby.optBoolean("ready"))
            .put("last_seen_epoch_ms", System.currentTimeMillis())
            .put("standby_namespace", standby.optString("standby_namespace"))
            .put("standby_heartbeat_event", standby.optString("standby_heartbeat_event"))
            .put("standby_dispatch_event", standby.optString("standby_dispatch_event"))
            .put("heartbeat_interval_seconds", standby.optInt("heartbeat_interval_seconds"))
            .put("automation_count", standby.optInt("automation_count"))
            .put("enabled_automation_count", standby.optInt("enabled_automation_count"))
            .put("remote_dispatch_count", standby.optInt("remote_dispatch_count"))
            .put("external_trigger_count", standby.optInt("external_trigger_count"))
            .put("recent_run_count", standby.optInt("recent_run_count"))
            .put("last_dispatch_epoch_ms", standby.opt("last_dispatch_epoch_ms"))
            .put("last_dispatch_task_name", standby.optString("last_dispatch_task_name"))
            .put("supported_dispatch_channels", standby.optJSONArray("supported_dispatch_channels") ?: JSONArray())
            .put("compatible_dispatch_payloads", standby.optJSONArray("compatible_dispatch_payloads") ?: JSONArray())
        return JSONObject()
            .put("success", true)
            .put("device_count", 1)
            .put("online_device_count", 1)
            .put("devices", JSONArray().put(device))
            .put("standby_dispatch", standby)
            .put(
                "compatible_device_queries",
                JSONArray(
                    listOf(
                        "OpenGUI devices",
                        "OpenGUI /devices",
                        "Hermes android_automation_tool operator_devices",
                        "Hermes android_automation_tool operator_standby_status",
                    )
                )
            )
            .toString()
    }

    fun operatorExecutionStatusJson(context: Context, arguments: JSONObject): String {
        val limit = arguments.optInt("limit", 5).coerceIn(1, 25)
        val executionId = stringArgument(arguments, "execution_id", "executionId", "remote_execution_id", allowEmpty = true)
            .orEmpty()
            .trim()
        val taskId = stringArgument(arguments, "task_id", "taskId", "remote_task_id", allowEmpty = true)
            .orEmpty()
            .trim()
        val taskName = stringArgument(arguments, "task_name", "taskName", "remote_task_name", "label", "name", allowEmpty = true)
            .orEmpty()
            .trim()
        val hasFilter = executionId.isNotBlank() || taskId.isNotBlank() || taskName.isNotBlank()
        val events = HermesAutomationStore(context).listRunEvents(50)
        val matchedEvents = events.filter { event ->
            val executionMatches = executionId.isBlank() ||
                event.remoteExecutionId == executionId ||
                event.id == executionId
            val taskIdMatches = taskId.isBlank() ||
                event.remoteTaskId == taskId ||
                event.automationId == taskId
            val taskNameMatches = taskName.isBlank() ||
                event.remoteTaskName.equals(taskName, ignoreCase = true) ||
                event.automationLabel.equals(taskName, ignoreCase = true) ||
                event.automationId.equals(taskName, ignoreCase = true)
            executionMatches && taskIdMatches && taskNameMatches
        }
        val latest = matchedEvents.firstOrNull()
        val status = when {
            latest == null && hasFilter -> "not_found"
            latest == null -> "idle"
            latest.success -> "completed"
            else -> "failed"
        }
        val execution = JSONObject()
            .put("status", status)
            .put("terminal", latest != null)
            .put("source", latest?.dispatchSource.orEmpty().ifBlank { "hermes_android" })
            .put("channel", latest?.dispatchChannel.orEmpty())
            .put("execution_id", latest?.remoteExecutionId.orEmpty().ifBlank { executionId })
            .put("task_id", latest?.remoteTaskId.orEmpty().ifBlank { taskId })
            .put("task_name", latest?.remoteTaskName.orEmpty().ifBlank { taskName.ifBlank { latest?.automationLabel.orEmpty() } })
            .put("automation_id", latest?.automationId.orEmpty())
            .put("automation_label", latest?.automationLabel.orEmpty())
            .put("started_at_epoch_ms", latest?.startedAtEpochMs ?: JSONObject.NULL)
            .put("finished_at_epoch_ms", latest?.finishedAtEpochMs ?: JSONObject.NULL)
            .put("success", latest?.success ?: JSONObject.NULL)
            .put("exit_code", latest?.exitCode ?: JSONObject.NULL)
            .put("result", latest?.result.orEmpty())
        return JSONObject()
            .put("success", latest != null || !hasFilter)
            .put("status", status)
            .put("matched_run_count", matchedEvents.size)
            .put("execution", execution)
            .put("runs", runEventsToJson(matchedEvents.take(limit)))
            .put(
                "compatible_status_queries",
                JSONArray(
                    listOf(
                        "OpenGUI /status [executionId]",
                        "Hermes android_automation_tool operator_execution_status",
                        "Hermes android_automation_tool run_history",
                    )
                )
            )
            .toString()
    }

    fun operatorCommandJson(context: Context, arguments: JSONObject): String {
        val command = operatorCommandTextFromArguments(arguments).trim()
        if (command.isBlank()) {
            return errorJson("operator_command requires command, text, message, or an OpenGUI slash subcommand payload")
        }
        if (command.indexOf('\u0000') >= 0) {
            return errorJson("operator_command text must not contain NUL bytes")
        }
        val prefix = stringArgument(arguments, "command_prefix", "prefix", allowEmpty = true)
            .orEmpty()
            .ifBlank { "!opengui" }
        val channel = stringArgument(arguments, "dispatch_channel", "channel", "platform", allowEmpty = true)
            .orEmpty()
            .ifBlank { "im" }
            .take(MAX_EVENT_VALUE_CHARS)
        val parsed = parseOperatorCommand(command, prefix)
        val access = operatorCommandAccess(arguments)
        if (!access.allowed) {
            return operatorCommandDeniedJson(parsed, access)
        }
        return when (parsed.type) {
            OperatorCommandType.HELP -> operatorCommandHelpJson(parsed)
            OperatorCommandType.LIST_TASKS -> withParsedOperatorCommand(listJson(context), parsed)
            OperatorCommandType.DEVICES -> withParsedOperatorCommand(operatorDevicesJson(context), parsed)
            OperatorCommandType.STATUS -> {
                val statusArgs = JSONObject()
                parsed.executionId?.let { statusArgs.put("executionId", it) }
                withParsedOperatorCommand(operatorExecutionStatusJson(context, statusArgs), parsed)
            }
            OperatorCommandType.RUN_TASK -> {
                val taskId = parsed.taskId.orEmpty()
                val dispatchArgs = JSONObject()
                    .put("taskId", taskId)
                    .put("taskName", taskId)
                    .put("executionId", stringArgument(arguments, "execution_id", "executionId", allowEmpty = true).orEmpty().ifBlank {
                        "opengui-command-${System.currentTimeMillis()}"
                    })
                    .put("dispatch_source", "opengui_im_command")
                    .put("dispatch_channel", channel)
                    .put("allow_disabled", arguments.optBoolean("allow_disabled", false))
                withParsedOperatorCommand(runRemoteDispatchJson(context, dispatchArgs), parsed)
            }
            OperatorCommandType.DO_TASK -> {
                val description = parsed.description.orEmpty().take(MAX_VARIABLE_VALUE_CHARS)
                val dispatchArgs = JSONObject()
                    .put("taskName", description)
                    .put("taskId", description.take(MAX_EVENT_VALUE_CHARS))
                    .put("executionId", stringArgument(arguments, "execution_id", "executionId", allowEmpty = true).orEmpty().ifBlank {
                        "opengui-command-${System.currentTimeMillis()}"
                    })
                    .put("dispatch_source", "opengui_im_command")
                    .put("dispatch_channel", channel)
                    .put("allow_disabled", arguments.optBoolean("allow_disabled", false))
                withParsedOperatorCommand(runRemoteDispatchJson(context, dispatchArgs), parsed)
            }
            OperatorCommandType.CANCEL,
            OperatorCommandType.PAUSE,
            OperatorCommandType.RESUME -> operatorCommandRecognizedButNotActiveJson(parsed)
            OperatorCommandType.FREE_TEXT -> operatorCommandFreeTextJson(parsed)
        }
    }

    private fun operatorCommandTextFromArguments(arguments: JSONObject): String {
        val rawCommand = stringArgument(arguments, "command", "text", "message", "raw_text", "rawText", allowEmpty = true)
            .orEmpty()
            .trim()
        val slashCommand = openguiSlashCommandTextFromArguments(arguments)
        if (slashCommand.isNotBlank() && rawCommand.isOpenGuiSlashEnvelope()) {
            return slashCommand
        }
        return rawCommand.ifBlank { slashCommand }
    }

    private fun String.isOpenGuiSlashEnvelope(): Boolean {
        val normalized = trim()
            .removePrefix("/")
            .lowercase(Locale.US)
            .replace("-", "_")
        return normalized == "opengui" || normalized == "hermes"
    }

    private fun openguiSlashCommandTextFromArguments(arguments: JSONObject): String {
        val subcommand = stringArgument(
            arguments,
            "subcommand",
            "sub_command",
            "slash_subcommand",
            "slashCommand",
            "opengui_subcommand",
            "openguiSubcommand",
            allowEmpty = true,
        ).orEmpty()
            .trim()
            .removePrefix("/")
            .lowercase(Locale.US)
            .replace("-", "_")
        if (subcommand.isBlank()) {
            return ""
        }
        return when (subcommand) {
            "help" -> "/help"
            "devices", "device" -> "/devices"
            "tasks", "task", "list", "list_tasks" -> "/tasks"
            "do", "do_task" -> {
                val task = stringArgument(
                    arguments,
                    "task",
                    "description",
                    "task_description",
                    "prompt",
                    "task_name",
                    "taskName",
                    allowEmpty = true,
                ).orEmpty().trim()
                if (task.isBlank()) "" else "/do $task"
            }
            "run", "run_task" -> {
                val taskId = stringArgument(
                    arguments,
                    "task_id",
                    "taskId",
                    "automation_id",
                    "automationId",
                    "id",
                    "task",
                    allowEmpty = true,
                ).orEmpty().trim()
                if (taskId.isBlank()) "" else "/run $taskId"
            }
            "status" -> {
                val executionId = stringArgument(
                    arguments,
                    "execution_id",
                    "executionId",
                    "remote_execution_id",
                    allowEmpty = true,
                ).orEmpty().trim()
                if (executionId.isBlank()) "/status" else "/status $executionId"
            }
            "cancel", "pause" -> {
                val executionId = stringArgument(
                    arguments,
                    "execution_id",
                    "executionId",
                    "remote_execution_id",
                    allowEmpty = true,
                ).orEmpty().trim()
                if (executionId.isBlank()) "" else "/$subcommand $executionId"
            }
            "resume" -> {
                val executionId = stringArgument(
                    arguments,
                    "execution_id",
                    "executionId",
                    "remote_execution_id",
                    allowEmpty = true,
                ).orEmpty().trim()
                val feedback = stringArgument(arguments, "feedback", "comment", "note", allowEmpty = true)
                    .orEmpty()
                    .trim()
                when {
                    executionId.isNotBlank() && feedback.isNotBlank() -> "/resume $executionId $feedback"
                    executionId.isNotBlank() -> "/resume $executionId"
                    feedback.isNotBlank() -> "/resume $feedback"
                    else -> ""
                }
            }
            else -> ""
        }
    }

    fun exportAutomationsJson(context: Context): String {
        val store = HermesAutomationStore(context)
        val records = store.list()
        val variables = store.listVariables()
        return JSONObject()
            .put("success", true)
            .put("kind", AUTOMATION_BUNDLE_KIND)
            .put("schema_version", AUTOMATION_BUNDLE_SCHEMA_VERSION)
            .put("exported_at_epoch_ms", System.currentTimeMillis())
            .put("automations", recordsToJson(records))
            .put("variables", variables)
            .put("automation_count", records.size)
            .put("variable_count", variables.length())
            .toString()
    }

    fun importAutomationsJson(context: Context, arguments: JSONObject): String {
        val bundle = runCatching { automationBundleFromArguments(arguments) }.getOrElse { error ->
            return errorJson("import_automations bundle_json must be valid JSON: ${error.message}")
        } ?: return errorJson("import_automations requires a bundle, bundle_json, automations, records, or variables argument")
        if (bundle.optString("kind").isNotBlank() && bundle.optString("kind") != AUTOMATION_BUNDLE_KIND) {
            return errorJson("Unsupported automation bundle kind: ${bundle.optString("kind")}")
        }
        if (arguments.optBoolean("enable_imported", false) && arguments.optBoolean("disable_imported", false)) {
            return errorJson("import_automations cannot set both enable_imported and disable_imported")
        }
        val automationArray = bundle.optJSONArray("automations") ?: bundle.optJSONArray("records") ?: JSONArray()
        val variables = variablesFromBundle(bundle)
        if (automationArray.length() == 0 && variables.length() == 0) {
            return errorJson("import_automations requires at least one automation record or variable")
        }
        if (automationArray.length() > MAX_IMPORTED_AUTOMATIONS) {
            return errorJson("import_automations accepts at most $MAX_IMPORTED_AUTOMATIONS automation records per bundle")
        }
        val enabledOverride = when {
            arguments.has("enable_imported") && !arguments.isNull("enable_imported") -> arguments.optBoolean("enable_imported")
            arguments.has("disable_imported") && !arguments.isNull("disable_imported") -> !arguments.optBoolean("disable_imported")
            else -> null
        }
        val now = System.currentTimeMillis()
        val importedRecords = mutableListOf<HermesAutomationRecord>()
        val importedIds = mutableSetOf<String>()
        for (index in 0 until automationArray.length()) {
            val recordJson = automationArray.optJSONObject(index)
                ?: return errorJson("import_automations automation at index $index must be a JSON object")
            val record = runCatching { sanitizeImportedRecord(recordJson, now, enabledOverride) }.getOrElse { error ->
                return errorJson("import_automations record $index is invalid: ${error.message}")
            }
            if (!importedIds.add(record.id)) {
                return errorJson("import_automations bundle contains duplicate automation id: ${record.id}")
            }
            importedRecords += record
        }
        val store = HermesAutomationStore(context)
        val replace = arguments.optBoolean("replace", false) ||
            arguments.optBoolean("replace_existing", false) ||
            arguments.optBoolean("clear_existing", false)
        if (replace) {
            store.list().forEach { existing -> HermesAutomationScheduler.cancel(context, existing.id) }
            store.replaceAll(importedRecords)
            store.replaceVariables(variables)
        } else {
            importedRecords.forEach { record -> store.upsert(record) }
            store.mergeVariables(variables)
        }
        importedRecords.forEach { record ->
            if (record.enabled) {
                HermesAutomationScheduler.schedule(context, record)
            } else {
                HermesAutomationScheduler.cancel(context, record.id)
            }
        }
        return JSONObject()
            .put("success", true)
            .put("kind", AUTOMATION_BUNDLE_KIND)
            .put("schema_version", AUTOMATION_BUNDLE_SCHEMA_VERSION)
            .put("replace", replace)
            .put("imported_automation_count", importedRecords.size)
            .put("imported_variable_count", variables.length())
            .put("automations", recordsToJson(importedRecords))
            .put("variables", variables)
            .toString()
    }

    fun importTaskerXmlJson(context: Context, arguments: JSONObject): String {
        val taskerImport = runCatching { HermesTaskerImportBridge.bundleFromArguments(arguments) }.getOrElse { error ->
            return errorJson("import_tasker_xml failed to parse Tasker XML: ${error.message}")
        } ?: return errorJson("import_tasker_xml requires tasker_xml, tasker_data_uri, or tasker_xml_base64")
        val importArguments = JSONObject(arguments.toString()).put("bundle", taskerImport.bundle)
        if (!importArguments.has("enable_imported") && !importArguments.has("disable_imported")) {
            importArguments.put("disable_imported", true)
        }
        val imported = JSONObject(importAutomationsJson(context, importArguments))
        if (!imported.optBoolean("success", false)) {
            return imported.toString()
        }
        return imported
            .put("source", "tasker_xml")
            .put("tasker_task_count", taskerImport.taskCount)
            .put("tasker_imported_action_count", taskerImport.importedActionCount)
            .put("tasker_skipped_actions", taskerImport.skippedActions)
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
        val payloadResult = intentPayloadFromArguments(arguments, defaultIntentTaskAction)
        if (payloadResult.error.isNotBlank()) {
            return errorJson(payloadResult.error)
        }
        val payload = payloadResult.payload
        val validation = HermesIntentBridge.performIntentJson(context, JSONObject(payload.toString()).put("__validate_only", true))
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

    fun performIntentNowJson(context: Context, arguments: JSONObject, defaultIntentTaskAction: String? = null): String {
        val payloadResult = intentPayloadFromArguments(arguments, defaultIntentTaskAction)
        if (payloadResult.error.isNotBlank()) {
            return errorJson(payloadResult.error)
        }
        return HermesIntentBridge.performIntentJson(context, payloadResult.payload).toString()
    }

    private fun intentPayloadFromArguments(
        arguments: JSONObject,
        defaultIntentTaskAction: String? = null,
    ): IntentPayloadResult {
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
            ?: return IntentPayloadResult(
                error = "Unsupported Android intent task action: $rawIntentTaskAction. Use start_activity, open_uri, or send_broadcast",
            )
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

        return IntentPayloadResult(payload = payload)
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

        val payload = JSONObject()
            .put("shizuku_action", shizukuAction)
        if (shizukuAction in SHIZUKU_PACKAGE_ACTIONS) {
            val packageName = stringArgument(
                arguments,
                "package_name",
                "packageName",
                "package",
                "app_package",
                "application_id",
            )?.trim() ?: return errorJson("create_shizuku_action_task $shizukuAction requires a package_name argument")
            if (packageName.indexOf('\u0000') >= 0) {
                return errorJson("create_shizuku_action_task package_name must not contain NUL bytes")
            }
            payload.put("package_name", packageName)
        }
        if (shizukuAction in SHIZUKU_PERMISSION_ACTIONS) {
            val permission = stringArgument(arguments, "permission", "permission_name", "permissionName", "android_permission")
                ?.trim()
                ?: return errorJson("create_shizuku_action_task $shizukuAction requires a permission argument")
            if (permission.indexOf('\u0000') >= 0) {
                return errorJson("create_shizuku_action_task permission must not contain NUL bytes")
            }
            payload.put("permission", permission)
        }
        if (shizukuAction in SHIZUKU_TOGGLE_STATE_ACTIONS) {
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
        if (shizukuAction in SHIZUKU_CUSTOM_SETTING_ACTIONS) {
            val namespace = stringArgument(
                arguments,
                "setting_namespace",
                "settings_namespace",
                "namespace",
                "setting_table",
                "table",
            )?.trim() ?: return errorJson("create_shizuku_action_task $shizukuAction requires a setting_namespace argument")
            if (namespace.indexOf('\u0000') >= 0) {
                return errorJson("create_shizuku_action_task setting_namespace must not contain NUL bytes")
            }
            val settingName = stringArgument(arguments, "setting_name", "setting_key", "name", "key")
                ?.trim()
                ?: return errorJson("create_shizuku_action_task $shizukuAction requires a setting_name argument")
            if (settingName.indexOf('\u0000') >= 0) {
                return errorJson("create_shizuku_action_task setting_name must not contain NUL bytes")
            }
            payload.put("setting_namespace", namespace)
            payload.put("setting_name", settingName)
            if (shizukuAction == "set_custom_setting") {
                val settingValue = stringArgument(
                    arguments,
                    "setting_value",
                    "value",
                    "settingValue",
                    allowEmpty = true,
                ) ?: return errorJson("create_shizuku_action_task set_custom_setting requires a setting_value argument")
                if (settingValue.indexOf('\u0000') >= 0) {
                    return errorJson("create_shizuku_action_task setting_value must not contain NUL bytes")
                }
                payload.put("setting_value", settingValue)
            }
        }
        if (shizukuAction in SHIZUKU_DND_ACTIONS) {
            val mode = stringArgument(arguments, "dnd_mode", "mode", "zen_mode", "state")
                ?.trim()
                ?: return errorJson("create_shizuku_action_task $shizukuAction requires a dnd_mode argument")
            if (mode.indexOf('\u0000') >= 0) {
                return errorJson("create_shizuku_action_task dnd_mode must not contain NUL bytes")
            }
            payload.put("dnd_mode", mode)
        }
        if (shizukuAction in SHIZUKU_USER_PROFILE_ACTIONS) {
            val userId = stringArgument(arguments, "user_id", "profile_user_id", "android_user_id", "work_profile_user_id")
                ?.trim()
                ?: return errorJson("create_shizuku_action_task $shizukuAction requires a user_id argument")
            if (userId.toIntOrNull()?.takeIf { it in 0..9999 } == null) {
                return errorJson("create_shizuku_action_task user_id must be an integer between 0 and 9999")
            }
            payload.put("user_id", userId)
        }
        if (shizukuAction in SHIZUKU_MOBILE_NETWORK_TYPE_ACTIONS) {
            val bitmask = stringArgument(
                arguments,
                "network_types_bitmask",
                "network_type_bitmask",
                "mobile_network_bitmask",
                "allowed_network_types",
                "bitmask",
            )?.trim() ?: return errorJson("create_shizuku_action_task $shizukuAction requires a network_types_bitmask argument")
            if (!Regex("[01]{1,64}").matches(bitmask)) {
                return errorJson("create_shizuku_action_task network_types_bitmask must be a binary bitmask string")
            }
            payload.put("network_types_bitmask", bitmask)
            stringArgument(arguments, "slot_id", "sim_slot_id", "subscription_slot")?.trim()?.let { slotId ->
                if (slotId.toIntOrNull()?.takeIf { it in 0..8 } == null) {
                    return errorJson("create_shizuku_action_task slot_id must be an integer between 0 and 8")
                }
                payload.put("slot_id", slotId)
            }
        }
        optionalPositiveInt(arguments, "timeout_seconds")?.let { timeout ->
            payload.put("timeout_seconds", timeout)
        }

        val recordArguments = JSONObject(arguments.toString())
            .put("use_shizuku", true)
        if (shizukuAction in SHIZUKU_TOGGLE_STATE_ACTIONS && !recordArguments.has("automation_enabled")) {
            recordArguments.put("automation_enabled", true)
        }
        return createRecordJson(
            context = context,
            arguments = recordArguments,
            actionType = ACTION_TYPE_SHIZUKU_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes Shizuku automation",
            forceUseShizuku = true,
        )
    }

    fun createSunriseSunsetTaskJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { sunriseSunsetPayloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "create_sunrise_sunset_task arguments are invalid")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_SUNRISE_SUNSET,
            payload = payload.toString(),
            defaultLabel = "Hermes sunrise/sunset automation",
        )
    }

    fun calculateSunriseSunsetJson(context: Context, arguments: JSONObject): String {
        val store = HermesAutomationStore(context)
        val variables = store.listVariables()
        val input = runCatching { sunriseSunsetInputFromArguments(arguments, variables) }.getOrElse { error ->
            return errorJson(error.message ?: "calculate_sunrise_sunset arguments are invalid")
        }
        val result = calculateSunriseSunset(input)
        setSunriseSunsetVariables(store, input, result)
        return sunriseSunsetResultJson(input, result, "calculate_sunrise_sunset").toString()
    }

    fun createNotificationTaskJson(context: Context, arguments: JSONObject): String {
        val notificationAction = HermesNotificationActionBridge.normalizeAction(
            stringArgument(arguments, "notification_action", "notify_action", "notification_mode", "mode")
                .orEmpty()
                .ifBlank { "post" },
        )
        val payload = JSONObject().put("notification_action", notificationAction)
        putOptionalExpandedPayloadString(payload, "title", arguments, "notification_title", "title")
        putOptionalExpandedPayloadString(payload, "text", arguments, "notification_text", "text", "content", "message", allowEmpty = true)
        putOptionalExpandedPayloadString(payload, "notification_id", arguments, "notification_id", "notify_id")
        putOptionalExpandedPayloadString(payload, "notification_tag", arguments, "notification_tag", "notify_tag", "tag")
        putOptionalExpandedPayloadString(payload, "channel_id", arguments, "channel_id", "notification_channel_id")
        putOptionalExpandedPayloadString(payload, "channel_name", arguments, "channel_name", "notification_channel_name")
        putOptionalExpandedPayloadString(payload, "priority", arguments, "priority", "notification_priority")
        putOptionalExpandedPayloadString(payload, "importance", arguments, "importance", "notification_importance")
        putOptionalExpandedPayloadString(payload, "group_key", arguments, "group_key", "notification_group", "group")
        putOptionalExpandedPayloadString(payload, "status_text", arguments, "status_text", "notification_status_text", "short_critical_text", "critical_text", allowEmpty = true)
        putOptionalExpandedPayloadString(payload, "progress_value", arguments, "progress_value", "notification_progress", "progress", "progress_current")
        putOptionalExpandedPayloadString(payload, "progress_max", arguments, "progress_max", "notification_progress_max")
        listOf("ongoing", "auto_cancel", "only_alert_once", "show_when", "group_summary", "progress_indeterminate").forEach { key ->
            if (arguments.has(key) && !arguments.isNull(key)) {
                payload.put(key, arguments.optBoolean(key))
            }
        }
        val buttonPayload = notificationButtonsFromArguments(arguments)
        if (buttonPayload.error != null) {
            return errorJson(buttonPayload.error)
        }
        if (buttonPayload.buttons.length() > 0) {
            payload.put("notification_buttons", buttonPayload.buttons)
        }
        if (notificationAction == "post" &&
            payload.optString("title").isBlank() &&
            payload.optString("text").isBlank()
        ) {
            return errorJson("create_notification_task requires notification_title/title or notification_text/text")
        }
        if (notificationAction != "post" && notificationAction != "cancel") {
            return errorJson("Unsupported notification action: $notificationAction")
        }
        val nulValues = listOf(
            "title",
            "text",
            "notification_id",
            "notification_tag",
            "channel_id",
            "channel_name",
            "priority",
            "importance",
            "group_key",
            "status_text",
            "progress_value",
            "progress_max",
        ).mapNotNull { key -> payload.optString(key).takeIf { it.indexOf('\u0000') >= 0 } }
        if (nulValues.isNotEmpty()) {
            return errorJson("create_notification_task fields must not contain NUL bytes")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_NOTIFICATION_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes notification automation",
        )
    }

    fun createVariableActionTaskJson(context: Context, arguments: JSONObject): String {
        val variableAction = normalizeVariableAction(
            stringArgument(arguments, "variable_action", "variableAction", "operation", "mode")
                .orEmpty()
                .ifBlank {
                    when {
                        arguments.has("value") || arguments.has("variable_value") -> VARIABLE_ACTION_SET
                        else -> VARIABLE_ACTION_CLEAR
                    }
                },
        ) ?: return errorJson("create_variable_action_task supports variable_action set, clear, append, add, subtract, or replace")
        val rawName = stringArgument(arguments, "name", "variable", "variable_name", "variableName")
            ?.trim()
            ?: return errorJson("create_variable_action_task requires a variable name like NAME or %NAME")
        val normalized = HermesAutomationStore.normalizeVariableName(rawName)
            ?: return errorJson("create_variable_action_task requires a variable name like NAME or %NAME")
        val payload = JSONObject()
            .put("variable_action", variableAction)
            .put("name", normalized)
        when (variableAction) {
            VARIABLE_ACTION_SET,
            VARIABLE_ACTION_APPEND,
            VARIABLE_ACTION_ADD,
            VARIABLE_ACTION_SUBTRACT -> {
                val value = stringArgument(arguments, "value", "variable_value", "text", "content", "operand", allowEmpty = true)
                    ?: return errorJson("create_variable_action_task $variableAction requires a value argument")
                if (value.indexOf('\u0000') >= 0) {
                    return errorJson("create_variable_action_task value must not contain NUL bytes")
                }
                payload.put("value", value)
            }
            VARIABLE_ACTION_REPLACE -> {
                val search = stringArgument(arguments, "search", "find", "pattern", "match")
                    ?: return errorJson("create_variable_action_task replace requires a search argument")
                val replacement = stringArgument(arguments, "replacement", "replace", "replace_with", "with", allowEmpty = true).orEmpty()
                if (search.isBlank() || search.indexOf('\u0000') >= 0 || replacement.indexOf('\u0000') >= 0) {
                    return errorJson("create_variable_action_task replace fields must not be blank or contain NUL bytes")
                }
                payload.put("search", search)
                    .put("replacement", replacement)
            }
            VARIABLE_ACTION_CLEAR -> Unit
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_VARIABLE_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes variable automation",
        )
    }

    fun createWaitTaskJson(context: Context, arguments: JSONObject): String {
        val durationMs = runCatching { waitDurationMsFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "create_wait_task duration is invalid")
        }
        val payload = JSONObject()
            .put("duration_ms", durationMs)
            .toString()
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_WAIT,
            payload = payload,
            defaultLabel = "Hermes wait automation",
        )
    }

    fun createClipboardTaskJson(context: Context, arguments: JSONObject): String {
        val clipboardAction = normalizeClipboardAction(
            stringArgument(arguments, "clipboard_action", "clipboardAction", "operation", "mode")
                .orEmpty()
                .ifBlank { "set" },
        ) ?: return errorJson("create_clipboard_task supports clipboard_action set")
        val text = stringArgument(
            arguments,
            "clipboard_text",
            "text",
            "content",
            "value",
            allowEmpty = true,
        ) ?: return errorJson("create_clipboard_task requires clipboard_text, text, content, or value")
        if (text.indexOf('\u0000') >= 0) {
            return errorJson("create_clipboard_task text must not contain NUL bytes")
        }
        val label = stringArgument(arguments, "clipboard_label", "label_text", "clip_label")
            ?.take(MAX_CLIPBOARD_LABEL_CHARS)
            ?: "Hermes"
        if (label.indexOf('\u0000') >= 0) {
            return errorJson("create_clipboard_task label must not contain NUL bytes")
        }
        val payload = JSONObject()
            .put("clipboard_action", clipboardAction)
            .put("text", text)
            .put("label", label)
            .toString()
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_CLIPBOARD_ACTION,
            payload = payload,
            defaultLabel = "Hermes clipboard automation",
        )
    }

    fun createVibrationTaskJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { vibrationPayloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "create_vibration_task arguments are invalid")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_VIBRATION_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes vibration automation",
        )
    }

    fun createAudioActionTaskJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { audioPayloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "create_audio_action_task arguments are invalid")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_AUDIO_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes audio automation",
        )
    }

    fun createHttpRequestTaskJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { HermesHttpRequestBridge.payloadFromArguments(arguments, allowVariableUrl = true) }.getOrElse { error ->
            return errorJson(error.message ?: "create_http_request_task arguments are invalid")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_HTTP_REQUEST,
            payload = payload.toString(),
            defaultLabel = "Hermes HTTP request automation",
        )
    }

    fun createOverlaySceneTaskJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { HermesOverlaySceneBridge.payloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "create_overlay_scene_task arguments are invalid")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_OVERLAY_SCENE,
            payload = payload.toString(),
            defaultLabel = "Hermes overlay scene automation",
        )
    }

    fun createToastTaskJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { toastPayloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "create_toast_task arguments are invalid")
        }
        return createRecordJson(
            context = context,
            arguments = arguments,
            actionType = ACTION_TYPE_TOAST_ACTION,
            payload = payload.toString(),
            defaultLabel = "Hermes toast automation",
        )
    }

    fun showToastJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { toastPayloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "show_toast arguments are invalid")
        }
        return HermesToastActionBridge.showToastJson(context, payload).toString()
    }

    fun performAudioActionJson(context: Context, requestedAction: String, arguments: JSONObject): String {
        val directArguments = JSONObject(arguments.toString())
        if (!directArguments.has("audio_action")) {
            when (requestedAction.lowercase().replace("-", "_")) {
                "set_audio_volume" -> directArguments.put("audio_action", "set_volume")
                "set_sound_mode", "set_ringer_mode" -> directArguments.put("audio_action", "set_ringer_mode")
                "set_microphone_mute" -> directArguments.put("audio_action", "set_microphone_mute")
                "set_speakerphone" -> directArguments.put("audio_action", "set_speakerphone")
            }
        }
        val payload = runCatching { audioPayloadFromArguments(directArguments) }.getOrElse { error ->
            return errorJson(error.message ?: "audio_action arguments are invalid")
        }
        return HermesAudioActionBridge.performAudioActionJson(context, payload).toString()
    }

    fun performHttpRequestJson(requestedAction: String, arguments: JSONObject): String {
        val directArguments = JSONObject(arguments.toString())
        if (!directArguments.has("method") && !directArguments.has("http_method")) {
            when (requestedAction.lowercase().replace("-", "_")) {
                "http_get" -> directArguments.put("method", "GET")
                "http_post" -> directArguments.put("method", "POST")
                "http_head" -> directArguments.put("method", "HEAD")
            }
        }
        val payload = runCatching { HermesHttpRequestBridge.payloadFromArguments(directArguments, allowVariableUrl = false) }.getOrElse { error ->
            return errorJson(error.message ?: "http_request arguments are invalid")
        }
        return HermesHttpRequestBridge.performHttpRequestJson(payload).toString()
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
        if (normalizedTrigger == TRIGGER_SENSOR) {
            return errorJson("sensor trigger requires run_sensor_trigger with sensor_type or sensor_name")
        }
        if (normalizedTrigger == TRIGGER_LOGCAT_ENTRY) {
            return errorJson("logcat_entry trigger requires run_logcat_entry_trigger with logcat_tag or logcat_message")
        }
        if (normalizedTrigger == TRIGGER_EXTERNAL) {
            return errorJson("external_trigger requires run_external_trigger with trigger_id and external_token")
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

    fun runSensorTriggerJson(context: Context, arguments: JSONObject): String {
        val sensorType = stringArgument(
            arguments,
            "sensor_type",
            "sensor",
            "sensor_name",
            "sensorType",
        )?.trim() ?: return errorJson("sensor trigger requires sensor_type or sensor_name")
        val sensorEvent = stringArgument(
            arguments,
            "sensor_event",
            "event",
            "event_type",
            "gesture",
            allowEmpty = true,
        ).orEmpty().trim()
        val valueName = stringArgument(
            arguments,
            "value_name",
            "sensor_value_name",
            "axis",
            "sensor_axis",
            allowEmpty = true,
        ).orEmpty().trim()
        val value = optionalDoubleArgument(
            arguments,
            "sensor_value",
            "value",
            "reading",
        )
        val unit = stringArgument(
            arguments,
            "sensor_unit",
            "unit",
            allowEmpty = true,
        ).orEmpty().trim()
        val accuracy = stringArgument(
            arguments,
            "sensor_accuracy",
            "accuracy",
            allowEmpty = true,
        ).orEmpty().trim()
        sensorEventNulError(sensorType, sensorEvent, valueName, unit, accuracy)?.let { error ->
            return errorJson(error)
        }
        val store = HermesAutomationStore(context)
        setSensorEventVariables(store, sensorType, sensorEvent, valueName, value, unit, accuracy)
        val variables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_SENSOR &&
                    sensorEventMatches(record.triggerData, variables, sensorType, sensorEvent, valueName, value)
            }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_SENSOR))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_SENSOR)
            .put("sensor_type", sensorType.take(MAX_EVENT_VALUE_CHARS))
            .put("sensor_event", sensorEvent.take(MAX_EVENT_VALUE_CHARS))
            .put("value_name", valueName.take(MAX_EVENT_VALUE_CHARS))
            .put("sensor_value", value ?: JSONObject.NULL)
            .put("sensor_unit", unit.take(MAX_EVENT_VALUE_CHARS))
            .put("sensor_accuracy", accuracy.take(MAX_EVENT_VALUE_CHARS))
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runLogcatEntryTriggerJson(context: Context, arguments: JSONObject): String {
        val tag = stringArgument(
            arguments,
            "logcat_tag",
            "log_tag",
            "tag",
            "component",
            allowEmpty = true,
        ).orEmpty().trim()
        val message = stringArgument(
            arguments,
            "logcat_message",
            "message",
            "text",
            "line",
            "log_message",
            allowEmpty = true,
        ).orEmpty()
        if (tag.isBlank() && message.isBlank()) {
            return errorJson("logcat_entry trigger requires logcat_tag or logcat_message")
        }
        val level = stringArgument(
            arguments,
            "logcat_level",
            "log_level",
            "priority",
            "level",
            allowEmpty = true,
        ).orEmpty().trim()
        val pid = stringArgument(
            arguments,
            "logcat_pid",
            "pid",
            "process_id",
            allowEmpty = true,
        ).orEmpty().trim()
        val packageName = stringArgument(
            arguments,
            "logcat_package_name",
            "trigger_package_name",
            "package_name",
            "packageName",
            "package",
            "app_package",
            allowEmpty = true,
        ).orEmpty().trim()
        val packageCandidates = stringArgument(
            arguments,
            "logcat_package_candidates",
            "package_candidates",
            "candidate_packages",
            "packages",
            allowEmpty = true,
        ).orEmpty().trim()
        val packageSource = stringArgument(
            arguments,
            "logcat_package_source",
            "package_source",
            "packageNameSource",
            allowEmpty = true,
        ).orEmpty().trim()
        val timestamp = stringArgument(
            arguments,
            "logcat_timestamp",
            "timestamp",
            "time",
            allowEmpty = true,
        ).orEmpty().trim()
        logcatEventNulError(tag, message, level, pid, packageName, packageCandidates, packageSource, timestamp)?.let { error ->
            return errorJson(error)
        }
        val store = HermesAutomationStore(context)
        setLogcatEventVariables(store, tag, message, level, pid, packageName, packageCandidates, packageSource, timestamp)
        val variables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_LOGCAT_ENTRY &&
                    logcatEntryMatches(record, variables, tag, message, level, pid, packageName, packageCandidates)
            }
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_LOGCAT_ENTRY))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_LOGCAT_ENTRY)
            .put("logcat_tag", tag.take(MAX_EVENT_VALUE_CHARS))
            .put("logcat_message", message.take(MAX_VARIABLE_VALUE_CHARS))
            .put("logcat_level", level.take(MAX_EVENT_VALUE_CHARS))
            .put("logcat_pid", pid.take(MAX_EVENT_VALUE_CHARS))
            .put("logcat_package_name", packageName.take(MAX_EVENT_VALUE_CHARS))
            .put("logcat_package_candidates", packageCandidates.take(MAX_EVENT_VALUE_CHARS))
            .put("logcat_package_source", packageSource.take(MAX_EVENT_VALUE_CHARS))
            .put("logcat_timestamp", timestamp.take(MAX_EVENT_VALUE_CHARS))
            .put("requires_shizuku_for_background_watch", true)
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runExternalTriggerJson(context: Context, arguments: JSONObject): String {
        val triggerId = stringArgument(
            arguments,
            "trigger_id",
            "external_trigger_id",
            "event_id",
            "triggerId",
        )?.trim() ?: return errorJson("external_trigger requires trigger_id")
        val externalToken = stringArgument(
            arguments,
            "external_token",
            "trigger_token",
            "token",
            "auth_token",
        )?.trim() ?: return errorJson("external_trigger requires external_token")
        val packageName = stringArgument(
            arguments,
            "trigger_package_name",
            "package_name",
            "packageName",
            "package",
            "caller_package",
            allowEmpty = true,
        ).orEmpty().trim()
        val referrer = stringArgument(
            arguments,
            "referrer",
            "source",
            "caller",
            "uri",
            allowEmpty = true,
        ).orEmpty().trim()
        val extrasText = externalExtrasText(arguments)
        externalEventNulError(triggerId, externalToken, packageName, referrer, extrasText)?.let { error ->
            return errorJson(error)
        }
        val store = HermesAutomationStore(context)
        val savedVariables = store.listVariables()
        val records = store.list()
            .filter { record ->
                record.enabled &&
                    record.triggerType == TRIGGER_EXTERNAL &&
                    externalTriggerMatches(record, savedVariables, triggerId, externalToken, packageName, referrer)
            }
        setExternalTriggerVariables(store, triggerId, packageName, referrer, extrasText)
        val results = JSONArray()
        records.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_EXTERNAL))
        }
        return JSONObject()
            .put("success", true)
            .put("trigger", TRIGGER_EXTERNAL)
            .put("trigger_id", triggerId.take(MAX_EVENT_VALUE_CHARS))
            .put("trigger_package_name", packageName.take(MAX_EVENT_VALUE_CHARS))
            .put("referrer", referrer.take(MAX_EVENT_VALUE_CHARS))
            .put("extras", extrasText.take(MAX_VARIABLE_VALUE_CHARS))
            .put("matched_count", records.size)
            .put("results", results)
            .toString()
    }

    fun runRemoteDispatchJson(context: Context, arguments: JSONObject): String {
        val payload = dispatchPayloadFromArguments(arguments)
        val store = HermesAutomationStore(context)
        val dispatch = dispatchContextFromPayload(payload)
        val automationId = stringArgument(payload, "automation_id", "automationId", "id").orEmpty().trim()
        val taskName = stringArgument(payload, "task_name", "taskName", "remote_task_name", "label", "name").orEmpty().trim()
        val allowDisabled = booleanArgument(payload, "allow_disabled", "allowDisabled") ?: false
        val records = store.list()
        val matchedRecords = when {
            automationId.isNotBlank() -> records.filter { record -> record.id == automationId }
            taskName.isNotBlank() -> records.filter { record ->
                record.label.equals(taskName, ignoreCase = true) || record.id.equals(taskName, ignoreCase = true)
            }
            else -> records.filter { record -> record.triggerType == TRIGGER_REMOTE_DISPATCH }
        }
        if (matchedRecords.isEmpty()) {
            val message =
                "No Android automation matched remote dispatch. Pass automation_id or taskName, or create an enabled automation with trigger remote_dispatch."
            val event = recordRemoteDispatchFailure(store, dispatch, message)
            return remoteDispatchFailureJson(dispatch, event, message, matchedRecords.size).toString()
        }
        val runnableRecords = if (allowDisabled) matchedRecords else matchedRecords.filter { it.enabled }
        if (runnableRecords.isEmpty()) {
            val message = "Remote dispatch matched only disabled automations; enable one or pass allow_disabled=true from a trusted local caller."
            val event = recordRemoteDispatchFailure(store, dispatch, message)
            return remoteDispatchFailureJson(dispatch, event, message, matchedRecords.size).toString()
        }
        setRemoteDispatchVariables(store, dispatch)
        val results = JSONArray()
        runnableRecords.forEach { record ->
            results.put(runRecordJson(context, store, record, TRIGGER_REMOTE_DISPATCH, dispatch))
        }
        val allSucceeded = (0 until results.length()).all { index ->
            results.optJSONObject(index)?.optBoolean("success", false) == true
        }
        return JSONObject()
            .put("success", true)
            .put("status", if (allSucceeded) "completed" else "failed")
            .put("terminal", true)
            .put("trigger", TRIGGER_REMOTE_DISPATCH)
            .put("dispatch_source", dispatch.source)
            .put("dispatch_channel", dispatch.channel)
            .put("remote_execution_id", dispatch.executionId)
            .put("remote_task_id", dispatch.taskId)
            .put("remote_task_name", dispatch.taskName)
            .put("matched_count", runnableRecords.size)
            .put("results", results)
            .toString()
    }

    private fun recordRemoteDispatchFailure(
        store: HermesAutomationStore,
        dispatch: HermesAutomationDispatchContext,
        message: String,
    ): HermesAutomationRunEvent {
        val now = System.currentTimeMillis()
        val label = dispatch.taskName
            .ifBlank { dispatch.taskId }
            .ifBlank { dispatch.executionId }
            .ifBlank { "Remote dispatch" }
            .take(MAX_EVENT_VALUE_CHARS)
        val event = HermesAutomationRunEvent(
            id = UUID.randomUUID().toString(),
            automationId = REMOTE_DISPATCH_FAILURE_AUTOMATION_ID,
            automationLabel = label,
            actionType = TRIGGER_REMOTE_DISPATCH,
            trigger = TRIGGER_REMOTE_DISPATCH,
            success = false,
            exitCode = -1,
            result = message.take(MAX_RESULT_CHARS),
            startedAtEpochMs = now,
            finishedAtEpochMs = now,
            dispatchSource = dispatch.source,
            dispatchChannel = dispatch.channel,
            remoteExecutionId = dispatch.executionId,
            remoteTaskId = dispatch.taskId,
            remoteTaskName = dispatch.taskName,
        )
        store.addRunEvent(event)
        return event
    }

    private fun remoteDispatchFailureJson(
        dispatch: HermesAutomationDispatchContext,
        event: HermesAutomationRunEvent,
        message: String,
        matchedCount: Int,
    ): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("status", "failed")
            .put("terminal", true)
            .put("trigger", TRIGGER_REMOTE_DISPATCH)
            .put("dispatch_source", dispatch.source)
            .put("dispatch_channel", dispatch.channel)
            .put("remote_execution_id", dispatch.executionId)
            .put("remote_task_id", dispatch.taskId)
            .put("remote_task_name", dispatch.taskName)
            .put("matched_count", matchedCount)
            .put("error", message)
            .put("execution", event.toJson())
    }

    private data class HermesAutomationDispatchContext(
        val source: String = "",
        val channel: String = "",
        val executionId: String = "",
        val taskId: String = "",
        val taskName: String = "",
    )

    private fun dispatchPayloadFromArguments(arguments: JSONObject): JSONObject {
        return jsonObjectArgument(arguments, "payload")
            ?: jsonObjectArgument(arguments, "dispatch")
            ?: jsonObjectArgument(arguments, "standby_dispatch")
            ?: arguments
    }

    private fun dispatchContextFromPayload(payload: JSONObject): HermesAutomationDispatchContext {
        return HermesAutomationDispatchContext(
            source = stringArgument(payload, "dispatch_source", "source", allowEmpty = true)
                .orEmpty()
                .ifBlank { "opengui_standby" }
                .take(MAX_EVENT_VALUE_CHARS),
            channel = stringArgument(payload, "dispatch_channel", "channel", allowEmpty = true)
                .orEmpty()
                .ifBlank { "standby" }
                .take(MAX_EVENT_VALUE_CHARS),
            executionId = stringArgument(payload, "execution_id", "executionId", "remote_execution_id", allowEmpty = true)
                .orEmpty()
                .take(MAX_EVENT_VALUE_CHARS),
            taskId = stringArgument(payload, "task_id", "taskId", "remote_task_id", allowEmpty = true)
                .orEmpty()
                .take(MAX_EVENT_VALUE_CHARS),
            taskName = stringArgument(payload, "task_name", "taskName", "remote_task_name", "label", "name", allowEmpty = true)
                .orEmpty()
                .take(MAX_EVENT_VALUE_CHARS),
        )
    }

    private fun setRemoteDispatchVariables(store: HermesAutomationStore, dispatch: HermesAutomationDispatchContext) {
        store.setVariable("DISPATCH_SOURCE", dispatch.source)
        store.setVariable("DISPATCH_CHANNEL", dispatch.channel)
        store.setVariable("DISPATCH_EXECUTION_ID", dispatch.executionId)
        store.setVariable("DISPATCH_TASK_ID", dispatch.taskId)
        store.setVariable("DISPATCH_TASK_NAME", dispatch.taskName)
        store.setVariable("REMOTE_EXECUTION_ID", dispatch.executionId)
        store.setVariable("REMOTE_TASK_ID", dispatch.taskId)
        store.setVariable("REMOTE_TASK_NAME", dispatch.taskName)
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
        runCatching {
            HermesTaskerEventBridge.notifyShizukuState(context, available, status)
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
        dispatch: HermesAutomationDispatchContext = HermesAutomationDispatchContext(),
    ): JSONObject {
        if (trigger == TRIGGER_TIME) {
            setTimeEventVariables(store)
        }
        val variables = store.listVariables()
        val startedAtEpochMs = System.currentTimeMillis()
        val rawResult = when (record.actionType) {
            ACTION_TYPE_SHELL -> runShellRecord(context, record, variables)
            ACTION_TYPE_FILE_WRITE -> runFileWriteRecord(context, record, variables)
            ACTION_TYPE_FILE_DELETE -> HermesWorkspaceFileBridge.deleteJson(context, expandVariables(record.command, variables))
            ACTION_TYPE_SYSTEM_ACTION -> runSystemActionRecord(context, record, variables)
            ACTION_TYPE_UI_ACTION -> runUiActionRecord(record, variables)
            ACTION_TYPE_APP_LAUNCH -> HermesAppControlBridge.launchPackage(context, expandVariables(record.command, variables))
            ACTION_TYPE_INTENT -> runIntentRecord(context, record, variables)
            ACTION_TYPE_SHIZUKU_ACTION -> runShizukuActionRecord(context, record, variables)
            ACTION_TYPE_SUNRISE_SUNSET -> runSunriseSunsetRecord(store, record, variables)
            ACTION_TYPE_NOTIFICATION_ACTION -> runNotificationActionRecord(context, record, variables)
            ACTION_TYPE_VARIABLE_ACTION -> runVariableActionRecord(store, record, variables)
            ACTION_TYPE_WAIT -> runWaitRecord(record, variables)
            ACTION_TYPE_CLIPBOARD_ACTION -> runClipboardActionRecord(context, record, variables)
            ACTION_TYPE_VIBRATION_ACTION -> runVibrationActionRecord(context, record, variables)
            ACTION_TYPE_AUDIO_ACTION -> runAudioActionRecord(context, record, variables)
            ACTION_TYPE_HTTP_REQUEST -> runHttpRequestRecord(store, record, variables)
            ACTION_TYPE_OVERLAY_SCENE -> runOverlaySceneRecord(context, record, variables)
            ACTION_TYPE_TOAST_ACTION -> runToastActionRecord(context, record, variables)
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
        val finishedAtEpochMs = System.currentTimeMillis()
        val updated = record.copy(
            updatedAtEpochMs = finishedAtEpochMs,
            lastRunEpochMs = finishedAtEpochMs,
            lastExitCode = exitCode,
            lastSuccess = success,
            lastResult = resultText,
        )
        store.upsert(updated)
        store.addRunEvent(
            HermesAutomationRunEvent(
                id = UUID.randomUUID().toString(),
                automationId = record.id,
                automationLabel = record.label,
                actionType = record.actionType,
                trigger = trigger,
                success = success,
                exitCode = exitCode,
                result = resultText,
                startedAtEpochMs = startedAtEpochMs,
                finishedAtEpochMs = finishedAtEpochMs,
                dispatchSource = dispatch.source,
                dispatchChannel = dispatch.channel,
                remoteExecutionId = dispatch.executionId,
                remoteTaskId = dispatch.taskId,
                remoteTaskName = dispatch.taskName,
            )
        )
        runCatching {
            HermesTaskerEventBridge.notifyAutomationFinished(
                context = context,
                record = updated,
                trigger = trigger,
                success = success,
                resultText = resultText,
            )
        }
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
        if (shizukuAction in SHIZUKU_PACKAGE_ACTIONS) {
            actionArguments.put("package_name", expandVariables(payload.optString("package_name"), variables))
        }
        if (shizukuAction in SHIZUKU_PERMISSION_ACTIONS && payload.has("permission")) {
            actionArguments.put("permission", expandVariables(payload.optString("permission"), variables))
        }
        if (payload.has("target_enabled") && !payload.isNull("target_enabled")) {
            actionArguments.put("enabled", payload.optBoolean("target_enabled"))
        }
        if (payload.has("state") && !payload.isNull("state")) {
            actionArguments.put("state", expandVariables(payload.optString("state"), variables))
        }
        if (shizukuAction in SHIZUKU_CUSTOM_SETTING_ACTIONS) {
            actionArguments.put("setting_namespace", expandVariables(payload.optString("setting_namespace"), variables))
            actionArguments.put("setting_name", expandVariables(payload.optString("setting_name"), variables))
            if (payload.has("setting_value") && !payload.isNull("setting_value")) {
                actionArguments.put("setting_value", expandVariables(payload.optString("setting_value"), variables))
            }
        }
        if (shizukuAction in SHIZUKU_DND_ACTIONS && payload.has("dnd_mode") && !payload.isNull("dnd_mode")) {
            actionArguments.put("dnd_mode", expandVariables(payload.optString("dnd_mode"), variables))
        }
        if (shizukuAction in SHIZUKU_USER_PROFILE_ACTIONS && payload.has("user_id") && !payload.isNull("user_id")) {
            actionArguments.put("user_id", expandVariables(payload.optString("user_id"), variables))
        }
        if (shizukuAction in SHIZUKU_MOBILE_NETWORK_TYPE_ACTIONS) {
            actionArguments.put("network_types_bitmask", expandVariables(payload.optString("network_types_bitmask"), variables))
            if (payload.has("slot_id") && !payload.isNull("slot_id")) {
                actionArguments.put("slot_id", expandVariables(payload.optString("slot_id"), variables))
            }
        }
        if (payload.has("timeout_seconds") && !payload.isNull("timeout_seconds")) {
            actionArguments.put("timeout_seconds", payload.optInt("timeout_seconds", AUTOMATION_TIMEOUT_SECONDS))
        }
        return JSONObject(HermesPrivilegedAccessBridge.performStructuredActionJson(context, shizukuAction, actionArguments))
    }

    private fun runSunriseSunsetRecord(
        store: HermesAutomationStore,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved sunrise_sunset automation payload is invalid"))
        val input = runCatching { sunriseSunsetInputFromPayload(payload, variables) }.getOrElse { error ->
            return JSONObject(errorJson(error.message ?: "Saved sunrise_sunset automation payload is invalid"))
        }
        val result = calculateSunriseSunset(input)
        setSunriseSunsetVariables(store, input, result)
        return sunriseSunsetResultJson(input, result, "sunrise_sunset")
    }

    private fun runNotificationActionRecord(
        context: Context,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved notification automation payload is invalid"))
        return JSONObject(HermesNotificationActionBridge.performNotificationJson(context, expandNotificationPayload(payload, variables)))
    }

    private fun runOverlaySceneRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved overlay_scene automation payload is invalid"))
        val expanded = expandOverlayScenePayload(payload, variables)
        val sceneAction = expanded.optString("scene_action").ifBlank { "show" }
        return JSONObject(HermesOverlaySceneBridge.performSceneJson(context, "${sceneAction}_overlay_scene", expanded))
    }

    private fun runToastActionRecord(context: Context, record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved toast automation payload is invalid"))
        return HermesToastActionBridge.showToastJson(context, expandToastPayload(payload, variables))
    }

    private fun runVariableActionRecord(
        store: HermesAutomationStore,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved variable_action automation payload is invalid"))
        val variableAction = normalizeVariableAction(expandVariables(payload.optString("variable_action"), variables))
            ?: return JSONObject(errorJson("Unsupported saved variable action: ${payload.optString("variable_action")}"))
        val normalized = HermesAutomationStore.normalizeVariableName(expandVariables(payload.optString("name"), variables))
            ?: return JSONObject(errorJson("Saved variable_action automation requires a variable name like NAME or %NAME"))
        return when (variableAction) {
            VARIABLE_ACTION_SET -> {
                val value = expandVariables(payload.optString("value"), variables)
                if (value.indexOf('\u0000') >= 0) {
                    return JSONObject(errorJson("Saved variable_action value must not contain NUL bytes"))
                }
                store.setVariable(normalized, value)
                JSONObject()
                    .put("success", true)
                    .put("exit_code", 0)
                    .put("action", VARIABLE_ACTION_SET)
                    .put("name", normalized)
                    .put("value", value.take(MAX_VARIABLE_VALUE_CHARS))
                    .put("message", "Set Android automation variable %$normalized")
            }
            VARIABLE_ACTION_CLEAR -> {
                val removed = store.removeVariable(normalized)
                JSONObject()
                    .put("success", true)
                    .put("exit_code", 0)
                    .put("action", VARIABLE_ACTION_CLEAR)
                    .put("name", normalized)
                    .put("removed", removed)
                    .put("message", "Cleared Android automation variable %$normalized")
            }
            VARIABLE_ACTION_APPEND -> {
                val value = expandVariables(payload.optString("value"), variables)
                if (value.indexOf('\u0000') >= 0) {
                    return JSONObject(errorJson("Saved variable_action value must not contain NUL bytes"))
                }
                val updatedValue = ((store.getVariable(normalized) ?: "") + value).take(MAX_VARIABLE_VALUE_CHARS)
                store.setVariable(normalized, updatedValue)
                JSONObject()
                    .put("success", true)
                    .put("exit_code", 0)
                    .put("action", VARIABLE_ACTION_APPEND)
                    .put("name", normalized)
                    .put("value", updatedValue)
                    .put("message", "Appended Android automation variable %$normalized")
            }
            VARIABLE_ACTION_ADD,
            VARIABLE_ACTION_SUBTRACT -> {
                val current = (store.getVariable(normalized) ?: "0").trim().ifBlank { "0" }.toDoubleOrNull()
                    ?: return JSONObject(errorJson("Saved variable_action $variableAction current value must be numeric"))
                val operand = expandVariables(payload.optString("value"), variables).trim().toDoubleOrNull()
                    ?: return JSONObject(errorJson("Saved variable_action $variableAction value must be numeric"))
                val updatedNumber = if (variableAction == VARIABLE_ACTION_ADD) current + operand else current - operand
                val updatedValue = formatVariableNumber(updatedNumber)
                store.setVariable(normalized, updatedValue)
                JSONObject()
                    .put("success", true)
                    .put("exit_code", 0)
                    .put("action", variableAction)
                    .put("name", normalized)
                    .put("value", updatedValue)
                    .put("message", "Updated numeric Android automation variable %$normalized")
            }
            VARIABLE_ACTION_REPLACE -> {
                val search = expandVariables(payload.optString("search"), variables)
                val replacement = expandVariables(payload.optString("replacement"), variables)
                if (search.isBlank() || search.indexOf('\u0000') >= 0 || replacement.indexOf('\u0000') >= 0) {
                    return JSONObject(errorJson("Saved variable_action replace fields must not be blank or contain NUL bytes"))
                }
                val updatedValue = (store.getVariable(normalized) ?: "")
                    .replace(search, replacement)
                    .take(MAX_VARIABLE_VALUE_CHARS)
                store.setVariable(normalized, updatedValue)
                JSONObject()
                    .put("success", true)
                    .put("exit_code", 0)
                    .put("action", VARIABLE_ACTION_REPLACE)
                    .put("name", normalized)
                    .put("value", updatedValue)
                    .put("message", "Replaced text in Android automation variable %$normalized")
            }
            else -> JSONObject(errorJson("Unsupported saved variable action: $variableAction"))
        }
    }

    private fun runWaitRecord(record: HermesAutomationRecord, variables: JSONObject): JSONObject {
        val durationMs = runCatching {
            val payload = JSONObject(record.command)
            waitDurationMsFromPayload(payload, variables)
        }.getOrElse { error ->
            return JSONObject(errorJson(error.message ?: "Saved wait automation payload is invalid"))
        }
        return try {
            Thread.sleep(durationMs)
            JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("duration_ms", durationMs)
                .put("message", "Waited ${durationMs} ms")
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
            JSONObject(errorJson("Saved wait automation was interrupted"))
        }
    }

    private fun runClipboardActionRecord(
        context: Context,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved clipboard_action automation payload is invalid"))
        val clipboardAction = normalizeClipboardAction(expandVariables(payload.optString("clipboard_action"), variables))
            ?: return JSONObject(errorJson("Unsupported saved clipboard action: ${payload.optString("clipboard_action")}"))
        if (clipboardAction != CLIPBOARD_ACTION_SET) {
            return JSONObject(errorJson("Unsupported saved clipboard action: $clipboardAction"))
        }
        val text = expandVariables(payload.optString("text"), variables)
        val label = expandVariables(payload.optString("label").ifBlank { "Hermes" }, variables)
            .take(MAX_CLIPBOARD_LABEL_CHARS)
        return HermesClipboardActionBridge.setClipboardJson(context, text, label)
    }

    private fun runVibrationActionRecord(
        context: Context,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved vibration_action automation payload is invalid"))
        val expanded = expandVibrationPayload(payload, variables)
        return HermesVibrationActionBridge.vibrateJson(context, expanded)
    }

    private fun runAudioActionRecord(
        context: Context,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved audio_action automation payload is invalid"))
        val expanded = expandAudioPayload(payload, variables)
        return HermesAudioActionBridge.performAudioActionJson(context, expanded)
    }

    private fun runHttpRequestRecord(
        store: HermesAutomationStore,
        record: HermesAutomationRecord,
        variables: JSONObject,
    ): JSONObject {
        val payload = runCatching { JSONObject(record.command) }.getOrNull()
            ?: return JSONObject(errorJson("Saved http_request automation payload is invalid"))
        val expanded = runCatching { expandHttpRequestPayload(payload, variables) }.getOrElse { error ->
            return JSONObject(errorJson(error.message ?: "Saved http_request automation payload is invalid"))
        }
        val result = HermesHttpRequestBridge.performHttpRequestJson(expanded)
        val statusText = if (result.has("status_code")) result.optInt("status_code").toString() else ""
        val bodyText = result.optString("body")
        store.setVariable("HTTPR", statusText)
        store.setVariable("HTTP_STATUS_CODE", statusText)
        store.setVariable("HTTPD", bodyText)
        store.setVariable("HTTP_RESPONSE_BODY", bodyText)
        val statusVariable = HermesAutomationStore.normalizeVariableName(expandVariables(expanded.optString("save_status_variable"), variables))
        if (statusVariable != null) {
            store.setVariable(statusVariable, statusText)
        }
        val responseVariable = HermesAutomationStore.normalizeVariableName(expandVariables(expanded.optString("save_response_variable"), variables))
        if (responseVariable != null) {
            store.setVariable(responseVariable, bodyText)
        }
        return result
            .put("saved_status_variable", statusVariable ?: JSONObject.NULL)
            .put("saved_response_variable", responseVariable ?: JSONObject.NULL)
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

    private fun notificationButtonsFromArguments(arguments: JSONObject): NotificationButtonsResult {
        val buttons = JSONArray()
        val rawButtons = arguments.optJSONArray("notification_buttons")
            ?: arguments.optJSONArray("buttons")
            ?: arguments.optJSONArray("notification_actions")
        if (rawButtons != null) {
            if (rawButtons.length() > MAX_NOTIFICATION_BUTTONS) {
                return NotificationButtonsResult(buttons, "notification_buttons supports at most $MAX_NOTIFICATION_BUTTONS buttons")
            }
            for (index in 0 until rawButtons.length()) {
                val raw = rawButtons.optJSONObject(index)
                    ?: return NotificationButtonsResult(buttons, "notification_buttons[$index] must be a JSON object")
                val parsed = notificationButtonFromJson(raw, index)
                if (parsed.error != null) {
                    return NotificationButtonsResult(buttons, parsed.error)
                }
                buttons.put(parsed.button)
            }
        }
        for (index in 1..MAX_NOTIFICATION_BUTTONS) {
            val title = stringArgument(
                arguments,
                "notification_button_${index}_title",
                "notification_action_${index}_title",
                "button_${index}_title",
                "button_${index}_label",
                allowEmpty = true,
            ) ?: continue
            val raw = JSONObject()
                .put("title", title)
                .put(
                    "action",
                    stringArgument(
                        arguments,
                        "notification_button_${index}_action",
                        "notification_action_${index}_action",
                        "button_${index}_action",
                    ).orEmpty().ifBlank { "open_app" },
                )
                .put(
                    "automation_id",
                    stringArgument(
                        arguments,
                        "notification_button_${index}_automation_id",
                        "notification_action_${index}_automation_id",
                        "button_${index}_automation_id",
                    ).orEmpty(),
                )
            if (arguments.has("notification_button_${index}_dismiss_on_tap")) {
                raw.put("dismiss_on_tap", arguments.optBoolean("notification_button_${index}_dismiss_on_tap"))
            }
            val parsed = notificationButtonFromJson(raw, buttons.length())
            if (parsed.error != null) {
                return NotificationButtonsResult(buttons, parsed.error)
            }
            buttons.put(parsed.button)
            if (buttons.length() > MAX_NOTIFICATION_BUTTONS) {
                return NotificationButtonsResult(buttons, "notification_buttons supports at most $MAX_NOTIFICATION_BUTTONS buttons")
            }
        }
        return NotificationButtonsResult(buttons)
    }

    private fun notificationButtonFromJson(raw: JSONObject, index: Int): NotificationButtonResult {
        val title = raw.optString("title")
            .ifBlank { raw.optString("label") }
            .ifBlank { raw.optString("text") }
            .take(MAX_NOTIFICATION_BUTTON_FIELD_CHARS)
        if (title.isBlank()) {
            return NotificationButtonResult(error = "notification_buttons[$index] requires title")
        }
        val action = HermesNotificationActionBridge.normalizeButtonAction(
            raw.optString("action")
                .ifBlank { raw.optString("button_action") }
                .ifBlank { raw.optString("type") }
                .ifBlank { "open_app" },
        )
        if (!HermesNotificationActionBridge.isSupportedButtonAction(action)) {
            return NotificationButtonResult(error = "Unsupported notification button action: $action")
        }
        val automationId = raw.optString("automation_id")
            .ifBlank { raw.optString("automation") }
            .ifBlank { raw.optString("id") }
            .take(MAX_AUTOMATION_ID_CHARS)
        if (action == "run_automation" && automationId.isBlank()) {
            return NotificationButtonResult(error = "notification run_automation button requires automation_id")
        }
        if (listOf(title, action, automationId).any { it.indexOf('\u0000') >= 0 }) {
            return NotificationButtonResult(error = "notification button fields must not contain NUL bytes")
        }
        return NotificationButtonResult(
            button = JSONObject()
                .put("title", title)
                .put("action", action)
                .put("automation_id", automationId)
                .put("dismiss_on_tap", raw.optBoolean("dismiss_on_tap", raw.optBoolean("cancel_notification", false))),
        )
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
            TRIGGER_SENSOR -> buildSensorTriggerData(arguments)
            TRIGGER_LOGCAT_ENTRY -> buildLogcatTriggerData(arguments)
            TRIGGER_EXTERNAL -> buildExternalTriggerData(arguments)
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

    private fun buildSensorTriggerData(arguments: JSONObject): TriggerDataResult {
        val payload = JSONObject()
        listOf(
            copyCalendarTriggerFilter(
                payload,
                "sensor_type",
                arguments,
                "sensor_type",
                "sensor",
                "sensor_name",
                "sensorType",
            ),
            copyCalendarTriggerFilter(
                payload,
                "sensor_event",
                arguments,
                "sensor_event",
                "event",
                "event_type",
                "gesture",
            ),
            copyCalendarTriggerFilter(
                payload,
                "value_name",
                arguments,
                "value_name",
                "sensor_value_name",
                "axis",
                "sensor_axis",
            ),
            copyLocationNumericTriggerFilter(
                payload,
                "min_value",
                arguments,
                "min_value",
                "sensor_min_value",
                "value_min",
            ),
            copyLocationNumericTriggerFilter(
                payload,
                "max_value",
                arguments,
                "max_value",
                "sensor_max_value",
                "value_max",
            ),
        ).firstOrNull { it != null }?.let { error ->
            return TriggerDataResult("", error)
        }
        validateSensorTriggerNumericPayload(payload)?.let { error ->
            return TriggerDataResult("", error)
        }
        return TriggerDataResult(payload.toString())
    }

    private fun buildLogcatTriggerData(arguments: JSONObject): TriggerDataResult {
        val payload = JSONObject()
        listOf(
            copyCalendarTriggerFilter(
                payload,
                "tag",
                arguments,
                "logcat_tag",
                "log_tag",
                "tag",
                "component",
            ),
            copyCalendarTriggerFilter(
                payload,
                "message_contains",
                arguments,
                "logcat_message_contains",
                "message_contains",
                "message_filter",
                "log_message_contains",
                "logcat_message",
                "log_message",
                "message",
                "filter",
                "grep_filter",
            ),
            copyCalendarTriggerFilter(
                payload,
                "level",
                arguments,
                "logcat_level",
                "log_level",
                "priority",
                "level",
            ),
            copyCalendarTriggerFilter(
                payload,
                "package_name",
                arguments,
                "logcat_package_name",
                "log_package_name",
                "source_package_name",
            ),
            copyCalendarTriggerFilter(
                payload,
                "pid",
                arguments,
                "logcat_pid",
                "pid",
                "process_id",
            ),
        ).firstOrNull { it != null }?.let { error ->
            return TriggerDataResult("", error)
        }
        val hasTriggerPackage = stringArgument(
            arguments,
            "trigger_package_name",
            "triggerPackageName",
            "profile_package_name",
            "context_package_name",
        )?.trim()?.isNotBlank() == true
        val hasBoundedFilter = payload.has("tag") ||
            payload.has("message_contains") ||
            payload.has("package_name") ||
            payload.has("pid") ||
            hasTriggerPackage
        if (!hasBoundedFilter) {
            return TriggerDataResult("", "logcat_entry trigger requires logcat_tag, logcat_message_contains, logcat_package_name, trigger_package_name, or logcat_pid")
        }
        validateLogcatTriggerData(payload)?.let { error ->
            return TriggerDataResult("", error)
        }
        payload.put("requires_shizuku_for_background_watch", true)
        return TriggerDataResult(payload.toString())
    }

    private fun validateSensorTriggerNumericPayload(payload: JSONObject): String? {
        val minValue = literalDoubleFilter(payload, "min_value")
        val maxValue = literalDoubleFilter(payload, "max_value")
        if (minValue != null && maxValue != null && minValue > maxValue) {
            return "sensor trigger min_value must be less than or equal to max_value"
        }
        return null
    }

    private fun validateLogcatTriggerData(payload: JSONObject): String? {
        val level = payload.optString("level").trim()
        if (level.isNotBlank() && !looksLikeVariableReference(level) && normalizeLogcatLevel(level).isBlank()) {
            return "logcat_level must be verbose, debug, info, warn, error, assert, or one of V/D/I/W/E/A/F"
        }
        val pid = payload.optString("pid").trim()
        if (pid.isNotBlank() && !looksLikeVariableReference(pid) && pid.toIntOrNull()?.takeIf { it > 0 } == null) {
            return "logcat_pid must be a positive integer"
        }
        return null
    }

    private fun validateLogcatTriggerData(triggerData: String, triggerPackageName: String): String? {
        val payload = runCatching { JSONObject(triggerData.ifBlank { "{}" }) }.getOrElse {
            return "logcat_entry trigger_data must be a JSON object"
        }
        validateLogcatTriggerData(payload)?.let { error -> return error }
        val hasBoundedFilter = payload.has("tag") ||
            payload.has("message_contains") ||
            payload.has("package_name") ||
            payload.has("pid") ||
            triggerPackageName.isNotBlank()
        return if (hasBoundedFilter) {
            null
        } else {
            "logcat_entry trigger_data requires tag, message_contains, package_name, pid, or trigger_package_name"
        }
    }

    private fun buildExternalTriggerData(arguments: JSONObject): TriggerDataResult {
        val triggerId = stringArgument(
            arguments,
            "trigger_id",
            "external_trigger_id",
            "event_id",
            "triggerId",
        )?.trim()
        if (triggerId.isNullOrBlank()) {
            return TriggerDataResult("", "external_trigger requires trigger_id")
        }
        val externalToken = stringArgument(
            arguments,
            "external_token",
            "trigger_token",
            "token",
            "auth_token",
        )?.trim()
        if (externalToken.isNullOrBlank()) {
            return TriggerDataResult("", "external_trigger requires external_token")
        }
        val payload = JSONObject()
            .put("trigger_id", triggerId)
            .put("external_token", externalToken)
        copyCalendarTriggerFilter(
            payload,
            "referrer_contains",
            arguments,
            "referrer_contains",
            "referrer",
            "source_contains",
            "caller_contains",
        )?.let { error -> return TriggerDataResult("", error) }
        validateExternalTriggerData(payload.toString())?.let { error ->
            return TriggerDataResult("", error)
        }
        return TriggerDataResult(payload.toString())
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

    private fun sunriseSunsetPayloadFromArguments(arguments: JSONObject): JSONObject {
        val latitude = rawArgumentText(arguments, "latitude", "lat", "solar_latitude")
            ?: throw IllegalArgumentException("create_sunrise_sunset_task requires latitude")
        val longitude = rawArgumentText(arguments, "longitude", "lon", "lng", "solar_longitude")
            ?: throw IllegalArgumentException("create_sunrise_sunset_task requires longitude")
        validateSolarCoordinateText(latitude, "latitude", -90.0, 90.0, allowVariables = true)
        validateSolarCoordinateText(longitude, "longitude", -180.0, 180.0, allowVariables = true)

        val timezone = rawArgumentText(arguments, "timezone", "time_zone", "tz")
        timezone?.let { validateSolarTimeZoneText(it, allowVariables = true) }
        val validationTimeZone = if (timezone != null && !looksLikeVariableReference(timezone)) {
            parseSolarTimeZone(timezone)
        } else {
            TimeZone.getDefault()
        }
        val date = rawArgumentText(arguments, "date", "day", "solar_date")
        date?.let { validateSolarDateText(it, validationTimeZone, allowVariables = true) }

        return JSONObject()
            .put("latitude", latitude)
            .put("longitude", longitude)
            .apply {
                if (!date.isNullOrBlank()) {
                    put("date", date)
                }
                if (!timezone.isNullOrBlank()) {
                    put("timezone", timezone)
                }
            }
    }

    private fun sunriseSunsetInputFromArguments(arguments: JSONObject, variables: JSONObject): SunriseSunsetInput {
        val timezoneText = expandedArgumentText(arguments, variables, "timezone", "time_zone", "tz")
        val timeZone = parseSolarTimeZone(timezoneText)
        val date = parseSolarDate(expandedArgumentText(arguments, variables, "date", "day", "solar_date"), timeZone)
        return SunriseSunsetInput(
            latitude = parseSolarCoordinate(
                expandedArgumentText(arguments, variables, "latitude", "lat", "solar_latitude")
                    ?: throw IllegalArgumentException("calculate_sunrise_sunset requires latitude"),
                "latitude",
                -90.0,
                90.0,
            ),
            longitude = parseSolarCoordinate(
                expandedArgumentText(arguments, variables, "longitude", "lon", "lng", "solar_longitude")
                    ?: throw IllegalArgumentException("calculate_sunrise_sunset requires longitude"),
                "longitude",
                -180.0,
                180.0,
            ),
            date = date,
            timeZone = timeZone,
        )
    }

    private fun sunriseSunsetInputFromPayload(payload: JSONObject, variables: JSONObject): SunriseSunsetInput {
        val timezoneText = expandedArgumentText(payload, variables, "timezone", "time_zone", "tz")
        val timeZone = parseSolarTimeZone(timezoneText)
        val date = parseSolarDate(expandedArgumentText(payload, variables, "date", "day", "solar_date"), timeZone)
        return SunriseSunsetInput(
            latitude = parseSolarCoordinate(
                expandedArgumentText(payload, variables, "latitude", "lat", "solar_latitude")
                    ?: throw IllegalArgumentException("sunrise_sunset automation requires latitude"),
                "latitude",
                -90.0,
                90.0,
            ),
            longitude = parseSolarCoordinate(
                expandedArgumentText(payload, variables, "longitude", "lon", "lng", "solar_longitude")
                    ?: throw IllegalArgumentException("sunrise_sunset automation requires longitude"),
                "longitude",
                -180.0,
                180.0,
            ),
            date = date,
            timeZone = timeZone,
        )
    }

    private fun calculateSunriseSunset(input: SunriseSunsetInput): SunriseSunsetResult {
        val sunrise = calculateSolarEvent(input, SOLAR_OFFICIAL_ZENITH, isSunrise = true)
        val sunset = calculateSolarEvent(input, SOLAR_OFFICIAL_ZENITH, isSunrise = false)
        val dawn = calculateSolarEvent(input, SOLAR_CIVIL_ZENITH, isSunrise = true)
        val dusk = calculateSolarEvent(input, SOLAR_CIVIL_ZENITH, isSunrise = false)
        val daylightMinutes = if (sunrise.minutes != null && sunset.minutes != null) {
            positiveMinuteDelta(sunrise.minutes, sunset.minutes)
        } else {
            null
        }
        val solarNoonMinutes = if (sunrise.minutes != null && daylightMinutes != null) {
            normalizeMinutes(sunrise.minutes + (daylightMinutes / 2))
        } else {
            null
        }
        val officialSunStates = listOf(sunrise.polarState, sunset.polarState)
        val sunState = when {
            officialSunStates.contains(SOLAR_POLAR_DAY) -> SOLAR_POLAR_DAY
            officialSunStates.contains(SOLAR_POLAR_NIGHT) -> SOLAR_POLAR_NIGHT
            isSolarDateToday(input) && sunrise.minutes != null && sunset.minutes != null ->
                if (isCurrentLocalTimeBetween(input.timeZone, sunrise.minutes, sunset.minutes)) "day" else "night"
            else -> "normal"
        }
        return SunriseSunsetResult(
            sunriseMinutes = sunrise.minutes,
            sunsetMinutes = sunset.minutes,
            dawnMinutes = dawn.minutes,
            duskMinutes = dusk.minutes,
            solarNoonMinutes = solarNoonMinutes,
            daylightMinutes = daylightMinutes,
            sunState = sunState,
        )
    }

    private fun calculateSolarEvent(input: SunriseSunsetInput, zenithDegrees: Double, isSunrise: Boolean): SolarEventResult {
        val longitudeHour = input.longitude / 15.0
        val baseHour = if (isSunrise) 6.0 else 18.0
        val approximateTime = input.date.dayOfYear + ((baseHour - longitudeHour) / 24.0)
        val meanAnomaly = (0.9856 * approximateTime) - 3.289
        val trueLongitude = normalizeDegrees(
            meanAnomaly +
                (1.916 * sin(Math.toRadians(meanAnomaly))) +
                (0.020 * sin(2.0 * Math.toRadians(meanAnomaly))) +
                282.634,
        )
        var rightAscension = normalizeDegrees(Math.toDegrees(kotlin.math.atan(0.91764 * tan(Math.toRadians(trueLongitude)))))
        val longitudeQuadrant = floor(trueLongitude / 90.0) * 90.0
        val ascensionQuadrant = floor(rightAscension / 90.0) * 90.0
        rightAscension = (rightAscension + (longitudeQuadrant - ascensionQuadrant)) / 15.0

        val sinDeclination = 0.39782 * sin(Math.toRadians(trueLongitude))
        val cosDeclination = cos(asin(sinDeclination))
        val latitudeRadians = Math.toRadians(input.latitude)
        val cosHourAngle = (
            cos(Math.toRadians(zenithDegrees)) -
                (sinDeclination * sin(latitudeRadians))
            ) / (cosDeclination * cos(latitudeRadians))
        if (cosHourAngle > 1.0) {
            return SolarEventResult(null, SOLAR_POLAR_NIGHT)
        }
        if (cosHourAngle < -1.0) {
            return SolarEventResult(null, SOLAR_POLAR_DAY)
        }

        var hourAngle = Math.toDegrees(acos(cosHourAngle))
        if (isSunrise) {
            hourAngle = 360.0 - hourAngle
        }
        val localMeanTime = (hourAngle / 15.0) + rightAscension - (0.06571 * approximateTime) - 6.622
        val universalTime = normalizeHours(localMeanTime - longitudeHour)
        val offsetMinutes = input.timeZone.getOffset(input.date.middayEpochMs) / 60_000.0
        return SolarEventResult(normalizeMinutes((universalTime * 60.0 + offsetMinutes).roundToInt()), null)
    }

    private fun sunriseSunsetResultJson(
        input: SunriseSunsetInput,
        result: SunriseSunsetResult,
        action: String,
    ): JSONObject {
        val json = JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", action)
            .put("date", input.date.text)
            .put("timezone", input.timeZone.id)
            .put("latitude", input.latitude)
            .put("longitude", input.longitude)
            .put("sun_state", result.sunState)
            .put("variables", sunriseSunsetVariablesJson(input, result))
            .put("message", sunriseSunsetMessage(result))
        putNullableSolarTime(json, "sunrise", result.sunriseMinutes)
        putNullableSolarTime(json, "sunset", result.sunsetMinutes)
        putNullableSolarTime(json, "dawn", result.dawnMinutes)
        putNullableSolarTime(json, "dusk", result.duskMinutes)
        putNullableSolarTime(json, "solar_noon", result.solarNoonMinutes)
        if (result.daylightMinutes == null) {
            json.put("daylight_minutes", JSONObject.NULL)
        } else {
            json.put("daylight_minutes", result.daylightMinutes)
        }
        return json
    }

    private fun setSunriseSunsetVariables(
        store: HermesAutomationStore,
        input: SunriseSunsetInput,
        result: SunriseSunsetResult,
    ) {
        sunriseSunsetVariableValues(input, result).forEach { (name, value) ->
            store.setVariable(name, value.take(MAX_EVENT_VALUE_CHARS))
        }
    }

    private fun sunriseSunsetVariablesJson(input: SunriseSunsetInput, result: SunriseSunsetResult): JSONObject {
        return JSONObject().apply {
            sunriseSunsetVariableValues(input, result).forEach { (name, value) -> put(name, value) }
        }
    }

    private fun sunriseSunsetVariableValues(
        input: SunriseSunsetInput,
        result: SunriseSunsetResult,
    ): Map<String, String> {
        val latitudeText = formatLocationNumber(input.latitude)
        val longitudeText = formatLocationNumber(input.longitude)
        val sunriseText = formatSolarClockTime(result.sunriseMinutes)
        val sunsetText = formatSolarClockTime(result.sunsetMinutes)
        val dawnText = formatSolarClockTime(result.dawnMinutes)
        val duskText = formatSolarClockTime(result.duskMinutes)
        val noonText = formatSolarClockTime(result.solarNoonMinutes)
        val daylightText = result.daylightMinutes?.toString().orEmpty()
        return linkedMapOf(
            "SUNRISE" to sunriseText,
            "SUNSET" to sunsetText,
            "SUN_DAWN" to dawnText,
            "SUN_DUSK" to duskText,
            "CIVIL_DAWN" to dawnText,
            "CIVIL_DUSK" to duskText,
            "SOLAR_NOON" to noonText,
            "SUN_DAYLIGHT_MINUTES" to daylightText,
            "DAYLIGHT_MINUTES" to daylightText,
            "SUN_STATE" to result.sunState,
            "SUN_DATE" to input.date.text,
            "SUN_TIMEZONE" to input.timeZone.id,
            "SUN_LAT" to latitudeText,
            "SUN_LON" to longitudeText,
            "SUN_LOCATION" to "$latitudeText,$longitudeText",
        )
    }

    private fun sunriseSunsetMessage(result: SunriseSunsetResult): String {
        return when (result.sunState) {
            SOLAR_POLAR_DAY -> "Sun does not set on this date at this location"
            SOLAR_POLAR_NIGHT -> "Sun does not rise on this date at this location"
            else -> "Sunrise ${formatSolarClockTime(result.sunriseMinutes)}, sunset ${formatSolarClockTime(result.sunsetMinutes)}"
        }
    }

    private fun putNullableSolarTime(json: JSONObject, key: String, minutes: Int?) {
        if (minutes == null) {
            json.put(key, JSONObject.NULL)
        } else {
            json.put(key, formatSolarClockTime(minutes))
        }
    }

    private fun rawArgumentText(arguments: JSONObject, vararg keys: String): String? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        val raw = arguments.opt(key) ?: return null
        return when (raw) {
            is Number -> raw.toDouble().toString()
            else -> raw.toString()
        }.trim().takeIf { it.isNotBlank() }
    }

    private fun expandedArgumentText(arguments: JSONObject, variables: JSONObject, vararg keys: String): String? {
        return rawArgumentText(arguments, *keys)
            ?.let { value -> expandVariables(value, variables).trim() }
            ?.takeIf { it.isNotBlank() }
    }

    private fun validateSolarCoordinateText(
        raw: String,
        label: String,
        min: Double,
        max: Double,
        allowVariables: Boolean,
    ) {
        rejectNul(raw, label)
        if (allowVariables && looksLikeVariableReference(raw)) {
            return
        }
        parseSolarCoordinate(raw, label, min, max)
    }

    private fun validateSolarDateText(raw: String, timeZone: TimeZone, allowVariables: Boolean) {
        rejectNul(raw, "date")
        if (allowVariables && looksLikeVariableReference(raw)) {
            return
        }
        parseSolarDate(raw, timeZone)
    }

    private fun validateSolarTimeZoneText(raw: String, allowVariables: Boolean) {
        rejectNul(raw, "timezone")
        if (allowVariables && looksLikeVariableReference(raw)) {
            return
        }
        parseSolarTimeZone(raw)
    }

    private fun parseSolarCoordinate(raw: String, label: String, min: Double, max: Double): Double {
        rejectNul(raw, label)
        val value = raw.trim().toDoubleOrNull()
            ?: throw IllegalArgumentException("sunrise/sunset $label must be a number")
        require(value.isFinite() && value in min..max) {
            "sunrise/sunset $label must be between ${formatLocationNumber(min)} and ${formatLocationNumber(max)}"
        }
        return value
    }

    private fun parseSolarTimeZone(raw: String?): TimeZone {
        val requested = raw?.trim().orEmpty()
        if (requested.indexOf('\u0000') >= 0) {
            throw IllegalArgumentException("timezone must not contain NUL bytes")
        }
        val zoneId = when (requested.lowercase(Locale.US)) {
            "", "local", "device", "system" -> TimeZone.getDefault().id
            "z", "utc" -> "UTC"
            else -> if (requested.startsWith("UTC+", ignoreCase = true) || requested.startsWith("UTC-", ignoreCase = true)) {
                "GMT${requested.substring(3)}"
            } else {
                requested
            }
        }
        if (zoneId !in AVAILABLE_TIME_ZONE_IDS && !GMT_OFFSET_ZONE_PATTERN.matches(zoneId)) {
            throw IllegalArgumentException("timezone must be an IANA id like Europe/London, UTC, local, or a GMT offset")
        }
        return TimeZone.getTimeZone(zoneId)
    }

    private fun parseSolarDate(raw: String?, timeZone: TimeZone): SolarDate {
        val value = raw?.trim().orEmpty().ifBlank { currentSolarDateText(timeZone) }
        rejectNul(value, "date")
        val match = SOLAR_DATE_PATTERN.matchEntire(value)
            ?: throw IllegalArgumentException("sunrise/sunset date must use YYYY-MM-DD")
        val year = match.groupValues[1].toInt()
        val month = match.groupValues[2].toInt()
        val day = match.groupValues[3].toInt()
        val calendar = Calendar.getInstance(timeZone).apply {
            isLenient = false
            clear()
            set(Calendar.YEAR, year)
            set(Calendar.MONTH, month - 1)
            set(Calendar.DAY_OF_MONTH, day)
            set(Calendar.HOUR_OF_DAY, 12)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val middayEpochMs = runCatching { calendar.timeInMillis }.getOrElse {
            throw IllegalArgumentException("sunrise/sunset date must be a valid calendar date")
        }
        return SolarDate(
            year = year,
            month = month,
            day = day,
            text = String.format(Locale.US, "%04d-%02d-%02d", year, month, day),
            dayOfYear = calendar.get(Calendar.DAY_OF_YEAR),
            middayEpochMs = middayEpochMs,
        )
    }

    private fun currentSolarDateText(timeZone: TimeZone): String {
        val calendar = Calendar.getInstance(timeZone)
        return String.format(
            Locale.US,
            "%04d-%02d-%02d",
            calendar.get(Calendar.YEAR),
            calendar.get(Calendar.MONTH) + 1,
            calendar.get(Calendar.DAY_OF_MONTH),
        )
    }

    private fun normalizeDegrees(value: Double): Double {
        val normalized = value % 360.0
        return if (normalized < 0.0) normalized + 360.0 else normalized
    }

    private fun normalizeHours(value: Double): Double {
        val normalized = value % 24.0
        return if (normalized < 0.0) normalized + 24.0 else normalized
    }

    private fun normalizeMinutes(value: Int): Int {
        val normalized = value % 1440
        return if (normalized < 0) normalized + 1440 else normalized
    }

    private fun positiveMinuteDelta(start: Int, end: Int): Int {
        val delta = normalizeMinutes(end - start)
        return if (delta == 0) 1440 else delta
    }

    private fun formatSolarClockTime(minutes: Int?): String {
        return minutes?.let { String.format(Locale.US, "%02d:%02d", it / 60, it % 60) }.orEmpty()
    }

    private fun isSolarDateToday(input: SunriseSunsetInput): Boolean {
        val calendar = Calendar.getInstance(input.timeZone)
        return calendar.get(Calendar.YEAR) == input.date.year &&
            calendar.get(Calendar.MONTH) + 1 == input.date.month &&
            calendar.get(Calendar.DAY_OF_MONTH) == input.date.day
    }

    private fun isCurrentLocalTimeBetween(timeZone: TimeZone, startMinutes: Int, endMinutes: Int): Boolean {
        val now = Calendar.getInstance(timeZone)
        val nowMinutes = now.get(Calendar.HOUR_OF_DAY) * 60 + now.get(Calendar.MINUTE)
        return if (startMinutes <= endMinutes) {
            nowMinutes in startMinutes until endMinutes
        } else {
            nowMinutes >= startMinutes || nowMinutes < endMinutes
        }
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

    private fun expandNotificationPayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        NOTIFICATION_STRING_PAYLOAD_KEYS.forEach { key ->
            if (payload.has(key) && !payload.isNull(key)) {
                expanded.put(key, expandVariables(payload.optString(key), variables))
            }
        }
        NOTIFICATION_BOOLEAN_PAYLOAD_KEYS.forEach { key ->
            if (payload.has(key) && !payload.isNull(key)) {
                expanded.put(key, payload.optBoolean(key))
            }
        }
        payload.optJSONArray("notification_buttons")?.let { buttons ->
            val expandedButtons = JSONArray()
            for (index in 0 until buttons.length()) {
                val raw = buttons.optJSONObject(index) ?: continue
                val button = JSONObject()
                listOf("title", "action", "automation_id").forEach { key ->
                    if (raw.has(key) && !raw.isNull(key)) {
                        button.put(key, expandVariables(raw.optString(key), variables))
                    }
                }
                if (raw.has("dismiss_on_tap") && !raw.isNull("dismiss_on_tap")) {
                    button.put("dismiss_on_tap", raw.optBoolean("dismiss_on_tap"))
                }
                expandedButtons.put(button)
            }
            expanded.put("notification_buttons", expandedButtons)
        }
        return expanded
    }

    private fun expandOverlayScenePayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        payload.keys().forEach { key ->
            when (val value = payload.opt(key)) {
                is String -> expanded.put(key, expandVariables(value, variables))
                else -> expanded.put(key, value)
            }
        }
        return HermesOverlaySceneBridge.payloadFromArguments(expanded)
    }

    private fun expandToastPayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        payload.keys().forEach { key ->
            when (val value = payload.opt(key)) {
                is String -> expanded.put(key, expandVariables(value, variables))
                else -> expanded.put(key, value)
            }
        }
        return toastPayloadFromArguments(expanded)
    }

    private fun waitDurationMsFromPayload(payload: JSONObject, variables: JSONObject): Long {
        val expanded = JSONObject()
        payload.keys().forEach { key ->
            when (val value = payload.opt(key)) {
                is String -> expanded.put(key, expandVariables(value, variables))
                else -> expanded.put(key, value)
            }
        }
        return waitDurationMsFromArguments(expanded)
    }

    private fun expandVibrationPayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        payload.keys().forEach { key ->
            when (val value = payload.opt(key)) {
                is String -> expanded.put(key, expandVariables(value, variables))
                is JSONArray -> {
                    val array = JSONArray()
                    for (index in 0 until value.length()) {
                        when (val entry = value.opt(index)) {
                            is String -> array.put(expandVariables(entry, variables))
                            else -> array.put(entry)
                        }
                    }
                    expanded.put(key, array)
                }
                else -> expanded.put(key, value)
            }
        }
        return vibrationPayloadFromArguments(expanded)
    }

    private fun expandAudioPayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        payload.keys().forEach { key ->
            when (val value = payload.opt(key)) {
                is String -> expanded.put(key, expandVariables(value, variables))
                else -> expanded.put(key, value)
            }
        }
        return audioPayloadFromArguments(expanded)
    }

    private fun expandHttpRequestPayload(payload: JSONObject, variables: JSONObject): JSONObject {
        val expanded = JSONObject()
        payload.keys().forEach { key ->
            when (val value = payload.opt(key)) {
                is String -> expanded.put(key, expandVariables(value, variables))
                is JSONObject -> {
                    val child = JSONObject()
                    value.keys().forEach { childKey ->
                        when (val childValue = value.opt(childKey)) {
                            is String -> child.put(childKey, expandVariables(childValue, variables))
                            else -> child.put(childKey, childValue)
                        }
                    }
                    expanded.put(key, child)
                }
                else -> expanded.put(key, value)
            }
        }
        return HermesHttpRequestBridge.payloadFromArguments(expanded, allowVariableUrl = false)
    }

    private fun audioPayloadFromArguments(arguments: JSONObject): JSONObject {
        val rawAction = stringArgument(
            arguments,
            "audio_action",
            "audioAction",
            "volume_action",
            "sound_action",
            "operation",
            "mode",
        ) ?: inferAudioAction(arguments)
            ?: throw IllegalArgumentException("create_audio_action_task requires audio_action")
        val audioAction = HermesAudioActionBridge.normalizeAudioAction(rawAction)
            ?: throw IllegalArgumentException("Unsupported audio action: $rawAction")
        val payload = JSONObject().put("audio_action", audioAction)
        when (audioAction) {
            "set_volume" -> {
                val stream = HermesAudioActionBridge.normalizeAudioStream(
                    stringArgument(arguments, "stream", "audio_stream", "volume_stream", "channel")
                        .orEmpty()
                        .ifBlank { "media" },
                ) ?: throw IllegalArgumentException("set_volume requires a supported audio stream")
                val level = audioLevelArgument(arguments, "level", "volume_level", "volume", "stream_volume")
                    ?: throw IllegalArgumentException("set_volume requires level")
                payload.put("stream", stream)
                    .put("level", level)
            }
            "set_ringer_mode" -> {
                val mode = HermesAudioActionBridge.normalizeRingerMode(
                    stringArgument(arguments, "ringer_mode", "sound_mode", "target_mode", "state").orEmpty(),
                ) ?: throw IllegalArgumentException("set_ringer_mode requires normal, vibrate, or silent")
                payload.put("ringer_mode", mode)
            }
            "set_microphone_mute",
            "set_speakerphone" -> {
                val enabled = booleanArgument(arguments, "target_enabled", "enabled", "state")
                    ?: throw IllegalArgumentException("$audioAction requires target_enabled")
                payload.put("target_enabled", enabled)
            }
        }
        return payload
    }

    private fun audioLevelArgument(arguments: JSONObject, vararg keys: String): Any? {
        keys.forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            return when (val raw = arguments.opt(key)) {
                is Number -> {
                    val value = raw.toLong()
                    require(value >= 0L) { "$key must be 0 or greater" }
                    require(value <= MAX_AUDIO_LEVEL) { "$key cannot exceed $MAX_AUDIO_LEVEL" }
                    value.toInt()
                }
                else -> {
                    val text = raw?.toString()?.trim().orEmpty()
                    val numeric = text.toLongOrNull()
                    if (numeric != null) {
                        require(numeric >= 0L) { "$key must be 0 or greater" }
                        require(numeric <= MAX_AUDIO_LEVEL) { "$key cannot exceed $MAX_AUDIO_LEVEL" }
                        numeric.toInt()
                    } else {
                        require(text.indexOf('\u0000') < 0) { "$key must not contain NUL bytes" }
                        require(text.contains('%') || text.contains("{{")) { "$key must be an integer or saved variable expression" }
                        text.take(MAX_AUDIO_LEVEL_TEXT_CHARS)
                    }
                }
            }
        }
        return null
    }

    private fun inferAudioAction(arguments: JSONObject): String? {
        return when {
            arguments.has("level") || arguments.has("volume_level") || arguments.has("volume") || arguments.has("stream_volume") -> "set_volume"
            arguments.has("ringer_mode") || arguments.has("sound_mode") || arguments.has("target_mode") -> "set_ringer_mode"
            else -> null
        }
    }

    private fun vibrationPayloadFromArguments(arguments: JSONObject): JSONObject {
        vibrationPatternMsFromArguments(arguments)?.let { pattern ->
            return if (pattern.size == 1) {
                JSONObject()
                    .put("vibration_action", VIBRATION_ACTION_VIBRATE)
                    .put("duration_ms", boundedVibrationDurationMs(pattern.first()))
            } else {
                JSONObject()
                    .put("vibration_action", VIBRATION_ACTION_VIBRATE)
                    .put("duration_ms", pattern.sum())
                    .put("pattern_ms", JSONArray(pattern))
            }
        }
        val durationMs = optionalNonNegativeLongArgument(
            arguments,
            "vibration_duration_ms",
            "duration_ms",
            "vibrate_ms",
            "milliseconds",
            "ms",
        ) ?: optionalNonNegativeLongArgument(arguments, "vibration_duration_seconds", "duration_seconds")
            ?.let { seconds -> scaleVibrationComponent(seconds, 1_000L, "seconds") }
            ?: throw IllegalArgumentException("create_vibration_task requires vibration_duration_ms or vibration_pattern_ms")
        return JSONObject()
            .put("vibration_action", VIBRATION_ACTION_VIBRATE)
            .put("duration_ms", boundedVibrationDurationMs(durationMs))
    }

    private fun toastPayloadFromArguments(arguments: JSONObject): JSONObject {
        val text = stringArgument(
            arguments,
            "toast_text",
            "flash_text",
            "message",
            "text",
            "content",
            "value",
            allowEmpty = true,
        ) ?: throw IllegalArgumentException("show_toast requires toast_text, flash_text, message, text, content, or value")
        require(text.indexOf('\u0000') < 0) { "toast text must not contain NUL bytes" }
        require(text.isNotBlank()) { "toast text must not be blank" }
        val long = booleanArgument(arguments, "toast_long", "flash_long", "long", "duration_long") ?: when (
            stringArgument(arguments, "toast_duration", "flash_duration", "duration").orEmpty().trim().lowercase()
        ) {
            "long", "longer", "1", "true" -> true
            else -> false
        }
        return JSONObject()
            .put("text", text.take(MAX_TOAST_TEXT_CHARS))
            .put("long", long)
    }

    private fun vibrationPatternMsFromArguments(arguments: JSONObject): List<Long>? {
        listOf("vibration_pattern_ms", "pattern_ms", "vibration_pattern", "pattern").forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            val values = when (val raw = arguments.opt(key)) {
                is JSONArray -> {
                    val parsed = mutableListOf<Long>()
                    for (index in 0 until raw.length()) {
                        val value = when (val item = raw.opt(index)) {
                            is Number -> item.toLong()
                            else -> item?.toString()?.trim()?.toLongOrNull()
                        } ?: throw IllegalArgumentException("$key[$index] must be an integer")
                        parsed += value
                    }
                    parsed
                }
                else -> raw?.toString()
                    ?.split(Regex("[,\\s]+"))
                    ?.filter { it.isNotBlank() }
                    ?.mapIndexed { index, entry ->
                        entry.toLongOrNull() ?: throw IllegalArgumentException("$key[$index] must be an integer")
                    }
                    .orEmpty()
            }
            return boundedVibrationPatternMs(values, key)
        }
        return null
    }

    private fun scaleVibrationComponent(value: Long, factor: Long, label: String): Long {
        require(value <= MAX_VIBRATION_TOTAL_MS / factor) { "$label exceeds the maximum vibration duration of ${MAX_VIBRATION_TOTAL_MS} ms" }
        return value * factor
    }

    private fun boundedVibrationDurationMs(durationMs: Long): Long {
        require(durationMs > 0L) { "vibration duration must be greater than 0 ms" }
        require(durationMs <= MAX_VIBRATION_TOTAL_MS) {
            "vibration duration cannot exceed ${MAX_VIBRATION_TOTAL_MS} ms"
        }
        return durationMs
    }

    private fun boundedVibrationPatternMs(values: List<Long>, label: String): List<Long> {
        require(values.isNotEmpty()) { "$label must contain at least one timing value" }
        require(values.size <= MAX_VIBRATION_PATTERN_ENTRIES) {
            "$label supports at most $MAX_VIBRATION_PATTERN_ENTRIES timing values"
        }
        values.forEachIndexed { index, value ->
            require(value >= 0L) { "$label[$index] must be 0 or greater" }
            require(value <= MAX_VIBRATION_TOTAL_MS) {
                "$label[$index] cannot exceed ${MAX_VIBRATION_TOTAL_MS} ms"
            }
        }
        require(values.any { it > 0L }) { "$label must contain at least one non-zero timing value" }
        val total = values.sum()
        require(total <= MAX_VIBRATION_TOTAL_MS) {
            "$label total duration cannot exceed ${MAX_VIBRATION_TOTAL_MS} ms"
        }
        return values
    }

    private fun waitDurationMsFromArguments(arguments: JSONObject): Long {
        optionalNonNegativeLongArgument(arguments, "duration_ms", "wait_ms", "milliseconds", "ms")?.let { durationMs ->
            return boundedWaitDurationMs(durationMs)
        }
        optionalNonNegativeLongArgument(arguments, "duration_seconds", "wait_seconds")?.let { seconds ->
            return boundedWaitDurationMs(scaleWaitComponent(seconds, 1_000L, "seconds"))
        }
        val milliseconds = optionalNonNegativeLongArgument(arguments, "millisecond_component", "milliseconds_component") ?: 0L
        val seconds = optionalNonNegativeLongArgument(arguments, "seconds", "second", "second_component", "seconds_component") ?: 0L
        val minutes = optionalNonNegativeLongArgument(arguments, "minutes", "minute", "wait_minutes") ?: 0L
        val hours = optionalNonNegativeLongArgument(arguments, "hours", "hour", "wait_hours") ?: 0L
        val days = optionalNonNegativeLongArgument(arguments, "days", "day", "wait_days") ?: 0L
        val total = listOf(
            scaleWaitComponent(milliseconds, 1L, "milliseconds"),
            scaleWaitComponent(seconds, 1_000L, "seconds"),
            scaleWaitComponent(minutes, 60_000L, "minutes"),
            scaleWaitComponent(hours, 3_600_000L, "hours"),
            scaleWaitComponent(days, 86_400_000L, "days"),
        ).sum()
        return boundedWaitDurationMs(total)
    }

    private fun optionalNonNegativeLongArgument(arguments: JSONObject, vararg keys: String): Long? {
        keys.forEach { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                return@forEach
            }
            val value = when (val raw = arguments.opt(key)) {
                is Number -> raw.toLong()
                else -> raw?.toString()?.trim()?.toLongOrNull()
            } ?: throw IllegalArgumentException("$key must be an integer")
            require(value >= 0L) { "$key must be 0 or greater" }
            return value
        }
        return null
    }

    private fun scaleWaitComponent(value: Long, factor: Long, label: String): Long {
        require(value <= MAX_WAIT_DURATION_MS / factor) { "$label exceeds the maximum wait duration of ${MAX_WAIT_DURATION_MS} ms" }
        return value * factor
    }

    private fun boundedWaitDurationMs(durationMs: Long): Long {
        require(durationMs > 0L) { "wait duration must be greater than 0 ms" }
        require(durationMs <= MAX_WAIT_DURATION_MS) { "wait duration cannot exceed ${MAX_WAIT_DURATION_MS} ms" }
        return durationMs
    }

    private fun recordsToJson(records: List<HermesAutomationRecord>): JSONArray {
        return JSONArray().apply {
            records.forEach { record -> put(record.toJson()) }
        }
    }

    private fun runEventsToJson(events: List<HermesAutomationRunEvent>): JSONArray {
        return JSONArray().apply {
            events.forEach { event -> put(event.toJson()) }
        }
    }

    private fun standbySummaryJson(context: Context, records: List<HermesAutomationRecord>, store: HermesAutomationStore): JSONObject {
        val enabledRecords = records.filter { it.enabled }
        val externalRecords = enabledRecords.filter { it.triggerType == TRIGGER_EXTERNAL }
        val remoteDispatchRecords = enabledRecords.filter { it.triggerType == TRIGGER_REMOTE_DISPATCH }
        val recentRuns = store.listRunEvents(5)
        val latestRun = recentRuns.firstOrNull()
        val latestDispatch = recentRuns.firstOrNull { event ->
            event.trigger == TRIGGER_REMOTE_DISPATCH || event.dispatchSource.isNotBlank()
        }
        val deviceId = runCatching {
            Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
        }.getOrNull().orEmpty().ifBlank { "unknown" }
        val deviceName = listOf(Build.MANUFACTURER, Build.MODEL)
            .map { it.orEmpty().trim() }
            .filter { it.isNotBlank() }
            .joinToString(" ")
            .ifBlank { Build.DEVICE.orEmpty().ifBlank { "Android device" } }
        return JSONObject()
            .put("ready", enabledRecords.isNotEmpty())
            .put("device_id", deviceId)
            .put("device_name", deviceName)
            .put("standby_namespace", "/standby")
            .put("standby_register_event", "standby:register")
            .put("standby_heartbeat_event", "standby:heartbeat")
            .put("standby_dispatch_event", "standby:dispatch")
            .put("heartbeat_interval_seconds", 30)
            .put("automation_count", records.size)
            .put("enabled_automation_count", enabledRecords.size)
            .put("external_trigger_count", externalRecords.size)
            .put("remote_dispatch_count", remoteDispatchRecords.size)
            .put("recent_run_count", recentRuns.size)
            .put("last_run_epoch_ms", latestRun?.finishedAtEpochMs ?: JSONObject.NULL)
            .put("last_run_success", latestRun?.success ?: JSONObject.NULL)
            .put("last_run_label", latestRun?.automationLabel.orEmpty())
            .put("last_run_trigger", latestRun?.trigger.orEmpty())
            .put("last_run_result", latestRun?.result.orEmpty())
            .put("last_dispatch_epoch_ms", latestDispatch?.finishedAtEpochMs ?: JSONObject.NULL)
            .put("last_dispatch_source", latestDispatch?.dispatchSource.orEmpty())
            .put("last_dispatch_channel", latestDispatch?.dispatchChannel.orEmpty())
            .put("last_dispatch_execution_id", latestDispatch?.remoteExecutionId.orEmpty())
            .put("last_dispatch_task_id", latestDispatch?.remoteTaskId.orEmpty())
            .put("last_dispatch_task_name", latestDispatch?.remoteTaskName.orEmpty())
            .put(
                "supported_dispatch_channels",
                JSONArray(
                    listOf(
                        "manual",
                        "external_broadcast",
                        "OpenGUI standby payload",
                        "feishu",
                        "telegram",
                        "discord",
                        "rest",
                        "quick_settings_tile",
                        "home_screen_widget",
                        "launcher_shortcut",
                        "Tasker plugin",
                    )
                )
            )
            .put(
                "compatible_dispatch_payloads",
                JSONArray(
                    listOf(
                        "OpenGUI standby:dispatch {executionId, taskId, taskName}",
                        "OpenGUI /status [executionId]",
                        "Hermes android_automation_tool run_remote_dispatch",
                        "Hermes android_automation_tool operator_execution_status",
                        "token-protected Hermes external broadcast",
                    )
                )
            )
    }

    private fun automationBundleFromArguments(arguments: JSONObject): JSONObject? {
        jsonObjectArgument(arguments, "bundle")?.let { return it }
        jsonObjectArgument(arguments, "bundle_json")?.let { return it }
        jsonObjectArgument(arguments, "json")?.let { return it }
        jsonObjectArgument(arguments, "payload")?.let { return it }
        if (arguments.has("automations") || arguments.has("records") || arguments.has("variables")) {
            return JSONObject()
                .put("automations", arguments.optJSONArray("automations") ?: arguments.optJSONArray("records") ?: JSONArray())
                .put("variables", arguments.optJSONObject("variables") ?: JSONObject())
        }
        return null
    }

    private fun jsonObjectArgument(arguments: JSONObject, key: String): JSONObject? {
        if (!arguments.has(key) || arguments.isNull(key)) {
            return null
        }
        return when (val value = arguments.opt(key)) {
            is JSONObject -> value
            is String -> JSONObject(value)
            else -> null
        }
    }

    private fun variablesFromBundle(bundle: JSONObject): JSONObject {
        val source = bundle.optJSONObject("variables") ?: return JSONObject()
        return JSONObject().apply {
            source.keys().forEach { key ->
                val normalized = HermesAutomationStore.normalizeVariableName(key) ?: return@forEach
                if (!source.isNull(key)) {
                    put(normalized, source.optString(key).take(MAX_VARIABLE_VALUE_CHARS))
                }
            }
        }
    }

    private fun sanitizeImportedRecord(json: JSONObject, now: Long, enabledOverride: Boolean?): HermesAutomationRecord {
        val id = json.optString("id").trim().ifBlank {
            "auto_${UUID.randomUUID().toString().replace("-", "").take(16)}"
        }
        rejectNul(id, "automation id")
        require(id.length <= MAX_AUTOMATION_ID_CHARS) { "automation id must be $MAX_AUTOMATION_ID_CHARS characters or shorter" }
        val actionType = normalizeActionType(
            json.optString("action_type")
                .ifBlank { json.optString("type") }
                .ifBlank { ACTION_TYPE_SHELL },
        )
        require(actionType in AUTOMATION_ACTION_TYPES) {
            "unsupported action_type $actionType; use one of ${AUTOMATION_ACTION_TYPES.joinToString()}"
        }
        val command = json.optString("command").ifBlank { json.optString("payload") }
        require(command.isNotBlank()) { "command must not be blank" }
        rejectNul(command, "automation command")
        val triggerType = normalizeTrigger(
            json.optString("trigger_type")
                .ifBlank { json.optString("trigger") }
                .ifBlank { TRIGGER_MANUAL },
        ) ?: throw IllegalArgumentException("unsupported trigger_type ${json.optString("trigger_type")}")
        val triggerPackageName = json.optString("trigger_package_name").trim()
        rejectNul(triggerPackageName, "trigger_package_name")
        require(triggerType != TRIGGER_APP_FOREGROUND || triggerPackageName.isNotBlank()) {
            "app_foreground trigger requires trigger_package_name"
        }
        require(triggerType != TRIGGER_NOTIFICATION_POSTED || triggerPackageName.isNotBlank()) {
            "notification_posted trigger requires trigger_package_name"
        }
        val intervalMinutes = importedOptionalInt(json, "interval_minutes", minValue = 1)
        require(triggerType != TRIGGER_INTERVAL || intervalMinutes != null) { "interval trigger requires interval_minutes" }
        if (intervalMinutes != null) {
            require(intervalMinutes >= HermesAutomationScheduler.MIN_INTERVAL_MINUTES) {
                "interval_minutes must be at least ${HermesAutomationScheduler.MIN_INTERVAL_MINUTES}"
            }
        }
        val triggerTimeMinutes = importedOptionalInt(json, "trigger_time_minutes", minValue = 0)
        if (triggerTimeMinutes != null) {
            require(triggerTimeMinutes in 0..1439) { "trigger_time_minutes must be between 0 and 1439" }
        }
        require(triggerType != TRIGGER_TIME || triggerTimeMinutes != null) { "time trigger requires trigger_time_minutes" }
        val triggerDaysOfWeek = json.optString("trigger_days_of_week").trim().uppercase()
        rejectNul(triggerDaysOfWeek, "trigger_days_of_week")
        validateImportedDaysOfWeek(triggerDaysOfWeek)
        val triggerData = importedTriggerData(json)
        if (triggerType == TRIGGER_EXTERNAL) {
            validateExternalTriggerData(triggerData)?.let { error -> throw IllegalArgumentException(error) }
        }
        if (triggerType == TRIGGER_LOGCAT_ENTRY) {
            validateLogcatTriggerData(triggerData, triggerPackageName)?.let { error -> throw IllegalArgumentException(error) }
        }
        val createdAt = json.optLong("created_at_epoch_ms", now).takeIf { it > 0L } ?: now
        return HermesAutomationRecord(
            id = id,
            label = json.optString("label").ifBlank { "Imported Hermes automation" }.take(80),
            actionType = actionType,
            command = command,
            useShizuku = json.optBoolean("use_shizuku", actionType == ACTION_TYPE_SHIZUKU_ACTION),
            triggerType = triggerType,
            triggerPackageName = triggerPackageName,
            triggerTimeMinutes = triggerTimeMinutes.takeIf { triggerType == TRIGGER_TIME },
            triggerDaysOfWeek = triggerDaysOfWeek.takeIf { triggerType == TRIGGER_TIME }.orEmpty(),
            intervalMinutes = intervalMinutes.takeIf { triggerType == TRIGGER_INTERVAL },
            enabled = enabledOverride ?: recordEnabled(json),
            createdAtEpochMs = createdAt,
            updatedAtEpochMs = now,
            lastRunEpochMs = null,
            lastExitCode = null,
            lastSuccess = null,
            lastResult = "",
            triggerData = triggerData,
        )
    }

    private fun importedOptionalInt(json: JSONObject, key: String, minValue: Int): Int? {
        if (!json.has(key) || json.isNull(key)) {
            return null
        }
        val value = when (val raw = json.opt(key)) {
            is Number -> raw.toInt()
            else -> raw?.toString()?.trim()?.toIntOrNull()
        } ?: throw IllegalArgumentException("$key must be an integer")
        require(value >= minValue) { "$key must be at least $minValue" }
        return value
    }

    private fun importedTriggerData(json: JSONObject): String {
        if (!json.has("trigger_data") || json.isNull("trigger_data")) {
            return ""
        }
        val triggerData = when (val raw = json.opt("trigger_data")) {
            is JSONObject -> raw.toString()
            is String -> raw.trim()
            else -> raw?.toString()?.trim().orEmpty()
        }
        rejectNul(triggerData, "trigger_data")
        if (triggerData.isNotBlank()) {
            JSONObject(triggerData)
        }
        return triggerData
    }

    private fun validateImportedDaysOfWeek(daysCsv: String) {
        if (daysCsv.isBlank()) {
            return
        }
        daysCsv.split(',').forEach { rawDay ->
            val day = rawDay.trim()
            require(day in DAY_ORDER) { "trigger_days_of_week contains unsupported day: $day" }
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

    private fun triggerPackageMatches(
        savedPackageName: String,
        foregroundPackageName: String,
        variables: JSONObject,
        packageCandidates: String = "",
    ): Boolean {
        val expanded = expandVariables(savedPackageName, variables).trim()
        return expanded.isNotBlank() &&
            packageCandidateList(foregroundPackageName, packageCandidates).any { packageName ->
                expanded.equals(packageName, ignoreCase = true)
            }
    }

    private fun packageCandidateList(packageName: String, packageCandidates: String): List<String> {
        return sequenceOf(packageName, packageCandidates)
            .flatMap { value -> value.splitToSequence(',', ';', '|', '\n', '\t') }
            .map { value -> value.trim() }
            .filter { value -> value.isNotBlank() && value.indexOf('\u0000') < 0 }
            .distinct()
            .toList()
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

    private fun sensorEventMatches(
        triggerData: String,
        variables: JSONObject,
        sensorType: String,
        sensorEvent: String,
        valueName: String,
        value: Double?,
    ): Boolean {
        val filters = runCatching { JSONObject(triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
        if (!textFilterMatches(filters.optString("sensor_type"), sensorType, variables)) {
            return false
        }
        if (!textFilterMatches(filters.optString("sensor_event"), sensorEvent, variables)) {
            return false
        }
        if (!textFilterMatches(filters.optString("value_name"), valueName, variables)) {
            return false
        }
        val minValue = expandedDoubleFilter(filters, "min_value", variables)
        val maxValue = expandedDoubleFilter(filters, "max_value", variables)
        if ((minValue != null || maxValue != null) && value == null) {
            return false
        }
        if (minValue != null && value != null && value < minValue) {
            return false
        }
        if (maxValue != null && value != null && value > maxValue) {
            return false
        }
        return true
    }

    private fun logcatEntryMatches(
        record: HermesAutomationRecord,
        variables: JSONObject,
        tag: String,
        message: String,
        level: String,
        pid: String,
        packageName: String,
        packageCandidates: String,
    ): Boolean {
        val filters = runCatching { JSONObject(record.triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
        if (
            record.triggerPackageName.isNotBlank() &&
            !triggerPackageMatches(record.triggerPackageName, packageName, variables, packageCandidates)
        ) {
            return false
        }
        if (!textFilterMatches(filters.optString("tag"), tag, variables)) {
            return false
        }
        if (!textFilterMatches(filters.optString("message_contains"), message, variables)) {
            return false
        }
        val savedPackageFilter = filters.optString("package_name")
        if (
            !textFilterMatches(savedPackageFilter, packageName, variables) &&
            !textFilterMatches(savedPackageFilter, packageCandidates, variables)
        ) {
            return false
        }
        if (!logcatLevelMatches(filters.optString("level"), level, variables)) {
            return false
        }
        return exactFilterMatches(filters.optString("pid"), pid, variables)
    }

    private fun logcatLevelMatches(filter: String, level: String, variables: JSONObject): Boolean {
        val expanded = expandVariables(filter, variables).trim()
        if (expanded.isBlank() || expanded == "*") {
            return true
        }
        if (expanded.indexOf('\u0000') >= 0) {
            return false
        }
        val normalizedFilter = normalizeLogcatLevel(expanded)
        val normalizedLevel = normalizeLogcatLevel(level)
        return normalizedFilter.isNotBlank() &&
            normalizedLevel.isNotBlank() &&
            normalizedFilter == normalizedLevel
    }

    private fun exactFilterMatches(filter: String, value: String, variables: JSONObject): Boolean {
        val expanded = expandVariables(filter, variables).trim()
        if (expanded.isBlank() || expanded == "*") {
            return true
        }
        if (expanded.indexOf('\u0000') >= 0) {
            return false
        }
        return expanded.equals(value.trim(), ignoreCase = true)
    }

    private fun normalizeLogcatLevel(level: String): String {
        return when (level.trim().lowercase(Locale.US)) {
            "v", "verbose" -> "v"
            "d", "debug" -> "d"
            "i", "info", "information" -> "i"
            "w", "warn", "warning" -> "w"
            "e", "error" -> "e"
            "a", "assert", "wtf" -> "a"
            "f", "fatal" -> "f"
            else -> ""
        }
    }

    private fun externalTriggerMatches(
        record: HermesAutomationRecord,
        variables: JSONObject,
        triggerId: String,
        externalToken: String,
        packageName: String,
        referrer: String,
    ): Boolean {
        val filters = runCatching { JSONObject(record.triggerData.ifBlank { "{}" }) }.getOrDefault(JSONObject())
        val savedTriggerId = expandVariables(filters.optString("trigger_id"), variables).trim()
        val savedToken = expandVariables(
            filters.optString("external_token").ifBlank { filters.optString("token") },
            variables,
        ).trim()
        if (savedTriggerId.isBlank() || savedToken.isBlank()) {
            return false
        }
        if (savedTriggerId.indexOf('\u0000') >= 0 || savedToken.indexOf('\u0000') >= 0) {
            return false
        }
        if (!savedTriggerId.equals(triggerId, ignoreCase = true)) {
            return false
        }
        if (!constantTimeEquals(savedToken, externalToken)) {
            return false
        }
        if (record.triggerPackageName.isNotBlank() && !triggerPackageMatches(record.triggerPackageName, packageName, variables)) {
            return false
        }
        return textFilterMatches(filters.optString("referrer_contains"), referrer, variables)
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

    private fun sensorEventNulError(
        sensorType: String,
        sensorEvent: String,
        valueName: String,
        unit: String,
        accuracy: String,
    ): String? {
        return when {
            sensorType.indexOf('\u0000') >= 0 -> "sensor_type must not contain NUL bytes"
            sensorEvent.indexOf('\u0000') >= 0 -> "sensor_event must not contain NUL bytes"
            valueName.indexOf('\u0000') >= 0 -> "sensor_value_name must not contain NUL bytes"
            unit.indexOf('\u0000') >= 0 -> "sensor_unit must not contain NUL bytes"
            accuracy.indexOf('\u0000') >= 0 -> "sensor_accuracy must not contain NUL bytes"
            else -> null
        }
    }

    private fun logcatEventNulError(
        tag: String,
        message: String,
        level: String,
        pid: String,
        packageName: String,
        packageCandidates: String,
        packageSource: String,
        timestamp: String,
    ): String? {
        return when {
            tag.indexOf('\u0000') >= 0 -> "logcat_tag must not contain NUL bytes"
            message.indexOf('\u0000') >= 0 -> "logcat_message must not contain NUL bytes"
            level.indexOf('\u0000') >= 0 -> "logcat_level must not contain NUL bytes"
            pid.indexOf('\u0000') >= 0 -> "logcat_pid must not contain NUL bytes"
            packageName.indexOf('\u0000') >= 0 -> "logcat_package_name must not contain NUL bytes"
            packageCandidates.indexOf('\u0000') >= 0 -> "logcat_package_candidates must not contain NUL bytes"
            packageSource.indexOf('\u0000') >= 0 -> "logcat_package_source must not contain NUL bytes"
            timestamp.indexOf('\u0000') >= 0 -> "logcat_timestamp must not contain NUL bytes"
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

    private fun validateExternalTriggerData(triggerData: String): String? {
        val payload = runCatching { JSONObject(triggerData.ifBlank { "{}" }) }.getOrElse {
            return "external_trigger trigger_data must be a JSON object"
        }
        val triggerId = payload.optString("trigger_id").trim()
        val token = payload.optString("external_token").ifBlank { payload.optString("token") }.trim()
        return when {
            triggerId.isBlank() -> "external_trigger trigger_data requires trigger_id"
            token.isBlank() -> "external_trigger trigger_data requires external_token"
            triggerId.indexOf('\u0000') >= 0 -> "external_trigger trigger_id must not contain NUL bytes"
            token.indexOf('\u0000') >= 0 -> "external_trigger external_token must not contain NUL bytes"
            payload.optString("referrer_contains").indexOf('\u0000') >= 0 -> "external_trigger referrer_contains must not contain NUL bytes"
            else -> null
        }
    }

    private fun externalExtrasText(arguments: JSONObject): String {
        val raw = listOf("extras", "trigger_extras", "event_extras", "data")
            .firstNotNullOfOrNull { key ->
                if (arguments.has(key) && !arguments.isNull(key)) arguments.opt(key) else null
            } ?: return ""
        return when (raw) {
            is JSONObject -> raw.toString()
            is JSONArray -> raw.toString()
            else -> raw.toString()
        }.take(MAX_VARIABLE_VALUE_CHARS)
    }

    private fun externalEventNulError(
        triggerId: String,
        externalToken: String,
        packageName: String,
        referrer: String,
        extras: String,
    ): String? {
        return when {
            triggerId.indexOf('\u0000') >= 0 -> "external_trigger trigger_id must not contain NUL bytes"
            externalToken.indexOf('\u0000') >= 0 -> "external_trigger external_token must not contain NUL bytes"
            packageName.indexOf('\u0000') >= 0 -> "external_trigger package name must not contain NUL bytes"
            referrer.indexOf('\u0000') >= 0 -> "external_trigger referrer must not contain NUL bytes"
            extras.indexOf('\u0000') >= 0 -> "external_trigger extras must not contain NUL bytes"
            else -> null
        }
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

    private fun setSensorEventVariables(
        store: HermesAutomationStore,
        sensorType: String,
        sensorEvent: String,
        valueName: String,
        value: Double?,
        unit: String,
        accuracy: String,
    ) {
        val valueText = value?.let { formatLocationNumber(it) }.orEmpty()
        store.setVariable("SENSOR", sensorType.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SENSOR_TYPE", sensorType.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SENSOR_NAME", sensorType.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SENSOR_EVENT", sensorEvent.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SENSOR_VALUE", valueText)
        store.setVariable("SENSOR_VALUE_NAME", valueName.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SENSOR_UNIT", unit.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SENSOR_ACCURACY", accuracy.take(MAX_EVENT_VALUE_CHARS))
    }

    private fun setLogcatEventVariables(
        store: HermesAutomationStore,
        tag: String,
        message: String,
        level: String,
        pid: String,
        packageName: String,
        packageCandidates: String,
        packageSource: String,
        timestamp: String,
    ) {
        store.setVariable("LOGCAT_TAG", tag.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOGCAT_MESSAGE", message.take(MAX_VARIABLE_VALUE_CHARS))
        store.setVariable("LOGCAT_LEVEL", level.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOGCAT_PID", pid.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOGCAT_PACKAGE", packageName.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOGCAT_PACKAGE_CANDIDATES", packageCandidates.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOGCAT_PACKAGE_SOURCE", packageSource.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOGCAT_TIME", timestamp.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOG_TAG", tag.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOG_MESSAGE", message.take(MAX_VARIABLE_VALUE_CHARS))
        store.setVariable("LOG_LEVEL", level.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOG_PACKAGE_CANDIDATES", packageCandidates.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("LOG_PACKAGE_SOURCE", packageSource.take(MAX_EVENT_VALUE_CHARS))
    }

    private fun setExternalTriggerVariables(
        store: HermesAutomationStore,
        triggerId: String,
        packageName: String,
        referrer: String,
        extras: String,
    ) {
        store.setVariable("SA_TRIGGER_ID", triggerId.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SA_TRIGGER_PACKAGE_NAME", packageName.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SA_REFERRER", referrer.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("SA_EXTRAS", extras.take(MAX_VARIABLE_VALUE_CHARS))
        store.setVariable("EXTERNAL_TRIGGER_ID", triggerId.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("EXTERNAL_TRIGGER_PACKAGE", packageName.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("EXTERNAL_TRIGGER_REFERRER", referrer.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("EXTERNAL_TRIGGER_EXTRAS", extras.take(MAX_VARIABLE_VALUE_CHARS))
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

    private fun normalizeActionType(actionType: String): String {
        val normalized = actionType.trim().lowercase().replace("-", "_").replace(" ", "_")
        return ACTION_TYPE_SYNONYMS[normalized] ?: normalized
    }

    private fun rejectNul(value: String, label: String) {
        require(value.indexOf('\u0000') < 0) { "$label must not contain NUL bytes" }
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

    private fun normalizeVariableAction(action: String): String? {
        return when (action.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "set", "variable_set", "set_variable", "assign" -> VARIABLE_ACTION_SET
            "clear", "delete", "remove", "variable_clear", "variable_delete", "delete_variable" -> VARIABLE_ACTION_CLEAR
            "append", "concat", "concatenate", "variable_append", "append_variable" -> VARIABLE_ACTION_APPEND
            "add", "increment", "plus", "variable_add" -> VARIABLE_ACTION_ADD
            "subtract", "sub", "decrement", "minus", "variable_subtract" -> VARIABLE_ACTION_SUBTRACT
            "replace", "search_replace", "search_and_replace", "variable_replace" -> VARIABLE_ACTION_REPLACE
            else -> null
        }
    }

    private fun formatVariableNumber(value: Double): String {
        return if (value.isFinite() && value % 1.0 == 0.0) {
            value.toLong().toString()
        } else {
            value.toString()
        }
    }

    private fun normalizeClipboardAction(action: String): String? {
        return when (action.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "set", "copy", "write", "set_clipboard", "clipboard_set" -> CLIPBOARD_ACTION_SET
            else -> null
        }
    }

    private enum class OperatorCommandType(val wireName: String) {
        HELP("help"),
        LIST_TASKS("list_tasks"),
        RUN_TASK("run_task"),
        DO_TASK("do_task"),
        STATUS("status"),
        CANCEL("cancel"),
        PAUSE("pause"),
        RESUME("resume"),
        DEVICES("devices"),
        FREE_TEXT("free_text"),
    }

    private data class ParsedOperatorCommand(
        val type: OperatorCommandType,
        val rawText: String,
        val strippedText: String,
        val taskId: String? = null,
        val executionId: String? = null,
        val description: String? = null,
        val feedback: String? = null,
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("type", type.wireName)
                .put("raw_text", rawText)
                .put("stripped_text", strippedText)
                .put("task_id", taskId ?: JSONObject.NULL)
                .put("execution_id", executionId ?: JSONObject.NULL)
                .put("description", description ?: JSONObject.NULL)
                .put("feedback", feedback ?: JSONObject.NULL)
        }
    }

    private data class OperatorCommandAccess(
        val allowed: Boolean,
        val reason: String,
        val guildId: String,
        val channelId: String,
        val userId: String,
        val guildAllowListSize: Int,
        val channelAllowListSize: Int,
        val userAllowListSize: Int,
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("allowed", allowed)
                .put("reason", reason)
                .put("guild_id", guildId.ifBlank { JSONObject.NULL })
                .put("channel_id", channelId.ifBlank { JSONObject.NULL })
                .put("user_id", userId.ifBlank { JSONObject.NULL })
                .put("guild_allowlist_size", guildAllowListSize)
                .put("channel_allowlist_size", channelAllowListSize)
                .put("user_allowlist_size", userAllowListSize)
        }
    }

    private fun parseOperatorCommand(text: String, prefix: String): ParsedOperatorCommand {
        val rawText = text.trim()
        val stripped = stripOperatorCommandPrefix(rawText, prefix)
        if (Regex("^/?(?:tasks?|list)\\b", RegexOption.IGNORE_CASE).containsMatchIn(stripped)) {
            return ParsedOperatorCommand(OperatorCommandType.LIST_TASKS, rawText, stripped)
        }
        Regex("^/?run\\s+(\\S+)", RegexOption.IGNORE_CASE).find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.RUN_TASK,
                rawText = rawText,
                strippedText = stripped,
                taskId = match.groupValues[1].trim(),
            )
        }
        Regex("^/?do\\s+(.+)", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)).find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.DO_TASK,
                rawText = rawText,
                strippedText = stripped,
                description = match.groupValues[1].trim(),
            )
        }
        Regex("^/?status(?:\\s+(\\S+))?\\b", RegexOption.IGNORE_CASE).find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.STATUS,
                rawText = rawText,
                strippedText = stripped,
                executionId = match.groupValues.getOrNull(1)?.trim()?.ifBlank { null },
            )
        }
        Regex("^/?cancel(?:\\s+(\\S+))?\\b", RegexOption.IGNORE_CASE).find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.CANCEL,
                rawText = rawText,
                strippedText = stripped,
                executionId = match.groupValues.getOrNull(1)?.trim()?.ifBlank { null },
            )
        }
        Regex("^/?pause(?:\\s+(\\S+))?\\b", RegexOption.IGNORE_CASE).find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.PAUSE,
                rawText = rawText,
                strippedText = stripped,
                executionId = match.groupValues.getOrNull(1)?.trim()?.ifBlank { null },
            )
        }
        Regex("^/?resume(?:\\s+(.+))?$", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)).find(stripped)?.let { match ->
            val args = match.groupValues.getOrNull(1).orEmpty().trim()
            val idAndFeedback = Regex("^(\\S+)(?:\\s+([\\s\\S]+))?$").find(args)
            return ParsedOperatorCommand(
                type = OperatorCommandType.RESUME,
                rawText = rawText,
                strippedText = stripped,
                executionId = idAndFeedback?.groupValues?.getOrNull(1)?.trim()?.ifBlank { null },
                feedback = idAndFeedback?.groupValues?.getOrNull(2)?.trim()?.ifBlank { null } ?: args.ifBlank { null },
            )
        }
        if (Regex("^/?devices?\\b", RegexOption.IGNORE_CASE).containsMatchIn(stripped)) {
            return ParsedOperatorCommand(OperatorCommandType.DEVICES, rawText, stripped)
        }
        if (Regex("^/?help\\b", RegexOption.IGNORE_CASE).containsMatchIn(stripped)) {
            return ParsedOperatorCommand(OperatorCommandType.HELP, rawText, stripped)
        }
        if (Regex("^(?:task\\s+list|\u4efb\u52a1\u5217\u8868)$", RegexOption.IGNORE_CASE).matches(stripped)) {
            return ParsedOperatorCommand(OperatorCommandType.LIST_TASKS, rawText, stripped)
        }
        Regex("^(?:\u6267\u884c|\u8fd0\u884c)\\s*[:\uFF1A]?\\s*(\\S+)").find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.RUN_TASK,
                rawText = rawText,
                strippedText = stripped,
                taskId = match.groupValues[1].trim(),
            )
        }
        Regex("^\u505a\\s*[:\uFF1A]?\\s*(.+)", RegexOption.DOT_MATCHES_ALL).find(stripped)?.let { match ->
            return ParsedOperatorCommand(
                type = OperatorCommandType.DO_TASK,
                rawText = rawText,
                strippedText = stripped,
                description = match.groupValues[1].trim(),
            )
        }
        return when (stripped) {
            "\u72b6\u6001" -> ParsedOperatorCommand(OperatorCommandType.STATUS, rawText, stripped)
            "\u53d6\u6d88" -> ParsedOperatorCommand(OperatorCommandType.CANCEL, rawText, stripped)
            "\u6682\u505c" -> ParsedOperatorCommand(OperatorCommandType.PAUSE, rawText, stripped)
            "\u6062\u590d" -> ParsedOperatorCommand(OperatorCommandType.RESUME, rawText, stripped)
            "\u5e2e\u52a9" -> ParsedOperatorCommand(OperatorCommandType.HELP, rawText, stripped)
            else -> ParsedOperatorCommand(OperatorCommandType.FREE_TEXT, rawText, stripped)
        }
    }

    private fun stripOperatorCommandPrefix(text: String, prefix: String): String {
        val trimmed = text.trim()
        val safePrefix = prefix.trim()
        val prefixes = listOf(safePrefix, "/opengui", "opengui")
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .distinct()
        val lowerText = trimmed.lowercase(Locale.US)
        for (candidate in prefixes) {
            val lowerPrefix = candidate.lowercase(Locale.US)
            if (lowerText == lowerPrefix || lowerText.startsWith("$lowerPrefix ")) {
                return trimmed.substring(candidate.length).trim().ifBlank { "help" }
            }
        }
        return trimmed
    }

    private fun operatorCommandAccess(arguments: JSONObject): OperatorCommandAccess {
        val guildId = stringArgument(
            arguments,
            "guild_id",
            "guildId",
            "discord_guild_id",
            "server_id",
            "serverId",
            allowEmpty = true,
        ).orEmpty().trim()
        val channelId = stringArgument(
            arguments,
            "channel_id",
            "channelId",
            "conversation_id",
            "conversationId",
            "discord_channel_id",
            "chat_id",
            "chatId",
            allowEmpty = true,
        ).orEmpty().trim()
        val userId = stringArgument(
            arguments,
            "user_id",
            "userId",
            "platform_user_id",
            "platformUserId",
            "discord_user_id",
            "open_id",
            "openId",
            allowEmpty = true,
        ).orEmpty().trim()
        val allowedGuildIds = stringSetArgument(arguments, "allowed_guild_ids", "allowedGuildIds", "discord_allowed_guild_ids")
        val allowedChannelIds = stringSetArgument(arguments, "allowed_channel_ids", "allowedChannelIds", "discord_allowed_channel_ids")
        val allowedUserIds = stringSetArgument(arguments, "allowed_user_ids", "allowedUserIds", "discord_allowed_user_ids")
        val failures = mutableListOf<String>()
        if (!matchesAllowList(allowedGuildIds, guildId)) {
            failures += "guild_id is not allowlisted"
        }
        if (!matchesAllowList(allowedChannelIds, channelId)) {
            failures += "channel_id is not allowlisted"
        }
        if (!matchesAllowList(allowedUserIds, userId)) {
            failures += "user_id is not allowlisted"
        }
        return OperatorCommandAccess(
            allowed = failures.isEmpty(),
            reason = failures.joinToString("; "),
            guildId = guildId.take(MAX_EVENT_VALUE_CHARS),
            channelId = channelId.take(MAX_EVENT_VALUE_CHARS),
            userId = userId.take(MAX_EVENT_VALUE_CHARS),
            guildAllowListSize = allowedGuildIds.size,
            channelAllowListSize = allowedChannelIds.size,
            userAllowListSize = allowedUserIds.size,
        )
    }

    private fun matchesAllowList(allowList: Set<String>, value: String): Boolean {
        return allowList.isEmpty() || (value.isNotBlank() && value in allowList)
    }

    private fun stringSetArgument(arguments: JSONObject, vararg keys: String): Set<String> {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return emptySet()
        return when (val value = arguments.opt(key)) {
            is JSONArray -> buildSet {
                for (index in 0 until value.length()) {
                    value.optString(index).trim().take(MAX_EVENT_VALUE_CHARS).takeIf { it.isNotBlank() }?.let { add(it) }
                }
            }
            else -> value?.toString().orEmpty()
                .split(',', ';', '\n', '\t', ' ')
                .map { it.trim().take(MAX_EVENT_VALUE_CHARS) }
                .filter { it.isNotBlank() }
                .toSet()
        }
    }

    private fun operatorCommandHelpJson(parsed: ParsedOperatorCommand): String {
        return JSONObject()
            .put("success", true)
            .put("handled", true)
            .put("parsed_command", parsed.toJson())
            .put("reply_lines", JSONArray(OPENGUI_COMPATIBLE_COMMAND_HELP))
            .put("compatible_prefixes", JSONArray(listOf("!opengui", "/opengui", "/")))
            .put("slash_command_schema", openguiSlashCommandSchemaJson())
            .put(
                "allowlist_arguments",
                JSONArray(
                    listOf(
                        "allowed_guild_ids",
                        "allowed_channel_ids",
                        "allowed_user_ids",
                        "guild_id",
                        "channel_id",
                        "user_id",
                    )
                )
            )
            .toString()
    }

    private fun operatorCommandDeniedJson(parsed: ParsedOperatorCommand, access: OperatorCommandAccess): String {
        return JSONObject()
            .put("success", false)
            .put("handled", false)
            .put("status", "not_allowed")
            .put("parsed_command", parsed.toJson())
            .put("message", "Hermes rejected this OpenGUI-compatible IM command because it did not match the supplied allowlist.")
            .put("access", access.toJson())
            .put("compatible_prefixes", JSONArray(listOf("!opengui", "/opengui", "/")))
            .toString()
    }

    private fun operatorCommandRecognizedButNotActiveJson(parsed: ParsedOperatorCommand): String {
        return JSONObject()
            .put("success", true)
            .put("handled", false)
            .put("status", "recognized_not_active")
            .put("parsed_command", parsed.toJson())
            .put(
                "message",
                "Hermes recognized this OpenGUI IM command, but local Android automations run synchronously in this bridge. Use operator_execution_status for completed runs, or disable/delete saved automations.",
            )
            .toString()
    }

    private fun operatorCommandFreeTextJson(parsed: ParsedOperatorCommand): String {
        return JSONObject()
            .put("success", true)
            .put("handled", false)
            .put("status", "free_text")
            .put("parsed_command", parsed.toJson())
            .put("message", "Send /help, /opengui help, or !opengui help to view supported remote commands.")
            .put("compatible_prefixes", JSONArray(listOf("!opengui", "/opengui", "/")))
            .toString()
    }

    private fun openguiSlashCommandSchemaJson(): JSONObject {
        val subcommands = JSONArray()
        for ((name, description) in OPENGUI_SLASH_COMMANDS) {
            subcommands.put(JSONObject().put("name", name).put("description", description))
        }
        return JSONObject()
            .put("name", "opengui")
            .put("description", "Control Hermes Android remote automations")
            .put("subcommands", subcommands)
    }

    private fun withParsedOperatorCommand(resultJson: String, parsed: ParsedOperatorCommand): String {
        return JSONObject(resultJson)
            .put("parsed_command", parsed.toJson())
            .put("handled", true)
            .toString()
    }

    private fun booleanArgument(arguments: JSONObject, vararg keys: String): Boolean? {
        for (key in keys) {
            if (!arguments.has(key) || arguments.isNull(key)) {
                continue
            }
            return when (val raw = arguments.opt(key)) {
                is Boolean -> raw
                is Number -> raw.toInt() != 0
                else -> when (raw?.toString()?.trim()?.lowercase()) {
                    "1", "true", "yes", "on", "long" -> true
                    "0", "false", "no", "off", "short" -> false
                    else -> null
                }
            }
        }
        return null
    }

    private fun constantTimeEquals(left: String, right: String): Boolean {
        return MessageDigest.isEqual(left.toByteArray(Charsets.UTF_8), right.toByteArray(Charsets.UTF_8))
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .put("available_actions", JSONArray(AUTOMATION_ACTIONS))
            .put("available_triggers", JSONArray(AUTOMATION_TRIGGERS))
            .toString()
    }

    private val OPENGUI_COMPATIBLE_COMMAND_HELP = listOf(
        "Hermes OpenGUI-compatible remote commands",
        "/opengui <subcommand> - raw slash-command compatible form",
        "/tasks - list saved Hermes automations",
        "/run <id> - run a matching saved remote-dispatch automation",
        "/do <description> - dispatch to an enabled automation with the same label",
        "/status [executionId] - inspect recent execution status",
        "/devices - list this standby Hermes device",
        "/pause, /resume, and /cancel are recognized for compatibility; current local runs are synchronous",
    )

    private val OPENGUI_SLASH_COMMANDS = listOf(
        "help" to "Show Hermes OpenGUI-compatible commands",
        "devices" to "List online Hermes standby devices",
        "tasks" to "List saved Hermes automations",
        "run" to "Run an existing remote-dispatch automation by id or label",
        "do" to "Dispatch a natural-language task to a matching enabled automation",
        "status" to "Show recent execution status",
        "cancel" to "Recognized for OpenGUI compatibility",
        "pause" to "Recognized for OpenGUI compatibility",
        "resume" to "Recognized for OpenGUI compatibility",
    )

    private val AUTOMATION_ACTIONS = listOf(
        "list",
        "operator_devices",
        "operator_standby_status",
        "operator_execution_status",
        "operator_command",
        "run_history",
        "create_shell_task",
        "create_file_write_task",
        "create_file_delete_task",
        "create_system_action_task",
        "create_ui_action_task",
        "create_app_launch_task",
        "create_intent_task",
        "open_uri",
        "open_url",
        "open_browser",
        "launch_browser",
        "start_activity",
        "send_broadcast",
        "create_shizuku_action_task",
        "create_sunrise_sunset_task",
        "create_notification_task",
        "create_variable_action_task",
        "create_wait_task",
        "create_clipboard_task",
        "create_vibration_task",
        "create_audio_action_task",
        "perform_audio_action",
        "set_audio_volume",
        "set_sound_mode",
        "set_microphone_mute",
        "set_speakerphone",
        "create_http_request_task",
        "perform_http_request",
        "http_get",
        "http_post",
        "http_head",
        "create_overlay_scene_task",
        "create_toast_task",
        "show_toast",
        "overlay_scene_status",
        "show_overlay_scene",
        "hide_overlay_scene",
        "create_launcher_shortcut",
        "list_launcher_shortcuts",
        "remove_launcher_shortcut",
        "set_quick_settings_tile_automation",
        "get_quick_settings_tile_automation",
        "clear_quick_settings_tile_automation",
        "run_quick_settings_tile",
        "set_home_screen_widget_automation",
        "get_home_screen_widget_automation",
        "list_home_screen_widgets",
        "clear_home_screen_widget_automation",
        "run_home_screen_widget",
        "calculate_sunrise_sunset",
        "export_automations",
        "import_automations",
        "import_tasker_xml",
        "logcat_watcher_status",
        "start_logcat_watcher",
        "stop_logcat_watcher",
        "scan_logcat_entries",
        "reset_logcat_watcher_cursor",
        "sensor_watcher_status",
        "start_sensor_watcher",
        "stop_sensor_watcher",
        "calendar_watcher_status",
        "start_calendar_watcher",
        "stop_calendar_watcher",
        "scan_calendar_events",
        "reset_calendar_watcher_cursor",
        "location_watcher_status",
        "start_location_watcher",
        "stop_location_watcher",
        "scan_location",
        "run",
        "run_trigger",
        "run_app_foreground_trigger",
        "run_notification_posted_trigger",
        "run_calendar_event_trigger",
        "run_location_trigger",
        "run_sensor_trigger",
        "run_external_trigger",
        "run_remote_dispatch",
        "submit_standby_dispatch",
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
    private val AUTOMATION_ACTION_TYPES = setOf(
        ACTION_TYPE_SHELL,
        ACTION_TYPE_FILE_WRITE,
        ACTION_TYPE_FILE_DELETE,
        ACTION_TYPE_SYSTEM_ACTION,
        ACTION_TYPE_UI_ACTION,
        ACTION_TYPE_APP_LAUNCH,
        ACTION_TYPE_INTENT,
        ACTION_TYPE_SHIZUKU_ACTION,
        ACTION_TYPE_SUNRISE_SUNSET,
        ACTION_TYPE_NOTIFICATION_ACTION,
        ACTION_TYPE_VARIABLE_ACTION,
        ACTION_TYPE_WAIT,
        ACTION_TYPE_CLIPBOARD_ACTION,
        ACTION_TYPE_VIBRATION_ACTION,
        ACTION_TYPE_AUDIO_ACTION,
        ACTION_TYPE_HTTP_REQUEST,
        ACTION_TYPE_OVERLAY_SCENE,
        ACTION_TYPE_TOAST_ACTION,
    )
    private val ACTION_TYPE_SYNONYMS = mapOf(
        "shizuku" to ACTION_TYPE_SHIZUKU_ACTION,
        "privileged_action" to ACTION_TYPE_SHIZUKU_ACTION,
        "sun" to ACTION_TYPE_SUNRISE_SUNSET,
        "solar" to ACTION_TYPE_SUNRISE_SUNSET,
        "sunrise" to ACTION_TYPE_SUNRISE_SUNSET,
        "sunset" to ACTION_TYPE_SUNRISE_SUNSET,
        "sun_times" to ACTION_TYPE_SUNRISE_SUNSET,
        "solar_times" to ACTION_TYPE_SUNRISE_SUNSET,
        "notification" to ACTION_TYPE_NOTIFICATION_ACTION,
        "notify" to ACTION_TYPE_NOTIFICATION_ACTION,
        "post_notification" to ACTION_TYPE_NOTIFICATION_ACTION,
        "variable" to ACTION_TYPE_VARIABLE_ACTION,
        "variable_action" to ACTION_TYPE_VARIABLE_ACTION,
        "variable_set" to ACTION_TYPE_VARIABLE_ACTION,
        "variable_clear" to ACTION_TYPE_VARIABLE_ACTION,
        "delay" to ACTION_TYPE_WAIT,
        "sleep" to ACTION_TYPE_WAIT,
        "clipboard" to ACTION_TYPE_CLIPBOARD_ACTION,
        "set_clipboard" to ACTION_TYPE_CLIPBOARD_ACTION,
        "vibrate" to ACTION_TYPE_VIBRATION_ACTION,
        "vibration" to ACTION_TYPE_VIBRATION_ACTION,
        "audio" to ACTION_TYPE_AUDIO_ACTION,
        "volume" to ACTION_TYPE_AUDIO_ACTION,
        "sound_mode" to ACTION_TYPE_AUDIO_ACTION,
        "http" to ACTION_TYPE_HTTP_REQUEST,
        "http_request" to ACTION_TYPE_HTTP_REQUEST,
        "http_get" to ACTION_TYPE_HTTP_REQUEST,
        "http_post" to ACTION_TYPE_HTTP_REQUEST,
        "scene" to ACTION_TYPE_OVERLAY_SCENE,
        "overlay" to ACTION_TYPE_OVERLAY_SCENE,
        "overlay_scene" to ACTION_TYPE_OVERLAY_SCENE,
        "toast" to ACTION_TYPE_TOAST_ACTION,
        "flash" to ACTION_TYPE_TOAST_ACTION,
        "flash_message" to ACTION_TYPE_TOAST_ACTION,
        "android_intent" to ACTION_TYPE_INTENT,
        "start_activity" to ACTION_TYPE_INTENT,
        "open_uri" to ACTION_TYPE_INTENT,
        "send_broadcast" to ACTION_TYPE_INTENT,
        "ui" to ACTION_TYPE_UI_ACTION,
        "accessibility" to ACTION_TYPE_UI_ACTION,
        "launch_app" to ACTION_TYPE_APP_LAUNCH,
        "system" to ACTION_TYPE_SYSTEM_ACTION,
        "file" to ACTION_TYPE_FILE_WRITE,
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
        TRIGGER_SENSOR,
        TRIGGER_LOGCAT_ENTRY,
        TRIGGER_EXTERNAL,
        TRIGGER_REMOTE_DISPATCH,
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
        "sensor" to TRIGGER_SENSOR,
        "sensor_event" to TRIGGER_SENSOR,
        "sensor_profile" to TRIGGER_SENSOR,
        "sensor_trigger" to TRIGGER_SENSOR,
        "sensor_state" to TRIGGER_SENSOR,
        "shake" to TRIGGER_SENSOR,
        "orientation" to TRIGGER_SENSOR,
        "motion" to TRIGGER_SENSOR,
        "logcat" to TRIGGER_LOGCAT_ENTRY,
        "logcat_entry" to TRIGGER_LOGCAT_ENTRY,
        "logcat_event" to TRIGGER_LOGCAT_ENTRY,
        "logcat_profile" to TRIGGER_LOGCAT_ENTRY,
        "log_entry" to TRIGGER_LOGCAT_ENTRY,
        "android_log" to TRIGGER_LOGCAT_ENTRY,
        "external" to TRIGGER_EXTERNAL,
        "external_event" to TRIGGER_EXTERNAL,
        "external_trigger" to TRIGGER_EXTERNAL,
        "extra_trigger" to TRIGGER_EXTERNAL,
        "trigger_app" to TRIGGER_EXTERNAL,
        "external_app" to TRIGGER_EXTERNAL,
        "third_party_trigger" to TRIGGER_EXTERNAL,
        "tasker_trigger_app" to TRIGGER_EXTERNAL,
        "remote" to TRIGGER_REMOTE_DISPATCH,
        "remote_dispatch" to TRIGGER_REMOTE_DISPATCH,
        "standby" to TRIGGER_REMOTE_DISPATCH,
        "standby_dispatch" to TRIGGER_REMOTE_DISPATCH,
        "opengui" to TRIGGER_REMOTE_DISPATCH,
        "opengui_standby" to TRIGGER_REMOTE_DISPATCH,
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
    private data class IntentPayloadResult(
        val payload: JSONObject = JSONObject(),
        val error: String = "",
    )
    private data class TimeParseResult(val minutes: Int?, val error: String? = null)
    private data class DaysParseResult(val daysCsv: String, val error: String? = null)
    private data class TriggerDataResult(val data: String, val error: String? = null)
    private data class NotificationButtonsResult(val buttons: JSONArray, val error: String? = null)
    private data class NotificationButtonResult(val button: JSONObject = JSONObject(), val error: String? = null)
    private data class SolarDate(
        val year: Int,
        val month: Int,
        val day: Int,
        val text: String,
        val dayOfYear: Int,
        val middayEpochMs: Long,
    )
    private data class SunriseSunsetInput(
        val latitude: Double,
        val longitude: Double,
        val date: SolarDate,
        val timeZone: TimeZone,
    )
    private data class SolarEventResult(val minutes: Int?, val polarState: String?)
    private data class SunriseSunsetResult(
        val sunriseMinutes: Int?,
        val sunsetMinutes: Int?,
        val dawnMinutes: Int?,
        val duskMinutes: Int?,
        val solarNoonMinutes: Int?,
        val daylightMinutes: Int?,
        val sunState: String,
    )

    private const val DEFAULT_LOCATION_RADIUS_METERS = 100.0
    private const val EARTH_RADIUS_METERS = 6_371_008.8
    private const val SOLAR_OFFICIAL_ZENITH = 90.833
    private const val SOLAR_CIVIL_ZENITH = 96.0
    private const val SOLAR_POLAR_DAY = "polar_day"
    private const val SOLAR_POLAR_NIGHT = "polar_night"
    private val SOLAR_DATE_PATTERN = Regex("(\\d{4})-(\\d{2})-(\\d{2})")
    private val GMT_OFFSET_ZONE_PATTERN = Regex("GMT[+-]\\d{1,2}(:?\\d{2})?")
    private val AVAILABLE_TIME_ZONE_IDS = TimeZone.getAvailableIDs().toSet()

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
        "clear_app_data",
        "enable_app",
        "disable_app",
        "set_app_enabled",
        "set_wifi_enabled",
        "enable_wifi",
        "disable_wifi",
        "set_bluetooth_enabled",
        "enable_bluetooth",
        "disable_bluetooth",
        "set_mobile_data_enabled",
        "enable_mobile_data",
        "disable_mobile_data",
        "set_airplane_mode_enabled",
        "enable_airplane_mode",
        "disable_airplane_mode",
        "set_wifi_tethering_enabled",
        "enable_wifi_tethering",
        "disable_wifi_tethering",
        "set_dnd_mode",
        "enable_dnd",
        "disable_dnd",
        "set_power_save_mode",
        "enable_power_save_mode",
        "disable_power_save_mode",
        "turn_screen_off",
        "end_call",
        "global_back",
        "global_home",
        "global_recents",
        "global_notifications",
        "global_quick_settings",
        "collapse_status_bar",
        "set_mobile_network_type",
        "start_user_profile",
        "stop_user_profile",
        "switch_user_profile",
        "set_custom_setting",
        "get_custom_setting",
        "delete_custom_setting",
    )
    private val SHIZUKU_PACKAGE_ACTIONS = setOf(
        "grant_runtime_permission",
        "revoke_runtime_permission",
        "force_stop_app",
        "clear_app_data",
        "enable_app",
        "disable_app",
        "set_app_enabled",
    )
    private val SHIZUKU_PERMISSION_ACTIONS = setOf("grant_runtime_permission", "revoke_runtime_permission")
    private val SHIZUKU_TOGGLE_STATE_ACTIONS = setOf(
        "set_app_enabled",
        "set_wifi_enabled",
        "set_bluetooth_enabled",
        "set_mobile_data_enabled",
        "set_airplane_mode_enabled",
        "set_wifi_tethering_enabled",
        "set_power_save_mode",
    )
    private val SHIZUKU_CUSTOM_SETTING_ACTIONS = setOf("set_custom_setting", "get_custom_setting", "delete_custom_setting")
    private val SHIZUKU_DND_ACTIONS = setOf("set_dnd_mode")
    private val SHIZUKU_USER_PROFILE_ACTIONS = setOf("start_user_profile", "stop_user_profile", "switch_user_profile")
    private val SHIZUKU_MOBILE_NETWORK_TYPE_ACTIONS = setOf("set_mobile_network_type")
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
    private val NOTIFICATION_STRING_PAYLOAD_KEYS = listOf(
        "notification_action",
        "title",
        "text",
        "notification_id",
        "notification_tag",
        "channel_id",
        "channel_name",
        "priority",
        "importance",
        "group_key",
        "status_text",
        "progress_value",
        "progress_max",
    )
    private val NOTIFICATION_BOOLEAN_PAYLOAD_KEYS = listOf(
        "ongoing",
        "auto_cancel",
        "only_alert_once",
        "show_when",
        "group_summary",
        "progress_indeterminate",
    )
    private val TASKER_VARIABLE_PATTERN = Regex("%([A-Za-z_][A-Za-z0-9_]{1,63})")
    private val BRACE_VARIABLE_PATTERN = Regex("\\{\\{([A-Za-z_][A-Za-z0-9_]{0,63})\\}\\}")
    private const val VARIABLE_ACTION_SET = "set"
    private const val VARIABLE_ACTION_CLEAR = "clear"
    private const val VARIABLE_ACTION_APPEND = "append"
    private const val VARIABLE_ACTION_ADD = "add"
    private const val VARIABLE_ACTION_SUBTRACT = "subtract"
    private const val VARIABLE_ACTION_REPLACE = "replace"
    private const val CLIPBOARD_ACTION_SET = "set"
    private const val VIBRATION_ACTION_VIBRATE = "vibrate"
    private const val REMOTE_DISPATCH_FAILURE_AUTOMATION_ID = "remote_dispatch_failure"
    private const val AUTOMATION_TIMEOUT_SECONDS = 30
    private const val MAX_WAIT_DURATION_MS = 60_000L
    private const val MAX_VIBRATION_TOTAL_MS = 60_000L
    private const val MAX_VIBRATION_PATTERN_ENTRIES = 32
    private const val MAX_AUDIO_LEVEL = 100L
    private const val MAX_AUDIO_LEVEL_TEXT_CHARS = 64
    private const val MAX_VARIABLE_VALUE_CHARS = 4_000
    private const val MAX_CLIPBOARD_LABEL_CHARS = 80
    private const val MAX_TOAST_TEXT_CHARS = 1_000
    private const val MAX_RESULT_CHARS = 2_000
    private const val MAX_EVENT_VALUE_CHARS = 500
    private const val MAX_NOTIFICATION_BUTTONS = 3
    private const val MAX_NOTIFICATION_BUTTON_FIELD_CHARS = 120
    private const val MAX_IMPORTED_AUTOMATIONS = 200
    private const val MAX_AUTOMATION_ID_CHARS = 120
    private const val AUTOMATION_BUNDLE_KIND = "hermes_android_automation_bundle"
    private const val AUTOMATION_BUNDLE_SCHEMA_VERSION = 1
}
