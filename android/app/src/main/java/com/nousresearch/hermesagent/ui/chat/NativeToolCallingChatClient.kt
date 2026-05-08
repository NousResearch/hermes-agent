package com.nousresearch.hermesagent.ui.chat

import android.content.Context
import com.nousresearch.hermesagent.api.ChatContentPart
import com.nousresearch.hermesagent.api.ChatMessage
import com.nousresearch.hermesagent.api.HermesApiClient
import com.nousresearch.hermesagent.api.toJsonObject
import com.nousresearch.hermesagent.device.HermesAccessibilityController
import com.nousresearch.hermesagent.device.HermesAccessibilityUiBridge
import com.nousresearch.hermesagent.device.HermesAutomationBridge
import com.nousresearch.hermesagent.device.HermesPrivilegedAccessBridge
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import com.nousresearch.hermesagent.device.HermesWorkspaceFileBridge
import com.nousresearch.hermesagent.device.NativeAndroidShellTool
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.TimeUnit

class NativeToolCallingChatClient(
    context: Context,
    private val httpClient: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.MINUTES)
        .build(),
) {
    private val appContext = context.applicationContext

    data class Result(
        val content: String,
        val executedToolCalls: Int,
    )

    fun send(
        baseUrl: String,
        modelName: String,
        sessionId: String,
        userText: String,
        userContentParts: List<ChatContentPart> = emptyList(),
    ): Result {
        val normalizedBaseUrl = baseUrl.trimEnd('/')
        val messages = JSONArray()
            .put(systemMessage())
            .put(
                ChatMessage(
                    role = "user",
                    content = userText,
                    contentParts = userContentParts,
                ).toJsonObject()
            )

        var executedToolCalls = 0
        var latestToolResult = ""
        var assistant = postChatCompletion(
            normalizedBaseUrl = normalizedBaseUrl,
            modelName = modelName,
            sessionId = sessionId,
            messages = messages,
            includeTools = true,
            maxTokens = NATIVE_TOOL_MAX_TOKENS,
        )

        if (assistant.toolCalls.isEmpty()) {
            val content = assistant.content.ifBlank {
                latestToolResult.ifBlank { "Done." }
            }
            return Result(content = content, executedToolCalls = executedToolCalls)
        }

        messages.put(assistant.toJsonMessage())
        assistant.toolCalls.forEach { toolCall ->
            val toolResult = executeToolCall(toolCall)
            executedToolCalls += 1
            latestToolResult = toolResult
            messages.put(
                JSONObject()
                    .put("role", "tool")
                    .put("tool_call_id", toolCall.id)
                    .put("name", toolCall.name)
                    .put("content", toolResult)
            )
        }

        val followUp = postChatCompletion(
            normalizedBaseUrl = normalizedBaseUrl,
            modelName = modelName,
            sessionId = sessionId,
            messages = messages,
            includeTools = false,
            maxTokens = NATIVE_TOOL_MAX_TOKENS,
        )
        return Result(
            content = followUp.content.ifBlank { toolCompletionReply(latestToolResult) },
            executedToolCalls = executedToolCalls,
        )
    }

    private fun toolCompletionReply(toolResult: String): String {
        val parsed = runCatching { JSONObject(toolResult) }.getOrNull() ?: return toolResult.ifBlank { "Tool call completed." }
        val output = parsed.optString("output").trim()
        if (output.isNotBlank()) {
            return output
        }
        val error = parsed.optString("error").trim()
        if (parsed.optInt("exit_code", 0) != 0 && error.isNotBlank()) {
            return error
        }
        val path = parsed.optString("path").trim()
        if (path.isNotBlank()) {
            return "Tool call completed: $path"
        }
        return toolResult.ifBlank { "Tool call completed." }
    }

    private fun postChatCompletion(
        normalizedBaseUrl: String,
        modelName: String,
        sessionId: String,
        messages: JSONArray,
        includeTools: Boolean,
        maxTokens: Int,
    ): AssistantMessage {
        val payload = JSONObject()
            .put("model", modelName)
            .put("stream", false)
            .put("temperature", 0.0)
            .put("max_tokens", maxTokens)
            .put("timeout_ms", NATIVE_TOOL_GENERATION_TIMEOUT_MS)
            .put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
            .put("messages", messages)
        if (includeTools) {
            payload.put("tools", toolSpecs())
        }

        val request = Request.Builder()
            .url("$normalizedBaseUrl/v1/chat/completions")
            .header(HermesApiClient.SESSION_HEADER, sessionId)
            .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()

        httpClient.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            require(response.isSuccessful) { "Native chat request failed: ${response.code} $body" }
            val root = JSONObject(body)
            val message = root
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
            return AssistantMessage.fromJson(message)
        }
    }

    private fun executeToolCall(toolCall: ToolCall): String {
        return when (toolCall.name) {
            "terminal_tool", "terminal", "shell" -> executeTerminalTool(toolCall)
            "file_write_tool", "write_file", "file_tool" -> executeFileWriteTool(toolCall)
            "android_system_tool", "android_system_action", "system_tool", "settings_tool", "phone_tool" ->
                executeAndroidSystemTool(toolCall)
            "android_automation_tool", "automation_tool", "tasker_tool" -> executeAndroidAutomationTool(toolCall)
            "android_ui_tool", "ui_tool", "screen_tool", "accessibility_tool" -> executeAndroidUiTool(toolCall)
            else -> JSONObject()
                .put("exit_code", 127)
                .put("error", "Unsupported native Hermes tool: ${toolCall.name}")
                .toString()
        }
    }

    private fun executeTerminalTool(toolCall: ToolCall): String {
        val command = listOf("command", "cmd", "input")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?: return JSONObject()
                .put("exit_code", 2)
                .put("error", "terminal_tool requires a command argument")
                .toString()

        val result = NativeAndroidShellTool.run(
            context = appContext,
            command = command,
            timeoutSeconds = TOOL_TIMEOUT_SECONDS.toLong(),
        )
        return JSONObject()
            .put("exit_code", result.optInt("exit_code", -1))
            .put("output", truncate(result.optString("output")))
            .put("error", truncate(result.optString("error")))
            .put("cwd", result.optString("cwd"))
            .toString()
    }

    private fun executeFileWriteTool(toolCall: ToolCall): String {
        val rawPath = listOf("path", "file_path", "filename", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            ?: return JSONObject()
                .put("exit_code", 2)
                .put("error", "file_write_tool requires a path argument")
                .toString()
        if (rawPath.indexOf('\u0000') >= 0) {
            return JSONObject()
                .put("exit_code", 2)
                .put("error", "file_write_tool path must not contain NUL bytes")
                .toString()
        }

        val content = listOf("content", "text", "data")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotEmpty() } }
            ?: return JSONObject()
                .put("exit_code", 2)
                .put("error", "file_write_tool requires a content argument")
                .toString()
        val append = toolCall.arguments.optBoolean("append", false)

        return HermesWorkspaceFileBridge.writeTextJson(appContext, rawPath, content, append).toString()
    }

    private fun executeAndroidSystemTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            .orEmpty()
        return if (action.isBlank() || action == "status" || action == "read_status") {
            HermesSystemControlBridge.statusJson()
        } else if (action == "run_privileged_shell" || action == "shizuku_shell" || action == "privileged_shell") {
            val command = listOf("command", "cmd", "input")
                .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
                .orEmpty()
            HermesPrivilegedAccessBridge.runShellCommandJson(
                context = appContext,
                command = command,
                timeoutSeconds = toolCall.arguments.optInt("timeout_seconds", PRIVILEGED_TOOL_TIMEOUT_SECONDS),
            )
        } else if (HermesPrivilegedAccessBridge.handlesStructuredAction(action)) {
            HermesPrivilegedAccessBridge.performStructuredActionJson(
                context = appContext,
                action = action,
                arguments = toolCall.arguments,
            )
        } else {
            HermesSystemControlBridge.performActionJson(action)
        }
    }

    private fun executeAndroidAutomationTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            ?.lowercase()
            .orEmpty()
        return HermesAutomationBridge.performActionJson(appContext, action, toolCall.arguments)
    }

    private fun executeAndroidUiTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            ?.lowercase()
            .orEmpty()
        return when (action.ifBlank { "status" }) {
            "status", "read_status" -> androidUiStatusJson()
            "snapshot", "screen_snapshot", "read_screen" -> HermesAccessibilityUiBridge.snapshotJson(
                limit = toolCall.arguments.optInt("limit", DEFAULT_UI_SNAPSHOT_LIMIT),
            )
            "click",
            "long_click",
            "focus",
            "scroll_forward",
            "scroll_backward",
            "set_text" -> HermesAccessibilityUiBridge.performActionJson(
                action = action,
                textContains = toolCall.arguments.optString("text_contains"),
                contentDescriptionContains = toolCall.arguments.optString("content_description_contains"),
                viewId = toolCall.arguments.optString("view_id"),
                packageName = toolCall.arguments.optString("package_name"),
                value = toolCall.arguments.optString("value"),
                index = toolCall.arguments.optInt("index", 0),
            )
            "back", "global_back" -> HermesAccessibilityUiBridge.performGlobalActionJson("back")
            "home", "global_home" -> HermesAccessibilityUiBridge.performGlobalActionJson("home")
            "recents", "global_recents" -> HermesAccessibilityUiBridge.performGlobalActionJson("recents")
            "notifications", "global_notifications" -> HermesAccessibilityUiBridge.performGlobalActionJson("notifications")
            "quick_settings", "global_quick_settings" -> HermesAccessibilityUiBridge.performGlobalActionJson("quick_settings")
            "open_accessibility_settings" -> HermesSystemControlBridge.performActionJson("open_accessibility_settings")
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported Android UI action: $action")
                .put("available_ui_actions", JSONArray(ANDROID_UI_ACTIONS))
                .toString()
        }
    }

    private fun androidUiStatusJson(): String {
        return JSONObject()
            .put("accessibility_service_enabled", HermesAccessibilityController.isServiceEnabled(appContext))
            .put("accessibility_connected", HermesAccessibilityController.isServiceConnected())
            .put("available_ui_actions", JSONArray(ANDROID_UI_ACTIONS))
            .put("selection_arguments", JSONArray(UI_SELECTOR_ARGUMENTS))
            .put("message", "Enable the Hermes accessibility service before using snapshot, click, set_text, scroll, or global navigation actions.")
            .toString()
    }

    private fun systemMessage(): JSONObject {
        return JSONObject()
            .put("role", "system")
            .put(
                "content",
                "You are Hermes running inside the native Android app. " +
                    "You have functions named terminal_tool, file_write_tool, android_system_tool, android_automation_tool, and android_ui_tool. " +
                    "When the user asks to write or replace a text file, prefer file_write_tool so multiline content is written exactly. " +
                    "When the user asks to run a command, inspect the filesystem, read a file, or use a device command, call terminal_tool instead of simulating the result. " +
                    "terminal_tool runs through /system/bin/sh in the Hermes app workspace. " +
                    "file_write_tool writes UTF-8 text files in the Hermes app workspace; file_write_tool can only write inside the Hermes app workspace. " +
                    "When the user asks about Android settings, phone connectivity, Wi-Fi hotspot/tethering, permissions, background runtime, app enable/disable, app data clearing, app force-stop, or safe system panels, call android_system_tool. " +
                    "android_system_tool status includes Shizuku/Sui privileged-access state, and it can open Shizuku, wireless debugging, and developer settings setup flows. " +
                    "If Shizuku/Sui is running and the user granted Hermes permission, android_system_tool can run explicit ADB/root-identity shell commands with action run_privileged_shell and a command argument. " +
                    "For Tasker-style Shizuku actions, android_system_tool can run grant_runtime_permission, revoke_runtime_permission, force_stop_app, clear_app_data, enable_app, disable_app, set_app_enabled, Wi-Fi/Bluetooth/mobile-data toggles, airplane-mode toggles, Wi-Fi tethering toggles, DND mode, power saver, end call, screen off, global navigation/statusbar actions, mobile network type bitmask changes, work-profile/user actions, and custom Android settings get/set/delete with explicit package_name, permission, enabled, dnd_mode, user_id, network_types_bitmask, or setting arguments where needed. " +
                    "When the user asks to create a recurring phone automation, reusable Android task, Tasker-like variable, time/day trigger, phone-state trigger, app-foreground trigger, notification trigger, calendar event trigger, location trigger, sensor trigger, logcat entry trigger, Shizuku availability trigger, sunrise/sunset calculation, saved notification action, saved variable set/clear action, saved clipboard action, saved Tasker Flash/toast action, saved vibration action, saved file action, safe saved Android settings action, saved visible-UI action, saved Android intent, broadcast, URI, activity launch, saved app-launch action, Quick Settings tile automation, home-screen widget automation, or backup/restore Hermes automations, call android_automation_tool. It can save shell, file-write, file-delete, variable set/clear, clipboard set, Tasker Flash/toast messages, vibration, safe Android system-action, accessibility UI-action, app-launch, Android intent, Shizuku package-permission/data-clear/connectivity/custom-setting/tethering/DND/power/global-navigation/mobile-network/user-profile, offline sunrise/sunset, and notification post/cancel tasks, configure a user-added Hermes Quick Settings tile or home-screen widget to run a saved automation, calculate sun times directly, export/import Hermes automation bundles, run tasks manually, enable/disable/delete them, schedule interval and time-of-day tasks with Android alarms, run boot/power/battery/time/app-foreground/notification-posted/calendar-event/location/sensor/logcat-entry/Shizuku-state triggers, and expand saved variables in commands, file content, clipboard text, toast messages, UI selectors, intent fields, package names, trigger packages, Shizuku state fields, sunrise/sunset location/date/timezone fields, notification action fields, variable action fields, and notification, calendar, location, sensor, logcat, or time event fields. " +
                    "When the user asks to inspect the visible phone screen, click, type, scroll, or use Back/Home/Recents/Quick Settings, call android_ui_tool. " +
                    "android_ui_tool requires the user-enabled Hermes accessibility service for screen snapshots and UI actions. " +
                    "Protected Android settings require user-granted permissions, Shizuku/Sui, accessibility service, or an opened settings panel.",
            )
    }

    private fun toolSpecs(): JSONArray {
        return JSONArray()
            .put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject()
                            .put("name", "terminal_tool")
                            .put(
                                "description",
                                "Run a short Android native shell command through /system/bin/sh in the Hermes app workspace and return stdout, stderr, exit code, and cwd.",
                            )
                            .put(
                                "parameters",
                                JSONObject()
                                    .put("type", "object")
                                    .put(
                                        "properties",
                                        JSONObject()
                                            .put(
                                                "command",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "The shell command to run."),
                                            ),
                                    )
                                    .put("required", JSONArray().put("command")),
                            ),
                    ),
            )
            .put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject()
                            .put("name", "android_automation_tool")
                            .put(
                                "description",
                                "Create, list, run, enable, disable, delete, export, or import saved Android automations and variables. Supports shell, file-write, file-delete, variable set/clear, clipboard set, Tasker Flash/toast messages, vibration, safe Android system-action, accessibility UI-action, app-launch, Android intent, Shizuku/Sui package-permission/data-clear/connectivity-toggle, offline sunrise/sunset, notification post/cancel tasks, overlay scene show/hide tasks, launcher shortcuts, a user-added Hermes Quick Settings tile, and a Hermes home-screen widget bound to a saved automation; direct sunrise/sunset calculation; safe Tasker XML/Data URI import; Shizuku-backed logcat scan/watch actions with a bounded scan cursor; manual tasks; interval tasks; Tasker-style time/day triggers; boot/power/battery/app-foreground/notification-posted/calendar-event/location/sensor/logcat-entry/Shizuku-state/external-trigger phone triggers; Tasker-style %VARIABLE expansion; and Hermes automation bundle backup/restore. Shizuku execution must be explicitly requested per shell task or by create_shizuku_action_task. Tasker import supports a safe subset of exported Tasker actions including global UI navigation, safe settings panels, Flash, Vibrate, Vibrate Pattern, Set Clipboard, Variable Set, and Variable Clear, and leaves records disabled unless enable_imported is set. Overlay scenes require Android draw-over-other-apps permission and support bounded title/text/button/position payloads, not arbitrary scene code. Notification post actions require Android notification permission on Android 13+. App-foreground triggers require the user-enabled Hermes accessibility service. Notification-posted triggers require user-enabled Hermes notification access. Calendar-event, location, sensor, logcat-entry, and external triggers are explicit event dispatches; external triggers also have an exported broadcast receiver guarded by a required shared token. Quick Settings tile actions can set, get, clear, or run the configured tile automation; the user still has to add the Hermes tile from Android Quick Settings. Home-screen widget actions can set, get, list, clear, request pinning for, or run the configured widget automation; Android launchers still control final widget placement. Location triggers can match latitude/longitude/radius, provider, name, and accuracy, and expose %LOC, %LAT, %LON, %LOCACC, %LOCPROVIDER, %LOCNAME, and LOCATION_* aliases. Sunrise/sunset actions accept latitude, longitude, optional date, and optional timezone, and expose %SUNRISE, %SUNSET, %SUN_DAWN, %SUN_DUSK, %SOLAR_NOON, %SUN_DAYLIGHT_MINUTES, %SUN_STATE, %SUN_DATE, %SUN_TIMEZONE, %SUN_LAT, and %SUN_LON. Notification actions can post, update, or cancel app notifications with title, text, channel, priority, group, ongoing, and only-alert-once fields. Variable actions can set or clear a saved Hermes automation variable at run time and expand existing variables in the target name and value. Clipboard actions set Android clipboard text and expand saved variables at run time. Toast actions show bounded Android toast/Tasker Flash messages and expand saved variables at run time. Vibration actions use Android's normal vibrator permission and cap duration/pattern totals. Sensor triggers can match type/name, event, value name, and min/max value, and expose %SENSOR, %SENSOR_EVENT, %SENSOR_VALUE, %SENSOR_VALUE_NAME, %SENSOR_UNIT, and %SENSOR_ACCURACY. Logcat-entry triggers can match tag, message text, level, pid, and package filters, expose %LOGCAT_TAG, %LOGCAT_MESSAGE, %LOGCAT_LEVEL, %LOGCAT_PID, %LOGCAT_PACKAGE, and %LOGCAT_TIME. start_logcat_watcher and scan_logcat_entries require Shizuku/Sui running with Hermes permission and at least one enabled logcat_entry record; watcher scans dedupe recently seen log lines and reset_logcat_watcher_cursor clears that cursor. External triggers can match trigger_id, external_token, optional trigger_package_name, and optional referrer_contains, and expose %SA_TRIGGER_ID, %SA_TRIGGER_PACKAGE_NAME, %SA_REFERRER, and %SA_EXTRAS. Shizuku-state triggers expose %SHIZUKU_AVAILABLE, %SHIZUKU_INSTALLED, %SUI_INSTALLED, %SHIZUKU_RUNNING, %SHIZUKU_PERMISSION_GRANTED, %SHIZUKU_PRIVILEGE_LABEL, and %SHIZUKU_UID.",
                            )
                            .put(
                                "parameters",
                                JSONObject()
                                    .put("type", "object")
                                    .put(
                                        "properties",
                                        JSONObject()
                                            .put(
                                                "action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "list, create_shell_task, create_file_write_task, create_file_delete_task, create_system_action_task, create_ui_action_task, create_app_launch_task, create_intent_task, create_shizuku_action_task, create_sunrise_sunset_task, create_notification_task, create_variable_action_task, create_wait_task, create_clipboard_task, create_toast_task, show_toast, create_vibration_task, create_overlay_scene_task, overlay_scene_status, show_overlay_scene, hide_overlay_scene, create_launcher_shortcut, list_launcher_shortcuts, remove_launcher_shortcut, set_quick_settings_tile_automation, get_quick_settings_tile_automation, clear_quick_settings_tile_automation, run_quick_settings_tile, set_home_screen_widget_automation, get_home_screen_widget_automation, list_home_screen_widgets, clear_home_screen_widget_automation, run_home_screen_widget, calculate_sunrise_sunset, export_automations, import_automations, import_tasker_xml, logcat_watcher_status, start_logcat_watcher, stop_logcat_watcher, scan_logcat_entries, reset_logcat_watcher_cursor, run, run_trigger, run_app_foreground_trigger, run_notification_posted_trigger, run_calendar_event_trigger, run_location_trigger, run_sensor_trigger, run_logcat_entry_trigger, run_external_trigger, run_shizuku_state_trigger, run_time_trigger, delete, enable, disable, list_variables, set_variable, get_variable, or delete_variable."),
                                            )
                                            .put(
                                                "bundle",
                                                JSONObject()
                                                    .put("type", "object")
                                                    .put("description", "Hermes automation export bundle for import_automations. Use export_automations to create one."),
                                            )
                                            .put(
                                                "bundle_json",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Stringified Hermes automation export bundle for import_automations."),
                                            )
                                            .put(
                                                "tasker_xml",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Raw exported Tasker XML for import_tasker_xml. Imports only a safe subset: Run Shell, Write File, Delete File, Launch App, Browse URL, Notify, Flash, Vibrate, Vibrate Pattern, Set Clipboard, Variable Set, Variable Clear, Go Home, Back Button, Show Recents, Quick Settings, and safe settings-panel actions."),
                                            )
                                            .put(
                                                "tasker_data_uri",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Tasker XML data URI for import_tasker_xml. Imported records are disabled by default unless enable_imported is true."),
                                            )
                                            .put(
                                                "replace",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For import_automations, replace existing saved automations and variables instead of merging/upserting."),
                                            )
                                            .put(
                                                "enable_imported",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For import_automations, force imported automation records enabled."),
                                            )
                                            .put(
                                                "disable_imported",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For import_automations, force imported automation records disabled."),
                                            )
                                            .put(
                                                "id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Automation id for run, delete, enable, or disable."),
                                            )
                                            .put(
                                                "label",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Human-readable label for create_*_task actions."),
                                            )
                                            .put(
                                                "command",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Shell command for create_shell_task, alternate system action value for create_system_action_task, alternate package name for create_app_launch_task, or alternate intent action for create_intent_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "path",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Workspace path for create_file_write_task or create_file_delete_task. Relative paths resolve inside the Hermes app shell home."),
                                            )
                                            .put(
                                                "content",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Text content for create_file_write_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "append",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Append instead of replacing content for create_file_write_task."),
                                            )
                                            .put(
                                                "system_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Safe Android system action for create_system_action_task, such as start_background_runtime, stop_background_runtime, open_wifi_panel, or open_accessibility_settings."),
                                            )
                                            .put(
                                                "ui_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved accessibility UI action for create_ui_action_task: click, long_click, focus, set_text, scroll_forward, scroll_backward, back, home, recents, notifications, or quick_settings."),
                                            )
                                            .put(
                                                "shizuku_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved Shizuku/Sui action for create_shizuku_action_task. Package actions: grant_runtime_permission, revoke_runtime_permission, force_stop_app, clear_app_data, enable_app, disable_app, set_app_enabled. Device actions: Wi-Fi/Bluetooth/mobile-data/airplane-mode/Wi-Fi tethering toggles, set_dnd_mode/enable_dnd/disable_dnd, set_power_save_mode/enable_power_save_mode/disable_power_save_mode, turn_screen_off, end_call, global_back/global_home/global_recents/global_notifications/global_quick_settings/collapse_status_bar, set_mobile_network_type, and start_user_profile/stop_user_profile/switch_user_profile. Requires user-started Shizuku/Sui and granted Hermes permission when run."),
                                            )
                                            .put(
                                                "notification_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved notification action for create_notification_task: post, update, show, notify, cancel, clear, or dismiss. Posting requires Android notification permission on Android 13+."),
                                            )
                                            .put(
                                                "variable_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved variable action for create_variable_action_task: set or clear. Set requires value; clear removes the saved variable. Saved variables can be referenced as %NAME or {{NAME}} in the target name and value."),
                                            )
                                            .put(
                                                "clipboard_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Text for create_clipboard_task. Saved variables can be referenced as %NAME or {{NAME}} and are expanded when the automation runs."),
                                            )
                                            .put(
                                                "clipboard_label",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional Android clipboard clip label for create_clipboard_task."),
                                            )
                                            .put(
                                                "toast_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Toast or Tasker Flash message text for create_toast_task or show_toast. Saved variables can be referenced as %NAME or {{NAME}} and are expanded when a saved automation runs."),
                                            )
                                            .put(
                                                "toast_long",
                                                JSONObject()
                                                    .put("type", JSONArray().put("boolean").put("string"))
                                                    .put("description", "Whether create_toast_task or show_toast should use Android's longer toast duration."),
                                            )
                                            .put(
                                                "vibration_duration_ms",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Duration in milliseconds for create_vibration_task. Capped to a short bounded vibration."),
                                            )
                                            .put(
                                                "vibration_pattern_ms",
                                                JSONObject()
                                                    .put("type", "array")
                                                    .put("description", "Optional vibration timing pattern in milliseconds for create_vibration_task, for example [0, 100, 50, 100].")
                                                    .put("items", JSONObject().put("type", "integer")),
                                            )
                                            .put(
                                                "scene_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Overlay scene action for create_overlay_scene_task, show_overlay_scene, or hide_overlay_scene: show or hide."),
                                            )
                                            .put(
                                                "scene_title",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Short overlay scene title. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "scene_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Overlay scene body text. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "scene_button_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Overlay scene dismiss button label."),
                                            )
                                            .put(
                                                "scene_position",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Overlay scene position: top, center, or bottom."),
                                            )
                                            .put(
                                                "scene_hide_after_ms",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Optional overlay scene auto-dismiss timeout in milliseconds, capped at 600000."),
                                            )
                                            .put(
                                                "notification_id",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Stable notification id for create_notification_task post/update/cancel. Use the same id to update or cancel a notification. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "notification_tag",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional notification tag for create_notification_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "channel_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Notification channel id for create_notification_task. Defaults to the Hermes automation channel."),
                                            )
                                            .put(
                                                "channel_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Notification channel display name for create_notification_task when the channel is first created."),
                                            )
                                            .put(
                                                "priority",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Notification priority for create_notification_task: min, low, default, high, max, or critical."),
                                            )
                                            .put(
                                                "importance",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android notification-channel importance for create_notification_task: min, low, default, high, max, or critical."),
                                            )
                                            .put(
                                                "group_key",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional notification group key for create_notification_task grouping."),
                                            )
                                            .put(
                                                "intent_task_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved Android intent mode for create_intent_task: start_activity, open_uri, or send_broadcast. Aliases include create_activity_task, create_uri_task, and create_broadcast_task."),
                                            )
                                            .put(
                                                "intent_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android Intent action for create_intent_task, for example android.intent.action.VIEW or a custom broadcast action. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "data_uri",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android Intent data URI for create_intent_task, for example https://example.com or package:com.example.app. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "text_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved UI selector: match a visible node whose text contains this value. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "content_description_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved UI selector: match a visible node whose accessibility description contains this value. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "view_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved UI selector: match a node by full or partial Android view id."),
                                            )
                                            .put(
                                                "package_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Package name for create_app_launch_task, create_shizuku_action_task package actions, create_intent_task, or saved UI selector package-name fragment for create_ui_action_task. Shizuku connectivity toggles do not require package_name. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "class_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Activity class name for create_intent_task start_activity. Use a fully qualified class or a package-relative .ClassName."),
                                            )
                                            .put(
                                                "component",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Flattened Android component for create_intent_task, such as com.example/.MainActivity."),
                                            )
                                            .put(
                                                "category",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional Android Intent category for create_intent_task."),
                                            )
                                            .put(
                                                "categories",
                                                JSONObject()
                                                    .put("type", "array")
                                                    .put("description", "Optional Android Intent categories for create_intent_task.")
                                                    .put("items", JSONObject().put("type", "string")),
                                            )
                                            .put(
                                                "extras",
                                                JSONObject()
                                                    .put("type", "object")
                                                    .put("description", "Optional primitive string, number, or boolean extras for create_intent_task or run_external_trigger. String values can reference saved variables as %NAME or {{NAME}} when saved in intent tasks."),
                                            )
                                            .put(
                                                "permission",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android runtime permission name for create_shizuku_action_task grant_runtime_permission or revoke_runtime_permission, for example android.permission.POST_NOTIFICATIONS. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "index",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Zero-based saved UI match index when multiple nodes match."),
                                            )
                                            .put(
                                                "trigger",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional trigger for create_*_task or run_trigger: manual, time, boot, power_connected, power_disconnected, battery_low, battery_okay, app_foreground, notification_posted, calendar_event, location, sensor, logcat_entry, external_trigger, shizuku_available, or shizuku_unavailable. interval_minutes creates an interval trigger. time requires a time argument such as 08:30 and can use days_of_week. app_foreground and notification_posted require trigger_package_name and the relevant Android service permission. calendar_event can filter by calendar_name, title_contains, description_contains, or location_contains. location can filter by latitude, longitude, radius_meters, location_provider, location_name, or max_accuracy_meters and must be run with run_location_trigger. sensor can filter by sensor_type, sensor_event, value_name, min_value, or max_value and must be run with run_sensor_trigger. logcat_entry can filter by logcat_tag, logcat_message_contains, logcat_level, logcat_pid, logcat_package_name, or trigger_package_name and must be run with run_logcat_entry_trigger. Shizuku logcat scans expose UID package candidates through logcat_package_candidates. external_trigger requires trigger_id and external_token and must be run with run_external_trigger or the exported Hermes external-trigger broadcast."),
                                            )
                                            .put(
                                                "trigger_package_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Package name that must become foreground for app_foreground trigger records, post a notification for notification_posted trigger records, or identify an external_trigger caller, such as com.android.settings. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "trigger_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "External trigger id for external_trigger saved records or run_external_trigger events. Saved variables can be referenced as %NAME or {{NAME}} in saved external_trigger records."),
                                            )
                                            .put(
                                                "external_token",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Required shared secret for external_trigger saved records and run_external_trigger events. A saved external trigger does not run unless this token matches."),
                                            )
                                            .put(
                                                "referrer",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional referrer/source value for run_external_trigger, or saved referrer filter for external_trigger records. Runs expose %SA_REFERRER."),
                                            )
                                            .put(
                                                "notification_title",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Notification title for create_notification_task, or optional notification title for run_notification_posted_trigger tests. Real listener events also expose %NOTIFICATION_TITLE."),
                                            )
                                            .put(
                                                "notification_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Notification text for create_notification_task, or optional notification text for run_notification_posted_trigger tests. Real listener events also expose %NOTIFICATION_TEXT."),
                                            )
                                            .put(
                                                "calendar_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar name for create_*_task calendar_event filters or run_calendar_event_trigger. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "calendar_title",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar event title for run_calendar_event_trigger. Calendar-event runs expose %CALTITLE and %CALENDAR_TITLE."),
                                            )
                                            .put(
                                                "calendar_description",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar event description for run_calendar_event_trigger. Calendar-event runs expose %CALDESCR and %CALENDAR_DESCRIPTION."),
                                            )
                                            .put(
                                                "calendar_location",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar event location for run_calendar_event_trigger. Calendar-event runs expose %CALLOC and %CALENDAR_LOCATION."),
                                            )
                                            .put(
                                                "title_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar-event trigger title filter for create_*_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "description_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar-event trigger description filter for create_*_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "location_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Calendar-event trigger location filter for create_*_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "latitude",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Latitude for create_*_task location filters, run_location_trigger, calculate_sunrise_sunset, or create_sunrise_sunset_task. Saved variables can be referenced as %NAME or {{NAME}} in saved filters and saved sunrise/sunset tasks."),
                                            )
                                            .put(
                                                "longitude",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Longitude for create_*_task location filters, run_location_trigger, calculate_sunrise_sunset, or create_sunrise_sunset_task. Saved variables can be referenced as %NAME or {{NAME}} in saved filters and saved sunrise/sunset tasks."),
                                            )
                                            .put(
                                                "date",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional YYYY-MM-DD date for calculate_sunrise_sunset or create_sunrise_sunset_task. Defaults to the current date in the selected timezone; saved variables can be referenced as %NAME or {{NAME}} in saved tasks."),
                                            )
                                            .put(
                                                "timezone",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional timezone for calculate_sunrise_sunset or create_sunrise_sunset_task, such as Europe/London, UTC, local, or GMT+01:00. Saved variables can be referenced as %NAME or {{NAME}} in saved tasks."),
                                            )
                                            .put(
                                                "radius_meters",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Radius in meters for saved location trigger coordinate matching. Defaults to 100 meters when latitude and longitude are provided."),
                                            )
                                            .put(
                                                "accuracy_meters",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Current event accuracy in meters for run_location_trigger. Runs expose %LOCACC and %LOCATION_ACCURACY_METERS."),
                                            )
                                            .put(
                                                "max_accuracy_meters",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Optional saved location trigger filter. The event only matches when accuracy_meters is present and no greater than this value."),
                                            )
                                            .put(
                                                "location_provider",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Provider/source filter or event value for location triggers, such as gps, network, or fused. Runs expose %LOCPROVIDER and %LOCATION_PROVIDER."),
                                            )
                                            .put(
                                                "location_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Place/name filter or event value for location triggers. Runs expose %LOCNAME and %LOCATION_NAME."),
                                            )
                                            .put(
                                                "sensor_type",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Sensor type/name filter or event value for sensor triggers, such as accelerometer, gyroscope, orientation, shake, light, or proximity. Runs expose %SENSOR, %SENSOR_TYPE, and %SENSOR_NAME."),
                                            )
                                            .put(
                                                "sensor_event",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional event/gesture filter or event value for sensor triggers, such as changed, shake, face_down, or threshold. Runs expose %SENSOR_EVENT."),
                                            )
                                            .put(
                                                "value_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional sensor value name, axis, or channel for sensor triggers, such as x, y, z, lux, or distance. Runs expose %SENSOR_VALUE_NAME."),
                                            )
                                            .put(
                                                "sensor_value",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Current numeric value for run_sensor_trigger. Runs expose %SENSOR_VALUE."),
                                            )
                                            .put(
                                                "min_value",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Optional saved sensor trigger lower bound. Saved variables can be referenced as %NAME or {{NAME}} in saved filters."),
                                            )
                                            .put(
                                                "max_value",
                                                JSONObject()
                                                    .put("type", JSONArray().put("number").put("string"))
                                                    .put("description", "Optional saved sensor trigger upper bound. Saved variables can be referenced as %NAME or {{NAME}} in saved filters."),
                                            )
                                            .put(
                                                "sensor_unit",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional unit label for run_sensor_trigger, such as m/s^2, lux, or cm. Runs expose %SENSOR_UNIT."),
                                            )
                                            .put(
                                                "sensor_accuracy",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional accuracy/status label for run_sensor_trigger. Runs expose %SENSOR_ACCURACY."),
                                            )
                                            .put(
                                                "logcat_tag",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Logcat tag/component filter for saved logcat_entry triggers, or event tag for run_logcat_entry_trigger. Saved variables can be referenced as %NAME or {{NAME}} in saved filters."),
                                            )
                                            .put(
                                                "logcat_message_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Saved logcat_entry message substring filter. Use logcat_message for the current event text when calling run_logcat_entry_trigger."),
                                            )
                                            .put(
                                                "logcat_message",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Current log line message for run_logcat_entry_trigger. Runs expose %LOGCAT_MESSAGE and %LOG_MESSAGE."),
                                            )
                                            .put(
                                                "logcat_level",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Logcat priority filter or event value: verbose, debug, info, warn, error, assert, fatal, or V/D/I/W/E/A/F. Runs expose %LOGCAT_LEVEL."),
                                            )
                                            .put(
                                                "logcat_pid",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Optional process id filter or event value for logcat_entry triggers. Saved variables can be referenced as %NAME or {{NAME}} in saved filters."),
                                            )
                                            .put(
                                                "logcat_package_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional package-name filter or event value for logcat_entry triggers. Shizuku logcat scans choose a single package when the log line identifies one, otherwise this can contain a comma-separated shared-UID candidate list. Runs expose %LOGCAT_PACKAGE."),
                                            )
                                            .put(
                                                "logcat_package_candidates",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional comma-separated package candidates for run_logcat_entry_trigger. The Shizuku logcat watcher fills this from UID lookup so trigger_package_name and package filters can match shared-UID candidates. Runs expose %LOGCAT_PACKAGE_CANDIDATES."),
                                            )
                                            .put(
                                                "logcat_package_source",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional package attribution source for run_logcat_entry_trigger, such as uid, uid_shared, or message. Runs expose %LOGCAT_PACKAGE_SOURCE."),
                                            )
                                            .put(
                                                "logcat_timestamp",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional timestamp for run_logcat_entry_trigger. Runs expose %LOGCAT_TIME."),
                                            )
                                            .put(
                                                "scan_interval_seconds",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Background logcat watcher scan interval for start_logcat_watcher. Clamped between 5 and 3600 seconds."),
                                            )
                                            .put(
                                                "max_lines",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Maximum recent logcat lines for scan_logcat_entries or each watcher scan. Clamped between 10 and 1000."),
                                            )
                                            .put(
                                                "use_cursor",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Optional scan_logcat_entries cursor flag. When true, Hermes suppresses recently seen log lines so polling does not rerun the same logcat event."),
                                            )
                                            .put(
                                                "reset_cursor",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Optional start_logcat_watcher or scan_logcat_entries flag that clears the logcat scan cursor before running."),
                                            )
                                            .put(
                                                "shizuku_state",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional state for run_shizuku_state_trigger: available, unavailable, shizuku_available, or shizuku_unavailable. When omitted, Hermes uses the current Shizuku/Sui binder and permission state."),
                                            )
                                            .put(
                                                "interval_minutes",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Optional interval schedule in minutes. Minimum is 15. Omit for a manual task."),
                                            )
                                            .put(
                                                "time",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Tasker-style time trigger for create_*_task, such as 08:30 or 8.30. Supplying time without trigger also creates a time trigger."),
                                            )
                                            .put(
                                                "days_of_week",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional day restriction for time triggers. Use daily, weekday, weekend, or comma-separated days like mon,wed,fri. Time runs daily when omitted."),
                                            )
                                            .put(
                                                "enabled",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Whether a created automation starts enabled. For create_shizuku_action_task set_app_enabled or set_*_enabled connectivity toggles, prefer target_enabled for the target state and automation_enabled for the record state."),
                                            )
                                            .put(
                                                "ongoing",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For create_notification_task, make the notification ongoing until explicitly cancelled."),
                                            )
                                            .put(
                                                "auto_cancel",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For create_notification_task, allow tapping the notification to auto-cancel. Defaults to true."),
                                            )
                                            .put(
                                                "only_alert_once",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For create_notification_task updates, avoid alerting again after the first post."),
                                            )
                                            .put(
                                                "group_summary",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For create_notification_task with group_key, mark this notification as the group summary."),
                                            )
                                            .put(
                                                "automation_enabled",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Explicit record enabled state for create_*_task. Useful when create_shizuku_action_task set_app_enabled also needs a target_enabled value."),
                                            )
                                            .put(
                                                "use_shizuku",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Run this automation through Shizuku/Sui privileged shell. Requires Shizuku setup and user-granted permission."),
                                            )
                                            .put(
                                                "target_enabled",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Target enabled state for create_shizuku_action_task with set_app_enabled or Shizuku toggles such as set_wifi_enabled or set_power_save_mode."),
                                            )
                                            .put(
                                                "dnd_mode",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "DND mode for create_shizuku_action_task set_dnd_mode: on, none, priority, alarms, all, or off."),
                                            )
                                            .put(
                                                "user_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android user/work-profile id for Shizuku user-profile actions such as start_user_profile, stop_user_profile, or switch_user_profile."),
                                            )
                                            .put(
                                                "network_types_bitmask",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Binary Android TelephonyManager network-type bitmask for Shizuku set_mobile_network_type."),
                                            )
                                            .put(
                                                "slot_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional SIM slot id for Shizuku set_mobile_network_type."),
                                            )
                                            .put(
                                                "timeout_seconds",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Optional timeout for Shizuku-backed saved package or permission actions, clamped by Hermes."),
                                            )
                                            .put(
                                                "name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Variable name for set_variable, get_variable, delete_variable, or create_variable_action_task. Leading % is optional."),
                                            )
                                            .put(
                                                "value",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Variable value for set_variable or create_variable_action_task set, or text value for create_ui_action_task set_text."),
                                            ),
                                    )
                                    .put("required", JSONArray().put("action")),
                            ),
                    ),
            )
            .put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject()
                            .put("name", "android_ui_tool")
                            .put(
                                "description",
                                "Inspect or control the visible Android UI through the user-enabled Hermes accessibility service. Supports status, screen snapshots, selector-based click/type/scroll/focus, and global Back/Home/Recents/notifications/quick-settings actions.",
                            )
                            .put(
                                "parameters",
                                JSONObject()
                                    .put("type", "object")
                                    .put(
                                        "properties",
                                        JSONObject()
                                            .put(
                                                "action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "status, snapshot, click, long_click, focus, set_text, scroll_forward, scroll_backward, back, home, recents, notifications, quick_settings, or open_accessibility_settings."),
                                            )
                                            .put(
                                                "text_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Match a visible node whose text contains this value."),
                                            )
                                            .put(
                                                "content_description_contains",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Match a visible node whose accessibility description contains this value."),
                                            )
                                            .put(
                                                "view_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Match a node by full or partial Android view id."),
                                            )
                                            .put(
                                                "package_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Restrict matching to a package name fragment."),
                                            )
                                            .put(
                                                "value",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Text value for set_text."),
                                            )
                                            .put(
                                                "index",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Zero-based match index when multiple nodes match."),
                                            )
                                            .put(
                                                "limit",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Maximum nodes to return for snapshot."),
                                            ),
                                    )
                                    .put("required", JSONArray().put("action")),
                            ),
                    ),
            )
            .put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject()
                            .put("name", "file_write_tool")
                            .put(
                                "description",
                                "Write or replace a UTF-8 text file inside the Hermes app workspace without shell quoting.",
                            )
                            .put(
                                "parameters",
                                JSONObject()
                                    .put("type", "object")
                                    .put(
                                        "properties",
                                        JSONObject()
                                            .put(
                                                "path",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Relative workspace path or app-workspace absolute path to write."),
                                            )
                                            .put(
                                                "content",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Exact UTF-8 text content to write."),
                                            )
                                            .put(
                                                "append",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Append instead of replacing the file."),
                                            ),
                                    )
                                    .put("required", JSONArray().put("path").put("content")),
                            ),
                    ),
            )
            .put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject()
                            .put("name", "android_system_tool")
                            .put(
                                "description",
                                            "Read Hermes Android phone/device status, open safe settings/setup panels, start/stop the background runtime, or perform explicit user-granted Shizuku/Sui app, permission, and connectivity toggle actions.",
                            )
                            .put(
                                "parameters",
                                JSONObject()
                                    .put("type", "object")
                                    .put(
                                        "properties",
                                        JSONObject()
                                            .put(
                                                "action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put(
                                                        "description",
                                                        "Use status to read device state, run_privileged_shell with a command argument, grant_runtime_permission/revoke_runtime_permission with package_name and permission, force_stop_app/clear_app_data/enable_app/disable_app/set_app_enabled with package_name, Shizuku connectivity toggles such as set_wifi_enabled/set_bluetooth_enabled/set_mobile_data_enabled/set_airplane_mode_enabled, set_wifi_tethering_enabled with enabled/state, set_dnd_mode with dnd_mode, set_power_save_mode with enabled/state, turn_screen_off, end_call, global navigation/statusbar actions, set_mobile_network_type with network_types_bitmask and optional slot_id, user/work-profile actions with user_id, custom setting actions set_custom_setting/get_custom_setting/delete_custom_setting with setting_namespace and setting_name, or one of the available actions returned in status.",
                                                    ),
                                            )
                                            .put(
                                                "command",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Shell command for action run_privileged_shell. Requires running Shizuku/Sui and user-granted Hermes permission."),
                                            )
                                            .put(
                                                "package_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android package name for Shizuku app-management actions such as force_stop_app, clear_app_data, enable_app, disable_app, set_app_enabled, grant_runtime_permission, and revoke_runtime_permission. Not needed for Shizuku Wi-Fi, Bluetooth, mobile-data, airplane-mode, Wi-Fi tethering, DND, power, global navigation, work-profile, mobile-network, or custom setting actions."),
                                            )
                                            .put(
                                                "permission",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android runtime permission name for grant_runtime_permission or revoke_runtime_permission, for example android.permission.POST_NOTIFICATIONS."),
                                            )
                                            .put(
                                                "enabled",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Desired enabled state for set_app_enabled, Shizuku connectivity toggles, set_wifi_tethering_enabled, or set_power_save_mode."),
                                            )
                                            .put(
                                                "setting_namespace",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android settings namespace for custom setting actions: system, secure, or global."),
                                            )
                                            .put(
                                                "setting_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android settings key for custom setting actions."),
                                            )
                                            .put(
                                                "setting_value",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Value for set_custom_setting."),
                                            )
                                            .put(
                                                "dnd_mode",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "DND mode for set_dnd_mode: on, none, priority, alarms, all, or off."),
                                            )
                                            .put(
                                                "user_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Android user/work-profile id for Shizuku start_user_profile, stop_user_profile, or switch_user_profile."),
                                            )
                                            .put(
                                                "network_types_bitmask",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Binary Android TelephonyManager network-type bitmask for set_mobile_network_type."),
                                            )
                                            .put(
                                                "slot_id",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional SIM slot id for set_mobile_network_type."),
                                            )
                                            .put(
                                                "timeout_seconds",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Optional timeout for run_privileged_shell and Shizuku-backed package-manager, setting, tethering, connectivity, DND, power, profile, mobile-network, or global navigation actions, clamped by Hermes."),
                                            )
                                    )
                                    .put("required", JSONArray().put("action")),
                            ),
                    ),
            )
    }

    private fun truncate(value: String): String {
        return if (value.length <= MAX_TOOL_RESULT_CHARS) {
            value
        } else {
            value.take(MAX_TOOL_RESULT_CHARS) + "\n[truncated]"
        }
    }

    private data class AssistantMessage(
        val content: String,
        val toolCalls: List<ToolCall>,
    ) {
        fun toJsonMessage(): JSONObject {
            val json = JSONObject()
                .put("role", "assistant")
                .put("content", content.ifBlank { JSONObject.NULL })
            if (toolCalls.isNotEmpty()) {
                json.put(
                    "tool_calls",
                    JSONArray().apply {
                        toolCalls.forEach { put(it.toJson()) }
                    },
                )
            }
            return json
        }

        companion object {
            fun fromJson(json: JSONObject): AssistantMessage {
                val toolCalls = mutableListOf<ToolCall>()
                val rawToolCalls = json.optJSONArray("tool_calls") ?: JSONArray()
                for (index in 0 until rawToolCalls.length()) {
                    val rawToolCall = rawToolCalls.optJSONObject(index) ?: continue
                    val function = rawToolCall.optJSONObject("function") ?: JSONObject()
                    val name = function.optString("name").ifBlank { "terminal_tool" }
                    val arguments = runCatching {
                        JSONObject(function.optString("arguments", "{}"))
                    }.getOrDefault(JSONObject())
                    toolCalls += ToolCall(
                        id = rawToolCall.optString("id").ifBlank { "call_${UUID.randomUUID()}_$index" },
                        name = name,
                        arguments = arguments,
                    )
                }
                return AssistantMessage(
                    content = json.optString("content").takeUnless { json.isNull("content") }.orEmpty(),
                    toolCalls = toolCalls,
                )
            }
        }
    }

    private data class ToolCall(
        val id: String,
        val name: String,
        val arguments: JSONObject,
    ) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("id", id)
                .put("type", "function")
                .put(
                    "function",
                    JSONObject()
                        .put("name", name)
                        .put("arguments", arguments.toString()),
                )
        }
    }

    private companion object {
        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
        private const val TOOL_TIMEOUT_SECONDS = 60
        private const val NATIVE_TOOL_GENERATION_TIMEOUT_MS = 300_000L
        private const val NATIVE_TOOL_MAX_TOKENS = 512
        private const val PRIVILEGED_TOOL_TIMEOUT_SECONDS = 30
        private const val MAX_TOOL_RESULT_CHARS = 12_000
        private const val DEFAULT_UI_SNAPSHOT_LIMIT = 80
        private val ANDROID_UI_ACTIONS = listOf(
            "status",
            "snapshot",
            "click",
            "long_click",
            "focus",
            "set_text",
            "scroll_forward",
            "scroll_backward",
            "back",
            "home",
            "recents",
            "notifications",
            "quick_settings",
            "open_accessibility_settings",
        )
        private val UI_SELECTOR_ARGUMENTS = listOf(
            "text_contains",
            "content_description_contains",
            "view_id",
            "package_name",
            "value",
            "index",
            "limit",
        )
    }
}
