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
                    "When the user asks about Android settings, phone connectivity, permissions, background runtime, or safe system panels, call android_system_tool. " +
                    "android_system_tool status includes Shizuku/Sui privileged-access state, and it can open Shizuku, wireless debugging, and developer settings setup flows. " +
                    "If Shizuku/Sui is running and the user granted Hermes permission, android_system_tool can run explicit ADB/root-identity shell commands with action run_privileged_shell and a command argument. " +
                    "When the user asks to create a recurring phone automation, reusable Android task, Tasker-like variable, phone-state trigger, app-foreground trigger, notification trigger, saved file action, safe saved Android settings action, saved visible-UI action, or saved app-launch action, call android_automation_tool. It can save shell, file-write, file-delete, safe Android system-action, accessibility UI-action, and app-launch tasks, run them manually, enable/disable/delete them, schedule interval tasks with Android alarms, run boot/power/battery/app-foreground/notification-posted triggers, and expand saved variables in commands, file content, UI selectors, package names, trigger packages, and notification event fields. " +
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
                                "Create, list, run, enable, disable, or delete saved Android automations and variables. Supports shell, file-write, file-delete, safe Android system-action, accessibility UI-action, and app-launch tasks; manual tasks; interval tasks; boot/power/battery/app-foreground/notification-posted phone-state triggers; and Tasker-style %VARIABLE expansion. Shizuku execution must be explicitly requested per shell task. App-foreground triggers require the user-enabled Hermes accessibility service. Notification-posted triggers require user-enabled Hermes notification access.",
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
                                                    .put("description", "list, create_shell_task, create_file_write_task, create_file_delete_task, create_system_action_task, create_ui_action_task, create_app_launch_task, run, run_trigger, run_app_foreground_trigger, run_notification_posted_trigger, delete, enable, disable, list_variables, set_variable, get_variable, or delete_variable."),
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
                                                    .put("description", "Shell command for create_shell_task, alternate system action value for create_system_action_task, or alternate package name for create_app_launch_task. Saved variables can be referenced as %NAME or {{NAME}}."),
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
                                                    .put("description", "Package name for create_app_launch_task, or saved UI selector package-name fragment for create_ui_action_task. Saved variables can be referenced as %NAME or {{NAME}}."),
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
                                                    .put("description", "Optional trigger for create_*_task or run_trigger: manual, boot, power_connected, power_disconnected, battery_low, battery_okay, app_foreground, or notification_posted. interval_minutes creates an interval trigger. app_foreground and notification_posted require trigger_package_name and the relevant Android service permission."),
                                            )
                                            .put(
                                                "trigger_package_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Package name that must become foreground for app_foreground trigger records or post a notification for notification_posted trigger records, such as com.android.settings. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "notification_title",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional notification title for run_notification_posted_trigger tests. Real listener events also expose %NOTIFICATION_TITLE."),
                                            )
                                            .put(
                                                "notification_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional notification text for run_notification_posted_trigger tests. Real listener events also expose %NOTIFICATION_TEXT."),
                                            )
                                            .put(
                                                "interval_minutes",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Optional interval schedule in minutes. Minimum is 15. Omit for a manual task."),
                                            )
                                            .put(
                                                "enabled",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Whether a created automation starts enabled."),
                                            )
                                            .put(
                                                "use_shizuku",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "Run this automation through Shizuku/Sui privileged shell. Requires Shizuku setup and user-granted permission."),
                                            )
                                            .put(
                                                "name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Variable name for set_variable, get_variable, or delete_variable. Leading % is optional."),
                                            )
                                            .put(
                                                "value",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Variable value for set_variable, or text value for create_ui_action_task set_text."),
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
                                "Read Hermes Android phone/device status or perform a safe system action such as opening settings panels, Shizuku/wireless-debugging setup, or starting/stopping the background runtime.",
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
                                                        "Use status to read device state, run_privileged_shell with a command argument for user-granted Shizuku/Sui shell execution, or one of the available system actions returned in status.",
                                                    ),
                                            )
                                            .put(
                                                "command",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Shell command for action run_privileged_shell. Requires running Shizuku/Sui and user-granted Hermes permission."),
                                            )
                                            .put(
                                                "timeout_seconds",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Optional timeout for run_privileged_shell, clamped by Hermes."),
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
