package com.nousresearch.hermesagent.ui.chat

import android.content.Context
import com.nousresearch.hermesagent.api.ChatContentPart
import com.nousresearch.hermesagent.api.ChatMessage
import com.nousresearch.hermesagent.api.HermesApiClient
import com.nousresearch.hermesagent.api.toJsonObject
import com.nousresearch.hermesagent.device.HermesAccessibilityController
import com.nousresearch.hermesagent.device.HermesAccessibilityUiBridge
import com.nousresearch.hermesagent.device.HermesAppControlBridge
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
    private val openGuiWorkingMemoryPrefs = appContext.getSharedPreferences("hermes_opengui_working_memory", Context.MODE_PRIVATE)
    private var activeOpenGuiMemorySessionId: String = ""
    private val openGuiActionHistory = OpenGuiActionHistory()

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
        require(normalizedBaseUrl.startsWith("http://") || normalizedBaseUrl.startsWith("https://")) {
            "Native tool chat requires a local HTTP base URL; got '${baseUrl.ifBlank { "<blank>" }}'."
        }
        activeOpenGuiMemorySessionId = sessionId
        executeExplicitDirectToolRequest(userText)?.let { return it }

        var executedToolCalls = 0
        var latestToolResult = ""
        val initialToolSpecs = compactToolSpecsFor(userText)
        val messages = JSONArray()
            .put(systemMessage(toolsEnabled = initialToolSpecs.length() > 0))
            .put(
                ChatMessage(
                    role = "user",
                    content = userText,
                    contentParts = userContentParts,
                ).toJsonObject()
            )
        var assistant = postChatCompletion(
            normalizedBaseUrl = normalizedBaseUrl,
            modelName = modelName,
            sessionId = sessionId,
            messages = messages,
            toolSpecs = initialToolSpecs,
            maxTokens = NATIVE_TOOL_MAX_TOKENS,
        )

        repeat(MAX_NATIVE_TOOL_ROUNDS) {
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
                toolSpecs = initialToolSpecs,
                maxTokens = NATIVE_TOOL_MAX_TOKENS,
            )
            if (followUp.toolCalls.isEmpty()) {
                return Result(
                    content = followUp.content.ifBlank { toolCompletionReply(latestToolResult) },
                    executedToolCalls = executedToolCalls,
                )
            }
            assistant = followUp
        }
        return Result(
            content = toolCompletionReply(latestToolResult),
            executedToolCalls = executedToolCalls,
        )
    }

    private fun executeExplicitDirectToolRequest(userText: String): Result? {
        val command = extractExactTerminalCommand(userText) ?: return null
        val toolResult = executeTerminalTool(
            ToolCall(
                id = "direct_${UUID.randomUUID()}",
                name = "terminal_tool",
                arguments = JSONObject()
                    .put("command", command)
                    .put("timeout_seconds", TOOL_TIMEOUT_SECONDS),
            )
        )
        return Result(
            content = toolCompletionReply(toolResult),
            executedToolCalls = 1,
        )
    }

    private fun extractExactTerminalCommand(userText: String): String? {
        val lower = userText.lowercase()
        if ("terminal_tool" !in lower) {
            return null
        }
        val markers = listOf("run exactly this command:", "run exactly:")
        val marker = markers.firstOrNull { it in lower } ?: return null
        val markerIndex = lower.indexOf(marker)
        if (markerIndex < 0) {
            return null
        }
        val start = markerIndex + marker.length
        val tail = userText.substring(start).trim()
        val endMarkers = listOf(". After terminal_tool", "\nAfter terminal_tool", " After terminal_tool")
        val end = endMarkers
            .map { tail.indexOf(it) }
            .filter { it >= 0 }
            .minOrNull()
            ?: tail.length
        return tail.substring(0, end)
            .trim()
            .trim('`')
            .takeIf { it.isNotBlank() }
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
        toolSpecs: JSONArray?,
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
        if (toolSpecs != null && toolSpecs.length() > 0) {
            payload.put("tools", toolSpecs)
        }

        val request = Request.Builder()
            .url("$normalizedBaseUrl/v1/chat/completions")
            .header(HermesApiClient.SESSION_HEADER, sessionId)
            .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()

        httpClient.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw IllegalStateException(formatNativeChatError(response.code, body))
            }
            val root = JSONObject(body)
            val message = root
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
            return AssistantMessage.fromJson(message)
        }
    }

    private fun formatNativeChatError(statusCode: Int, body: String): String {
        val parsedMessage = runCatching {
            val root = JSONObject(body)
            val error = root.optJSONObject("error")
            error?.optString("message")?.takeIf { it.isNotBlank() }
                ?: root.optString("message").takeIf { it.isNotBlank() }
        }.getOrNull().orEmpty()
        val diagnostic = parsedMessage.ifBlank { body }.trim()
        val lower = diagnostic.lowercase()
        if ("exceed_context_size" in lower || "exceeds the available context size" in lower) {
            return "The local model ran out of context. Hermes now uses a compact Android tool prompt, but this model still could not fit the request. Start a new chat, clear history, or choose a model with a larger context window."
        }
        return if (diagnostic.isNotBlank()) {
            "Native chat request failed ($statusCode): ${diagnostic.take(MAX_NATIVE_ERROR_CHARS)}"
        } else {
            "Native chat request failed ($statusCode)."
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
        val rawAction = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            .orEmpty()
        val action = rawAction.lowercase()
        return when (action.ifBlank { "status" }) {
            "status", "read_status" -> androidUiStatusJson()
            "snapshot", "screen_snapshot", "read_screen" -> executeAndroidSnapshotTool(toolCall)
            "screenshot",
            "screen_image",
            "visual_snapshot",
            "capture_screenshot" -> executeAndroidScreenshotTool(toolCall)
            "opengui_history",
            "open_gui_history",
            "gui_history",
            "semantic_history",
            "action_history" -> openGuiActionHistory.snapshotJson().toString()
            "clear_opengui_history",
            "clear_open_gui_history",
            "reset_gui_history" -> openGuiActionHistory.clearJson().toString()
            "parse_opengui_action",
            "parse_open_gui_action",
            "parse_gui_action" -> executeOpenGuiActionTool(toolCall, parseOnly = true)
            "opengui_action",
            "open_gui_action",
            "gui_action",
            "vlm_action",
            "execute_opengui_action",
            "execute_gui_action" -> executeOpenGuiActionTool(toolCall, parseOnly = false)
            "tap",
            "tap_at",
            "coordinate_tap",
            "coordinate_click",
            "click_at" -> executeAndroidCoordinateGesture(toolCall, "tap")
            "long_press",
            "long_press_at",
            "coordinate_long_press" -> executeAndroidCoordinateGesture(toolCall, "long_press")
            "swipe",
            "drag",
            "coordinate_swipe" -> executeAndroidCoordinateGesture(toolCall, "swipe")
            "scroll",
            "scroll_up",
            "scroll_down",
            "scroll_left",
            "scroll_right" -> executeAndroidScrollGesture(toolCall, action)
            "type",
            "type_text" -> HermesAccessibilityUiBridge.performTextInputJson(
                value = stringArgument(toolCall.arguments, "value", "text", "content").orEmpty(),
                textContains = toolCall.arguments.optString("text_contains"),
                contentDescriptionContains = toolCall.arguments.optString("content_description_contains"),
                viewId = toolCall.arguments.optString("view_id"),
                packageName = toolCall.arguments.optString("package_name"),
                className = toolCall.arguments.optString("class_name"),
                index = toolCall.arguments.optInt("index", 0),
            )
            "click" -> if (hasCoordinateGestureArguments(toolCall.arguments)) {
                executeAndroidCoordinateGesture(toolCall, "tap")
            } else {
                executeAndroidSelectorAction(toolCall, action)
            }
            "long_click",
            "focus",
            "scroll_forward",
            "scroll_backward",
            "set_text" -> executeAndroidSelectorAction(toolCall, action)
            "back", "global_back", "press_back" -> HermesAccessibilityUiBridge.performGlobalActionJson("back")
            "home", "global_home", "press_home" -> HermesAccessibilityUiBridge.performGlobalActionJson("home")
            "recents", "global_recents" -> HermesAccessibilityUiBridge.performGlobalActionJson("recents")
            "notifications", "global_notifications" -> HermesAccessibilityUiBridge.performGlobalActionJson("notifications")
            "quick_settings", "global_quick_settings" -> HermesAccessibilityUiBridge.performGlobalActionJson("quick_settings")
            "open_app", "launch_app" -> HermesAppControlBridge.launchApp(
                context = appContext,
                packageName = stringArgument(
                    toolCall.arguments,
                    "package_name",
                    "packageName",
                    "package",
                    "app_package",
                    "application_id",
                ).orEmpty(),
                appName = stringArgument(toolCall.arguments, "app_name", "appName", "application_name", "label").orEmpty(),
            ).toString()
            "open_accessibility_settings" -> HermesSystemControlBridge.performActionJson("open_accessibility_settings")
            else -> if (rawAction.looksLikeOpenGuiAction()) {
                executeOpenGuiActionTool(toolCall, parseOnly = false, fallbackRawAction = rawAction)
            } else {
                JSONObject()
                    .put("success", false)
                    .put("error", "Unsupported Android UI action: $action")
                    .put("available_ui_actions", JSONArray(ANDROID_UI_ACTIONS))
                    .toString()
            }
        }
    }

    private fun executeAndroidSnapshotTool(toolCall: ToolCall): String {
        val result = HermesAccessibilityUiBridge.snapshotJson(
            limit = toolCall.arguments.optInt("limit", DEFAULT_UI_SNAPSHOT_LIMIT),
        )
        rememberOpenGuiScreenHashFromResult(result)
        return result
    }

    private fun executeAndroidScreenshotTool(toolCall: ToolCall): String {
        val arguments = toolCall.arguments
        val result = HermesAccessibilityUiBridge.captureScreenshotJson(
            saveFile = optionalBooleanArgument(arguments, "save_file", "save", "persist") ?: true,
            includeBase64 = optionalBooleanArgument(arguments, "include_base64", "base64", "inline_image") ?: false,
            maxImageEdgePx = optionalIntArgument(arguments, "max_image_edge_px", "max_edge_px", "max_edge") ?: 0,
        )
        rememberOpenGuiScreenHashFromResult(result)
        return result
    }

    private fun executeOpenGuiActionTool(
        toolCall: ToolCall,
        parseOnly: Boolean,
        fallbackRawAction: String = "",
    ): String {
        val rawAction = stringArgument(
            toolCall.arguments,
            "raw_action",
            "action_text",
            "prediction",
            "vlm_prediction",
            "open_gui_action",
            "opengui_action",
            "screen_hash",
            "snapshot_hash",
            "ui_state_hash",
            "screenshot_hash",
            "phash",
        )
            ?: fallbackRawAction.takeIf { it.isNotBlank() }
            ?: return JSONObject()
                .put("success", false)
                .put("error", "OpenGUI action compatibility requires raw_action, action_text, prediction, or a function-call action string")
                .put("available_ui_actions", JSONArray(ANDROID_UI_ACTIONS))
                .toString()

        val parsed = runCatching { OpenGuiActionCompat.parse(rawAction) }.getOrElse { error ->
            return JSONObject()
                .put("success", false)
                .put("error", error.message ?: error.javaClass.simpleName)
                .put("opengui_action_compat", true)
                .put("raw_action", rawAction)
                .toString()
        }
        rememberOpenGuiScreenHash(
            stringArgument(
                toolCall.arguments,
                "screen_hash",
                "snapshot_hash",
                "ui_state_hash",
                "screenshot_hash",
                "phash",
            ).orEmpty(),
        )
        if (parseOnly) {
            return JSONObject()
                .put("success", true)
                .put("opengui_action_compat", true)
                .put("parse_only", true)
                .put("parsed_action", parsed.toJson())
                .put("execution_review_supported", true)
                .put("screen_review_supported", true)
                .put("recent_screen_hash_count", openGuiActionHistory.screenHashList().size)
                .toString()
        }

        val review = OpenGuiExecutionReview.review(
            recentActions = openGuiActionHistory.actionsList(),
            nextAction = parsed,
            recentScreenHashes = openGuiActionHistory.screenHashList(),
        )
        if (review.detected) {
            return OpenGuiExecutionReview.blockedActionJson(parsed, review).toString()
        }
        val result = when (parsed.actionType) {
            "click" -> executeParsedOpenGuiTap(parsed, "tap")
            "long_press" -> executeParsedOpenGuiTap(parsed, "long_press")
            "swipe" -> executeParsedOpenGuiSwipe(parsed)
            "scroll" -> executeParsedOpenGuiScroll(parsed)
            "type" -> HermesAccessibilityUiBridge.performTextInputJson(
                value = parsed.content,
                textContains = "",
                contentDescriptionContains = "",
                viewId = "",
                packageName = "",
                className = "",
                index = 0,
            )
            "open_app" -> HermesAppControlBridge.launchApp(
                context = appContext,
                packageName = parsed.packageName,
                appName = parsed.appName,
            ).toString()
            "press_back" -> HermesAccessibilityUiBridge.performGlobalActionJson("back")
            "press_home" -> HermesAccessibilityUiBridge.performGlobalActionJson("home")
            "wait" -> executeParsedOpenGuiWait(parsed)
            "finished" -> JSONObject()
                .put("success", true)
                .put("action", "finished")
                .put("terminal", true)
                .put("message", "OpenGUI action marked the mobile task finished.")
                .toString()
            "call_user" -> OpenGuiUserHandoff.execute(
                context = appContext,
                parsed = parsed,
                sessionId = activeOpenGuiMemorySessionId,
            ).toString()
            "update_working_memory" -> executeParsedOpenGuiWorkingMemoryUpdate(parsed)
            "get_working_memory" -> executeParsedOpenGuiWorkingMemoryRead()
            "request_visual",
            "downgrade_to_a11y" -> JSONObject()
                .put("success", true)
                .put("action", parsed.actionType)
                .put("message", "Parsed OpenGUI recovery/text-side action; Hermes stayed on the accessibility UI path and no gesture was needed.")
                .toString()
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported parsed OpenGUI action type: ${parsed.actionType}")
                .put("available_ui_actions", JSONArray(ANDROID_UI_ACTIONS))
                .toString()
        }
        recordOpenGuiAction(parsed)
        return attachOpenGuiParsedAction(result, parsed, review)
    }

    private fun rememberOpenGuiScreenHashFromResult(result: String) {
        runCatching {
            val json = JSONObject(result)
            rememberOpenGuiScreenHash(
                stringArgument(json, "ui_state_hash", "screen_hash", "snapshot_hash", "screenshot_hash", "phash").orEmpty(),
            )
        }
    }

    private fun rememberOpenGuiScreenHash(hash: String) {
        openGuiActionHistory.rememberScreenHash(hash)
    }

    private fun recordOpenGuiAction(parsed: ParsedOpenGuiAction) {
        openGuiActionHistory.recordAction(parsed)
    }

    private fun executeParsedOpenGuiTap(parsed: ParsedOpenGuiAction, action: String): String {
        val point = parsed.startCoords
            ?: return JSONObject()
                .put("success", false)
                .put("error", "Parsed OpenGUI $action action did not include start_box, point, x/y, or equivalent coordinates")
                .toString()
        return HermesAccessibilityUiBridge.performCoordinateGestureJson(
            action = action,
            x = point.x,
            y = point.y,
            x1 = null,
            y1 = null,
            x2 = null,
            y2 = null,
            durationMs = parsed.durationMs ?: 0L,
            coordinateSpace = "normalized",
        )
    }

    private fun executeParsedOpenGuiSwipe(parsed: ParsedOpenGuiAction): String {
        val start = parsed.startCoords
            ?: return JSONObject()
                .put("success", false)
                .put("error", "Parsed OpenGUI swipe action did not include start coordinates")
                .toString()
        val end = parsed.endCoords
            ?: return JSONObject()
                .put("success", false)
                .put("error", "Parsed OpenGUI swipe action did not include end coordinates")
                .toString()
        return HermesAccessibilityUiBridge.performCoordinateGestureJson(
            action = "swipe",
            x = null,
            y = null,
            x1 = start.x,
            y1 = start.y,
            x2 = end.x,
            y2 = end.y,
            durationMs = parsed.durationMs ?: 0L,
            coordinateSpace = "normalized",
        )
    }

    private fun executeParsedOpenGuiScroll(parsed: ParsedOpenGuiAction): String {
        val start = parsed.startCoords
        return HermesAccessibilityUiBridge.performScrollGestureJson(
            direction = parsed.direction,
            x = start?.x,
            y = start?.y,
            distancePx = null,
            durationMs = parsed.durationMs ?: 0L,
            coordinateSpace = if (start != null) "normalized" else "",
        )
    }

    private fun executeParsedOpenGuiWait(parsed: ParsedOpenGuiAction): String {
        val duration = (parsed.durationMs ?: 1_000L).coerceIn(0L, 5_000L)
        if (duration > 0L) {
            Thread.sleep(duration)
        }
        return JSONObject()
            .put("success", true)
            .put("action", "wait")
            .put("duration_ms", duration)
            .put("message", "Waited for $duration ms.")
            .toString()
    }

    private fun attachOpenGuiParsedAction(
        result: String,
        parsed: ParsedOpenGuiAction,
        review: OpenGuiExecutionReviewResult? = null,
    ): String {
        val json = runCatching { JSONObject(result) }.getOrElse {
            JSONObject()
                .put("success", false)
                .put("raw_result", result)
        }
        return json
            .put("opengui_action_compat", true)
            .put("parsed_action", parsed.toJson())
            .also { output ->
                review?.let { output.put("execution_review", it.toJson()) }
            }
            .toString()
    }

    private fun executeParsedOpenGuiWorkingMemoryUpdate(parsed: ParsedOpenGuiAction): String {
        val content = parsed.content.trim()
        if (content.isBlank()) {
            return JSONObject()
                .put("success", false)
                .put("action", "update_working_memory")
                .put("error", "update_working_memory requires content, text, memory, or value")
                .toString()
        }
        val key = openGuiWorkingMemoryKey()
        val existing = openGuiWorkingMemoryPrefs.getString(key, "").orEmpty().trim()
        val next = listOf(existing, content)
            .filter { it.isNotBlank() }
            .joinToString("\n")
            .takeLast(MAX_OPEN_GUI_WORKING_MEMORY_CHARS)
        openGuiWorkingMemoryPrefs.edit().putString(key, next).apply()
        return JSONObject()
            .put("success", true)
            .put("action", "update_working_memory")
            .put("stored_chars", content.length)
            .put("memory_chars", next.length)
            .put("max_memory_chars", MAX_OPEN_GUI_WORKING_MEMORY_CHARS)
            .toString()
    }

    private fun executeParsedOpenGuiWorkingMemoryRead(): String {
        val content = openGuiWorkingMemoryPrefs.getString(openGuiWorkingMemoryKey(), "").orEmpty()
        return JSONObject()
            .put("success", true)
            .put("action", "get_working_memory")
            .put("content", content)
            .put("memory_chars", content.length)
            .put("max_memory_chars", MAX_OPEN_GUI_WORKING_MEMORY_CHARS)
            .toString()
    }

    private fun openGuiWorkingMemoryKey(): String {
        val sessionKey = activeOpenGuiMemorySessionId.ifBlank { "default" }
        return "session_$sessionKey"
    }

    private fun executeAndroidCoordinateGesture(toolCall: ToolCall, action: String): String {
        val arguments = toolCall.arguments
        return HermesAccessibilityUiBridge.performCoordinateGestureJson(
            action = action,
            x = optionalDoubleArgument(arguments, "x", "screen_x", "tap_x"),
            y = optionalDoubleArgument(arguments, "y", "screen_y", "tap_y"),
            x1 = optionalDoubleArgument(arguments, "x1", "start_x", "from_x"),
            y1 = optionalDoubleArgument(arguments, "y1", "start_y", "from_y"),
            x2 = optionalDoubleArgument(arguments, "x2", "end_x", "to_x"),
            y2 = optionalDoubleArgument(arguments, "y2", "end_y", "to_y"),
            durationMs = optionalLongArgument(arguments, "duration_ms", "duration", "gesture_duration_ms") ?: 0L,
            coordinateSpace = stringArgument(arguments, "coordinate_space", "coordinates", "coord_space").orEmpty(),
        )
    }

    private fun hasCoordinateGestureArguments(arguments: JSONObject): Boolean {
        return optionalDoubleArgument(arguments, "x", "screen_x", "tap_x") != null ||
            optionalDoubleArgument(arguments, "y", "screen_y", "tap_y") != null ||
            optionalDoubleArgument(arguments, "x1", "start_x", "from_x") != null ||
            optionalDoubleArgument(arguments, "y1", "start_y", "from_y") != null
    }

    private fun executeAndroidSelectorAction(toolCall: ToolCall, action: String): String {
        return HermesAccessibilityUiBridge.performActionJson(
            action = action,
            textContains = toolCall.arguments.optString("text_contains"),
            contentDescriptionContains = toolCall.arguments.optString("content_description_contains"),
            viewId = toolCall.arguments.optString("view_id"),
            packageName = toolCall.arguments.optString("package_name"),
            className = toolCall.arguments.optString("class_name"),
            value = toolCall.arguments.optString("value"),
            index = toolCall.arguments.optInt("index", 0),
        )
    }

    private fun executeAndroidScrollGesture(toolCall: ToolCall, action: String): String {
        val arguments = toolCall.arguments
        val direction = when (action) {
            "scroll_down" -> "down"
            "scroll_left" -> "left"
            "scroll_right" -> "right"
            "scroll_up" -> "up"
            else -> stringArgument(arguments, "direction", "scroll_direction").orEmpty()
        }
        return HermesAccessibilityUiBridge.performScrollGestureJson(
            direction = direction,
            x = optionalDoubleArgument(arguments, "x", "start_x"),
            y = optionalDoubleArgument(arguments, "y", "start_y"),
            distancePx = optionalDoubleArgument(arguments, "distance_px", "distance", "scroll_distance_px"),
            durationMs = optionalLongArgument(arguments, "duration_ms", "duration", "gesture_duration_ms") ?: 0L,
            coordinateSpace = stringArgument(arguments, "coordinate_space", "coordinates", "coord_space").orEmpty(),
        )
    }

    private fun androidUiStatusJson(): String {
        return JSONObject()
            .put("accessibility_service_enabled", HermesAccessibilityController.isServiceEnabled(appContext))
            .put("accessibility_connected", HermesAccessibilityController.isServiceConnected())
            .put("available_ui_actions", JSONArray(ANDROID_UI_ACTIONS))
            .put("selection_arguments", JSONArray(UI_SELECTOR_ARGUMENTS))
            .put("coordinate_arguments", JSONArray(UI_COORDINATE_ARGUMENTS))
            .put("opengui_action_arguments", JSONArray(OPEN_GUI_ACTION_ARGUMENTS))
            .put("snapshot_hash_support", true)
            .put("visual_screenshot_hash_support", true)
            .put("screenshot_capture_supported", android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R)
            .put("screenshot_capture_api_level", 30)
            .put("screenshot_file_output", true)
            .put("opengui_screen_review_supported", true)
            .put("opengui_history_supported", true)
            .put("opengui_user_handoff_supported", true)
            .put("opengui_user_handoff_surfaces", JSONArray(listOf("notification", "toast", "vibration")))
            .put("recent_opengui_action_count", openGuiActionHistory.actionsList().size)
            .put("recent_snapshot_hash_count", openGuiActionHistory.screenHashList().size)
            .put("active_package", HermesAccessibilityController.currentForegroundPackageName())
            .put("current_app_name", HermesAccessibilityController.currentForegroundPackageName())
            .put("normalized_coordinate_support", true)
            .also { json ->
                HermesAccessibilityController.screenMetrics()?.let { metrics ->
                    json.put("screen_width", metrics.width)
                    json.put("screen_height", metrics.height)
                    json.put("density", metrics.density.toDouble())
                    json.put("scale_factor", 1.0)
                }
            }
            .put("message", "Enable the Hermes accessibility service before using snapshot, selector actions, coordinate tap/swipe, or global navigation actions.")
            .toString()
    }

    private fun stringArgument(arguments: JSONObject, vararg keys: String): String? {
        return keys.firstNotNullOfOrNull { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                null
            } else {
                arguments.optString(key).takeIf { it.isNotBlank() }
            }
        }
    }

    private fun optionalDoubleArgument(arguments: JSONObject, vararg keys: String): Double? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is Number -> value.toDouble()
            else -> value?.toString()?.trim()?.toDoubleOrNull()
        }
    }

    private fun optionalIntArgument(arguments: JSONObject, vararg keys: String): Int? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is Number -> value.toInt()
            else -> value?.toString()?.trim()?.toIntOrNull()
        }
    }

    private fun optionalLongArgument(arguments: JSONObject, vararg keys: String): Long? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is Number -> value.toLong()
            else -> value?.toString()?.trim()?.toLongOrNull()
        }
    }

    private fun optionalBooleanArgument(arguments: JSONObject, vararg keys: String): Boolean? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is Boolean -> value
            is Number -> value.toInt() != 0
            else -> when (value?.toString()?.trim()?.lowercase()) {
                "1", "true", "yes", "on", "enabled" -> true
                "0", "false", "no", "off", "disabled" -> false
                else -> null
            }
        }
    }

    private fun String.looksLikeOpenGuiAction(): Boolean {
        return contains('(') &&
            contains(')') &&
            Regex("""^[A-Za-z_][A-Za-z0-9_]*\s*\(""").containsMatchIn(trim())
    }

    private fun systemMessage(toolsEnabled: Boolean): JSONObject {
        val content = if (toolsEnabled) {
            "You are Hermes running inside the native Android app. " +
                "Use tools when work requires real files, shell commands, Android UI, Android settings, Shizuku/Sui, or saved Tasker-style automation. " +
                "Use terminal_tool for shell commands and inspection, file_write_tool for exact text file creation, android_ui_tool for visible-screen selectors and coordinate gestures, android_system_tool for device/settings/Shizuku operations, and android_automation_tool for saved tasks, triggers, notifications, variables, widgets, and Tasker/Locale plugin actions. " +
                "If a planner emits OpenGUI-style raw GUI actions such as click(start_box=...), pass that text to android_ui_tool action=parse_opengui_action or action=opengui_action; call_user/need_login/asset_risk/delete_confirm are surfaced as a visible phone handoff. " +
                "When the user asks to write or replace multiline text, prefer file_write_tool so multiline content is written exactly; file_write_tool can only write inside the Hermes app workspace. " +
                "Ask for or report missing Android permissions instead of pretending protected settings changed. Keep replies brief and report real tool results."
        } else {
            "You are Hermes running inside the native Android app. Keep replies brief and direct."
        }
        return JSONObject()
            .put("role", "system")
            .put("content", content)
    }

    private fun compactToolSpecs(): JSONArray {
        return JSONArray()
            .put(
                functionSpec(
                    name = "terminal_tool",
                    description = "Run /system/bin/sh in the Hermes workspace.",
                    properties = JSONObject()
                        .put("command", stringProp("Shell command."))
                        .put("timeout_seconds", intProp("Optional timeout.")),
                    required = JSONArray().put("command"),
                ),
            )
            .put(
                functionSpec(
                    name = "file_write_tool",
                    description = "Write UTF-8 text inside the Hermes workspace.",
                    properties = JSONObject()
                        .put("path", stringProp("Workspace path."))
                        .put("content", stringProp("Exact text content."))
                        .put("append", boolProp("Append instead of replace.")),
                    required = JSONArray().put("path").put("content"),
                ),
            )
            .put(
                functionSpec(
                    name = "android_system_tool",
                    description = "Read phone state, open safe settings panels, or run explicit Shizuku/Sui actions.",
                    properties = JSONObject()
                        .put("action", stringProp("status, run_privileged_shell, open_*_settings, grant/revoke permission, force_stop_app, clear_app_data, enable/disable app, connectivity/DND/power/profile/custom-setting actions."))
                        .put("command", stringProp("Command for run_privileged_shell."))
                        .put("package_name", stringProp("Android package name."))
                        .put("permission", stringProp("Android permission."))
                        .put("enabled", boolProp("Desired enabled state."))
                        .put("setting_namespace", stringProp("system, secure, or global."))
                        .put("setting_name", stringProp("Android settings key."))
                        .put("setting_value", stringProp("Android settings value."))
                        .put("dnd_mode", stringProp("DND mode."))
                        .put("user_id", stringProp("Android user/profile id."))
                        .put("network_types_bitmask", stringProp("Mobile network bitmask."))
                        .put("slot_id", stringProp("SIM slot id."))
                        .put("timeout_seconds", intProp("Optional timeout.")),
                    required = JSONArray().put("action"),
                ),
            )
            .put(
                functionSpec(
                    name = "android_ui_tool",
                    description = "Inspect or control the visible Android UI through Hermes accessibility. OpenGUI-compatible execution includes local repeated-action and screen-state review guards that can return requires_replan before a likely loop continues, plus user-visible call_user handoff notifications/toasts/vibration.",
                    properties = JSONObject()
                        .put("action", stringProp("status, snapshot, screenshot, visual_snapshot, parse_opengui_action, opengui_action, click, long_click, focus, set_text, type, scroll_forward, scroll_backward, scroll, scroll_up, scroll_down, scroll_left, scroll_right, tap, long_press, swipe, drag, open_app, launch_app, back, home, press_back, press_home, recents, notifications, quick_settings, open_accessibility_settings."))
                        .put("raw_action", stringProp("OpenGUI-style VLM action text for parse_opengui_action or opengui_action, such as Action: click(start_box='<point>500 250</point>')."))
                        .put("screen_hash", stringProp("Optional OpenGUI pHash or Hermes snapshot ui_state_hash for screen-state loop review."))
                        .put("text_contains", stringProp("Visible text selector."))
                        .put("content_description_contains", stringProp("Accessibility description selector."))
                        .put("view_id", stringProp("Android view id selector."))
                        .put("package_name", stringProp("Package filter for selectors, or package name for action=open_app."))
                        .put("app_name", stringProp("OpenGUI-style launcher app label for action=open_app when package_name is unknown."))
                        .put("class_name", stringProp("Android accessibility class-name filter, such as EditText, Button, or RecyclerView."))
                        .put("value", stringProp("Text for set_text."))
                        .put("index", intProp("Zero-based match index."))
                        .put("limit", intProp("Snapshot node limit."))
                        .put("x", scalarProp("Tap x coordinate, or swipe start x when x1 is omitted."))
                        .put("y", scalarProp("Tap y coordinate, or swipe start y when y1 is omitted."))
                        .put("x1", scalarProp("Swipe start x coordinate."))
                        .put("y1", scalarProp("Swipe start y coordinate."))
                        .put("x2", scalarProp("Swipe end x coordinate."))
                        .put("y2", scalarProp("Swipe end y coordinate."))
                        .put("coordinate_space", stringProp("absolute_px by default, or normalized/percent."))
                        .put("duration_ms", intProp("Gesture duration in milliseconds."))
                        .put("direction", stringProp("Scroll finger direction: up, down, left, or right."))
                        .put("distance_px", scalarProp("Optional scroll distance in screen pixels."))
                        .put("save_file", boolProp("For screenshot/visual_snapshot, save PNG in the Hermes app files directory. Defaults true."))
                        .put("include_base64", boolProp("For screenshot/visual_snapshot, include base64 PNG bytes inline. Defaults false to keep tool results small."))
                        .put("max_image_edge_px", intProp("For screenshot/visual_snapshot, resize the longest image edge before returning or saving. The response includes screen_width, screen_height, scale_factor, and a 64-bit visual screen_hash for OpenGUI-style progress review.")),
                    required = JSONArray().put("action"),
                ),
            )
            .put(
                functionSpec(
                    name = "android_automation_tool",
                    description = "Open URLs/files immediately or create, run, manage, import, or trigger saved Hermes/Tasker-style Android automations.",
                    properties = JSONObject()
                        .put("action", stringProp("open_uri/open_url/open_browser for immediate browser/file launch; list, run, delete, enable, disable, export/import, import_tasker_xml, create_*_task, set/get/delete_variable, watcher status/start/stop/scan, run_*_trigger, widget/tile actions."))
                        .put("id", stringProp("Automation id."))
                        .put("label", stringProp("Automation label."))
                        .put("command", stringProp("Shell/system/intent command."))
                        .put("path", stringProp("Workspace path."))
                        .put("content", stringProp("File content."))
                        .put("append", boolProp("Append file content."))
                        .put("intent_task_action", stringProp("Intent mode: open_uri, start_activity, or send_broadcast."))
                        .put("data_uri", stringProp("URL or Hermes workspace file path for open_uri/open_browser."))
                        .put("intent_action", stringProp("Android intent action."))
                        .put("system_action", stringProp("Safe Android system action."))
                        .put("ui_action", stringProp("Saved UI action."))
                        .put("shizuku_action", stringProp("Saved Shizuku/Sui action."))
                        .put("notification_action", stringProp("post, update, cancel, or clear."))
                        .put("notification_id", scalarProp("Notification id."))
                        .put("notification_title", stringProp("Notification title."))
                        .put("notification_text", stringProp("Notification text."))
                        .put("status_text", stringProp("Short notification status text."))
                        .put("progress_value", scalarProp("Notification progress value."))
                        .put("progress_max", scalarProp("Notification progress max."))
                        .put("progress_indeterminate", boolProp("Indeterminate progress."))
                        .put("package_name", stringProp("Android package name."))
                        .put("class_name", stringProp("Android class name for intents, or accessibility class-name filter for saved UI actions."))
                        .put("permission", stringProp("Android permission."))
                        .put("enabled", boolProp("Automation enabled state."))
                        .put("target_enabled", boolProp("Target enabled state."))
                        .put("trigger", stringProp("manual, time, boot, power, app, notification, calendar, location, sensor, logcat, external, or Shizuku trigger."))
                        .put("time", stringProp("Time trigger such as 08:30."))
                        .put("days_of_week", stringProp("Day filter."))
                        .put("name", stringProp("Variable name."))
                        .put("value", stringProp("Variable or UI value."))
                        .put("clipboard_text", stringProp("Clipboard text."))
                        .put("toast_text", stringProp("Toast or Tasker Flash text."))
                        .put("latitude", scalarProp("Latitude."))
                        .put("longitude", scalarProp("Longitude."))
                        .put("radius_meters", scalarProp("Location radius."))
                        .put("tasker_xml", stringProp("Tasker XML to import."))
                        .put("tasker_data_uri", stringProp("Tasker data URI to import."))
                        .put("bundle_json", stringProp("Hermes automation bundle JSON.")),
                    required = JSONArray().put("action"),
                ),
            )
    }

    private fun compactToolSpecsFor(userText: String): JSONArray {
        val selectedNames = explicitlyRequestedToolNames(userText)
            .ifEmpty { inferredToolNames(userText) }
        if (selectedNames.isEmpty()) {
            return JSONArray()
        }
        val allTools = compactToolSpecs()
        return JSONArray().apply {
            for (index in 0 until allTools.length()) {
                val tool = allTools.optJSONObject(index) ?: continue
                val name = tool.optJSONObject("function")?.optString("name").orEmpty()
                if (name in selectedNames) {
                    put(tool)
                }
            }
        }
    }

    private fun inferredToolNames(userText: String): Set<String> {
        val lower = userText.lowercase()
        return buildSet {
            if (
                listOf(
                    "write file",
                    "create file",
                    "save file",
                    "edit file",
                    "delete file",
                    "remove file",
                    "html file",
                    ".html",
                    ".txt",
                    ".json",
                    ".md",
                ).any { it in lower }
            ) {
                add("file_write_tool")
            }
            if (
                listOf(
                    "shell command",
                    "terminal command",
                    "run command",
                    "execute command",
                    "command output",
                    "mkdir",
                    "rm -",
                    "ls ",
                    "cat ",
                    "python ",
                ).any { it in lower }
            ) {
                add("terminal_tool")
            }
            if (
                listOf(
                    "android setting",
                    "system setting",
                    "phone setting",
                    "shizuku",
                    "sui",
                    "permission",
                    "force stop",
                    "clear app data",
                ).any { it in lower }
            ) {
                add("android_system_tool")
            }
            if (
                listOf(
                    "tap ",
                    "click ",
                    "type ",
                    "screen",
                    "screenshot",
                    "visible ui",
                    "accessibility",
                    "quick settings",
                    "notifications shade",
                ).any { it in lower }
            ) {
                add("android_ui_tool")
            }
            if (
                listOf(
                    "automation",
                    "tasker",
                    "cron",
                    "schedule",
                    "trigger",
                    "watcher",
                    "widget",
                    "tile",
                    "profile",
                    "open browser",
                    "launch browser",
                    "open url",
                    "browse url",
                ).any { it in lower }
            ) {
                add("android_automation_tool")
            }
            if (
                "browser" in lower &&
                listOf("open", "launch", "show", "view", ".html", "html").any { it in lower }
            ) {
                add("android_automation_tool")
            }
        }
    }

    private fun explicitlyRequestedToolNames(userText: String): Set<String> {
        val lower = userText.lowercase()
        return buildSet {
            if ("terminal_tool" in lower || "shell tool" in lower) {
                add("terminal_tool")
            }
            if ("file_write_tool" in lower || "write_file" in lower) {
                add("file_write_tool")
            }
            if ("android_system_tool" in lower || "settings_tool" in lower || "phone_tool" in lower) {
                add("android_system_tool")
            }
            if ("android_ui_tool" in lower || "screen_tool" in lower || "accessibility_tool" in lower) {
                add("android_ui_tool")
            }
            if ("android_automation_tool" in lower || "tasker_tool" in lower) {
                add("android_automation_tool")
            }
        }
    }

    private fun functionSpec(
        name: String,
        description: String,
        properties: JSONObject,
        required: JSONArray = JSONArray(),
    ): JSONObject {
        val parameters = JSONObject()
            .put("type", "object")
            .put("properties", properties)
        if (required.length() > 0) {
            parameters.put("required", required)
        }
        return JSONObject()
            .put("type", "function")
            .put(
                "function",
                JSONObject()
                    .put("name", name)
                    .put("description", description)
                    .put("parameters", parameters),
            )
    }

    private fun stringProp(description: String): JSONObject {
        return JSONObject().put("type", "string").put("description", description)
    }

    private fun intProp(description: String): JSONObject {
        return JSONObject().put("type", "integer").put("description", description)
    }

    private fun boolProp(description: String): JSONObject {
        return JSONObject().put("type", "boolean").put("description", description)
    }

    private fun scalarProp(description: String): JSONObject {
        return JSONObject()
            .put("type", JSONArray().put("string").put("integer").put("number"))
            .put("description", description)
    }

    @Suppress("UNREACHABLE_CODE")
    private fun toolSpecs(): JSONArray {
        return compactToolSpecs()
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
                                "Create, list, run, enable, disable, delete, export, or import saved Android automations and variables. Supports shell, file-write, file-delete, variable set/clear/append/add/subtract/literal-replace, clipboard set, Tasker Flash/toast messages, vibration, safe Android system-action, accessibility UI-action, app-launch, Android intent, Shizuku/Sui package-permission/data-clear/connectivity-toggle, offline sunrise/sunset, notification post/cancel tasks, screen-aware overlay scene show/hide tasks, launcher shortcuts, a user-added Hermes Quick Settings tile, a Hermes home-screen widget, and token-protected Tasker/Locale action, condition, and event plugins bound to saved automations or Hermes/Shizuku state; direct sunrise/sunset calculation; safe Tasker XML/Data URI import; provider-backed calendar scan/watch actions; provider-backed location scan/watch actions; Shizuku-backed logcat scan/watch actions with a bounded scan cursor; manual tasks; interval tasks; Tasker-style time/day triggers; boot/power/battery/app-foreground/notification-posted/calendar-event/location/sensor/logcat-entry/Shizuku-state/external-trigger/remote-dispatch phone triggers; OpenGUI-style standby heartbeat, standby device listing, standby dispatch payloads, execution status queries, lifecycle state for /pause /resume /cancel, raw slash payloads, and IM command strings such as !opengui devices, /opengui devices, /status, /run, and /do; Tasker-style %VARIABLE expansion; and Hermes automation bundle backup/restore. Shizuku execution must be explicitly requested per shell task or by create_shizuku_action_task. The Tasker condition plugin can expose Shizuku availability, saved automation enabled/disabled state, last-run success/failure, and saved Hermes variable set/equality state to Tasker profiles. The Tasker event plugin can trigger Tasker profiles from verified Hermes automation finished/succeeded/failed events and Shizuku available/unavailable updates while returning Tasker-local %hermes_* event variables. Tasker import supports a safe subset of exported Tasker actions including global UI navigation, safe settings panels, Flash, Vibrate, Vibrate Pattern, Set Clipboard, HTTP request, audio, Variable Set, Variable Clear, Variable Add, Variable Subtract, and replace-enabled Variable Search Replace, and leaves records disabled unless enable_imported is set. Overlay scenes require Android draw-over-other-apps permission and support bounded title/text/button/position/width payloads, not arbitrary scene code. Notification post actions require Android notification permission on Android 13+. App-foreground triggers require the user-enabled Hermes accessibility service. Notification-posted triggers require user-enabled Hermes notification access. Calendar-event triggers can be explicit event dispatches or scanned/watched from Android Calendar after the user grants calendar access. Location triggers can be explicit event dispatches or scanned/watched from Android location providers after the user grants location access. Sensor, logcat-entry, and external triggers are explicit event dispatches; external triggers also have an exported broadcast receiver guarded by a required shared token. Remote dispatch can list this phone with operator_devices, record OpenGUI standby heartbeats with operator_heartbeat, parse OpenGUI-style messages with operator_command, run enabled records by automation_id, by OpenGUI taskName/label, or by trigger remote_dispatch, exposes %DISPATCH_SOURCE, %DISPATCH_CHANNEL, %DISPATCH_EXECUTION_ID, %DISPATCH_TASK_ID, and %DISPATCH_TASK_NAME, and can be inspected with operator_execution_status or lifecycle actions. Quick Settings tile actions can set, get, clear, or run the configured tile automation; the user still has to add the Hermes tile from Android Quick Settings. Home-screen widget actions can set, get, list, clear, request pinning for, or run the configured widget automation; Android launchers still control final widget placement. Location triggers can match latitude/longitude/radius, provider, name, and accuracy, and expose %LOC, %LAT, %LON, %LOCACC, %LOCPROVIDER, %LOCNAME, and LOCATION_* aliases. start_location_watcher and scan_location require Android location permission and at least one enabled location record. Sunrise/sunset actions accept latitude, longitude, optional date, and optional timezone, and expose %SUNRISE, %SUNSET, %SUN_DAWN, %SUN_DUSK, %SOLAR_NOON, %SUN_DAYLIGHT_MINUTES, %SUN_STATE, %SUN_DATE, %SUN_TIMEZONE, %SUN_LAT, and %SUN_LON. Notification actions can post, update, or cancel app notifications with title, text, channel, priority, group, ongoing, and only-alert-once fields. Variable actions can set, clear, append, add, subtract, or literal-replace a saved Hermes automation variable at run time and expand existing variables in the target name and value. Clipboard actions set Android clipboard text and expand saved variables at run time. Toast actions show bounded Android toast/Tasker Flash messages and expand saved variables at run time. Vibration actions use Android's normal vibrator permission and cap duration/pattern totals. Sensor triggers can match type/name, event, value name, and min/max value, and expose %SENSOR, %SENSOR_EVENT, %SENSOR_VALUE, %SENSOR_VALUE_NAME, %SENSOR_UNIT, and %SENSOR_ACCURACY. start_calendar_watcher and scan_calendar_events require calendar permission and at least one enabled calendar_event record; watcher scans dedupe recently seen events and reset_calendar_watcher_cursor clears that cursor. Logcat-entry triggers can match tag, message text, level, pid, and package filters, expose %LOGCAT_TAG, %LOGCAT_MESSAGE, %LOGCAT_LEVEL, %LOGCAT_PID, %LOGCAT_PACKAGE, and %LOGCAT_TIME. start_logcat_watcher and scan_logcat_entries require Shizuku/Sui running with Hermes permission and at least one enabled logcat_entry record; watcher scans dedupe recently seen log lines and reset_logcat_watcher_cursor clears that cursor. External triggers can match trigger_id, external_token, optional trigger_package_name, and optional referrer_contains, and expose %SA_TRIGGER_ID, %SA_TRIGGER_PACKAGE_NAME, %SA_REFERRER, and %SA_EXTRAS. Shizuku-state triggers expose %SHIZUKU_AVAILABLE, %SHIZUKU_INSTALLED, %SUI_INSTALLED, %SHIZUKU_RUNNING, %SHIZUKU_PERMISSION_GRANTED, %SHIZUKU_PRIVILEGE_LABEL, and %SHIZUKU_UID.",
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
                                                    .put("description", "list, operator_devices, operator_standby_status, operator_heartbeat, operator_execution_status, operator_cancel_execution, operator_pause_execution, operator_resume_execution, operator_command, run_history, create_shell_task, create_file_write_task, create_file_delete_task, create_system_action_task, create_ui_action_task, create_app_launch_task, create_intent_task, create_shizuku_action_task, create_sunrise_sunset_task, create_notification_task, create_variable_action_task, create_wait_task, create_clipboard_task, create_toast_task, show_toast, create_vibration_task, create_overlay_scene_task, overlay_scene_status, show_overlay_scene, hide_overlay_scene, create_launcher_shortcut, list_launcher_shortcuts, remove_launcher_shortcut, set_quick_settings_tile_automation, get_quick_settings_tile_automation, clear_quick_settings_tile_automation, run_quick_settings_tile, set_home_screen_widget_automation, get_home_screen_widget_automation, list_home_screen_widgets, clear_home_screen_widget_automation, run_home_screen_widget, calculate_sunrise_sunset, export_automations, import_automations, import_tasker_xml, calendar_watcher_status, start_calendar_watcher, stop_calendar_watcher, scan_calendar_events, reset_calendar_watcher_cursor, location_watcher_status, start_location_watcher, stop_location_watcher, scan_location, sensor_watcher_status, start_sensor_watcher, stop_sensor_watcher, logcat_watcher_status, start_logcat_watcher, stop_logcat_watcher, scan_logcat_entries, reset_logcat_watcher_cursor, run, run_trigger, run_app_foreground_trigger, run_notification_posted_trigger, run_calendar_event_trigger, run_location_trigger, run_sensor_trigger, run_logcat_entry_trigger, run_external_trigger, run_remote_dispatch, submit_standby_dispatch, run_shizuku_state_trigger, run_time_trigger, delete, enable, disable, list_variables, set_variable, get_variable, or delete_variable."),
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
                                                    .put("description", "Shell command for create_shell_task, OpenGUI-style IM text for operator_command, alternate system action value for create_system_action_task, alternate package name for create_app_launch_task, or alternate intent action for create_intent_task. Saved variables can be referenced as %NAME or {{NAME}}."),
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
                                                "scene_width",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("number").put("string"))
                                                    .put("description", "Optional overlay width. Accepts dp numbers, pixel strings like 960px, or percentages like 94%. Hermes clamps it to the current safe screen area."),
                                            )
                                            .put(
                                                "scene_width_dp",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Optional overlay width in dp, clamped between 220 and 560 before safe-screen fitting."),
                                            )
                                            .put(
                                                "scene_width_px",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Optional overlay width in physical pixels. Useful when sizing from screenshot dimensions; Hermes clamps it to the current safe screen area."),
                                            )
                                            .put(
                                                "scene_width_percent",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("number").put("string"))
                                                    .put("description", "Optional overlay width as a percent or fraction of the usable screen, for example 94, 94%, or 0.94."),
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
                                                "status_text",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional short status/subtext for create_notification_task, useful for Tasker-style live notification updates. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "progress_value",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Optional bounded progress value for create_notification_task post/update. Use the same notification_id to update progress. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "progress_max",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "Optional progress maximum for create_notification_task. Defaults to 100 and is capped."),
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
                                                    .put("description", "Activity class name for create_intent_task start_activity, or accessibility class-name selector for create_ui_action_task. Use a fully qualified activity class or a package-relative .ClassName for intents; use fragments such as EditText, Button, or RecyclerView for UI selectors."),
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
                                                    .put("description", "Optional trigger for create_*_task or run_trigger: manual, time, boot, power_connected, power_disconnected, battery_low, battery_okay, app_foreground, notification_posted, calendar_event, location, sensor, logcat_entry, external_trigger, remote_dispatch, shizuku_available, or shizuku_unavailable. interval_minutes creates an interval trigger. time requires a time argument such as 08:30 and can use days_of_week. app_foreground and notification_posted require trigger_package_name and the relevant Android service permission. calendar_event can filter by calendar_name, title_contains, description_contains, or location_contains. location can filter by latitude, longitude, radius_meters, location_provider, location_name, or max_accuracy_meters and can be run with run_location_trigger or watched from enabled saved location records with start_location_watcher after Android location permission is granted. sensor can filter by sensor_type, sensor_event, value_name, or min/max value and can be run with run_sensor_trigger or watched from enabled saved sensor records with start_sensor_watcher. logcat_entry can filter by logcat_tag, logcat_message_contains, logcat_level, logcat_pid, logcat_package_name, or trigger_package_name and must be run with run_logcat_entry_trigger. Shizuku logcat scans expose UID package candidates through logcat_package_candidates. external_trigger requires trigger_id and external_token and must be run with run_external_trigger or the exported Hermes external-trigger broadcast. remote_dispatch records can be run by run_remote_dispatch with OpenGUI-style executionId, taskId, and taskName payloads."),
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
                                                "executionId",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "OpenGUI-compatible remote execution id for run_remote_dispatch. Also accepts execution_id."),
                                            )
                                            .put(
                                                "taskId",
                                                JSONObject()
                                                    .put("type", JSONArray().put("integer").put("string"))
                                                    .put("description", "OpenGUI-compatible remote task id for run_remote_dispatch. Also accepts task_id."),
                                            )
                                            .put(
                                                "taskName",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "OpenGUI-compatible task name for run_remote_dispatch. Hermes matches this against enabled automation labels or ids."),
                                            )
                                            .put(
                                                "dispatch_source",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional remote dispatch source, such as opengui_standby, discord, telegram, feishu, or rest."),
                                            )
                                            .put(
                                                "dispatch_channel",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional remote dispatch channel for run_remote_dispatch run-history metadata."),
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
                                                "lookahead_minutes",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Future calendar scan window for start_calendar_watcher or scan_calendar_events. Clamped between 1 minute and 7 days."),
                                            )
                                            .put(
                                                "lookback_minutes",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Past calendar scan window for start_calendar_watcher or scan_calendar_events. Clamped between 0 minutes and 1 day."),
                                            )
                                            .put(
                                                "max_events",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Maximum provider calendar events to inspect during scan_calendar_events. Clamped between 1 and 200."),
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
                                                    .put("description", "Background scan interval for start_logcat_watcher or start_calendar_watcher. Logcat clamps between 5 and 3600 seconds; calendar clamps between 60 and 3600 seconds."),
                                            )
                                            .put(
                                                "min_interval_ms",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Minimum debounce interval for start_sensor_watcher sensor events or start_location_watcher location updates. Sensor clamps between 250 and 60000 milliseconds; location clamps between 1000 and 3600000 milliseconds."),
                                            )
                                            .put(
                                                "min_distance_meters",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Minimum movement distance for start_location_watcher provider updates. Clamped between 0 and 10000 meters."),
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
                                                "progress_indeterminate",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For create_notification_task, show an indeterminate notification progress bar instead of a numeric progress value."),
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
                                "Inspect or control the visible Android UI through the user-enabled Hermes accessibility service. Supports status, screen snapshots with stable ui_state_hash values, selector-based click/type/scroll/focus, OpenGUI-style raw VLM action parsing/execution, deterministic OpenGUI action history, user-visible call_user handoffs, repeated-action and screen-state review guards, scroll/type/press/open-app aliases, coordinate tap/long-press/swipe gestures, and global Back/Home/Recents/notifications/quick-settings actions.",
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
                                                    .put("description", "status, snapshot, opengui_history, clear_opengui_history, parse_opengui_action, opengui_action, click, long_click, focus, set_text, type, scroll_forward, scroll_backward, scroll, scroll_up, scroll_down, scroll_left, scroll_right, tap, long_press, swipe, open_app, launch_app, back, home, press_back, press_home, recents, notifications, quick_settings, or open_accessibility_settings."),
                                            )
                                            .put(
                                                "raw_action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "OpenGUI-style VLM action text for parse_opengui_action or opengui_action, such as Action: click(start_box='<point>500 250</point>')."),
                                            )
                                            .put(
                                                "screen_hash",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Optional OpenGUI pHash or Hermes snapshot ui_state_hash for screen-state loop review."),
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
                                                    .put("description", "Restrict matching to a package name fragment, or launch this package for action=open_app."),
                                            )
                                            .put(
                                                "app_name",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "OpenGUI-style launcher app label for action=open_app when package_name is unknown."),
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
                                            )
                                            .put(
                                                "x",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Tap x coordinate, or swipe start x when x1 is omitted."),
                                            )
                                            .put(
                                                "y",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Tap y coordinate, or swipe start y when y1 is omitted."),
                                            )
                                            .put(
                                                "x1",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Swipe start x coordinate."),
                                            )
                                            .put(
                                                "y1",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Swipe start y coordinate."),
                                            )
                                            .put(
                                                "x2",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Swipe end x coordinate."),
                                            )
                                            .put(
                                                "y2",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Swipe end y coordinate."),
                                            )
                                            .put(
                                                "coordinate_space",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "absolute_px by default, or normalized/percent."),
                                            )
                                            .put(
                                                "duration_ms",
                                                JSONObject()
                                                    .put("type", "integer")
                                                    .put("description", "Gesture duration in milliseconds."),
                                            )
                                            .put(
                                                "direction",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Scroll finger direction for action=scroll: up, down, left, or right. Defaults to up."),
                                            )
                                            .put(
                                                "distance_px",
                                                JSONObject()
                                                    .put("type", "number")
                                                    .put("description", "Optional scroll distance in screen pixels. Defaults to half the relevant screen axis."),
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
        private const val NATIVE_TOOL_MAX_TOKENS = 1024
        private const val MAX_NATIVE_TOOL_ROUNDS = 3
        private const val PRIVILEGED_TOOL_TIMEOUT_SECONDS = 30
        private const val MAX_TOOL_RESULT_CHARS = 12_000
        private const val MAX_NATIVE_ERROR_CHARS = 360
        private const val DEFAULT_UI_SNAPSHOT_LIMIT = 80
        private const val MAX_OPEN_GUI_WORKING_MEMORY_CHARS = 16_000
        private val ANDROID_UI_ACTIONS = listOf(
            "status",
            "snapshot",
            "screenshot",
            "screen_image",
            "visual_snapshot",
            "capture_screenshot",
            "opengui_history",
            "clear_opengui_history",
            "parse_opengui_action",
            "opengui_action",
            "open_gui_action",
            "gui_action",
            "vlm_action",
            "click",
            "long_click",
            "focus",
            "set_text",
            "type",
            "scroll_forward",
            "scroll_backward",
            "scroll",
            "scroll_up",
            "scroll_down",
            "scroll_left",
            "scroll_right",
            "tap",
            "long_press",
            "swipe",
            "drag",
            "open_app",
            "launch_app",
            "back",
            "home",
            "press_back",
            "press_home",
            "recents",
            "notifications",
            "quick_settings",
            "open_accessibility_settings",
        )
        private val OPEN_GUI_ACTION_ARGUMENTS = listOf(
            "raw_action",
            "action_text",
            "prediction",
            "vlm_prediction",
            "open_gui_action",
            "opengui_action",
        )
        private val UI_SELECTOR_ARGUMENTS = listOf(
            "text_contains",
            "content_description_contains",
            "view_id",
            "package_name",
            "app_name",
            "class_name",
            "value",
            "index",
            "limit",
        )
        private val UI_COORDINATE_ARGUMENTS = listOf(
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "coordinate_space",
            "duration_ms",
            "direction",
            "distance_px",
            "save_file",
            "include_base64",
            "max_image_edge_px",
        )
    }
}
