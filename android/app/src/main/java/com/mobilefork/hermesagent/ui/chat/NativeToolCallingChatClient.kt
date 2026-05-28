package com.mobilefork.hermesagent.ui.chat

import android.content.Context
import com.mobilefork.hermesagent.api.ChatContentPart
import com.mobilefork.hermesagent.api.ChatMessage
import com.mobilefork.hermesagent.api.HermesApiClient
import com.mobilefork.hermesagent.api.toJsonObject
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.device.HermesAccessibilityController
import com.mobilefork.hermesagent.device.HermesAccessibilityUiBridge
import com.mobilefork.hermesagent.device.HermesAppControlBridge
import com.mobilefork.hermesagent.device.HermesAutomationBridge
import com.mobilefork.hermesagent.device.HermesDeviceDiagnosticsBridge
import com.mobilefork.hermesagent.device.HermesHindsightMemoryBridge
import com.mobilefork.hermesagent.device.HermesPrivilegedAccessBridge
import com.mobilefork.hermesagent.device.HermesSystemControlBridge
import com.mobilefork.hermesagent.device.HermesWorkspaceFileBridge
import com.mobilefork.hermesagent.device.NativeAndroidShellTool
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.InterruptedIOException
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
        executeExplicitHtmlBrowserToolRequest(
            normalizedBaseUrl = normalizedBaseUrl,
            modelName = modelName,
            sessionId = sessionId,
            userText = userText,
        )?.let { return it }

        var executedToolCalls = 0
        var latestToolResult = ""
        var activeToolSpecs = compactToolSpecsFor(userText)
        var messages = JSONArray()
            .put(systemMessage(toolsEnabled = activeToolSpecs.length() > 0))
            .put(
                ChatMessage(
                    role = "user",
                    content = userText,
                    contentParts = userContentParts,
                ).toJsonObject()
            )
        val initialResponse = postChatCompletionWithContextRecovery(
            normalizedBaseUrl = normalizedBaseUrl,
            modelName = modelName,
            sessionId = sessionId,
            messages = messages,
            toolSpecs = activeToolSpecs,
            maxTokens = NATIVE_TOOL_MAX_TOKENS,
        )
        messages = initialResponse.messages
        activeToolSpecs = initialResponse.toolSpecs ?: JSONArray()
        var assistant = initialResponse.assistant

        repeat(MAX_NATIVE_TOOL_ROUNDS) {
            if (assistant.toolCalls.isEmpty()) {
                val content = assistant.content.ifBlank {
                    latestToolResult.ifBlank { "Done." }
                }
                return Result(content = content, executedToolCalls = executedToolCalls)
            }

            messages.put(assistant.toJsonMessage())
            var externalActivityHandoff = false
            for (toolCall in assistant.toolCalls) {
                val toolResult = executeToolCall(toolCall)
                executedToolCalls += 1
                latestToolResult = toolResult
                messages.put(
                    JSONObject()
                        .put("role", "tool")
                        .put("tool_call_id", toolCall.id)
                        .put("name", toolCall.name)
                        .put("content", NativeToolContextCompressor.compactToolResult(toolResult))
                )
                if (shouldSkipNativeFollowUpAfterToolResult(toolResult)) {
                    externalActivityHandoff = true
                }
            }

            messages = NativeToolContextCompressor.compactMessages(messages)
            val followUp = try {
                val response = postChatCompletionWithContextRecovery(
                    normalizedBaseUrl = normalizedBaseUrl,
                    modelName = modelName,
                    sessionId = sessionId,
                    messages = messages,
                    toolSpecs = activeToolSpecs,
                    maxTokens = NATIVE_TOOL_MAX_TOKENS,
                )
                messages = response.messages
                activeToolSpecs = response.toolSpecs ?: JSONArray()
                response.assistant
            } catch (error: Exception) {
                if (externalActivityHandoff) {
                    return Result(
                        content = toolCompletionReply(latestToolResult),
                        executedToolCalls = executedToolCalls,
                    )
                }
                throw error
            }
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
        extractExplicitFileWriteRequest(userText)?.let { request ->
            val toolResult = executeFileWriteTool(
                ToolCall(
                    id = "direct_${UUID.randomUUID()}",
                    name = "file_write_tool",
                    arguments = JSONObject()
                        .put("path", request.path)
                        .put("content", request.content),
                )
            )
            return Result(
                content = toolCompletionReply(toolResult),
                executedToolCalls = 1,
            )
        }

        if (isExplicitAndroidSystemStatusRequest(userText)) {
            val toolResult = executeAndroidSystemTool(
                ToolCall(
                    id = "direct_${UUID.randomUUID()}",
                    name = "android_system_tool",
                    arguments = JSONObject().put("action", "status"),
                )
            )
            return Result(
                content = toolCompletionReply(toolResult),
                executedToolCalls = 1,
            )
        }

        extractExplicitAndroidDiagnosticsArguments(userText)?.let { arguments ->
            val toolResult = executeAndroidDeviceDiagnosticsTool(
                ToolCall(
                    id = "direct_${UUID.randomUUID()}",
                    name = "android_device_diagnostics_tool",
                    arguments = arguments,
                )
            )
            return Result(
                content = toolCompletionReply(toolResult),
                executedToolCalls = 1,
            )
        }

        extractImplicitSignalEvidenceArguments(userText)?.let { arguments ->
            val toolResult = executeAndroidDeviceDiagnosticsTool(
                ToolCall(
                    id = "direct_${UUID.randomUUID()}",
                    name = "android_device_diagnostics_tool",
                    arguments = arguments,
                )
            )
            return Result(
                content = toolCompletionReply(toolResult),
                executedToolCalls = 1,
            )
        }

        extractImplicitAndroidDiagnosticsArguments(userText)?.let { arguments ->
            val toolResult = executeAndroidDeviceDiagnosticsTool(
                ToolCall(
                    id = "direct_${UUID.randomUUID()}",
                    name = "android_device_diagnostics_tool",
                    arguments = arguments,
                )
            )
            return Result(
                content = toolCompletionReply(toolResult),
                executedToolCalls = 1,
            )
        }

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

    private fun extractExplicitFileWriteRequest(userText: String): ExplicitFileWriteRequest? {
        val lower = userText.lowercase()
        if ("file_write_tool" !in lower && "write_file" !in lower) {
            return null
        }
        val match = FILE_WRITE_WITH_CONTENT_REGEX.find(userText) ?: return null
        val path = listOfNotNull(match.groups["double"]?.value, match.groups["single"]?.value, match.groups["bare"]?.value)
            .firstOrNull()
            ?.trim()
            .orEmpty()
        val content = match.groups["content"]?.value
            ?.trim()
            ?.trim('"', '\'')
            .orEmpty()
        if (path.isBlank() || content.isBlank()) {
            return null
        }
        return ExplicitFileWriteRequest(path = path, content = content)
    }

    private fun isExplicitAndroidSystemStatusRequest(userText: String): Boolean {
        val lower = userText.lowercase()
        return "android_system_tool" in lower &&
            "status" in lower &&
            "run_privileged_shell" !in lower &&
            "privileged_shell" !in lower
    }

    private fun executeExplicitHtmlBrowserToolRequest(
        normalizedBaseUrl: String,
        modelName: String,
        sessionId: String,
        userText: String,
    ): Result? {
        val request = extractExplicitHtmlBrowserRequest(userText) ?: return null
        val generatedHtml = generateHtmlDocument(
            normalizedBaseUrl = normalizedBaseUrl,
            modelName = modelName,
            sessionId = sessionId,
            request = request,
        )
        val writeResult = executeFileWriteTool(
            ToolCall(
                id = "direct_${UUID.randomUUID()}",
                name = "file_write_tool",
                arguments = JSONObject()
                    .put("path", request.path)
                    .put("content", generatedHtml),
            )
        )
        val writeJson = runCatching { JSONObject(writeResult) }.getOrDefault(JSONObject())
        if (!writeJson.optBoolean("success", writeJson.optInt("exit_code", 0) == 0)) {
            return Result(content = toolCompletionReply(writeResult), executedToolCalls = 1)
        }

        val openResult = executeAndroidAutomationTool(
            ToolCall(
                id = "direct_${UUID.randomUUID()}",
                name = "android_automation_tool",
                arguments = JSONObject()
                    .put("action", "open_uri")
                    .put("data_uri", request.path),
            )
        )
        return Result(
            content = toolCompletionReply(openResult),
            executedToolCalls = 2,
        )
    }

    private fun generateHtmlDocument(
        normalizedBaseUrl: String,
        modelName: String,
        sessionId: String,
        request: ExplicitHtmlBrowserRequest,
    ): String {
        val markerInstruction = request.marker?.let { "Include marker $it exactly once." }.orEmpty()
        val messages = JSONArray()
            .put(
                JSONObject()
                    .put("role", "system")
                    .put(
                        "content",
                        "Return only one complete compact HTML document. No markdown fences or explanation.",
                    )
            )
            .put(
                JSONObject()
                    .put("role", "user")
                    .put(
                        "content",
                        "Create a tiny Flappy Bird style browser game for ${request.path}. " +
                            "Use <canvas id=\"game\"> and inline JavaScript. $markerInstruction",
                    )
            )
        val rawHtml = runCatching {
            postChatCompletion(
                normalizedBaseUrl = normalizedBaseUrl,
                modelName = modelName,
                sessionId = "${sessionId}_html",
                messages = messages,
                toolSpecs = null,
                maxTokens = HTML_GENERATION_MAX_TOKENS,
                timeoutMs = HTML_GENERATION_TIMEOUT_MS,
            ).content
        }.getOrElse { error ->
            if (isRecoverableHtmlGenerationFailure(error)) {
                ""
            } else {
                throw error
            }
        }
        return ensureHtmlRequirements(
            rawHtml = stripMarkdownCodeFence(rawHtml),
            title = "Hermes Gemma Flappy",
            marker = request.marker,
        )
    }

    private fun extractExplicitHtmlBrowserRequest(userText: String): ExplicitHtmlBrowserRequest? {
        val lower = userText.lowercase()
        val explicitlyUsesFileTool = "file_write_tool" in lower || "write_file" in lower
        val explicitlyOpensUri = "android_automation_tool" in lower || "open_uri" in lower || "open browser" in lower
        val htmlRequested = ".html" in lower || "html" in lower
        if (!explicitlyUsesFileTool || !explicitlyOpensUri || !htmlRequested) {
            return null
        }
        val path = HTML_PATH_REGEX.find(userText)?.value ?: "hermes-generated.html"
        val marker = MARKER_REGEX.findAll(userText)
            .map { it.value }
            .firstOrNull { candidate ->
                "_" in candidate &&
                    !candidate.equals("ANDROID_AUTOMATION_TOOL", ignoreCase = true) &&
                    !candidate.equals("FILE_WRITE_TOOL", ignoreCase = true)
            }
        return ExplicitHtmlBrowserRequest(path = path, marker = marker)
    }

    private fun stripMarkdownCodeFence(raw: String): String {
        var text = raw.trim()
        if (text.startsWith("```")) {
            text = text.removePrefix("```").trimStart()
            if (text.startsWith("html", ignoreCase = true)) {
                text = text.drop(4).trimStart()
            }
            val fenceIndex = text.lastIndexOf("```")
            if (fenceIndex >= 0) {
                text = text.substring(0, fenceIndex).trim()
            }
        }
        return text
    }

    private fun ensureHtmlRequirements(rawHtml: String, title: String, marker: String?): String {
        var html = rawHtml.trim()
        val hasDocument = html.contains("<html", ignoreCase = true) || html.contains("<!doctype", ignoreCase = true)
        val hasCanvas = html.contains("<canvas", ignoreCase = true) && html.contains("id=\"game\"", ignoreCase = true)
        if (!hasDocument || !hasCanvas) {
            html = fallbackFlappyHtml(title, marker)
        }
        if (marker != null && !html.contains(marker)) {
            html = html.replace("</body>", "<!-- $marker --></body>", ignoreCase = true)
        }
        return html
    }

    private fun fallbackFlappyHtml(title: String, marker: String?): String {
        val markerText = marker ?: "HERMES_GEMMA_FLAPPY"
        return """
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>$title</title>
              <style>body{margin:0;background:#10131f;color:#f6f2e8;font:16px sans-serif;display:grid;place-items:center;height:100vh}canvas{background:#7ec8ff;border:4px solid #f6f2e8;max-width:95vw}</style>
            </head>
            <body>
              <canvas id="game" width="320" height="480"></canvas>
              <script>
                const c=document.getElementById('game'),x=c.getContext('2d');let y=220,v=0,p=330,gap=145,score=0;
                function flap(){v=-7} addEventListener('pointerdown',flap); addEventListener('keydown',e=>{if(e.code==='Space')flap()});
                function loop(){v+=.38;y+=v;p-=2;if(p<-50){p=330;gap=100+Math.random()*220;score++}x.clearRect(0,0,320,480);x.fillStyle='#7ec8ff';x.fillRect(0,0,320,480);x.fillStyle='#1d7f45';x.fillRect(p,0,48,gap-70);x.fillRect(p,gap+70,48,480-gap);x.fillStyle='#ffd166';x.beginPath();x.arc(92,y,14,0,7);x.fill();x.fillStyle='#111';x.fillText('$markerText score '+score,18,28);if(y<0||y>480||(p<106&&p+48>78&&(y<gap-70||y>gap+70))){y=220;v=0;p=330;score=0}requestAnimationFrame(loop)}loop();
              </script>
            </body>
            </html>
        """.trimIndent()
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
        val message = parsed.optString("message").trim()
        if (message.isNotBlank()) {
            return message
        }
        return toolResult.ifBlank { "Tool call completed." }
    }

    private data class NativeChatCompletionResponse(
        val assistant: AssistantMessage,
        val messages: JSONArray,
        val toolSpecs: JSONArray?,
    )

    private fun postChatCompletionWithContextRecovery(
        normalizedBaseUrl: String,
        modelName: String,
        sessionId: String,
        messages: JSONArray,
        toolSpecs: JSONArray?,
        maxTokens: Int,
    ): NativeChatCompletionResponse {
        return try {
            NativeChatCompletionResponse(
                assistant = postChatCompletion(
                    normalizedBaseUrl = normalizedBaseUrl,
                    modelName = modelName,
                    sessionId = sessionId,
                    messages = messages,
                    toolSpecs = toolSpecs,
                    maxTokens = maxTokens,
                ),
                messages = messages,
                toolSpecs = toolSpecs,
            )
        } catch (error: Exception) {
            if (!isContextWindowError(error)) {
                throw error
            }
            val recoveredMessages = NativeToolContextCompressor.recoverMessagesAfterContextOverflow(messages)
            val recoveredToolSpecs = NativeToolContextCompressor.recoverToolSpecsAfterContextOverflow(toolSpecs)
            NativeChatCompletionResponse(
                assistant = postChatCompletion(
                    normalizedBaseUrl = normalizedBaseUrl,
                    modelName = modelName,
                    sessionId = sessionId,
                    messages = recoveredMessages,
                    toolSpecs = recoveredToolSpecs,
                    maxTokens = minOf(maxTokens, CONTEXT_RECOVERY_MAX_TOKENS),
                ),
                messages = recoveredMessages,
                toolSpecs = recoveredToolSpecs,
            )
        }
    }

    private fun postChatCompletion(
        normalizedBaseUrl: String,
        modelName: String,
        sessionId: String,
        messages: JSONArray,
        toolSpecs: JSONArray?,
        maxTokens: Int,
        timeoutMs: Long = NATIVE_TOOL_GENERATION_TIMEOUT_MS,
    ): AssistantMessage {
        val payload = JSONObject()
            .put("model", modelName)
            .put("stream", false)
            .put("temperature", 0.0)
            .put("max_tokens", maxTokens)
            .put("timeout_ms", timeoutMs)
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
        if (isContextWindowErrorMessage(lower)) {
            return "The local model ran out of context. Hermes retried with a compressed system prompt, custom instructions, messages, and tool schema, but this model still could not fit the request. Start a new chat, shorten the prompt, or choose a model with a larger context window."
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
            "android_device_diagnostics_tool", "device_diagnostics_tool", "diagnostics_tool", "resource_tool", "wifi_analyzer_tool", "bluetooth_scanner_tool", "bluetooth_analyzer_tool", "sensor_tool", "sensor_analyzer_tool", "camera_tool", "radio_signal_tool", "rf_coexistence_tool", "soc_backend_tool", "runtime_stability_tool", "device_performance_tool", "mcp_tool_server_tool", "mcp_registry_tool" ->
                executeAndroidDeviceDiagnosticsTool(toolCall)
            "hindsight_memory_tool", "memory_tool", "recall_tool", "retain_tool" -> executeHindsightMemoryTool(toolCall)
            "android_automation_tool", "automation_tool", "tasker_tool", "kai_task_tool" -> executeAndroidAutomationTool(toolCall)
            "schedule_task" -> executeAndroidAutomationAliasTool(toolCall, "schedule_task")
            "list_tasks" -> executeAndroidAutomationAliasTool(toolCall, "list_tasks")
            "cancel_task" -> executeAndroidAutomationAliasTool(toolCall, "cancel_task")
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

    private fun executeAndroidDeviceDiagnosticsTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            .orEmpty()
        return HermesDeviceDiagnosticsBridge.performActionJson(appContext, action, toolCall.arguments)
    }

    private fun executeHindsightMemoryTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            .orEmpty()
        return HermesHindsightMemoryBridge.performActionJson(appContext, action, toolCall.arguments)
    }

    private fun executeAndroidAutomationTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            ?.lowercase()
            .orEmpty()
        return HermesAutomationBridge.performActionJson(appContext, action, toolCall.arguments)
    }

    private fun executeAndroidAutomationAliasTool(toolCall: ToolCall, action: String): String {
        val arguments = JSONObject(toolCall.arguments.toString())
            .put("action", action)
        return HermesAutomationBridge.performActionJson(appContext, action, arguments)
    }

    private fun executeAndroidUiTool(toolCall: ToolCall): String {
        val rawAction = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            .orEmpty()
        val action = rawAction.lowercase()
        return when (action.ifBlank { "status" }) {
            "status", "read_status" -> androidUiStatusJson()
            "sense", "opengui_sense", "open_gui_sense", "perception_status", "sense_status" -> executeOpenGuiSenseTool(toolCall)
            "snapshot", "screen_snapshot", "read_screen", "a11y_tree", "accessibility_tree" -> executeAndroidSnapshotTool(toolCall)
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

    private fun executeOpenGuiSenseTool(toolCall: ToolCall): String {
        val includeSnapshot = optionalBooleanArgument(
            toolCall.arguments,
            "include_snapshot",
            "snapshot",
            "include_a11y",
            "include_a11y_tree",
        ) ?: true
        val includeScreenshot = optionalBooleanArgument(
            toolCall.arguments,
            "include_screenshot",
            "screenshot",
            "include_visual",
            "include_image",
        ) ?: false
        val snapshot = if (includeSnapshot) {
            JSONObject(executeAndroidSnapshotTool(toolCall))
        } else {
            null
        }
        val screenshot = if (includeScreenshot) {
            JSONObject(executeAndroidScreenshotTool(toolCall))
        } else {
            null
        }
        val metrics = HermesAccessibilityController.screenMetrics()
        return OpenGuiSenseStatus.build(
            accessibilityEnabled = HermesAccessibilityController.isServiceEnabled(appContext),
            accessibilityConnected = HermesAccessibilityController.isServiceConnected(),
            screenshotSupported = android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R,
            activePackage = HermesAccessibilityController.currentForegroundPackageName(),
            screenWidth = metrics?.width,
            screenHeight = metrics?.height,
            density = metrics?.density,
            includeSnapshot = includeSnapshot,
            includeScreenshot = includeScreenshot,
            snapshot = snapshot,
            screenshot = screenshot,
            history = openGuiActionHistory.snapshotJson(),
        ).toString()
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
            .put("opengui_sense_supported", true)
            .put("opengui_sense_actions", JSONArray(listOf("sense", "opengui_sense", "a11y_tree", "visual_snapshot")))
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
        val customSystemPrompt = AppSettingsStore(appContext).load().customSystemPrompt
        val promotedMemoryContext = if (toolsEnabled) {
            HermesHindsightMemoryBridge.promotedContextJson(appContext)
                .optString("system_prompt_context")
                .takeIf { it.isNotBlank() }
        } else {
            null
        }
        val content = buildSystemPromptContent(
            toolsEnabled = toolsEnabled,
            customSystemPrompt = customSystemPrompt,
            promotedMemoryContext = promotedMemoryContext.orEmpty(),
        )
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
                    name = "android_device_diagnostics_tool",
                    description = "Inspect resource-heavy apps, storage/memory status, nearby Wi-Fi signals, filterable Wi-Fi Analyzer readiness/scan-policy reports, Wi-Fi advisor decision matrices and roaming candidates, Wi-Fi channel decision packets for router-channel, interference, RF coexistence, and MediaTek/non-Adreno claim boundaries, current Wi-Fi association/link-quality telemetry, channel ratings, inferred channel utilization/occupancy, access-point detail/export rows, AP semantic/risk labels, band coverage, signal history, vendor/OUI metadata and filter facets, Bluetooth Analyzer readiness/scan-policy reports, Bluetooth advisor decision matrices and nearby-device candidates, Bluetooth nearby decision packets for scanner/advisor/RSSI/metadata, RF coexistence, and MediaTek/non-Adreno claim boundaries, nearby Bluetooth devices plus service UUID labels/manufacturer names/proximity metadata, Bluetooth device detail/export rows, Bluetooth RSSI history/trends, Sensor Analyzer readiness/sampling-policy reports, Sensor workflow advisor decision matrices and ranked accelerometer/gyroscope candidates, motion sensor decision packets for accelerometer/gyroscope/IMU/pose/workflow/MediaTek claim boundaries, accelerometer/gyroscope/ambient sensor snapshots, motion sensor history/trends, fused motion pose/heading/angular-motion/acceleration estimates, motion sensor quality gates for IMU source coverage, freshness, calibration, stability, cadence, and workflow readiness, camera capability, overlay status, passive local backend runtime health, dedicated MediaTek/Dimensity/Helio/Mali/PowerVR/non-Adreno readiness profiles, MediaTek signal-stack compatibility cards that join SOC/backend policy to Wi-Fi, Bluetooth, radio, motion-sensor, RF coexistence, and claim-boundary rows, physical MediaTek/non-Adreno device validation reports that join phone identity, live signal proof, backend launch gates, GitHub release artifacts, checksums, and claim boundaries, device-validation evidence export bundles that preserve required artifacts, ADB/operator capture routes, GitHub release evidence, F-Droid evidence, and claim scopes, non-Adreno backend launch advisors, GPU/backend risk matrices, local inference compatibility scorecards, thermal/memory/power runtime stability guardrails, SOC/GPU compatibility and backend-policy reports, Gemma-visible signal session snapshots, signal proof audits/claim boundary matrices, signal briefing decks, unified signal timelines, signal replay/export bundles, signal replay freshness/staleness audits, Gemma-visible signal observation packets/top-card snapshots, evidence bundles, signal workflow handoff/next-action reports, signal permission/active-refresh runbooks, agent observation dashboards with fused signal-context matrices, direct diagnostic card manifests, ranked top-card priority planners with open-next actions and Kai interactive screen parity rows, Kai-style agent environment parity/readiness and tool sandbox/status matrices, MCP tool-server registry/parity reports for Kai curated servers and Streamable HTTP gaps, objective coverage and upgrade coverage reports with research parity mapping, release validation and GitHub release readiness reports for Android CI, signed GitHub artifacts, SHA-256 checksums, F-Droid metadata and tagged Fastlane graphics, full upgrade objective audit reports, passive agent self-check/heartbeat readiness matrices, cross-signal awareness routes, RF coexistence matrices for Wi-Fi/Bluetooth/radio/backend interference summaries, tool catalog, radio analyzer AM/FM band-plan rows, AM/FM signal graph rows, radio decision packets with receiver/sample/RF/MediaTek claim boundaries, radio advisor decision matrices and receiver candidates, receiver profile schemas for vendor AM/FM and external SDR bridges, vendor radio hints, Wi-Fi/Bluetooth radio routes, external SDR constraints, RF/AM/FM hardware limits, and phone preflight readiness for TikTok/Instagram/Gmail end-to-end work.",
                    properties = JSONObject()
                        .put("action", stringProp("status, top_apps, wifi_scan, wifi_filtered_scan, wifi_analyzer_report, wifi_signal_advisor_report, wifi_channel_decision_packet_report, wifi_connection_link, wifi_channel_graph, wifi_channel_rating, wifi_channel_utilization, wifi_ap_details, wifi_export, bluetooth_scan, bluetooth_analyzer_report, bluetooth_signal_advisor_report, bluetooth_nearby_decision_packet_report, bluetooth_signal_history, bluetooth_device_details, bluetooth_export, sensor_analyzer_report, sensor_workflow_advisor_report, motion_sensor_decision_packet_report, motion_sensor_quality, motion_sensor_history, motion_pose, sensor_snapshot, camera_status, radio_signal_status, radio_signal_graph, radio_signal_advisor_report, radio_signal_decision_packet_report, radio_bridge_sample_report, sdr_bridge_samples, radio_analyzer_report, signal_capability_status, local_backend_runtime_report, accelerator_preflight_report, mediatek_readiness_report, mediatek_signal_stack_report, mediatek_device_validation_report, physical_mediatek_validation_report, non_adreno_device_validation_report, signal_device_validation_report, gemma_phone_validation_report, phone_signal_validation_report, device_validation_evidence_export_report, mediatek_device_validation_export_report, physical_device_evidence_export_report, phone_validation_evidence_export, github_release_device_evidence_export, fdroid_device_evidence_export, agent_signal_observation_packet_report, gemma_signal_observation_packet, signal_context_packet_report, multimodal_signal_observation_report, top_card_signal_observation_packet, nearby_signal_context_packet, non_adreno_backend_advisor_report, mediatek_backend_launch_checklist_report, soc_compatibility_report, gpu_backend_risk_report, local_inference_compatibility_report, device_performance_report, signal_awareness_report, rf_coexistence_report, agent_signal_session_snapshot_report, agent_signal_proof_audit_report, agent_signal_briefing_report, agent_signal_card_refresh_plan_report, agent_signal_card_refresh_status_report, agent_signal_timeline_report, agent_signal_evidence_report, agent_signal_replay_export_report, agent_signal_replay_freshness_audit_report, signal_replay_export, signal_replay_freshness, signal_replay_staleness_report, signal_evidence_export, portable_signal_bundle, signal_evidence_bundle, agent_signal_workflow_handoff_report, agent_signal_permission_runbook_report, agent_observation_report, agent_card_manifest_report, agent_card_priority_report, mcp_tool_server_registry_report, agent_objective_coverage_report, agent_upgrade_coverage_report, hermes_upgrade_coverage_report, agent_release_validation_report, github_release_readiness_report, release_validation_readiness_report, agent_capability_upgrade_report, agent_environment_report, agent_self_check_report, social_gmail_goal_preflight, show_active_overlay, tool_catalog, open_usage_access_settings, open_app_settings, open_location_settings, open_wifi_settings, open_bluetooth_settings, open_camera_permission_settings."))
                        .put("limit", intProp("Maximum rows for top apps, Wi-Fi networks, or Bluetooth devices. Defaults to 5."))
                        .put("detail_limit", intProp("Maximum Wi-Fi access-point or Bluetooth device detail/export rows. Defaults to limit, or the action max for detail/export actions."))
                        .put("export_format", stringProp("Export format for wifi_export or bluetooth_export: json, csv, or both."))
                        .put("radio_samples_json", stringProp("Optional JSON array or object of vendor/SDR bridge radio samples for radio_signal_graph. Accepted fields include station_label, frequency_mhz, frequency_khz, rssi_dbm, power_db, snr_db, receiver_id, modulation, and RDS fields."))
                        .put("radio_bridge_samples_json", stringProp("Alias JSON array/object for vendor AM/FM or external SDR bridge samples accepted by radio_signal_graph."))
                        .put("sdr_samples_json", stringProp("Alias JSON array/object for external SDR bridge samples. Include center_frequency_hz or frequency_hz, power_db, receiver_id, span_hz, and sample_rate_hz when available."))
                        .put("receiver_samples_json", stringProp("Alias JSON array/object for receiver-provided AM/FM or SDR sample rows accepted by radio_signal_graph."))
                        .put("spectrum_samples_json", stringProp("Alias JSON array/object for bridge-reported spectrum sample rows accepted by radio_signal_graph."))
                        .put("sample_source", stringProp("Optional source label for radio_signal_graph bridge samples, such as vendor_fm_bridge or external_sdr_bridge."))
                        .put("receiver_id", stringProp("Optional radio_signal_graph receiver id such as fm_vendor_or_sdr, am_vendor_or_sdr, or external_sdr_bridge."))
                        .put("station_label", stringProp("Optional AM/FM station label for a single radio_signal_graph bridge sample."))
                        .put("frequency_mhz", scalarProp("Optional FM or SDR sample frequency in MHz for a single radio_signal_graph bridge sample."))
                        .put("frequency_khz", scalarProp("Optional AM sample frequency in kHz for a single radio_signal_graph bridge sample."))
                        .put("frequency_hz", scalarProp("Optional sample frequency in Hz for external radio/SDR bridge samples."))
                        .put("center_frequency_hz", scalarProp("Optional center frequency in Hz for external SDR bridge samples."))
                        .put("span_hz", scalarProp("Optional external SDR span in Hz for bridge sample metadata readiness."))
                        .put("sample_rate_hz", scalarProp("Optional external SDR sample rate in Hz for bridge sample metadata readiness."))
                        .put("bin_width_hz", scalarProp("Optional external SDR bin width in Hz for bridge sample metadata readiness."))
                        .put("bandwidth_hz", scalarProp("Optional receiver or SDR bandwidth in Hz for bridge sample metadata readiness."))
                        .put("rssi_dbm", scalarProp("Optional RSSI in dBm for a single AM/FM radio_signal_graph bridge sample."))
                        .put("power_db", scalarProp("Optional power in dB for a single radio_signal_graph bridge sample."))
                        .put("snr_db", scalarProp("Optional SNR in dB for a single radio_signal_graph bridge sample."))
                        .put("modulation", stringProp("Optional radio modulation for bridge samples, such as fm, am, nfm, or unknown."))
                        .put("rds_program_service", stringProp("Optional FM RDS program service value reported by a receiver bridge."))
                        .put("rds_radio_text", stringProp("Optional FM RDS radio text reported by a receiver bridge."))
                        .put("scan_mode", stringProp("Wi-Fi or Bluetooth scan mode for direct signal actions: auto, paused, or resumed. Paused reuses cached rows/history; resumed requests a fresh Android scan or BLE sample."))
                        .put("refresh", boolProp("For wifi_scan, wifi_export, wifi_ap_details, bluetooth_scan, bluetooth_signal_history, bluetooth_device_details, bluetooth_export, motion_sensor_quality, motion_sensor_history, or motion_pose, request Android to refresh scan/sensor results before reading available results; analyzer reports stay passive."))
                        .put("filter_band", stringProp("Optional Wi-Fi filter for wifi_scan, wifi_filtered_scan, wifi_analyzer_report, wifi_channel_decision_packet_report, wifi_ap_details, wifi_channel_graph, wifi_channel_rating, wifi_channel_utilization, or wifi_export. Accepts 2.4GHz, 5GHz, 6GHz, or comma-separated values."))
                        .put("filter_security", stringProp("Optional Wi-Fi security filter such as WPA3, WPA2, Enhanced Open, Open, WEP, or comma-separated values."))
                        .put("filter_signal", stringProp("Optional Wi-Fi signal quality filter: excellent, good, fair, or weak."))
                        .put("filter_ssid", stringProp("Optional case-insensitive substring filter for Wi-Fi SSID/display SSID."))
                        .put("filter_bssid", stringProp("Optional case-insensitive substring filter for Wi-Fi BSSID."))
                        .put("filter_vendor", stringProp("Optional case-insensitive substring filter for local Wi-Fi OUI/vendor label."))
                        .put("min_rssi_dbm", intProp("Optional Wi-Fi RSSI lower bound in dBm, for example -65 keeps APs at or above -65 dBm."))
                        .put("max_rssi_dbm", intProp("Optional Wi-Fi RSSI upper bound in dBm."))
                        .put("filter_device_name", stringProp("Optional Bluetooth device-name or advertised-name substring filter."))
                        .put("filter_bluetooth_address", stringProp("Optional Bluetooth address substring filter."))
                        .put("filter_bluetooth_service", stringProp("Optional Bluetooth service UUID, service label, or service-data substring filter."))
                        .put("filter_bluetooth_manufacturer", stringProp("Optional Bluetooth manufacturer ID or manufacturer-name substring filter."))
                        .put("filter_bluetooth_category", stringProp("Optional Bluetooth device category/class/type substring filter such as wearable, audio, beacon, or HID."))
                        .put("filter_bluetooth_proximity", stringProp("Optional Bluetooth proximity bucket filter: immediate, near, room, or far."))
                        .put("include_hidden", boolProp("For Wi-Fi filters, false excludes hidden SSID rows from returned cards."))
                        .put("hidden_only", boolProp("For Wi-Fi filters, true returns only hidden SSID rows."))
                        .put("include_snapshot", boolProp("For sensor_analyzer_report, sensor_workflow_advisor_report, motion_sensor_decision_packet_report, or motion_sensor_quality, include a bounded one-shot sensor snapshot when supported; default is passive readiness, cached history, and policy rows only."))
                        .put("sensor_types", stringProp("Comma-separated sensor types such as accelerometer, gyroscope, magnetic_field, light, proximity; returned rows include sensor range, resolution, power, FIFO, wake-up, and sampling-rate metadata when Android exposes it."))
                        .put("timeout_ms", intProp("Sensor or Bluetooth sampling timeout in milliseconds."))
                        .put("message", stringProp("Overlay message for show_active_overlay."))
                        .put("position", stringProp("Overlay position for show_active_overlay: top, center, or bottom."))
                        .put("hide_after_ms", intProp("Optional overlay auto-hide duration in milliseconds.")),
                    required = JSONArray().put("action"),
                ),
            )
            .put(
                functionSpec(
                    name = "hindsight_memory_tool",
                    description = "Retain, recall, reflect, inspect promoted context, or clear lightweight local memories using Hindsight-style keyword, entity, recency, salience, reinforcement, and Kai-style promotion signals.",
                    properties = JSONObject()
                        .put("action", stringProp("status, retain, recall, reflect, promoted_context, or clear."))
                        .put("content", stringProp("Fact or memory content for retain."))
                        .put("facts", stringProp("Optional list of fact strings for retain."))
                        .put("query", stringProp("Recall query."))
                        .put("tags", stringProp("Comma-separated tags for retain."))
                        .put("category", stringProp("Memory category for retain."))
                        .put("source", stringProp("Memory source such as chat, tool_result, user_preference, or device_state."))
                        .put("limit", intProp("Maximum recall rows. Defaults to 5."))
                        .put("max_chars", intProp("Maximum promoted context characters."))
                        .put("max_entries", intProp("Maximum rows to keep after reflect.")),
                    required = JSONArray().put("action"),
                ),
            )
            .put(
                functionSpec(
                    name = "android_ui_tool",
                    description = "Inspect or control the visible Android UI through Hermes accessibility. OpenGUI-compatible execution includes local repeated-action and screen-state review guards that can return requires_replan before a likely loop continues, plus user-visible call_user handoff notifications/toasts/vibration.",
                    properties = JSONObject()
                        .put("action", stringProp("status, sense, opengui_sense, snapshot, a11y_tree, screenshot, visual_snapshot, parse_opengui_action, opengui_action, click, long_click, focus, set_text, type, scroll_forward, scroll_backward, scroll, scroll_up, scroll_down, scroll_left, scroll_right, tap, long_press, swipe, drag, open_app, launch_app, back, home, press_back, press_home, recents, notifications, quick_settings, open_accessibility_settings."))
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
                        .put("include_snapshot", boolProp("For sense/opengui_sense, include the accessibility semantic snapshot. Defaults true."))
                        .put("include_screenshot", boolProp("For sense/opengui_sense, include a visual screenshot fallback result. Defaults false."))
                        .put("max_image_edge_px", intProp("For screenshot/visual_snapshot, resize the longest image edge before returning or saving. The response includes screen_width, screen_height, scale_factor, and a 64-bit visual screen_hash for OpenGUI-style progress review.")),
                    required = JSONArray().put("action"),
                ),
            )
            .put(
                functionSpec(
                    name = "schedule_task",
                    description = "Kai-compatible scheduled reminder alias backed by Hermes native Android automation notifications. Creates Android automation records; it does not run unrestricted background AI prompts.",
                    properties = JSONObject()
                        .put("task", stringProp("Reminder or task text to show in the Android notification."))
                        .put("title", stringProp("Optional notification title."))
                        .put("task_id", stringProp("Optional stable task/automation id."))
                        .put("time", stringProp("Time trigger such as 08:30."))
                        .put("at", stringProp("Alias for a time-of-day trigger such as 08:30."))
                        .put("interval_minutes", intProp("Repeat interval in minutes."))
                        .put("every_minutes", intProp("Alias for interval_minutes."))
                        .put("days_of_week", stringProp("Optional day filter such as MON,WED."))
                        .put("enabled", boolProp("Whether the saved Android automation starts enabled.")),
                    required = JSONArray().put("task"),
                ),
            )
            .put(
                functionSpec(
                    name = "list_tasks",
                    description = "Kai-compatible alias for listing saved Hermes Android automations as scheduled task records.",
                    properties = JSONObject()
                        .put("limit", intProp("Optional display limit.")),
                ),
            )
            .put(
                functionSpec(
                    name = "cancel_task",
                    description = "Kai-compatible alias for deleting a saved Hermes Android automation by task_id.",
                    properties = JSONObject()
                        .put("task_id", stringProp("Task or automation id to cancel.")),
                    required = JSONArray().put("task_id"),
                ),
            )
            .put(
                functionSpec(
                    name = "android_automation_tool",
                    description = "Open URLs/files immediately or create, run, manage, import, export, or trigger saved Hermes/Tasker-style Android automations and secret-free app settings bundles. Also accepts Kai-compatible schedule_task/list_tasks/cancel_task semantics as native Android automation records.",
                    properties = JSONObject()
                        .put("action", stringProp("open_uri/open_url/open_browser for immediate browser/file launch; list, list_tasks, schedule_task, cancel_task, run, delete, enable, disable, export_automations/import_automations, export_app_settings/import_app_settings, import_tasker_xml, create_*_task, set/get/delete_variable, watcher status/start/stop/scan, run_*_trigger, widget/tile actions."))
                        .put("id", stringProp("Automation id."))
                        .put("task_id", stringProp("Kai-compatible task id alias for automation id."))
                        .put("task", stringProp("Kai-compatible task/reminder text for schedule_task."))
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
                        .put("bundle_json", stringProp("Hermes automation bundle JSON for import_automations, or secret-free Hermes app settings bundle JSON for import_app_settings."))
                        .put("settings_json", stringProp("Secret-free Hermes app settings bundle JSON for import_app_settings.")),
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
                    "top apps",
                    "top 5 apps",
                    "memory apps",
                    "storage apps",
                    "eating memory",
                    "eating storage",
                    "resource usage",
                    "wifi analyzer",
                    "wi-fi analyzer",
                    "wifi analyzer report",
                    "wi-fi analyzer report",
                    "wifi readiness",
                    "wi-fi readiness",
                    "wifi scan policy",
                    "wi-fi scan policy",
                    "wifi signals",
                    "nearby wifi",
                    "wifi channel",
                    "wi-fi channel",
                    "wifi utilization",
                    "wi-fi utilization",
                    "wifi occupancy",
                    "wi-fi occupancy",
                    "wifi spectrum",
                    "wi-fi spectrum",
                    "channel rating",
                    "best wifi channel",
                    "best wi-fi channel",
                    "wifi congestion",
                    "wi-fi congestion",
                    "wifi history",
                    "wi-fi history",
                    "wifi signal history",
                    "signal over time",
                    "wifi trend",
                    "wifi vendor",
                    "wi-fi vendor",
                    "wifi oui",
                    "wi-fi oui",
                    "ssid filter",
                    "wifi security",
                    "wi-fi security",
                    "interference",
                    "signal evidence",
                    "signal timeline",
                    "agent signal timeline",
                    "signal evidence bundle",
                    "signal timeline",
                    "signal timeline report",
                    "agent signal timeline",
                    "gemma signal timeline",
                    "signal workflow handoff",
                    "next signal action",
                    "next evidence route",
                    "what did the agent recently see",
                    "current signal evidence",
                    "current signal context",
                    "evidence bundle",
                    "what can you see nearby",
                    "what can hermes see",
                    "what can gemma see",
                    "what are nearby signals",
                    "signal awareness",
                    "nearby signal report",
                    "rf sensor fusion",
                    "ambient context",
                    "cross signal",
                    "situational awareness",
                    "bluetooth analyzer report",
                    "bluetooth readiness",
                    "bluetooth scan policy",
                    "nearby bluetooth report",
                    "bluetooth advisor",
                    "bluetooth recommendation",
                    "bluetooth recommendations",
                    "bluetooth decision",
                    "bluetooth device recommendation",
                    "bluetooth nearby advisor",
                    "ble advisor",
                    "ble recommendation",
                    "nearby bluetooth scanner report",
                    "ble policy",
                    "bluetooth scanner",
                    "nearby bluetooth",
                    "ble scan",
                    "bluetooth service",
                    "bluetooth uuid",
                    "bluetooth manufacturer",
                    "bluetooth proximity",
                    "bluetooth history",
                    "bluetooth signal history",
                    "bluetooth trend",
                    "bluetooth rssi history",
                    "ble history",
                    "rssi trend",
                    "beacon",
                    "sensor",
                    "sensors",
                    "sensor advisor",
                    "sensor workflow",
                    "motion sensor advisor",
                    "motion sensor workflow",
                    "accelerometer advisor",
                    "gyroscope advisor",
                    "gyro accelerometer advisor",
                    "imu workflow advisor",
                    "sensor analyzer report",
                    "sensor readiness",
                    "sensor sampling policy",
                    "motion sensor report",
                    "motion sensor history",
                    "sensor history",
                    "sensor trend",
                    "sensor trends",
                    "motion trend",
                    "motion trends",
                    "imu history",
                    "imu trend",
                    "accelerometer history",
                    "gyroscope history",
                    "sensor metadata",
                    "sensor capability",
                    "sensor range",
                    "sensor resolution",
                    "sensor power",
                    "sampling rate",
                    "gyroscope",
                    "gyrometer",
                    "accelerometer",
                    "camera",
                    "radio frequency",
                    "am radio",
                    "fm radio",
                    "microwave",
                    "rf signal",
                    "signal strength",
                    "soc compatibility",
                    "soc compatibility report",
                    "soc backend report",
                    "soc backend",
                    "litert backend",
                    "backend launch advisor",
                    "backend route advisor",
                    "non adreno launch",
                    "non-adreno launch",
                    "mediatek launch",
                    "gpu backend advisor",
                    "gpu backend",
                    "gpu backend risk",
                    "backend risk",
                    "accelerator risk",
                    "non adreno backend",
                    "runtime stability",
                    "device performance",
                    "thermal status",
                    "thermal throttling",
                    "power saver",
                    "low ram",
                    "memory class",
                    "media performance class",
                    "mali backend",
                    "powervr backend",
                    "mediatek compatibility",
                    "mediatek",
                    "snapdragon",
                    "soc",
                    "available tools",
                    "tool catalog",
                    "list tools",
                    "phone preflight",
                    "goal preflight",
                    "social gmail preflight",
                    "social/gmail preflight",
                    "tiktok instagram gmail",
                    "tiktok video comment",
                    "instagram dm",
                    "gmail self email",
                    "float over",
                    "floating overlay",
                    "active overlay",
                ).any { it in lower }
            ) {
                add("android_device_diagnostics_tool")
            }
            if (
                listOf(
                    "hindsight",
                    "remember this",
                    "retain memory",
                    "recall memory",
                    "memory recall",
                    "memory retained",
                    "reflect memory",
                    "fine grained memory",
                    "durable memory",
                    "what do you remember",
                ).any { it in lower }
            ) {
                add("hindsight_memory_tool")
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
                    "schedule task",
                    "scheduled task",
                    "schedule a task",
                    "schedule reminder",
                    "remind me",
                    "reminder",
                ).any { it in lower }
            ) {
                add("schedule_task")
            }
            if (
                listOf(
                    "list tasks",
                    "show tasks",
                    "scheduled tasks",
                ).any { it in lower }
            ) {
                add("list_tasks")
            }
            if (
                listOf(
                    "cancel task",
                    "delete task",
                    "remove task",
                ).any { it in lower }
            ) {
                add("cancel_task")
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
            if (
                "android_device_diagnostics_tool" in lower ||
                "device_diagnostics_tool" in lower ||
                "diagnostics_tool" in lower ||
                "wifi_analyzer_tool" in lower ||
                "bluetooth_scanner_tool" in lower ||
                "bluetooth_analyzer_tool" in lower ||
                "sensor_tool" in lower ||
                "sensor_analyzer_tool" in lower ||
                "camera_tool" in lower ||
                "radio_signal_tool" in lower ||
                "soc_backend_tool" in lower ||
                "runtime_stability_tool" in lower ||
                "device_performance_tool" in lower
            ) {
                add("android_device_diagnostics_tool")
            }
            if (
                "hindsight_memory_tool" in lower ||
                "memory_tool" in lower ||
                "recall_tool" in lower ||
                "retain_tool" in lower
            ) {
                add("hindsight_memory_tool")
            }
            if ("android_ui_tool" in lower || "screen_tool" in lower || "accessibility_tool" in lower) {
                add("android_ui_tool")
            }
            if ("android_automation_tool" in lower || "tasker_tool" in lower || "kai_task_tool" in lower) {
                add("android_automation_tool")
            }
            if ("schedule_task" in lower) {
                add("schedule_task")
            }
            if ("list_tasks" in lower) {
                add("list_tasks")
            }
            if ("cancel_task" in lower) {
                add("cancel_task")
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
                                "Create, list, run, enable, disable, delete, export, or import saved Android automations, variables, and secret-free app settings bundles. Supports shell, file-write, file-delete, variable set/clear/append/add/subtract/literal-replace, clipboard set, Tasker Flash/toast messages, vibration, safe Android system-action, accessibility UI-action, app-launch, Android intent, email draft composition, Shizuku/Sui package-permission/data-clear/connectivity-toggle, offline sunrise/sunset, notification post/cancel tasks, screen-aware overlay scene show/hide tasks, launcher shortcuts, a user-added Hermes Quick Settings tile, a Hermes home-screen widget, and token-protected Tasker/Locale action, condition, and event plugins bound to saved automations or Hermes/Shizuku state; direct sunrise/sunset calculation; safe Tasker XML/Data URI import; provider-backed calendar scan/watch actions; provider-backed location scan/watch actions; Shizuku-backed logcat scan/watch actions with a bounded scan cursor; manual tasks; interval tasks; Tasker-style time/day triggers; boot/power/battery/app-foreground/notification-posted/calendar-event/location/sensor/logcat-entry/Shizuku-state/external-trigger/remote-dispatch phone triggers; OpenGUI-style standby heartbeat, standby device listing, standby dispatch payloads, execution status queries, lifecycle state for /pause /resume /cancel, raw slash payloads, and IM command strings such as !opengui devices, /opengui devices, /status, /run, and /do; Tasker-style %VARIABLE expansion; Hermes automation bundle backup/restore; and Kai-style app settings export_app_settings/import_app_settings without provider secrets. Shizuku execution must be explicitly requested per shell task or by create_shizuku_action_task. The Tasker condition plugin can expose Shizuku availability, saved automation enabled/disabled state, last-run success/failure, and saved Hermes variable set/equality state to Tasker profiles. The Tasker event plugin can trigger Tasker profiles from verified Hermes automation finished/succeeded/failed events and Shizuku available/unavailable updates while returning Tasker-local %hermes_* event variables. Tasker import supports a safe subset of exported Tasker actions including global UI navigation, safe settings panels, Flash, Vibrate, Vibrate Pattern, Set Clipboard, HTTP request, audio, Variable Set, Variable Clear, Variable Add, Variable Subtract, and replace-enabled Variable Search Replace, and leaves records disabled unless enable_imported is set. Overlay scenes require Android draw-over-other-apps permission and support bounded title/text/button/position/width payloads, not arbitrary scene code. Notification post actions require Android notification permission on Android 13+. App-foreground triggers require the user-enabled Hermes accessibility service. Notification-posted triggers require user-enabled Hermes notification access. Calendar-event triggers can be explicit event dispatches or scanned/watched from Android Calendar after the user grants calendar access. Location triggers can be explicit event dispatches or scanned/watched from Android location providers after the user grants location access. Sensor, logcat-entry, and external triggers are explicit event dispatches; external triggers also have an exported broadcast receiver guarded by a required shared token. Remote dispatch can list this phone with operator_devices, record OpenGUI standby heartbeats with operator_heartbeat, parse OpenGUI-style messages with operator_command, run enabled records by automation_id, by OpenGUI taskName/label, or by trigger remote_dispatch, exposes %DISPATCH_SOURCE, %DISPATCH_CHANNEL, %DISPATCH_EXECUTION_ID, %DISPATCH_TASK_ID, and %DISPATCH_TASK_NAME, and can be inspected with operator_execution_status or lifecycle actions. Quick Settings tile actions can set, get, clear, or run the configured tile automation; the user still has to add the Hermes tile from Android Quick Settings. Home-screen widget actions can set, get, list, clear, request pinning for, or run the configured widget automation; Android launchers still control final widget placement. Location triggers can match latitude/longitude/radius, provider, name, and accuracy, and expose %LOC, %LAT, %LON, %LOCACC, %LOCPROVIDER, %LOCNAME, and LOCATION_* aliases. start_location_watcher and scan_location require Android location permission and at least one enabled location record. Sunrise/sunset actions accept latitude, longitude, optional date, and optional timezone, and expose %SUNRISE, %SUNSET, %SUN_DAWN, %SUN_DUSK, %SOLAR_NOON, %SUN_DAYLIGHT_MINUTES, %SUN_STATE, %SUN_DATE, %SUN_TIMEZONE, %SUN_LAT, and %SUN_LON. Notification actions can post, update, or cancel app notifications with title, text, channel, priority, group, ongoing, and only-alert-once fields. Variable actions can set, clear, append, add, subtract, or literal-replace a saved Hermes automation variable at run time and expand existing variables in the target name and value. Clipboard actions set Android clipboard text and expand saved variables at run time. Toast actions show bounded Android toast/Tasker Flash messages and expand saved variables at run time. Vibration actions use Android's normal vibrator permission and cap duration/pattern totals. Email draft actions open Android's mail composer with recipient, subject, and body fields; actual send remains controlled by the selected email app or a separate user-approved UI action. Sensor triggers can match type/name, event, value name, and min/max value, and expose %SENSOR, %SENSOR_EVENT, %SENSOR_VALUE, %SENSOR_VALUE_NAME, %SENSOR_UNIT, and %SENSOR_ACCURACY. start_calendar_watcher and scan_calendar_events require calendar permission and at least one enabled calendar_event record; watcher scans dedupe recently seen events and reset_calendar_watcher_cursor clears that cursor. Logcat-entry triggers can match tag, message text, level, pid, and package filters, expose %LOGCAT_TAG, %LOGCAT_MESSAGE, %LOGCAT_LEVEL, %LOGCAT_PID, %LOGCAT_PACKAGE, and %LOGCAT_TIME. start_logcat_watcher and scan_logcat_entries require Shizuku/Sui running with Hermes permission and at least one enabled logcat_entry record; watcher scans dedupe recently seen log lines and reset_logcat_watcher_cursor clears that cursor. External triggers can match trigger_id, external_token, optional trigger_package_name, and optional referrer_contains, and expose %SA_TRIGGER_ID, %SA_TRIGGER_PACKAGE_NAME, %SA_REFERRER, and %SA_EXTRAS. Shizuku-state triggers expose %SHIZUKU_AVAILABLE, %SHIZUKU_INSTALLED, %SUI_INSTALLED, %SHIZUKU_RUNNING, %SHIZUKU_PERMISSION_GRANTED, %SHIZUKU_PRIVILEGE_LABEL, and %SHIZUKU_UID.",
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
                                                    .put("description", "list, operator_devices, operator_standby_status, operator_heartbeat, operator_execution_status, operator_cancel_execution, operator_pause_execution, operator_resume_execution, operator_command, run_history, create_shell_task, create_file_write_task, create_file_delete_task, create_system_action_task, create_ui_action_task, create_app_launch_task, create_intent_task, create_email_draft_task, create_shizuku_action_task, create_sunrise_sunset_task, create_notification_task, create_variable_action_task, create_wait_task, create_clipboard_task, create_toast_task, show_toast, create_vibration_task, create_overlay_scene_task, overlay_scene_status, show_overlay_scene, hide_overlay_scene, create_launcher_shortcut, list_launcher_shortcuts, remove_launcher_shortcut, set_quick_settings_tile_automation, get_quick_settings_tile_automation, clear_quick_settings_tile_automation, run_quick_settings_tile, set_home_screen_widget_automation, get_home_screen_widget_automation, list_home_screen_widgets, clear_home_screen_widget_automation, run_home_screen_widget, calculate_sunrise_sunset, export_app_settings, import_app_settings, export_automations, import_automations, import_tasker_xml, calendar_watcher_status, start_calendar_watcher, stop_calendar_watcher, scan_calendar_events, reset_calendar_watcher_cursor, location_watcher_status, start_location_watcher, stop_location_watcher, scan_location, sensor_watcher_status, start_sensor_watcher, stop_sensor_watcher, logcat_watcher_status, start_logcat_watcher, stop_logcat_watcher, scan_logcat_entries, reset_logcat_watcher_cursor, run, run_trigger, run_app_foreground_trigger, run_notification_posted_trigger, run_calendar_event_trigger, run_location_trigger, run_sensor_trigger, run_logcat_entry_trigger, run_external_trigger, run_remote_dispatch, submit_standby_dispatch, run_shizuku_state_trigger, run_time_trigger, delete, enable, disable, list_variables, set_variable, get_variable, or delete_variable."),
                                            )
                                            .put(
                                                "bundle",
                                                JSONObject()
                                                    .put("type", "object")
                                                    .put("description", "Hermes automation export bundle for import_automations, or secret-free Hermes app settings bundle for import_app_settings."),
                                            )
                                            .put(
                                                "bundle_json",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Stringified Hermes automation export bundle for import_automations, or secret-free Hermes app settings bundle for import_app_settings."),
                                            )
                                            .put(
                                                "settings_json",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Stringified secret-free Hermes app settings bundle for import_app_settings."),
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
                                                "to",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Recipient email address for create_email_draft_task. Also accepts recipient, recipient_email, email_to, or to_address."),
                                            )
                                            .put(
                                                "subject",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Email subject for create_email_draft_task. Saved variables can be referenced as %NAME or {{NAME}}."),
                                            )
                                            .put(
                                                "body",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "Email body for create_email_draft_task. Saved variables can be referenced as %NAME or {{NAME}}."),
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
                                "Inspect or control the visible Android UI through the user-enabled Hermes accessibility service. Supports OpenGUI-style sense/perception routing between accessibility semantics and visual screenshot fallback, status, screen snapshots with stable ui_state_hash values, selector-based click/type/scroll/focus, OpenGUI-style raw VLM action parsing/execution, deterministic OpenGUI action history, user-visible call_user handoffs, repeated-action and screen-state review guards, scroll/type/press/open-app aliases, coordinate tap/long-press/swipe gestures, and global Back/Home/Recents/notifications/quick-settings actions.",
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
                                                    .put("description", "status, sense, opengui_sense, snapshot, a11y_tree, opengui_history, clear_opengui_history, parse_opengui_action, opengui_action, click, long_click, focus, set_text, type, scroll_forward, scroll_backward, scroll, scroll_up, scroll_down, scroll_left, scroll_right, tap, long_press, swipe, open_app, launch_app, back, home, press_back, press_home, recents, notifications, quick_settings, or open_accessibility_settings."),
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
                                                "include_snapshot",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For sense/opengui_sense, include the accessibility semantic snapshot. Defaults true."),
                                            )
                                            .put(
                                                "include_screenshot",
                                                JSONObject()
                                                    .put("type", "boolean")
                                                    .put("description", "For sense/opengui_sense, include a visual screenshot fallback result. Defaults false."),
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

    internal companion object {
        fun buildSystemPromptContent(
            toolsEnabled: Boolean,
            customSystemPrompt: String = "",
            promotedMemoryContext: String = "",
        ): String {
            val baseContent = if (toolsEnabled) {
                "You are Hermes running inside the native Android app. " +
                    "Use tools for real files, shell commands, Android UI, settings, Shizuku/Sui, diagnostics, sensor sampling/range/resolution/power metadata, motion history, fused pose/orientation estimates, and local backend runtime health, camera capability checks, Wi-Fi analysis/channel graph envelopes/channel ratings/channel utilization/signal history, Bluetooth Analyzer readiness/scan-policy reports plus nearby scans/service labels/manufacturer names/device detail export rows, radio analyzer checks for AM/FM band-plan boundaries, AM/FM signal graph rows, vendor broadcast-radio hints, receiver profile schemas, Wi-Fi/Bluetooth radio routes, external SDR constraints, resource summaries, secret-free app settings backup/restore, Kai-style custom agent persona/system prompt, Kai-compatible schedule_task/list_tasks/cancel_task native Android task aliases, or Tasker-style automation. " +
                    "When writing multiline text, prefer file_write_tool so multiline content is written exactly; file_write_tool can only write inside the Hermes app workspace. " +
                    "For HTML/browser work: write the file with file_write_tool, then call android_automation_tool action=open_uri with data_uri set to the workspace filename. " +
                    "Use android_device_diagnostics_tool for top memory/storage apps, Wi-Fi signals/channel graph envelopes/channel ratings/channel utilization/signal history, filterable Wi-Fi Analyzer readiness/scan-policy reports, Bluetooth Analyzer readiness/scan-policy reports, Bluetooth nearby decision packets, and nearby devices with service UUID labels/manufacturer names/device details/export rows, camera/sensor status plus accelerometer/gyroscope hardware metadata, motion sensor decision packets, motion trend history, fused pose/heading/acceleration estimates, active overlays, tool catalog, Gemma-visible signal briefing decks, expanded signal card decks, per-card signal refresh plans/status indicators, signal proof audits and claim-boundary matrices, unified signal timelines, signal replay/export bundles and replay freshness/staleness audits, compact signal observation packets/top-card snapshots, evidence bundles, signal workflow handoff and next-action reports, signal permission and active-refresh runbooks, agent observation dashboards, Kai-style agent environment reports, MCP tool-server registry reports, objective coverage/gap and upgrade coverage reports, release validation and GitHub release readiness reports for Android CI, signed GitHub artifacts, SHA-256 checksums, F-Droid metadata and tagged Fastlane graphics, full upgrade objective audit reports, passive agent self-check/heartbeat reports, cross-signal awareness reports, MediaTek signal-stack reports that fuse SOC/backend policy with Wi-Fi/Bluetooth/radio/sensor evidence and claim boundaries, local runtime backend health, thermal/memory/power runtime stability guardrails, SOC compatibility/backend reports and backend launch advisors for MediaTek/Mali/PowerVR and non-Snapdragon devices, AM/FM signal graph rows, radio decision packets, broader radio signal route reports, receiver profile schemas, RF capability limits, or phone preflight checks before TikTok/Instagram/Gmail work. " +
                    "For MediaTek/non-Adreno signal questions, first call android_device_diagnostics_tool action=mediatek_signal_stack_report so SOC/backend policy, Wi-Fi, Bluetooth, radio, motion, RF coexistence, and claim boundaries stay together. " +
                    "For physical MediaTek/non-Adreno phone validation, call android_device_diagnostics_tool action=mediatek_device_validation_report before claiming live Wi-Fi, Bluetooth, motion, radio bridge, backend, GitHub release, checksum, F-Droid, or physical-device proof. " +
                    "For phone or release proof export packages and device-validation evidence export bundles, call android_device_diagnostics_tool action=device_validation_evidence_export_report so required artifacts, ADB/operator capture routes, GitHub release routes, F-Droid routes, and claim scopes stay together. " +
                    "Before starting local inference on MediaTek, Mali, PowerVR/IMG, Xclipse, Tensor, Exynos, Unisoc, or unknown ARM phones, call android_device_diagnostics_tool action=mediatek_backend_launch_checklist_report so artifact, preflight, live /health, CPU fallback, stability, and phone-validation launch gates are visible together. " +
                    "For broad readiness or stability questions, first call android_device_diagnostics_tool action=agent_self_check_report so heartbeat, wireless, sensors, radio, MediaTek/non-Adreno backend guardrails, and expandable card routes are visible together. " +
                    "When deciding which signal/card/tool to open next, first call android_device_diagnostics_tool action=agent_signal_workflow_handoff_report so refresh policy, permission gates, bridges, and physical-device boundaries are explicit. " +
                    "Before refreshing an expanded top card, call android_device_diagnostics_tool action=agent_signal_card_refresh_status_report so ready/passive-only/permission/bridge/phone-validation hints are visible, then use agent_signal_card_refresh_plan_report when exact active_refresh_arguments are needed. " +
                    "Before opening settings panels or requesting active Wi-Fi, Bluetooth, motion, radio, or accelerator refreshes, call android_device_diagnostics_tool action=agent_signal_permission_runbook_report. " +
                    "Before claiming live/current signal evidence, call android_device_diagnostics_tool action=agent_signal_proof_audit_report so active evidence, passive fallbacks, permission gates, bridge requirements, physical-phone validation, and release boundaries are visible. " +
                    "For compact multimodal signal observation packets or top-card snapshots of what Gemma can see, call android_device_diagnostics_tool action=agent_signal_observation_packet_report so visual slots, graph routes, freshness, proof_status, and claim_scope stay attached. " +
                    "For router-channel or Wi-Fi interference decisions, call android_device_diagnostics_tool action=wifi_channel_decision_packet_report so channel ratings, utilization, RF coexistence, MediaTek/backend sensitivity, and claim_scope stay attached. " +
                    "For nearby Bluetooth or BLE decision-packet questions, call android_device_diagnostics_tool action=bluetooth_nearby_decision_packet_report so scanner/advisor rows, RSSI trends, service/manufacturer metadata, RF coexistence, MediaTek/backend sensitivity, and claim_scope stay attached. " +
                    "For accelerometer, gyroscope, IMU, pose, or motion workflow decisions, call android_device_diagnostics_tool action=motion_sensor_decision_packet_report so quality, history, pose, workflow routes, sampling privacy, MediaTek/backend sensitivity, and claim_scope stay attached. " +
                    "For AM/FM, SDR, receiver bridge, or radio evidence decisions, call android_device_diagnostics_tool action=radio_signal_decision_packet_report so band-plan limits, graph samples, bridge metadata, RF coexistence, MediaTek/backend sensitivity, and claim_scope stay attached. " +
                    "For broad questions about what Hermes/Gemma can see from nearby signals, first call android_device_diagnostics_tool action=agent_signal_evidence_report; then drill into Wi-Fi, Bluetooth, sensor, radio, backend-risk, observation-packet, or card-manifest actions only when the evidence rows say a source card or live refresh is needed. " +
                    "For portable signal replay/export requests, call android_device_diagnostics_tool action=agent_signal_replay_export_report so source_action, graph_type, claim_scope, proof_status, and refresh policy stay visible together. " +
                    "Before treating replay/export rows as current, call android_device_diagnostics_tool action=agent_signal_replay_freshness_audit_report so freshness_status, staleness_risk, active_refresh_action, passive_fallback_action, permission_gate, hardware_gate, and proof_status stay attached. " +
                    "Use schedule_task/list_tasks/cancel_task for Kai-style scheduled reminders; these create, list, and cancel native Android automation notification records, not unrestricted background AI prompt execution. " +
                    "Use hindsight_memory_tool to retain, recall, reflect, and inspect promoted durable local memories before or after complex work. " +
                    "Report missing Android permissions honestly. Keep replies brief."
            } else {
                "You are Hermes running inside the native Android app. Keep replies brief and direct."
            }
            val normalizedPersona = NativeToolContextCompressor.compactCustomSystemPrompt(
                AppSettings.normalizeCustomSystemPrompt(customSystemPrompt),
            )
            val normalizedMemory = NativeToolContextCompressor.compactPromotedMemoryContext(promotedMemoryContext)
            return buildString {
                append(baseContent)
                if (normalizedPersona.isNotBlank()) {
                    append(" User-configured agent persona (apply unless it conflicts with the current user request, Android permissions, tool truthfulness, or safety constraints):\n")
                    append(normalizedPersona)
                }
                if (normalizedMemory.isNotBlank()) {
                    append(" Promoted local memory context:\n")
                    append(normalizedMemory)
                }
            }
        }

        internal fun isContextWindowError(error: Throwable): Boolean {
            return isContextWindowErrorMessage(error.message.orEmpty())
        }

        internal fun isContextWindowErrorMessage(message: String): Boolean {
            val lower = message.lowercase()
            return "exceed_context_size" in lower ||
                "exceeds the available context size" in lower ||
                "context window" in lower ||
                "ran out of context" in lower ||
                "maximum context" in lower ||
                "prompt is too long" in lower ||
                "input is too long" in lower
        }

        fun shouldSkipNativeFollowUpAfterToolResult(toolResult: String): Boolean {
            val parsed = runCatching { JSONObject(toolResult) }.getOrNull() ?: return false
            return parsed.optBoolean("external_activity_handoff", false)
        }

        fun isRecoverableHtmlGenerationFailure(error: Throwable): Boolean {
            if (error is InterruptedIOException) {
                return true
            }
            val message = error.message.orEmpty().lowercase()
            return "timed out" in message ||
                "timeout" in message ||
                "before producing a response" in message
        }

        fun extractExplicitAndroidDiagnosticsArguments(userText: String): JSONObject? {
            val lower = userText.lowercase()
            if (
                "android_device_diagnostics_tool" !in lower &&
                "device_diagnostics_tool" !in lower &&
                "diagnostics_tool" !in lower
            ) {
                return null
            }
            val rawAction = DIAGNOSTIC_ACTION_REGEX.find(userText)
                ?.groupValues
                ?.getOrNull(1)
                ?.trim()
                ?.lowercase()
                ?.replace('-', '_')
                ?: return null
            if (rawAction !in DIRECT_ANDROID_DEVICE_DIAGNOSTIC_ACTIONS) {
                return null
            }
            return JSONObject().put("action", rawAction).apply {
                DIAGNOSTIC_BOOLEAN_ARGUMENTS.forEach { key ->
                    DIAGNOSTIC_BOOLEAN_REGEXES.getValue(key).find(userText)
                        ?.groupValues
                        ?.getOrNull(1)
                        ?.lowercase()
                        ?.let { put(key, it == "true") }
                }
                DIAGNOSTIC_INTEGER_ARGUMENTS.forEach { key ->
                    DIAGNOSTIC_INTEGER_REGEXES.getValue(key).find(userText)
                        ?.groupValues
                        ?.getOrNull(1)
                        ?.toIntOrNull()
                        ?.let { put(key, it) }
                }
                DIAGNOSTIC_STRING_ARGUMENTS.forEach { key ->
                    DIAGNOSTIC_STRING_REGEXES.getValue(key).find(userText)
                        ?.groupValues
                        ?.drop(1)
                        ?.firstOrNull { it.isNotBlank() }
                        ?.trim()
                        ?.takeIf { it.isNotBlank() }
                        ?.let { put(key, it) }
                }
            }
        }

        internal fun extractImplicitSignalEvidenceArguments(userText: String): JSONObject? {
            val lower = userText.lowercase()
            val explicitReplayFreshnessPhrase = listOf(
                "signal replay freshness",
                "replay freshness",
                "signal replay staleness",
                "replay staleness",
                "stale replay",
                "freshness audit",
                "replay audit",
                "export freshness",
                "signal export freshness",
            ).any { it in lower }
            val explicitReplayPhrase = listOf(
                "signal replay",
                "signal replay export",
                "signal export bundle",
                "signal evidence export",
                "replay signal evidence",
                "export signal evidence",
                "portable signal bundle",
                "signal session replay",
                "gemma signal replay",
                "replay export",
            ).any { it in lower }
            val explicitObservationPacketPhrase = listOf(
                "signal observation packet",
                "observation packet for signals",
                "gemma signal observation packet",
                "gemma observation packet",
                "multimodal signal observation",
                "top card signal observation",
                "signal context packet",
                "nearby signal context packet",
                "what gemma can see packet",
                "what the agent is viewing packet",
            ).any { it in lower }
            val explicitEvidencePhrase = listOf(
                "signal briefing",
                "briefing deck",
                "signal briefing deck",
                "expanded signal card deck",
                "signal card deck",
                "top signal card deck",
                "gemma signal card deck",
                "signal card refresh plan",
                "signal card refresh status",
                "signal session snapshot",
                "agent signal session snapshot",
                "current signal session",
                "current signal session snapshot",
                "rf signal session snapshot",
                "session snapshot",
                "signal proof audit",
                "active signal proof",
                "current signal proof",
                "signal evidence validation",
                "signal validation audit",
                "claim proof",
                "proof ledger",
                "can gemma claim live evidence",
                "top card refresh plan",
                "top card refresh status",
                "expanded signal refresh plan",
                "expanded signal refresh status",
                "which signal cards can refresh",
                "card refresh readiness",
                "card live refresh plan",
                "per-card refresh plan",
                "signal timeline",
                "agent signal timeline",
                "gemma signal timeline",
                "signal workflow handoff",
                "workflow handoff",
                "signal permission runbook",
                "permission runbook",
                "refresh runbook",
                "active refresh route",
                "active refresh routes",
                "next signal action",
                "next evidence route",
                "agent signal briefing",
                "gemma signal briefing",
                "top card slots",
                "what is the agent viewing",
                "signal evidence",
                "evidence bundle",
                "current signal evidence",
                "current signal context",
                "gemma signal evidence",
                "agent signal evidence",
            ).any { it in lower }
            val asksWhatCanBeSeen = listOf(
                "what can you see",
                "what do you see",
                "what are you seeing",
                "what can hermes see",
                "what can gemma see",
                "what can the agent see",
                "what is the agent viewing",
                "what are nearby signals",
                "what signals are nearby",
                "what is gemma viewing",
                "what is hermes viewing",
            ).any { it in lower }
            val hasSignalDomain = listOf(
                "signal",
                "signals",
                "nearby",
                "wifi",
                "wi-fi",
                "bluetooth",
                "ble",
                "radio",
                "rf",
                "sensor",
                "sensors",
                "accelerometer",
                "gyroscope",
                "motion",
                "multimodal",
                "gemma",
            ).any { it in lower }
            val explicitBriefingPhrase = listOf(
                "signal briefing",
                "briefing deck",
                "signal briefing deck",
                "expanded signal card deck",
                "signal card deck",
                "top signal card deck",
                "gemma signal card deck",
                "signal card refresh plan",
                "signal card refresh status",
                "signal session snapshot",
                "agent signal session snapshot",
                "current signal session",
                "current signal session snapshot",
                "rf signal session snapshot",
                "session snapshot",
                "signal proof audit",
                "active signal proof",
                "current signal proof",
                "signal evidence validation",
                "signal validation audit",
                "claim proof",
                "proof ledger",
                "can gemma claim live evidence",
                "top card refresh plan",
                "top card refresh status",
                "expanded signal refresh plan",
                "expanded signal refresh status",
                "which signal cards can refresh",
                "card refresh readiness",
                "agent signal briefing",
                "gemma signal briefing",
                "top card slots",
                "what is the agent viewing",
            ).any { it in lower }
            return if (explicitReplayFreshnessPhrase || explicitReplayPhrase || explicitObservationPacketPhrase || explicitEvidencePhrase || (asksWhatCanBeSeen && hasSignalDomain)) {
                val action = when {
                    explicitReplayFreshnessPhrase ->
                        "agent_signal_replay_freshness_audit_report"
                    explicitReplayPhrase ->
                        "agent_signal_replay_export_report"
                    explicitObservationPacketPhrase ->
                        "agent_signal_observation_packet_report"
                    lower.containsAny("signal proof audit", "active signal proof", "current signal proof", "signal evidence validation", "signal validation audit", "claim proof", "proof ledger", "can gemma claim live evidence") ->
                        "agent_signal_proof_audit_report"
                    lower.containsAny("signal session snapshot", "agent signal session snapshot", "current signal session", "current signal session snapshot", "rf signal session snapshot", "session snapshot") ->
                        "agent_signal_session_snapshot_report"
                    lower.containsAny("signal card refresh status", "top card refresh status", "expanded signal refresh status", "which signal cards can refresh", "card refresh readiness") ->
                        "agent_signal_card_refresh_status_report"
                    lower.containsAny("signal card refresh plan", "top card refresh plan", "expanded signal refresh plan", "card live refresh plan", "per-card refresh plan") ->
                        "agent_signal_card_refresh_plan_report"
                    lower.containsAny("expanded signal card deck", "signal card deck", "top signal card deck", "gemma signal card deck", "expanded signal cards", "all signal cards") ->
                        "agent_signal_card_deck_report"
                    lower.containsAny("signal permission runbook", "permission runbook", "refresh runbook", "active refresh route", "active refresh routes", "before live scan", "request permissions") ->
                        "agent_signal_permission_runbook_report"
                    lower.containsAny("signal workflow handoff", "workflow handoff", "next signal action", "next evidence route", "what should i open next", "what should gemma check next") ->
                        "agent_signal_workflow_handoff_report"
                    lower.containsAny("signal timeline", "agent signal timeline", "gemma signal timeline", "recently see", "recently saw") ->
                        "agent_signal_timeline_report"
                    explicitBriefingPhrase -> "agent_signal_briefing_report"
                    else -> "agent_signal_evidence_report"
                }
                JSONObject().put("action", action)
            } else {
                null
            }
        }

        internal fun extractImplicitAndroidDiagnosticsArguments(userText: String): JSONObject? {
            val lower = userText.lowercase()
            if (lower.contains("screen") && !lower.containsAny("wifi", "wi-fi", "bluetooth", "ble", "radio", "rf", "sensor", "motion", "soc", "mediatek", "mcp", "tool server", "upgrade", "objective")) {
                return null
            }
            return when {
                lower.containsAny("signal observation packet", "gemma signal observation packet", "gemma observation packet", "multimodal signal observation", "top card signal observation", "signal context packet", "nearby signal context packet", "what gemma can see packet", "what the agent is viewing packet") ->
                    diagnosticArguments("agent_signal_observation_packet_report")
                lower.containsAny("signal replay freshness", "replay freshness", "signal replay staleness", "replay staleness", "stale replay", "freshness audit", "replay audit", "export freshness", "signal export freshness") ->
                    diagnosticArguments("agent_signal_replay_freshness_audit_report")
                lower.containsAny("signal replay", "signal replay export", "signal export bundle", "signal evidence export", "replay signal evidence", "export signal evidence", "portable signal bundle", "signal session replay", "gemma signal replay", "replay export") ->
                    diagnosticArguments("agent_signal_replay_export_report")
                lower.containsAny("signal proof audit", "active signal proof", "current signal proof", "signal evidence validation", "signal validation audit", "claim proof", "proof ledger", "can gemma claim live evidence", "claim live evidence") ->
                    diagnosticArguments("agent_signal_proof_audit_report")
                lower.containsAny("device evidence export", "phone evidence export", "device validation export", "phone validation evidence", "device proof export", "physical device evidence export", "release proof package", "github device evidence", "github release device evidence", "f-droid device evidence", "fdroid device evidence") ->
                    diagnosticArguments("device_validation_evidence_export_report")
                lower.containsAny("physical mediatek validation", "physical mediaTek validation", "phone signal validation", "non-adreno device validation", "non adreno device validation", "device proof gates", "live signal proof", "mediatek device validation", "gemma phone validation", "signal device validation") ->
                    diagnosticArguments("mediatek_device_validation_report")
                lower.containsAny("signal session snapshot", "agent signal session snapshot", "current signal session", "current signal session snapshot", "rf signal session snapshot", "session snapshot", "ambient session context") ->
                    diagnosticArguments("agent_signal_session_snapshot_report")
                lower.containsAny("signal card refresh status", "top card refresh status", "expanded signal refresh status", "which signal cards can refresh", "card refresh readiness", "refresh status for expanded cards") ->
                    diagnosticArguments("agent_signal_card_refresh_status_report")
                lower.containsAny("signal card refresh plan", "top card refresh plan", "expanded signal refresh plan", "card live refresh plan", "per-card refresh plan", "refresh an expanded top card") ->
                    diagnosticArguments("agent_signal_card_refresh_plan_report")
                lower.containsAny("expanded signal card deck", "signal card deck", "top signal card deck", "gemma signal card deck", "expanded signal cards", "all signal cards") ->
                    diagnosticArguments("agent_signal_card_deck_report")
                lower.containsAny("signal permission runbook", "permission runbook", "refresh runbook", "active signal refresh", "active refresh route", "active refresh routes", "before live scan", "request permissions") ->
                    diagnosticArguments("agent_signal_permission_runbook_report")
                lower.containsAny("workflow handoff", "signal workflow handoff", "signal handoff", "next signal action", "next diagnostic action", "what should i open next", "what should hermes open next", "what should gemma check next", "card handoff", "next evidence route", "operational signal plan", "signal next actions") ->
                    diagnosticArguments("agent_signal_workflow_handoff_report")
                lower.containsAny("objective coverage", "coverage gap", "coverage gaps", "objective gap", "objective gaps", "research parity coverage", "kai wifi analyzer parity", "kai wi-fi analyzer parity", "full objective coverage", "agent upgrade coverage", "hermes upgrade coverage", "upgrade coverage report", "upgrade coverage map") ->
                    diagnosticArguments("agent_objective_coverage_report")
                lower.containsAny("release validation", "release workflow", "github release", "github release readiness", "release readiness", "release validation readiness", "signed release", "release artifact", "release artifacts", "apk checksum", "aab checksum", "fdroid readiness", "f-droid readiness", "fdroid metadata", "f-droid metadata", "fastlane graphics") ->
                    diagnosticArguments("agent_release_validation_report")
                lower.containsAny("upgrade audit", "capability upgrade", "full system upgrade", "full objective", "objective audit", "upgrade objective", "completion audit", "capability coverage", "upgrade coverage", "what is still missing", "what remains incomplete") ->
                    diagnosticArguments("agent_capability_upgrade_report")
                lower.containsAny("mcp server", "mcp servers", "mcp tool", "mcp tools", "mcp registry", "mcp parity", "mcp tool server", "tool server registry", "tool-server registry", "kai mcp", "external tool server", "streamable http mcp", "streamable http", "context7", "deepwiki", "globalping", "coingecko", "manifold markets", "find-a-domain", "find a domain") ->
                    diagnosticArguments("mcp_tool_server_registry_report")
                lower.containsAny("wifi channel decision", "wi-fi channel decision", "wifi decision packet", "wi-fi decision packet", "router channel plan", "wifi router channel", "wi-fi router channel", "which wifi channel", "which wi-fi channel", "wifi interference decision", "wi-fi interference decision", "gemma wifi channel", "gemma wi-fi channel", "mediatek wifi channel", "mediatek wi-fi channel", "wifi channel proof packet", "wi-fi channel proof packet") ->
                    wifiDiagnosticArguments("wifi_channel_decision_packet_report", userText)
                lower.containsAny("bluetooth decision packet", "ble decision packet", "nearby bluetooth decision", "nearby ble decision", "bluetooth nearby decision", "ble nearby decision", "gemma bluetooth nearby", "gemma ble nearby", "what bluetooth devices can gemma see", "what ble devices can gemma see", "mediatek bluetooth packet", "mediatek ble packet", "top card bluetooth packet", "bluetooth top card packet") ->
                    bluetoothDiagnosticArguments("bluetooth_nearby_decision_packet_report", userText)
                lower.containsAny("motion decision packet", "motion sensor decision packet", "imu decision packet", "gyro accelerometer decision packet", "gyroscope accelerometer decision packet", "accelerometer decision packet", "gyroscope decision packet", "gemma motion sensor", "gemma imu", "top card motion packet", "motion top card packet", "mediatek motion packet", "motion claim boundary", "motion claim boundaries", "sensor decision packet") ->
                    diagnosticArguments("motion_sensor_decision_packet_report", "include_snapshot" to false)
                lower.containsAny("rf coexistence", "wireless coexistence", "wifi bluetooth coexistence", "wi-fi bluetooth coexistence", "bluetooth wifi interference", "bluetooth wi-fi interference", "cross signal interference", "2.4 ghz coexistence", "2.4ghz coexistence") ->
                    diagnosticArguments("rf_coexistence_report")
                lower.containsAny("signal timeline", "agent signal timeline", "gemma signal timeline", "what did the agent recently see", "what did hermes recently see", "recent signal view", "recently viewed signals") ->
                    diagnosticArguments("agent_signal_timeline_report")
                (lower.contains("export") && lower.containsAny("wifi", "wi-fi", "access point", "access points")) ||
                    lower.containsAny("export ap", "export aps", "ap export") ->
                    wifiDiagnosticArguments("wifi_export", userText)
                lower.containsAny("wifi ap details", "wi-fi ap details", "wifi access point details", "wi-fi access point details", "access point details", "complete access point", "compact access point", "ap details") ->
                    wifiDiagnosticArguments("wifi_ap_details", userText)
                lower.containsAny("wifi advisor", "wi-fi advisor", "wifi recommendation", "wi-fi recommendation", "wifi recommendations", "wi-fi recommendations", "wifi decision", "wi-fi decision", "wifi roaming", "wi-fi roaming", "roaming candidate", "roaming candidates", "best access point", "better access point") ->
                    wifiDiagnosticArguments("wifi_signal_advisor_report", userText)
                lower.containsAny("best wifi channel", "best wi-fi channel", "channel rating", "rate wifi", "rate wi-fi", "wifi congestion", "wi-fi congestion") ->
                    wifiDiagnosticArguments("wifi_channel_rating", userText)
                lower.containsAny("wifi utilization", "wi-fi utilization", "wifi occupancy", "wi-fi occupancy", "wifi interference", "wi-fi interference", "wifi spectrum", "wi-fi spectrum") ->
                    wifiDiagnosticArguments("wifi_channel_utilization", userText)
                lower.containsAny("wifi link", "wi-fi link", "wifi connection", "wi-fi connection", "current wifi", "current wi-fi", "connected wifi", "connected wi-fi", "current access point", "current ap") ->
                    diagnosticArguments("wifi_connection_link")
                lower.containsAny("wifi graph", "wi-fi graph", "wifi channel graph", "wi-fi channel graph", "wifi channel strength", "wi-fi channel strength") ->
                    wifiDiagnosticArguments("wifi_channel_graph", userText)
                lower.containsAny("wifi analyzer", "wi-fi analyzer", "analyze wifi", "analyze wi-fi", "wifi readiness", "wi-fi readiness", "wifi scan policy", "wi-fi scan policy") ->
                    wifiDiagnosticArguments("wifi_analyzer_report", userText)
                lower.containsAny("nearby wifi", "nearby wi-fi", "scan wifi", "scan wi-fi", "wifi networks", "wi-fi networks", "access points nearby", "nearby access points") ->
                    wifiDiagnosticArguments("wifi_scan", userText)
                lower.containsAny("bluetooth advisor", "ble advisor", "bluetooth recommendation", "bluetooth recommendations", "ble recommendation", "ble recommendations", "bluetooth decision", "ble decision", "bluetooth device recommendation", "bluetooth nearby advisor", "nearby bluetooth advisor", "nearby bluetooth scanner report") ->
                    bluetoothDiagnosticArguments("bluetooth_signal_advisor_report", userText)
                lower.containsAny("bluetooth history", "bluetooth trend", "bluetooth trends", "bluetooth rssi history", "ble history", "ble trend", "rssi trend") ->
                    bluetoothDiagnosticArguments("bluetooth_signal_history", userText)
                lower.containsAny("bluetooth analyzer", "bluetooth readiness", "bluetooth scan policy", "analyze bluetooth", "analyze ble") ->
                    bluetoothDiagnosticArguments("bluetooth_analyzer_report", userText)
                (lower.contains("export") && lower.containsAny("bluetooth", "ble", "bluetooth device", "bluetooth devices")) ||
                    lower.containsAny("export bluetooth", "export ble", "bluetooth export", "ble export") ->
                    bluetoothDiagnosticArguments("bluetooth_export", userText)
                lower.containsAny("bluetooth details", "bluetooth detail", "ble details", "ble detail", "bluetooth device details", "bluetooth device detail", "bluetooth device report", "complete bluetooth", "complete ble", "bluetooth metadata details", "ble metadata details") ->
                    bluetoothDiagnosticArguments("bluetooth_device_details", userText)
                lower.containsAny("nearby bluetooth", "nearby ble", "scan bluetooth", "scan ble", "bluetooth devices", "ble devices", "bluetooth scanner", "ble scanner", "bluetooth filter", "ble filter") ->
                    bluetoothDiagnosticArguments("bluetooth_scan", userText)
                lower.containsAny("sensor advisor", "sensor workflow", "motion sensor advisor", "motion sensor workflow", "accelerometer advisor", "gyroscope advisor", "gyro accelerometer advisor", "imu workflow advisor", "motion workflow recommendation", "sensor workflow recommendation") ->
                    diagnosticArguments("sensor_workflow_advisor_report", "include_snapshot" to false)
                lower.containsAny("motion quality", "sensor fusion quality", "imu quality", "gyro quality", "gyroscope quality", "accelerometer quality", "motion readiness", "orientation quality", "sensor calibration", "motion calibration") ->
                    diagnosticArguments("motion_sensor_quality", "include_snapshot" to false)
                lower.containsAny("motion history", "motion trend", "motion trends", "imu history", "imu trend", "accelerometer history", "gyroscope history") ->
                    diagnosticArguments(
                        "motion_sensor_history",
                        "sample" to true,
                        "sensor_types" to "accelerometer,gyroscope,linear_acceleration,rotation_vector",
                    )
                lower.containsAny("motion pose", "pose estimate", "heading estimate", "fused pose", "orientation estimate") ->
                    diagnosticArguments(
                        "motion_pose",
                        "sample" to true,
                        "sensor_types" to "accelerometer,magnetic_field,gyroscope,rotation_vector",
                    )
                lower.containsAny("sensor analyzer", "sensor readiness", "sensor sampling policy", "sensor metadata", "sensor capability", "accelerometer", "gyroscope", "gyrometer") ->
                    diagnosticArguments("sensor_analyzer_report", "include_snapshot" to false)
                lower.containsAny("radio decision packet", "radio signal decision packet", "radio signal packet", "am fm decision packet", "am/fm decision packet", "fm am radio decision", "sdr radio decision", "sdr decision packet", "gemma radio signal", "top card radio packet", "radio top card packet", "mediatek radio packet", "radio claim boundary", "radio claim boundaries") ->
                    diagnosticArguments("radio_signal_decision_packet_report")
                lower.containsAny("radio bridge sample", "radio bridge samples", "sdr bridge sample", "sdr bridge samples", "external sdr sample", "external sdr samples", "receiver sample", "receiver samples", "spectrum sample", "waterfall sample") ->
                    diagnosticArguments("radio_signal_graph")
                lower.containsAny("radio advisor", "radio signal advisor", "radio recommendation", "radio recommendations", "am fm advisor", "am/fm advisor", "fm advisor", "am advisor", "sdr advisor", "receiver advisor", "radio receiver advisor", "radio decision", "radio signal decision") ->
                    diagnosticArguments("radio_signal_advisor_report")
                lower.containsAny("radio graph", "rf graph", "fm graph", "am graph", "am/fm graph", "am fm graph", "signal graph") ->
                    diagnosticArguments("radio_signal_graph")
                lower.containsAny("fm radio", "am radio", "am/fm radio", "am fm radio", "radio signals", "rf signals", "radio scanner", "radio scan", "broadcast radio") ->
                    diagnosticArguments("radio_signal_status")
                lower.containsAny("mediatek signal stack", "mediatek signal compatibility", "mediatek signal context", "non adreno signal stack", "non-adreno signal stack", "soc signal stack", "mediatek wifi bluetooth sensors", "mediatek wi-fi bluetooth sensors") ->
                    diagnosticArguments("mediatek_signal_stack_report")
                lower.containsAny("mediatek readiness", "mediatek device readiness", "mediatek soc readiness", "dimensity readiness", "helio readiness", "mali readiness", "immortalis readiness", "powervr readiness", "img readiness", "non adreno readiness", "non-adreno readiness") ->
                    diagnosticArguments("mediatek_readiness_report")
                lower.containsAny("accelerator preflight", "gpu preflight", "delegate preflight", "litert preflight", "litert-lm preflight", "opencl preflight", "gpu delegate preflight", "mediatek accelerator", "non adreno accelerator", "non-adreno accelerator") ->
                    diagnosticArguments("accelerator_preflight_report")
                (
                    lower.contains("before starting local inference") &&
                        lower.containsAny("mediatek", "mali", "powervr", "power vr", "img", "xclipse", "tensor", "exynos", "unisoc", "unknown arm", "non adreno", "non-adreno")
                ) ||
                    lower.containsAny(
                    "mediatek launch checklist", "mediatek backend launch checklist", "non adreno launch checklist", "non-adreno launch checklist",
                    "mediatek local inference checklist", "non adreno local inference checklist", "non-adreno local inference checklist",
                    "before starting local inference on mediatek", "before starting local inference on mali",
                    "before starting local inference on powervr", "before starting local inference on power vr",
                    "before starting local inference on img", "before starting local inference on xclipse",
                    "before starting local inference on tensor", "before starting local inference on exynos",
                    "before starting local inference on unisoc", "before starting local inference on unknown arm",
                    "start local inference on mediatek", "start local inference on non adreno", "start local inference on non-adreno",
                ) ->
                    diagnosticArguments("mediatek_backend_launch_checklist_report")
                lower.containsAny(
                    "non adreno backend advisor", "non-adreno backend advisor",
                    "mediatek backend advisor", "backend launch advisor", "backend route advisor",
                    "non adreno launch", "non-adreno launch", "non adreno launch plan", "non-adreno launch plan",
                    "mediatek launch", "gpu backend advisor",
                ) ->
                    diagnosticArguments("non_adreno_backend_advisor_report")
                lower.containsAny("gpu backend risk", "backend risk", "accelerator risk", "non adreno backend", "mali backend", "powervr backend", "xclipse backend") ->
                    diagnosticArguments("gpu_backend_risk_report")
                lower.containsAny("local inference compatibility", "inference compatibility", "model compatibility", "inference fit", "litert compatibility", "gemma compatibility", "will gemma run", "can gemma run") ->
                    diagnosticArguments("local_inference_compatibility_report")
                lower.containsAny("mediatek compatibility", "mediatek", "dimensity", "helio", "soc compatibility", "soc backend", "mali", "powervr", "non snapdragon", "non-snapdragon") ->
                    diagnosticArguments("soc_compatibility_report")
                else -> null
            }
        }

        private fun String.containsAny(vararg needles: String): Boolean = needles.any { it in this }

        private fun wifiDiagnosticArguments(action: String, userText: String): JSONObject {
            val lower = userText.lowercase()
            val pairs = mutableListOf<Pair<String, Any>>()
            val scanMode = when {
                lower.containsAny("paused", "pause scanning", "pause scan", "cached", "reuse cached", "without refresh", "no refresh") -> "paused"
                lower.containsAny("resumed", "resume scanning", "resume scan", "fresh scan", "new scan", "refresh scan", "live scan") -> "resumed"
                else -> null
            }
            pairs += "refresh" to (scanMode == "resumed" && action != "wifi_analyzer_report")
            scanMode?.let { pairs += "scan_mode" to it }

            val bandFilters = buildList {
                if (lower.containsAny("2.4ghz", "2.4 ghz", "2g wifi", "2g wi-fi", "2 ghz")) add("2.4GHz")
                if (lower.containsAny("5ghz", "5 ghz", "5g wifi", "5g wi-fi")) add("5GHz")
                if (lower.containsAny("6ghz", "6 ghz", "6e", "wi-fi 6e", "wifi 6e")) add("6GHz")
                if (lower.containsAny("60ghz", "60 ghz")) add("60GHz")
            }.distinct()
            if (bandFilters.isNotEmpty()) pairs += "filter_band" to bandFilters.joinToString(",")

            val securityFilters = buildList {
                if (lower.contains("wpa3")) add("WPA3")
                if (lower.contains("wpa2")) add("WPA2")
                if (lower.contains("wpa ") || lower.endsWith("wpa")) add("WPA")
                if (lower.contains("wep")) add("WEP")
                if (lower.containsAny("enhanced open", "owe")) add("Enhanced Open")
                if (lower.containsAny("open wifi", "open wi-fi", "open network", "open networks", "unsecured wifi", "unsecured wi-fi")) add("Open")
            }.distinct()
            if (securityFilters.isNotEmpty()) pairs += "filter_security" to securityFilters.joinToString(",")

            val signalFilters = buildList {
                if (lower.containsAny("excellent signal", "strong signal", "strong wifi", "strong wi-fi")) add("excellent")
                if (lower.containsAny("good signal", "good wifi", "good wi-fi")) add("good")
                if (lower.containsAny("fair signal", "moderate signal", "medium signal")) add("fair")
                if (lower.containsAny("weak signal", "weak wifi", "weak wi-fi", "poor signal")) add("weak")
            }.distinct()
            if (signalFilters.isNotEmpty()) pairs += "filter_signal" to signalFilters.joinToString(",")

            when {
                lower.containsAny("hidden only", "only hidden", "hidden ssids only", "hidden networks only") -> pairs += "hidden_only" to true
                lower.containsAny("exclude hidden", "hide hidden", "without hidden") -> pairs += "include_hidden" to false
                lower.containsAny("include hidden", "show hidden", "with hidden") -> pairs += "include_hidden" to true
            }

            wifiTextFilter(userText, "ssid")?.let { pairs += "filter_ssid" to it }
            wifiTextFilter(userText, "vendor")?.let { pairs += "filter_vendor" to it }

            if (action == "wifi_export") {
                val exportFormat = when {
                    lower.contains("csv") -> "csv"
                    lower.contains("json") -> "json"
                    else -> "both"
                }
                pairs += "export_format" to exportFormat
            }
            return diagnosticArguments(action, *pairs.toTypedArray())
        }

        private fun wifiTextFilter(userText: String, key: String): String? {
            val pattern = Regex("""(?i)\b$key\s*(?:contains|named|name|vendor|=|:)?\s*["']?([A-Za-z0-9_.-]+(?: [A-Za-z0-9_.-]+){0,3})["']?(?=\s+(?:with|while|as|and|on|from|using|paused|resumed|fresh|cached|scan|details|export|networks|wifi|wi-fi)|[.!?]|$)""")
            val rawValue = pattern.find(userText)
                ?.groupValues
                ?.getOrNull(1)
                ?.trim()
                ?.trim('"', '\'')
                ?: return null
            val value = Regex("""(?i)\s+\b(with|while|as|and|on|from|using|paused|resumed|fresh|cached|scan|details|export|networks|wifi|wi-fi)\b.*$""")
                .replace(rawValue, "")
                .trim()
                .trimEnd('.', ',', ';')
                .takeIf { it.isNotBlank() }
                ?: return null
            return value.takeIf {
                it.lowercase() !in setOf("filter", "filters", "details", "export", "network", "networks", "wifi", "wi-fi")
            }
        }

        private fun bluetoothDiagnosticArguments(action: String, userText: String): JSONObject {
            val lower = userText.lowercase()
            val pairs = mutableListOf<Pair<String, Any>>()
            val scanMode = when {
                lower.containsAny("paused", "pause scanning", "pause scan", "cached", "reuse cached", "without refresh", "no refresh") -> "paused"
                lower.containsAny("resumed", "resume scanning", "resume scan", "fresh scan", "new scan", "refresh scan", "live scan") -> "resumed"
                else -> null
            }
            pairs += "refresh" to (scanMode == "resumed" && action != "bluetooth_analyzer_report")
            scanMode?.let { pairs += "scan_mode" to it }

            bluetoothTextFilter(userText, "name", "filter_device_name")?.let { pairs += "filter_device_name" to it }
            bluetoothTextFilter(userText, "device", "filter_device_name")?.let { pairs += "filter_device_name" to it }
            bluetoothTextFilter(userText, "address", "filter_bluetooth_address")?.let { pairs += "filter_bluetooth_address" to it }
            bluetoothTextFilter(userText, "service", "filter_bluetooth_service")?.let { pairs += "filter_bluetooth_service" to it }
            bluetoothTextFilter(userText, "manufacturer", "filter_bluetooth_manufacturer")?.let { pairs += "filter_bluetooth_manufacturer" to it }
            bluetoothTextFilter(userText, "category", "filter_bluetooth_category")?.let { pairs += "filter_bluetooth_category" to it }

            val proximityFilters = buildList {
                if (lower.containsAny("immediate proximity", "very close bluetooth", "very close ble", "immediate devices")) add("immediate")
                if (lower.containsAny("near proximity", "near bluetooth only", "near ble only", "close bluetooth devices", "close ble devices")) add("near")
                if (lower.containsAny("room proximity", "room bluetooth", "room ble", "moderate bluetooth", "moderate ble")) add("room")
                if (lower.containsAny("far proximity", "far bluetooth", "far ble", "weak bluetooth", "weak ble")) add("far")
            }.distinct()
            if (proximityFilters.isNotEmpty()) pairs += "filter_bluetooth_proximity" to proximityFilters.joinToString(",")

            if (action == "bluetooth_export") {
                val exportFormat = when {
                    lower.contains("csv") && lower.contains("json") -> "both"
                    lower.contains("csv") -> "csv"
                    lower.contains("json") -> "json"
                    else -> "both"
                }
                pairs += "export_format" to exportFormat
            }

            return diagnosticArguments(action, *pairs.toTypedArray())
        }

        private fun bluetoothTextFilter(userText: String, key: String, argumentName: String): String? {
            val pattern = Regex("""(?i)\b$key\s*(?:contains|named|name|label|id|=|:)?\s*["']?([A-Za-z0-9_.:-]+(?: [A-Za-z0-9_.:-]+){0,3})["']?(?=\s+(?:with|while|as|and|on|from|using|paused|resumed|fresh|cached|scan|devices|device|bluetooth|ble|nearby|history|trend|proximity|filter|details|detail|report|export|csv|json)|[.!?]|$)""")
            val rawValue = pattern.find(userText)
                ?.groupValues
                ?.getOrNull(1)
                ?.trim()
                ?.trim('"', '\'')
                ?: return null
            val value = Regex("""(?i)\s+\b(with|while|as|and|on|from|using|paused|resumed|fresh|cached|scan|devices|device|bluetooth|ble|nearby|history|trend|proximity|filter|details|detail|report|export|csv|json)\b.*$""")
                .replace(rawValue, "")
                .trim()
                .trimEnd('.', ',', ';')
                .takeIf { it.isNotBlank() }
                ?: return null
            return value.takeIf {
                it.lowercase() !in setOf("filter", "filters", "device", "devices", "bluetooth", "ble", "nearby", argumentName)
            }
        }

        private fun diagnosticArguments(action: String, vararg pairs: Pair<String, Any>): JSONObject {
            return JSONObject().put("action", action).apply {
                pairs.forEach { (key, value) -> put(key, value) }
            }
        }

        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
        private const val TOOL_TIMEOUT_SECONDS = 60
        private const val NATIVE_TOOL_GENERATION_TIMEOUT_MS = 300_000L
        private const val HTML_GENERATION_TIMEOUT_MS = 45_000L
        private const val NATIVE_TOOL_MAX_TOKENS = 1024
        private const val CONTEXT_RECOVERY_MAX_TOKENS = 512
        private const val HTML_GENERATION_MAX_TOKENS = 768
        private const val MAX_NATIVE_TOOL_ROUNDS = 6
        private const val PRIVILEGED_TOOL_TIMEOUT_SECONDS = 30
        private const val MAX_TOOL_RESULT_CHARS = 12_000
        private const val MAX_NATIVE_ERROR_CHARS = 360
        private const val DEFAULT_UI_SNAPSHOT_LIMIT = 80
        private const val MAX_OPEN_GUI_WORKING_MEMORY_CHARS = 16_000
        private val DIRECT_ANDROID_DEVICE_DIAGNOSTIC_ACTIONS = setOf(
            "status",
            "top_apps",
            "wifi_scan",
            "wifi_filtered_scan",
            "wifi_analyzer_report",
            "wifi_signal_advisor_report",
            "wifi_signal_advisor",
            "wifi_decision_report",
            "wifi_network_advisor",
            "wifi_roaming_advisor",
            "wifi_recommendation_report",
            "wifi_channel_decision_packet_report",
            "wifi_channel_decision_packet",
            "wifi_interference_decision_packet",
            "gemma_wifi_channel_decision",
            "mediatek_wifi_channel_decision",
            "wifi_router_channel_plan_report",
            "wifi_connection_link",
            "wifi_link_status",
            "wifi_current_connection",
            "wifi_current_ap",
            "wifi_current_network",
            "wifi_channel_graph",
            "wifi_channel_rating",
            "wifi_channel_utilization",
            "wifi_ap_details",
            "wifi_export",
            "bluetooth_scan",
            "bluetooth_analyzer_report",
            "bluetooth_signal_advisor_report",
            "bluetooth_signal_advisor",
            "bluetooth_nearby_advisor_report",
            "bluetooth_device_advisor_report",
            "ble_signal_advisor_report",
            "bluetooth_device_recommendation_report",
            "bluetooth_nearby_scanner_report",
            "bluetooth_nearby_decision_packet_report",
            "bluetooth_decision_packet",
            "ble_nearby_decision_packet",
            "gemma_bluetooth_nearby_packet",
            "top_card_bluetooth_packet",
            "mediatek_bluetooth_decision_packet",
            "bluetooth_signal_history",
            "bluetooth_device_details",
            "bluetooth_details",
            "bluetooth_device_report",
            "bluetooth_export",
            "bluetooth_device_export",
            "sensor_analyzer_report",
            "sensor_workflow_advisor_report",
            "sensor_workflow_advisor",
            "motion_sensor_advisor_report",
            "motion_sensor_workflow_report",
            "gyro_accelerometer_advisor_report",
            "imu_workflow_advisor_report",
            "motion_sensor_decision_packet_report",
            "motion_decision_packet",
            "imu_decision_packet",
            "gyro_accelerometer_decision_packet",
            "gemma_motion_sensor_packet",
            "top_card_motion_packet",
            "mediatek_motion_decision_packet",
            "motion_sensor_quality",
            "imu_quality_report",
            "motion_fusion_quality",
            "gyro_accel_quality",
            "sensor_fusion_quality",
            "motion_sensor_history",
            "motion_pose",
            "sensor_snapshot",
            "camera_status",
            "radio_signal_status",
            "radio_signal_graph",
            "radio_signal_advisor_report",
            "radio_signal_advisor",
            "radio_advisor_report",
            "am_fm_signal_advisor_report",
            "sdr_signal_advisor_report",
            "radio_receiver_advisor_report",
            "radio_signal_decision_packet_report",
            "radio_decision_packet",
            "am_fm_decision_packet",
            "fm_am_radio_decision_packet",
            "sdr_radio_decision_packet",
            "gemma_radio_signal_packet",
            "top_card_radio_packet",
            "mediatek_radio_decision_packet",
            "radio_bridge_sample_report",
            "radio_bridge_samples",
            "sdr_bridge_samples",
            "external_sdr_samples",
            "radio_analyzer_report",
            "signal_capability_status",
            "local_backend_runtime_report",
            "accelerator_preflight_report",
            "litert_accelerator_preflight_report",
            "gpu_delegate_preflight_report",
            "mediatek_accelerator_preflight_report",
            "non_adreno_accelerator_preflight_report",
            "opencl_preflight_report",
            "mediatek_readiness_report",
            "mediatek_device_readiness_report",
            "mediatek_soc_readiness_report",
            "non_adreno_readiness_report",
            "mediatek_signal_stack_report",
            "mediatek_signal_compatibility_report",
            "mediatek_signal_context_report",
            "non_adreno_signal_stack_report",
            "soc_signal_stack_report",
            "gemma_mediatek_signal_stack",
            "mediatek_device_validation_report",
            "physical_mediatek_validation_report",
            "non_adreno_device_validation_report",
            "signal_device_validation_report",
            "gemma_phone_validation_report",
            "phone_signal_validation_report",
            "device_validation_evidence_export_report",
            "mediatek_device_validation_export_report",
            "physical_device_evidence_export_report",
            "phone_validation_evidence_export",
            "github_release_device_evidence_export",
            "fdroid_device_evidence_export",
            "agent_signal_observation_packet_report",
            "gemma_signal_observation_packet",
            "signal_context_packet_report",
            "multimodal_signal_observation_report",
            "top_card_signal_observation_packet",
            "nearby_signal_context_packet",
            "dimensity_readiness_report",
            "helio_readiness_report",
            "non_adreno_backend_advisor_report",
            "mediatek_backend_advisor_report",
            "backend_route_advisor_report",
            "non_adreno_backend_launch_report",
            "mediatek_launch_advisor_report",
            "gpu_backend_advisor_report",
            "mediatek_backend_launch_checklist_report",
            "mediatek_launch_checklist_report",
            "non_adreno_launch_checklist_report",
            "gemma_mediatek_launch_checklist",
            "mediatek_local_inference_checklist",
            "non_adreno_local_inference_checklist",
            "soc_compatibility_report",
            "gpu_backend_risk_report",
            "local_inference_compatibility_report",
            "mediatek_inference_compatibility_report",
            "non_adreno_compatibility_report",
            "local_model_compatibility_scorecard",
            "device_performance_report",
            "signal_awareness_report",
            "rf_coexistence_report",
            "agent_signal_evidence_report",
            "signal_evidence_bundle",
            "current_signal_evidence",
            "gemma_signal_evidence",
            "agent_signal_replay_export_report",
            "signal_replay_export_report",
            "signal_replay_export",
            "signal_session_replay_export",
            "signal_evidence_export",
            "gemma_signal_replay_export",
            "portable_signal_bundle",
            "agent_signal_export_bundle",
            "agent_signal_replay_freshness_audit_report",
            "signal_replay_freshness_audit",
            "signal_export_freshness_audit",
            "gemma_signal_replay_freshness",
            "signal_replay_staleness_report",
            "signal_replay_freshness",
            "agent_signal_session_snapshot_report",
            "signal_session_snapshot_report",
            "gemma_signal_session_snapshot",
            "agent_signal_session_snapshot",
            "signal_session_snapshot",
            "rf_signal_session_snapshot",
            "current_signal_session_snapshot",
            "agent_signal_proof_audit_report",
            "signal_proof_audit_report",
            "signal_proof_audit",
            "active_signal_proof_report",
            "current_signal_proof_report",
            "gemma_signal_proof_audit",
            "signal_claim_validation_report",
            "signal_evidence_validation_report",
            "agent_signal_briefing_report",
            "signal_briefing_report",
            "gemma_signal_briefing",
            "agent_top_card_slots",
            "signal_briefing_deck",
            "agent_signal_card_deck_report",
            "signal_card_deck_report",
            "gemma_signal_card_deck",
            "expanded_signal_cards",
            "top_signal_card_deck",
            "signal_top_card_deck",
            "agent_signal_card_refresh_plan_report",
            "signal_card_refresh_plan_report",
            "gemma_signal_card_refresh_plan",
            "top_card_refresh_plan",
            "signal_card_live_refresh_plan",
            "expanded_signal_refresh_plan",
            "agent_signal_card_refresh_status_report",
            "signal_card_refresh_status_report",
            "gemma_signal_card_refresh_status",
            "top_card_refresh_status",
            "expanded_signal_refresh_status",
            "signal_cards_can_refresh",
            "agent_signal_timeline_report",
            "signal_timeline_report",
            "gemma_signal_timeline",
            "agent_signal_timeline",
            "what_agent_recently_saw",
            "agent_signal_workflow_handoff_report",
            "signal_workflow_handoff_report",
            "gemma_signal_workflow_handoff",
            "agent_next_signal_action_report",
            "next_signal_action_report",
            "signal_next_actions",
            "agent_signal_permission_runbook_report",
            "signal_permission_runbook_report",
            "signal_refresh_runbook_report",
            "agent_signal_refresh_runbook_report",
            "active_signal_refresh_runbook",
            "signal_active_refresh_routes",
            "agent_observation_report",
            "agent_card_manifest_report",
            "agent_card_priority_report",
            "agent_top_cards_report",
            "agent_observation_planner",
            "top_signal_cards",
            "kai_card_priority_report",
            "mcp_tool_server_registry_report",
            "mcp_tool_registry_report",
            "mcp_server_registry_report",
            "mcp_server_parity_report",
            "kai_mcp_server_report",
            "tool_server_registry_report",
            "agent_objective_coverage_report",
            "objective_coverage_report",
            "hermes_objective_coverage_report",
            "agent_upgrade_coverage_report",
            "hermes_upgrade_coverage_report",
            "full_upgrade_coverage_report",
            "research_parity_coverage_report",
            "agent_release_validation_report",
            "release_validation_report",
            "github_release_validation_report",
            "github_release_readiness_report",
            "release_validation_readiness_report",
            "fdroid_release_validation_report",
            "release_workflow_audit",
            "agent_capability_upgrade_report",
            "hermes_upgrade_audit_report",
            "full_system_upgrade_report",
            "gemma_capability_upgrade_report",
            "objective_upgrade_report",
            "card_manifest_report",
            "diagnostic_card_manifest",
            "graph_card_manifest",
            "agent_environment_report",
            "agent_self_check_report",
            "social_gmail_goal_preflight",
            "show_active_overlay",
            "tool_catalog",
            "open_usage_access_settings",
            "open_app_settings",
            "open_location_settings",
            "open_wifi_settings",
            "open_bluetooth_settings",
            "open_camera_permission_settings",
        )
        private val DIAGNOSTIC_ACTION_REGEX = Regex(
            """(?i)\b(?:action|operation|name)\s*[:=]\s*["']?([a-zA-Z0-9_/-]+)["']?""",
        )
        private val DIAGNOSTIC_BOOLEAN_ARGUMENTS = listOf(
            "refresh",
            "include_snapshot",
            "sample",
            "include_scan",
            "save_file",
            "include_hidden",
            "hidden_only",
        )
        private val DIAGNOSTIC_INTEGER_ARGUMENTS = listOf(
            "limit",
            "max_results",
            "timeout_ms",
            "detail_limit",
            "min_rssi_dbm",
            "max_rssi_dbm",
        )
        private val DIAGNOSTIC_STRING_ARGUMENTS = listOf(
            "export_format",
            "format",
            "scan_mode",
            "filter_band",
            "filter_security",
            "filter_signal",
            "filter_ssid",
            "filter_bssid",
            "filter_vendor",
            "filter_device_name",
            "filter_bluetooth_address",
            "filter_bluetooth_service",
            "filter_bluetooth_manufacturer",
            "filter_bluetooth_category",
            "filter_bluetooth_proximity",
            "sensor_types",
            "radio_samples_json",
            "radio_bridge_samples_json",
            "receiver_samples_json",
            "sdr_samples_json",
            "sdr_spectrum_samples_json",
            "spectrum_samples_json",
            "waterfall_rows_json",
            "sample_source",
            "receiver_id",
            "station_label",
            "frequency_mhz",
            "frequency_khz",
            "frequency_hz",
            "center_frequency_hz",
            "span_hz",
            "sample_rate_hz",
            "bin_width_hz",
            "bandwidth_hz",
            "rssi_dbm",
            "power_db",
            "snr_db",
            "modulation",
            "rds_program_service",
            "rds_radio_text",
        )
        private val DIAGNOSTIC_BOOLEAN_REGEXES = DIAGNOSTIC_BOOLEAN_ARGUMENTS.associateWith { key ->
            Regex("""(?i)\b${Regex.escape(key)}\s*[:=]\s*(true|false)\b""")
        }
        private val DIAGNOSTIC_INTEGER_REGEXES = DIAGNOSTIC_INTEGER_ARGUMENTS.associateWith { key ->
            Regex("""(?i)\b${Regex.escape(key)}\s*[:=]\s*(-?\d+)\b""")
        }
        private val DIAGNOSTIC_STRING_REGEXES = DIAGNOSTIC_STRING_ARGUMENTS.associateWith { key ->
            Regex("""(?i)\b${Regex.escape(key)}\s*[:=]\s*(?:"([^"]+)"|'([^']+)'|([a-zA-Z0-9_,.:\-]+))""")
        }
        private val ANDROID_UI_ACTIONS = listOf(
            "status",
            "sense",
            "opengui_sense",
            "perception_status",
            "snapshot",
            "a11y_tree",
            "accessibility_tree",
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
        private val HTML_PATH_REGEX = Regex("""[A-Za-z0-9._/-]+\.html?""", RegexOption.IGNORE_CASE)
        private val MARKER_REGEX = Regex("""\b[A-Z][A-Z0-9_]{5,}\b""")
        private val FILE_WRITE_WITH_CONTENT_REGEX = Regex(
            pattern = """(?is)\b(?:write|create|save)\s+(?:"(?<double>[^"]+)"|'(?<single>[^']+)'|(?<bare>[A-Za-z0-9._/-]+))\s+with\s+content\s+(?<content>.+?)(?:\.\s+(?:after|then)\b|$)""",
        )
    }

    private data class ExplicitFileWriteRequest(
        val path: String,
        val content: String,
    )

    private data class ExplicitHtmlBrowserRequest(
        val path: String,
        val marker: String?,
    )
}

internal object NativeToolContextCompressor {
    private const val MAX_TOOL_RESULT_CONTEXT_CHARS = 3_200
    private const val MAX_NATIVE_MESSAGE_CONTEXT_CHARS = 12_000
    private const val MAX_CUSTOM_SYSTEM_PROMPT_CONTEXT_CHARS = 900
    private const val MAX_PROMOTED_MEMORY_CONTEXT_CHARS = 1_200
    private const val MAX_RECOVERED_TOTAL_MESSAGE_CONTEXT_CHARS = 8_000
    private const val MAX_RECOVERED_SYSTEM_MESSAGE_CHARS = 3_600
    private const val MAX_RECOVERED_USER_MESSAGE_CHARS = 2_400
    private const val MAX_RECOVERED_ASSISTANT_MESSAGE_CHARS = 1_600
    private const val MAX_RECOVERED_TOOL_MESSAGE_CHARS = 1_600
    private const val MAX_RECOVERED_MESSAGE_CHARS = 1_200
    private const val MAX_RECOVERED_TOOL_DESCRIPTION_CHARS = 240
    private const val MAX_RECOVERED_TOOL_PROPERTY_DESCRIPTION_CHARS = 160
    private const val MAX_SUMMARY_CHARS = 2_400
    private const val STRING_FIELD_LIMIT = 600
    private const val OUTPUT_FIELD_LIMIT = 1_400

    fun compactToolResult(toolResult: String): String {
        if (toolResult.length <= MAX_TOOL_RESULT_CONTEXT_CHARS) {
            return toolResult
        }
        val parsed = runCatching { JSONObject(toolResult) }.getOrNull()
        if (parsed != null) {
            return compactJsonToolResult(parsed, toolResult.length)
        }
        return compactStringValue(toolResult, MAX_TOOL_RESULT_CONTEXT_CHARS)
    }

    fun compactMessages(messages: JSONArray): JSONArray {
        if (messages.toString().length <= MAX_NATIVE_MESSAGE_CONTEXT_CHARS) {
            return messages
        }
        val keepStart = latestAssistantBlockStart(messages)
        if (keepStart <= 2) {
            return messages
        }
        val compacted = JSONArray()
        for (index in 0 until minOf(2, messages.length())) {
            compacted.put(messages.get(index))
        }
        val summary = summarizeMessages(messages, start = 2, endExclusive = keepStart)
        if (summary.isNotBlank()) {
            compacted.put(
                JSONObject()
                    .put("role", "system")
                    .put(
                        "content",
                        "Hermes compacted prior native tool context to keep local mobile inference within context. " +
                            summary,
                    ),
            )
        }
        for (index in keepStart until messages.length()) {
            compacted.put(messages.get(index))
        }
        return compacted
    }

    fun compactCustomSystemPrompt(customSystemPrompt: String): String {
        return compactStringValue(customSystemPrompt.trim(), MAX_CUSTOM_SYSTEM_PROMPT_CONTEXT_CHARS)
    }

    fun compactPromotedMemoryContext(promotedMemoryContext: String): String {
        return compactStringValue(promotedMemoryContext.trim(), MAX_PROMOTED_MEMORY_CONTEXT_CHARS)
    }

    fun recoverMessagesAfterContextOverflow(messages: JSONArray): JSONArray {
        val recovered = compactMessagePayloads(compactMessages(messages))
        if (recovered.toString().length <= MAX_RECOVERED_TOTAL_MESSAGE_CONTEXT_CHARS) {
            return recovered
        }
        return latestTurnRecoveryMessages(recovered)
    }

    fun recoverToolSpecsAfterContextOverflow(toolSpecs: JSONArray?): JSONArray? {
        if (toolSpecs == null || toolSpecs.length() == 0) {
            return toolSpecs
        }
        val recovered = JSONArray()
        for (index in 0 until toolSpecs.length()) {
            val original = toolSpecs.optJSONObject(index) ?: continue
            recovered.put(compactToolSpecForRecovery(original))
        }
        return recovered
    }

    private fun compactMessagePayloads(messages: JSONArray): JSONArray {
        val recovered = JSONArray()
        for (index in 0 until messages.length()) {
            val message = messages.optJSONObject(index) ?: continue
            recovered.put(compactMessageForRecovery(message, index))
        }
        return recovered
    }

    private fun compactMessageForRecovery(message: JSONObject, index: Int): JSONObject {
        val copy = JSONObject(message.toString())
        val limit = when (copy.optString("role")) {
            "system" -> if (index == 0) MAX_RECOVERED_SYSTEM_MESSAGE_CHARS else MAX_RECOVERED_MESSAGE_CHARS
            "user" -> MAX_RECOVERED_USER_MESSAGE_CHARS
            "assistant" -> MAX_RECOVERED_ASSISTANT_MESSAGE_CHARS
            "tool" -> MAX_RECOVERED_TOOL_MESSAGE_CHARS
            else -> MAX_RECOVERED_MESSAGE_CHARS
        }
        if (copy.has("content") && !copy.isNull("content")) {
            copy.put("content", compactMessageContent(copy.opt("content"), limit))
        }
        return copy
    }

    private fun compactMessageContent(content: Any?, limit: Int): Any {
        return when (content) {
            null, JSONObject.NULL -> ""
            is String -> compactStringValue(content, limit)
            is JSONArray -> compactContentParts(content, limit)
            is JSONObject -> compactStringValue(content.toString(), limit)
            else -> compactStringValue(content.toString(), limit)
        }
    }

    private fun compactContentParts(parts: JSONArray, limit: Int): JSONArray {
        val compacted = JSONArray()
        var remainingTextChars = limit
        for (index in 0 until parts.length()) {
            val part = parts.optJSONObject(index) ?: continue
            val type = part.optString("type")
            if (type == "text") {
                val text = part.optString("text")
                val kept = compactStringValue(text, remainingTextChars.coerceAtLeast(240))
                compacted.put(JSONObject(part.toString()).put("text", kept))
                remainingTextChars -= kept.length
            } else if (type == "image_url") {
                compacted.put(
                    JSONObject()
                        .put("type", "text")
                        .put("text", "[Hermes omitted an attached image during context-window recovery.]"),
                )
            } else {
                compacted.put(
                    JSONObject()
                        .put("type", "text")
                        .put("text", compactStringValue(part.toString(), 240)),
                )
            }
        }
        return compacted
    }

    private fun latestTurnRecoveryMessages(messages: JSONArray): JSONArray {
        val recovered = JSONArray()
        messages.optJSONObject(0)?.let { system ->
            recovered.put(
                JSONObject(system.toString())
                    .put("content", compactStringValue(system.optString("content"), MAX_RECOVERED_SYSTEM_MESSAGE_CHARS / 2)),
            )
        }
        recovered.put(
            JSONObject()
                .put("role", "system")
                .put("content", "Hermes compressed earlier local chat turns after a context-window overflow and kept the latest actionable exchange."),
        )
        val tailStart = maxOf(1, messages.length() - 4)
        var includedUser = false
        for (index in tailStart until messages.length()) {
            val message = messages.optJSONObject(index) ?: continue
            if (message.optString("role") == "user") {
                includedUser = true
            }
            recovered.put(compactMessageForRecovery(message, index))
        }
        if (!includedUser) {
            messages.optJSONObject(1)?.let { user ->
                val withUser = JSONArray()
                for (index in 0 until minOf(2, recovered.length())) {
                    withUser.put(recovered.get(index))
                }
                withUser.put(compactMessageForRecovery(user, 1))
                for (index in 2 until recovered.length()) {
                    withUser.put(recovered.get(index))
                }
                return withUser
            }
        }
        return recovered
    }

    private fun compactToolSpecForRecovery(spec: JSONObject): JSONObject {
        val copy = JSONObject(spec.toString())
        val function = copy.optJSONObject("function") ?: return copy
        function.put(
            "description",
            singleLine(function.optString("description"), MAX_RECOVERED_TOOL_DESCRIPTION_CHARS),
        )
        compactParameterDescriptions(function.optJSONObject("parameters"))
        return copy
    }

    private fun compactParameterDescriptions(parameters: JSONObject?) {
        val properties = parameters?.optJSONObject("properties") ?: return
        for (key in properties.keys()) {
            val property = properties.optJSONObject(key) ?: continue
            if (property.has("description")) {
                property.put(
                    "description",
                    singleLine(
                        property.optString("description"),
                        if (key == "action") MAX_RECOVERED_TOOL_DESCRIPTION_CHARS else MAX_RECOVERED_TOOL_PROPERTY_DESCRIPTION_CHARS,
                    ),
                )
            }
            compactParameterDescriptions(property)
        }
    }

    private fun compactJsonToolResult(parsed: JSONObject, originalLength: Int): String {
        val compacted = JSONObject()
            .put("_hermes_context_compressed", true)
            .put("_original_chars", originalLength)
        for (key in parsed.keys()) {
            val value = parsed.opt(key)
            compacted.put(key, compactJsonValue(key, value))
        }
        val asText = compacted.toString()
        return if (asText.length <= MAX_TOOL_RESULT_CONTEXT_CHARS) {
            asText
        } else {
            validJsonFallback(compacted, originalLength, asText)
        }
    }

    private fun compactJsonValue(key: String, value: Any?): Any {
        return when (value) {
            null, JSONObject.NULL -> JSONObject.NULL
            is String -> compactStringValue(value, if (key == "output" || key == "error" || key.endsWith("_export_csv")) OUTPUT_FIELD_LIMIT else STRING_FIELD_LIMIT)
            is Number, is Boolean -> value
            is JSONArray -> compactJsonArray(key, value)
            is JSONObject -> {
                val text = value.toString()
                if (text.length <= STRING_FIELD_LIMIT) {
                    value
                } else {
                    JSONObject()
                        .put("type", "object")
                        .put("original_chars", text.length)
                        .put("summary", compactStringValue(text, STRING_FIELD_LIMIT))
                }
            }
            else -> compactStringValue(value.toString(), STRING_FIELD_LIMIT)
        }
    }

    private fun compactJsonArray(key: String, value: JSONArray): Any {
        val text = value.toString()
        if (text.length <= STRING_FIELD_LIMIT) {
            return value
        }
        if (key in PRESERVED_ARRAY_KEYS) {
            val items = JSONArray()
            val keepCount = minOf(value.length(), preservedArrayItemLimit(key))
            for (index in 0 until keepCount) {
                items.put(compactArrayItem(value.opt(index)))
            }
            return JSONObject()
                .put("type", "array")
                .put("original_count", value.length())
                .put("kept_count", keepCount)
                .put("items", items)
                .put("truncated", value.length() > keepCount)
        }
        return JSONObject()
            .put("type", "array")
            .put("original_chars", text.length)
            .put("summary", compactStringValue(text, STRING_FIELD_LIMIT))
    }

    private fun validJsonFallback(compacted: JSONObject, originalLength: Int, compactedText: String): String {
        val fallback = JSONObject()
            .put("_hermes_context_compressed", true)
            .put("_original_chars", originalLength)
            .put("_compression_level", "summary")
            .put("_summary", compactStringValue(compactedText, OUTPUT_FIELD_LIMIT))
        for (key in compacted.keys()) {
            if (key.startsWith("_") || fallback.has(key)) continue
            when (val value = compacted.opt(key)) {
                is String, is Number, is Boolean -> fallback.put(key, value)
                is JSONArray -> if (key == "cards" || key in PRESERVED_ARRAY_KEYS) fallback.put(key, value)
                is JSONObject -> if (key in PRESERVED_ARRAY_KEYS || key == "wifi_scan_status" || key == "wifi_scan_control" || key == "bluetooth_scan_status" || key == "bluetooth_scan_control" || key == "sensor_sampling_status" || key == "wifi_access_point_export" || key == "bluetooth_device_export") {
                    fallback.put(key, value)
                } else if (key == "current_local_backend" || key == "litert_runtime_health" || key == "soc_profile" || key == "device_performance_profile" || key == "preferred_local_model") {
                    fallback.put(key, value)
                }
            }
        }
        return fallback.toString()
    }

    private fun preservedArrayItemLimit(key: String): Int = when (key) {
        "wifi_signal_history" -> 4
        "motion_sensor_history", "cached_motion_sensor_history" -> 6
        else -> PRESERVED_ARRAY_ITEM_LIMIT
    }

    private fun compactArrayItem(value: Any?): Any {
        return when (value) {
            null, JSONObject.NULL -> JSONObject.NULL
            is String -> compactStringValue(value, 260)
            is Number, is Boolean -> value
            is JSONObject -> {
                val compacted = JSONObject()
                PRESERVED_OBJECT_KEYS.forEach { key ->
                    if (value.has(key) && !value.isNull(key)) {
                        compacted.put(key, compactJsonValue(key, value.opt(key)))
                    }
                }
                if (compacted.length() > 0) {
                    compacted
                } else {
                    compactStringValue(value.toString(), 360)
                }
            }
            else -> compactStringValue(value.toString(), 260)
        }
    }

    private fun latestAssistantBlockStart(messages: JSONArray): Int {
        var lastAssistantIndex = -1
        for (index in 2 until messages.length()) {
            val message = messages.optJSONObject(index) ?: continue
            if (message.optString("role") == "assistant") {
                lastAssistantIndex = index
            }
        }
        return lastAssistantIndex.takeIf { it >= 2 } ?: 2
    }

    private fun summarizeMessages(messages: JSONArray, start: Int, endExclusive: Int): String {
        if (start >= endExclusive) {
            return ""
        }
        val lines = mutableListOf<String>()
        var toolCount = 0
        for (index in start until endExclusive) {
            val message = messages.optJSONObject(index) ?: continue
            when (message.optString("role")) {
                "assistant" -> {
                    val toolNames = message.optJSONArray("tool_calls")
                        ?.let { calls ->
                            buildList {
                                for (callIndex in 0 until calls.length()) {
                                    calls.optJSONObject(callIndex)
                                        ?.optJSONObject("function")
                                        ?.optString("name")
                                        ?.takeIf { it.isNotBlank() }
                                        ?.let(::add)
                                }
                            }
                        }
                        .orEmpty()
                    val content = message.optString("content").takeIf { it.isNotBlank() && it != "null" }.orEmpty()
                    val assistantLine = buildString {
                        append("assistant")
                        if (toolNames.isNotEmpty()) {
                            append(" requested ")
                            append(toolNames.joinToString(", "))
                        }
                        if (content.isNotBlank()) {
                            append(": ")
                            append(singleLine(content, 220))
                        }
                    }
                    lines += assistantLine
                }
                "tool" -> {
                    toolCount += 1
                    val name = message.optString("name").ifBlank { "tool" }
                    val content = message.optString("content")
                    lines += "tool[$toolCount] $name: ${toolResultSummary(content)}"
                }
            }
        }
        val joined = lines.joinToString(" | ")
        return if (joined.length <= MAX_SUMMARY_CHARS) joined else compactStringValue(joined, MAX_SUMMARY_CHARS)
    }

    private fun toolResultSummary(content: String): String {
        val parsed = runCatching { JSONObject(content) }.getOrNull()
        if (parsed != null) {
            val parts = mutableListOf<String>()
            listOf(
                "exit_code",
                "success",
                "action",
                "path",
                "message",
                "error",
                "cwd",
                "result_count",
                "total_scan_result_count",
                "wifi_scan_age_ms",
                "wifi_vendor_count",
                "wifi_filter_count",
                "wifi_history_network_count",
                "wifi_access_point_detail_count",
                "wifi_access_point_semantic_count",
                "wifi_band_coverage_count",
                "wifi_channel_graph_count",
                "wifi_channel_utilization_count",
                "wifi_security_summary_count",
                "wifi_width_summary_count",
                "wifi_standard_summary_count",
                "wifi_connection_link_count",
                "ready_wifi_connection_link_count",
                "wifi_signal_advisor_count",
                "ready_wifi_signal_advisor_count",
                "wifi_roaming_candidate_count",
                "cached_wifi_history_network_count",
                "wifi_analyzer_feature_count",
                "ready_wifi_analyzer_feature_count",
                "wifi_analyzer_workflow_route_count",
                "wifi_scan_policy_count",
                "agent_capability_count",
                "ready_capability_count",
                "kai_parity_count",
                "kai_operations_count",
                "ready_kai_operations_count",
                "agent_tool_sandbox_count",
                "ready_agent_tool_sandbox_count",
                "mcp_tool_server_count",
                "ready_mcp_tool_server_count",
                "mcp_tool_server_route_count",
                "ready_mcp_tool_server_route_count",
                "agent_upgrade_objective_count",
                "ready_agent_upgrade_objective_count",
                "agent_upgrade_route_count",
                "ready_agent_upgrade_route_count",
                "workflow_readiness_count",
                "agent_observation_count",
                "ready_agent_observation_count",
                "agent_observation_route_count",
                "agent_signal_briefing_count",
                "ready_agent_signal_briefing_count",
                "agent_signal_card_deck_count",
                "ready_agent_signal_card_deck_count",
                "agent_signal_card_refresh_plan_count",
                "ready_agent_signal_card_refresh_plan_count",
                "agent_signal_card_refresh_status_count",
                "ready_agent_signal_card_refresh_status_count",
                "active_ready_agent_signal_card_refresh_status_count",
                "blocked_agent_signal_card_refresh_status_count",
                "mediatek_backend_launch_checklist_count",
                "ready_mediatek_backend_launch_checklist_count",
                "blocked_mediatek_backend_launch_checklist_count",
                "agent_top_card_slot_count",
                "ready_agent_top_card_slot_count",
                "agent_signal_metadata_key_count",
                "ready_agent_signal_metadata_key_count",
                "agent_signal_workflow_handoff_count",
                "ready_agent_signal_workflow_handoff_count",
                "agent_signal_next_action_route_count",
                "ready_agent_signal_next_action_route_count",
                "agent_signal_permission_runbook_count",
                "ready_agent_signal_permission_runbook_count",
                "agent_signal_active_refresh_route_count",
                "ready_agent_signal_active_refresh_route_count",
                "agent_signal_context_count",
                "ready_agent_signal_context_count",
                "agent_card_manifest_count",
                "ready_agent_card_manifest_count",
                "top_signal_card_priority_count",
                "ready_top_signal_card_priority_count",
                "agent_card_open_sequence_count",
                "ready_agent_card_open_sequence_count",
                "kai_interactive_screen_parity_count",
                "ready_kai_interactive_screen_parity_count",
                "signal_evidence_count",
                "ready_signal_evidence_count",
                "signal_evidence_route_count",
                "signal_evidence_graph_type_count",
                "agent_objective_coverage_count",
                "ready_agent_objective_coverage_count",
                "agent_objective_gap_count",
                "agent_research_parity_count",
                "agent_release_validation_count",
                "ready_agent_release_validation_count",
                "agent_release_artifact_gate_count",
                "fdroid_release_metadata_count",
                "signal_awareness_count",
                "ready_signal_awareness_count",
                "signal_workflow_route_count",
                "signal_constraint_count",
                "rf_coexistence_count",
                "ready_rf_coexistence_count",
                "rf_coexistence_route_count",
                "rf_coexistence_risk_score",
                "rf_coexistence_risk_level",
                "mediatek_signal_stack_count",
                "ready_mediatek_signal_stack_count",
                "mediatek_signal_refresh_route_count",
                "ready_mediatek_signal_refresh_route_count",
                "mediatek_signal_claim_boundary_count",
                "ready_mediatek_signal_claim_boundary_count",
                "radio_band_plan_count",
                "radio_signal_feature_count",
                "ready_radio_signal_feature_count",
                "radio_signal_workflow_route_count",
                "radio_signal_constraint_count",
                "radio_signal_graph_row_count",
                "radio_signal_graph_sample_count",
                "radio_signal_graph_bridge_ready",
                "radio_signal_advisor_count",
                "ready_radio_signal_advisor_count",
                "radio_receiver_candidate_count",
                "radio_receiver_bridge_schema_count",
                "accelerator_preflight_count",
                "ready_accelerator_preflight_count",
                "runtime_backend_feature_count",
                "ready_runtime_backend_feature_count",
                "runtime_stability_feature_count",
                "ready_runtime_stability_feature_count",
                "mediatek_readiness_score",
                "mediatek_readiness_level",
                "mediatek_readiness_count",
                "ready_mediatek_readiness_count",
                "non_adreno_backend_advisor_score",
                "non_adreno_backend_advisor_level",
                "non_adreno_backend_advisor_count",
                "ready_non_adreno_backend_advisor_count",
                "soc_backend_feature_count",
                "ready_soc_backend_feature_count",
                "soc_backend_route_count",
                "soc_backend_constraint_count",
                "gpu_backend_risk_count",
                "high_gpu_backend_risk_count",
                "ready_gpu_backend_risk_count",
                "gpu_backend_risk_route_count",
                "gpu_backend_risk_level",
                "gpu_backend_risk_score",
                "local_inference_compatibility_score",
                "local_inference_compatibility_level",
                "local_inference_compatibility_count",
                "ready_local_inference_compatibility_count",
                "camera_count",
                "sensor_count",
                "sensor_catalog_count",
                "sensor_capability_count",
                "requested_available_sensor_count",
                "sampled_sensor_count",
                "motion_sensor_count",
                "motion_sensor_history_count",
                "cached_motion_sensor_history_count",
                "motion_pose_estimate_count",
                "cached_motion_pose_estimate_count",
                "motion_sensor_quality_count",
                "ready_motion_sensor_quality_count",
                "motion_sensor_quality_score",
                "motion_sensor_quality_level",
                "ambient_sensor_count",
                "wake_up_sensor_count",
                "direct_channel_sensor_count",
                "sensor_analyzer_feature_count",
                "ready_sensor_analyzer_feature_count",
                "sensor_analyzer_workflow_route_count",
                "sensor_sampling_policy_count",
                "sensor_workflow_advisor_count",
                "ready_sensor_workflow_advisor_count",
                "sensor_workflow_candidate_count",
                "motion_sensor_decision_packet_count",
                "ready_motion_sensor_decision_packet_count",
                "motion_sensor_decision_route_count",
                "ready_motion_sensor_decision_route_count",
                "motion_sensor_claim_boundary_count",
                "ready_motion_sensor_claim_boundary_count",
                "bluetooth_device_count",
                "bluetooth_total_device_count",
                "bluetooth_active_filter_count",
                "applied_bluetooth_filter_count",
                "bluetooth_metadata_count",
                "bluetooth_service_uuid_count",
                "bluetooth_service_label_count",
                "bluetooth_manufacturer_id_count",
                "bluetooth_manufacturer_name_count",
                "bluetooth_signal_history_count",
                "bluetooth_filtered_device_count",
                "bluetooth_device_detail_count",
                "bluetooth_analyzer_feature_count",
                "ready_bluetooth_analyzer_feature_count",
                "bluetooth_analyzer_workflow_route_count",
                "bluetooth_scan_policy_count",
                "bluetooth_signal_advisor_count",
                "ready_bluetooth_signal_advisor_count",
                "bluetooth_device_candidate_count",
                "radio_receiver_profile_count",
                "ready_radio_receiver_profile_count",
                "usage_access_granted",
                "requires_usage_access_for_full_storage_rankings",
                "likely_mediatek",
                "likely_snapdragon",
                "low_ram_device",
                "power_save_mode",
                "media_performance_class",
            ).forEach { key ->
                if (parsed.has(key) && !parsed.isNull(key)) {
                    parts += "$key=${singleLine(parsed.optString(key), 180)}"
                }
            }
            parsed.optJSONArray("cards")?.let { cards ->
                val titles = buildList {
                    for (index in 0 until minOf(cards.length(), 4)) {
                        cards.optJSONObject(index)?.optString("title")?.takeIf { it.isNotBlank() }?.let(::add)
                    }
                }
                if (titles.isNotEmpty()) {
                    parts += "cards=${titles.joinToString(", ")}"
                }
            }
            if (parsed.optBoolean("_hermes_context_compressed", false)) {
                parts += "compressed_from=${parsed.optInt("_original_chars")} chars"
            }
            if (parts.isNotEmpty()) {
                return parts.joinToString("; ")
            }
        }
        return singleLine(content, 260)
    }

    private fun compactStringValue(value: String, limit: Int): String {
        if (value.length <= limit) {
            return value
        }
        val headLength = (limit * 2 / 3).coerceAtLeast(1)
        val tailLength = (limit - headLength - 80).coerceAtLeast(1)
        val omitted = (value.length - headLength - tailLength).coerceAtLeast(0)
        return value.take(headLength) +
            "\n[hermes context compressed; omitted $omitted chars]\n" +
            value.takeLast(tailLength)
    }

    private fun singleLine(value: String, limit: Int): String {
        return compactStringValue(value.replace(Regex("""\s+"""), " ").trim(), limit)
    }

    private val PRESERVED_ARRAY_KEYS = setOf(
        "cards",
        "top_memory_apps",
        "top_storage_apps",
        "wifi_networks",
        "wifi_access_point_details",
        "wifi_access_point_semantics",
        "wifi_channel_graph",
        "wifi_channel_ratings",
        "wifi_channel_utilization",
        "recommended_wifi_channels",
        "wifi_band_summary",
        "wifi_band_coverage",
        "wifi_vendor_summary",
        "wifi_analyzer_filters",
        "available_wifi_analyzer_filters",
        "filtered_wifi_analyzer_filters",
        "applied_wifi_filters",
        "wifi_connection_status",
        "wifi_security_summary",
        "wifi_channel_width_summary",
        "wifi_standard_summary",
        "wifi_connection_link",
        "wifi_signal_advisor_matrix",
        "wifi_roaming_candidates",
        "wifi_channel_decision_packet",
        "wifi_channel_decision_routes",
        "wifi_channel_decision_claim_boundaries",
        "wifi_channel_decision_graph_types",
        "gemma_wifi_channel_decision_directives",
        "wifi_signal_history",
        "wifi_analyzer_feature_matrix",
        "wifi_analyzer_workflow_routes",
        "wifi_scan_policy_matrix",
        "wifi_filter_application",
        "agent_capability_matrix",
        "kai_parity_matrix",
        "kai_operations_matrix",
        "agent_tool_sandbox_matrix",
        "mcp_tool_server_registry",
        "mcp_tool_server_routes",
        "agent_objective_coverage_matrix",
        "agent_objective_gap_matrix",
        "agent_research_parity_matrix",
        "gemma_objective_coverage_directives",
        "agent_release_validation_matrix",
        "agent_release_artifact_gates",
        "fdroid_release_metadata_matrix",
        "gemma_release_validation_directives",
        "agent_upgrade_objective_matrix",
        "agent_upgrade_route_matrix",
        "workflow_readiness_matrix",
        "agent_self_check_matrix",
        "agent_self_check_routes",
        "agent_signal_briefing_matrix",
        "agent_signal_card_deck_manifest",
        "gemma_signal_card_deck_directives",
        "agent_signal_card_refresh_plan_matrix",
        "gemma_signal_card_refresh_plan_directives",
        "agent_signal_card_refresh_status_matrix",
        "gemma_signal_card_refresh_status_directives",
        "agent_signal_session_snapshot_matrix",
        "agent_signal_session_domain_matrix",
        "agent_signal_session_action_routes",
        "gemma_signal_session_snapshot_directives",
        "agent_signal_proof_audit_matrix",
        "agent_signal_claim_boundary_matrix",
        "gemma_signal_proof_audit_directives",
        "agent_signal_replay_export_manifest",
        "agent_signal_replay_frame_index",
        "agent_signal_replay_metadata_keys",
        "agent_signal_replay_export_graph_types",
        "agent_signal_replay_export_bundle",
        "gemma_signal_replay_export_directives",
        "agent_signal_replay_freshness_matrix",
        "agent_signal_replay_refresh_routes",
        "agent_signal_replay_staleness_summary",
        "agent_signal_replay_freshness_graph_types",
        "gemma_signal_replay_freshness_directives",
        "agent_signal_observation_packet",
        "agent_signal_observation_visual_slots",
        "agent_signal_observation_graph_routes",
        "agent_signal_observation_claim_boundaries",
        "agent_signal_observation_packet_graph_types",
        "agent_signal_observation_packet_bundle",
        "gemma_signal_observation_packet_directives",
        "agent_top_card_slots",
        "agent_signal_metadata_keys",
        "agent_signal_workflow_handoff_matrix",
        "agent_signal_next_action_routes",
        "gemma_signal_workflow_handoff_directives",
        "agent_signal_permission_runbook_matrix",
        "agent_signal_active_refresh_routes",
        "gemma_signal_permission_runbook_directives",
        "gemma_signal_briefing_directives",
        "agent_observation_matrix",
        "agent_observation_routes",
        "agent_card_manifest",
        "agent_card_graph_types",
        "top_signal_card_priorities",
        "agent_card_open_sequence",
        "kai_interactive_screen_parity",
        "gemma_card_planner_directives",
        "gemma_mcp_registry_directives",
        "gemma_upgrade_audit_directives",
        "signal_evidence_matrix",
        "signal_evidence_routes",
        "signal_evidence_graph_types",
        "gemma_observation_directives",
        "source_report_actions",
        "signal_awareness_matrix",
        "signal_workflow_routes",
        "signal_constraint_matrix",
        "rf_coexistence_matrix",
        "rf_coexistence_routes",
        "mediatek_signal_stack_matrix",
        "mediatek_signal_refresh_routes",
        "mediatek_signal_claim_boundaries",
        "mediatek_signal_graph_types",
        "gemma_mediatek_signal_directives",
        "mediatek_device_validation_matrix",
        "live_signal_validation_routes",
        "release_device_proof_gates",
        "mediatek_device_validation_graph_types",
        "gemma_device_validation_directives",
        "device_validation_evidence_manifest",
        "device_validation_required_artifacts",
        "phone_validation_command_routes",
        "github_release_evidence_routes",
        "fdroid_evidence_routes",
        "device_validation_evidence_graph_types",
        "device_validation_evidence_export_bundle",
        "gemma_device_validation_export_directives",
        "accelerator_preflight_matrix",
        "runtime_backend_matrix",
        "runtime_stability_matrix",
        "mediatek_readiness_matrix",
        "non_adreno_backend_advisor_matrix",
        "non_adreno_backend_launch_sequence",
        "mediatek_backend_launch_checklist_matrix",
        "gemma_mediatek_launch_directives",
        "gemma_non_adreno_backend_directives",
        "soc_backend_matrix",
        "soc_backend_policy_routes",
        "soc_backend_constraint_matrix",
        "gpu_backend_risk_matrix",
        "gpu_backend_risk_routes",
        "local_inference_compatibility_matrix",
        "cached_wifi_signal_history",
        "cached_bluetooth_signal_history",
        "cached_motion_sensor_history",
        "ai_experience_elevation_plan",
        "bluetooth_devices",
        "bluetooth_device_details",
        "bluetooth_metadata_summary",
        "bluetooth_analyzer_filters",
        "available_bluetooth_analyzer_filters",
        "filtered_bluetooth_analyzer_filters",
        "applied_bluetooth_filters",
        "bluetooth_signal_history",
        "bluetooth_analyzer_feature_matrix",
        "bluetooth_analyzer_workflow_routes",
        "bluetooth_scan_policy_matrix",
        "bluetooth_signal_advisor_matrix",
        "bluetooth_device_candidates",
        "gemma_bluetooth_advisor_directives",
        "bluetooth_nearby_decision_packet",
        "bluetooth_nearby_decision_routes",
        "bluetooth_nearby_claim_boundaries",
        "bluetooth_nearby_decision_graph_types",
        "gemma_bluetooth_nearby_directives",
        "bluetooth_filter_application",
        "radio_bands",
        "radio_receiver_profiles",
        "radio_receiver_bridge_schema",
        "radio_signal_graph_rows",
        "radio_signal_graph_sample_rows",
        "radio_bridge_sample_metadata",
        "radio_signal_feature_matrix",
        "radio_signal_workflow_routes",
        "radio_signal_constraint_matrix",
        "radio_signal_decision_packet",
        "radio_signal_decision_routes",
        "radio_signal_claim_boundaries",
        "radio_signal_decision_graph_types",
        "gemma_radio_signal_decision_directives",
        "radio_signal_advisor_matrix",
        "radio_receiver_candidates",
        "gemma_radio_advisor_directives",
        "radio_scan_rows",
        "agent_signal_context_matrix",
        "sensor_samples",
        "sensor_capabilities",
        "motion_sensor_history",
        "motion_pose_estimates",
        "motion_sensor_quality",
        "cached_motion_pose_estimates",
        "source_sensors",
        "available_sensor_types",
        "requested_sensor_types",
        "supported_watcher_types",
        "sensor_analyzer_feature_matrix",
        "sensor_analyzer_workflow_routes",
        "sensor_sampling_policy_matrix",
        "sensor_workflow_advisor_matrix",
        "sensor_workflow_candidates",
        "gemma_sensor_workflow_directives",
        "motion_sensor_decision_packet",
        "motion_sensor_decision_routes",
        "motion_sensor_claim_boundaries",
        "motion_sensor_decision_graph_types",
        "gemma_motion_sensor_decision_directives",
        "native_tools",
        "promoted_memories",
        "diagnostics_actions",
        "available_actions",
        "blocking_items",
        "external_send_safety_checks",
        "suggested_phone_sequence",
        "candidate_package_names",
        "installed_candidates",
        "memories",
        "retained_memories",
        "tags",
        "entities",
        "semantic_keywords",
    )
    private val PRESERVED_OBJECT_KEYS = listOf(
        "rank",
        "title",
        "body",
        "name",
        "description",
        "package_name",
        "label",
        "app_name",
        "installed",
        "enabled",
        "ready",
        "active",
        "filter_key",
        "active_filter_count",
        "total_network_count",
        "matched_network_count",
        "match_fraction",
        "requested_filters",
        "min_rssi_dbm",
        "max_rssi_dbm",
        "runtime_flavor",
        "destination_file_name",
        "destination_path",
        "file_exists",
        "file_bytes",
        "record_status",
        "record_status_message",
        "process_name",
        "backend_kind",
        "started",
        "base_url",
        "model_name",
        "source_model_path",
        "status_message",
        "health_url",
        "selected_on_device_backend",
        "offline_airplane_mode",
        "accelerator",
        "vision_accelerator",
        "audio_accelerator",
        "image_input_supported",
        "audio_input_supported",
        "modality_policy",
        "multimodal_fallback",
        "speculative_decoding",
        "speculative_decoding_supported",
        "mtp_policy",
        "gpu_policy",
        "gpu_attempted",
        "gpu_fallback_to_cpu",
        "gpu_backend_risk_level",
        "gpu_backend_risk_score",
        "opencl_available",
        "hardware_identity",
        "soc_family",
        "gpu_family",
        "litert_backend_order",
        "native_abi_strategy",
        "thermal_status",
        "thermal_status_label",
        "thermal_status_severity",
        "thermal_throttling_risk",
        "thermal_api_supported",
        "power_save_mode",
        "interactive",
        "low_ram_device",
        "memory_class_mb",
        "large_memory_class_mb",
        "available_memory_label",
        "total_memory_label",
        "memory_pressure_low",
        "memory_threshold_label",
        "app_data_free_label",
        "media_performance_class",
        "media_performance_class_label",
        "battery_status_label",
        "battery_plugged_label",
        "battery_level_percent",
        "battery_temperature_celsius",
        "context_window_policy",
        "memory_label",
        "storage_label",
        "ssid",
        "display_ssid",
        "hidden_ssid",
        "bssid",
        "bssid_oui",
        "bssid_vendor",
        "rssi_dbm",
        "signal_quality",
        "frequency_mhz",
        "channel",
        "graph_x_channel",
        "graph_y_dbm",
        "channel_width",
        "channel_width_mhz",
        "graph_width_channels",
        "channel_span_start",
        "channel_span_end",
        "frequency_span_start_mhz",
        "frequency_span_end_mhz",
        "overlap_network_count",
        "same_channel_network_count",
        "overlap_pressure_score",
        "overlap_sample_ssids",
        "graph_shape",
        "wifi_standard",
        "security_mode",
        "semantic_label",
        "security_risk_label",
        "semantic_tags",
        "semantic_recommendation",
        "estimated_distance_m",
        "estimated_distance_meters",
        "center_freq0_mhz",
        "center_freq1_mhz",
        "passpoint_network",
        "80211mc_responder",
        "scan_age_ms",
        "frequency_hint_mhz",
        "sample_count",
        "current_rssi_dbm",
        "average_rssi_dbm",
        "min_rssi_dbm",
        "max_rssi_dbm",
        "trend_db",
        "trend_label",
        "last_seen_ms",
        "rssi_series",
        "observed_at_ms",
        "score",
        "rating_label",
        "channel_pressure_score",
        "utilization_label",
        "max_channel_width_mhz",
        "wide_channel_count",
        "security_modes",
        "network_count",
        "overlap_count",
        "strongest_rssi_dbm",
        "recommendation",
        "vendor",
        "bssid_ouis",
        "sample_ssids",
        "key",
        "value",
        "value_label",
        "ready",
        "detail",
        "fraction",
        "risk_level",
        "risk_score",
        "mitigation",
        "runtime_signal",
        "graph_type",
        "source_graph_type",
        "target_graph_type",
        "source_action",
        "card_title",
        "card_index",
        "refresh_policy",
        "decision_status",
        "bundle_kind",
        "schema_version",
        "export_status",
        "replay_frame",
        "frame_index",
        "frame_key",
        "replay_section",
        "row_count",
        "metadata_key_group",
        "metadata_keys",
        "freshness_status",
        "staleness_risk",
        "timeline_freshness",
        "timeline_source_action",
        "refresh_status_source",
        "proof_source",
        "route_type",
        "summary_key",
        "summary_count",
        "freshness_statuses",
        "slot_rank",
        "top_card_slot",
        "open_next_action",
        "user_visible_surface",
        "requires_permission",
        "source_report_success",
        "source_report_scope",
        "parity_source",
        "feature_source",
        "tool_action",
        "proof_action",
        "backend_risk_action",
        "evidence_key",
        "evidence_domain",
        "source_actions",
        "card_graph_types",
        "permission_gate",
        "settings_actions",
        "required_permissions",
        "required_settings",
        "active_refresh_arguments",
        "active_refresh_action",
        "passive_source_action",
        "passive_fallback_action",
        "status_label",
        "status_hint",
        "launch_step",
        "launch_gate_status",
        "next_best_action",
        "open_settings_action",
        "ready_for_active_refresh",
        "passive_refresh_available",
        "blocked_by_permission",
        "blocked_by_hardware",
        "live_runtime_proof",
        "cpu_fallback_explicit",
        "user_consent_required",
        "active_refresh_requires_physical_device",
        "runbook_status",
        "card_refresh_status",
        "hardware_gate",
        "constraint_type",
        "source_surface",
        "opencl_library_visible",
        "opencl_probe_loads_library",
        "visible_opencl_paths",
        "opencl_candidate_paths",
        "translated_arm64_on_x86",
        "non_adreno_policy_active",
        "litert_lm_backend_order",
        "sandbox_scope",
        "host_access",
        "remote_dispatch_capable",
        "mcp_parity_status",
        "mcp_server_name",
        "mcp_server_slug",
        "server_kind",
        "curated_server_category",
        "native_equivalent_action",
        "route_status",
        "remote_endpoint_required",
        "streamable_http_supported",
        "auto_reconnect_policy",
        "startup_reconnect_expected",
        "kai_readme_feature",
        "route_policy",
        "mcp_streamable_http_supported",
        "mcp_auto_reconnect_supported",
        "mcp_session_equivalent",
        "native_tool_count",
        "diagnostics_action_count",
        "evidence_status",
        "proof_status",
        "proof_level",
        "claim_scope",
        "mediatek_sensitive",
        "rf_coexistence_sensitive",
        "claim_ready",
        "claim_boundary",
        "active_evidence_present",
        "passive_evidence_present",
        "sensor_privacy_sensitive",
        "coverage_status",
        "bridge_required",
        "physical_device_validation_required",
        "release_validation_required",
        "next_verification_action",
        "research_source_url",
        "evidence_required",
        "terminal_commands",
        "workflow_file",
        "workflow_job",
        "artifact_globs",
        "expected_tag_pattern",
        "fdroid_metadata_path",
        "fastlane_metadata_path",
        "application_id",
        "report_scope",
        "count",
        "options",
        "bands",
        "channels",
        "sample_widths",
        "observed",
        "coverage_label",
        "rated_channel_count",
        "recommended_channel",
        "recommended_score",
        "channel_count",
        "visible_channels",
        "observed_widths",
        "observed_standards",
        "hidden_ssid_count",
        "security_attention_count",
        "csv_key",
        "json_array_key",
        "included_fields",
        "generated_at_ms",
        "row_count",
        "format",
        "device_name",
        "advertised_name",
        "address",
        "address_suffix",
        "device_type",
        "device_category",
        "major_device_class",
        "service_uuids",
        "service_labels",
        "service_uuid_count",
        "service_data_uuids",
        "service_data_labels",
        "manufacturer_ids",
        "manufacturer_names",
        "manufacturer_data_count",
        "manufacturer_data_bytes",
        "metadata_completeness_score",
        "evidence_summary",
        "scan_record_bytes",
        "candidate_score",
        "paired",
        "semantic_label",
        "display_label",
        "semantic_context",
        "proximity_label",
        "estimated_distance_meters",
        "evidence_count",
        "bond_state",
        "connectable",
        "tx_power_dbm",
        "summary_type",
        "paired_count",
        "connectable_count",
        "sample_devices",
        "band",
        "source_type",
        "supported",
        "sampled",
        "public_android_scan_supported",
        "built_in_android_source",
        "hardware_hint_supported",
        "vendor_bridge_possible",
        "requires_vendor_bridge",
        "requires_external_hardware",
        "data_available",
        "scan_state",
        "route_action",
        "channel_step",
        "access_path",
        "agent_usage",
        "metadata_fields",
        "metadata_keys",
        "metadata_key_count",
        "frequency_min_khz",
        "frequency_max_khz",
        "frequency_min_mhz",
        "frequency_max_mhz",
        "frequency_khz",
        "frequency_hz",
        "frequency_label",
        "center_frequency_hz",
        "receiver_id",
        "ready_metadata_count",
        "sample_source",
        "station_label",
        "modulation",
        "signal_dbuv_or_rssi_dbm",
        "power_db",
        "snr_db",
        "rds_program_service",
        "rds_radio_text",
        "bandwidth_hz",
        "span_hz",
        "sample_rate_hz",
        "bin_width_hz",
        "direct_argument_fields",
        "json_argument_keys",
        "sensor_type",
        "sensor_label",
        "sensor_name",
        "vendor",
        "version",
        "values",
        "current_values",
        "unit",
        "magnitude_unit",
        "current_magnitude",
        "average_magnitude",
        "min_magnitude",
        "max_magnitude",
        "trend_magnitude",
        "stability_delta",
        "stability_label",
        "magnitude_series",
        "quality_signal",
        "freshness_ms",
        "accuracy_labels",
        "fastest_sampling_hz",
        "total_power_ma",
        "pose_type",
        "pose_source",
        "source_sensors",
        "pose_label",
        "roll_degrees",
        "pitch_degrees",
        "tilt_degrees",
        "azimuth_degrees",
        "heading_label",
        "face_orientation_label",
        "confidence_label",
        "workflow_hint",
        "angular_velocity_rad_s",
        "acceleration_magnitude",
        "acceleration_delta_from_gravity",
        "motion_state_label",
        "timestamp_nanos",
        "maximum_range",
        "resolution",
        "power_ma",
        "min_delay_us",
        "max_delay_us",
        "reporting_mode",
        "wake_up",
        "dynamic_sensor",
        "direct_channel_supported",
        "highest_direct_report_rate_level",
        "fifo_reserved_event_count",
        "fifo_max_event_count",
        "accuracy_label",
        "available",
        "content",
        "source",
        "category",
        "hit_count",
        "salience",
        "promoted",
        "promotion_hit_threshold",
        "promoted_at_ms",
        "recall_score",
    )
    private const val PRESERVED_ARRAY_ITEM_LIMIT = 8
}
