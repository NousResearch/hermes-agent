package com.nousresearch.hermesagent.ui.chat

import android.content.Context
import com.nousresearch.hermesagent.api.ChatContentPart
import com.nousresearch.hermesagent.api.ChatMessage
import com.nousresearch.hermesagent.api.HermesApiClient
import com.nousresearch.hermesagent.api.toJsonObject
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import com.nousresearch.hermesagent.device.NativeAndroidShellTool
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
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
        )

        repeat(MAX_TOOL_ROUNDS) {
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
            assistant = postChatCompletion(
                normalizedBaseUrl = normalizedBaseUrl,
                modelName = modelName,
                sessionId = sessionId,
                messages = messages,
                includeTools = true,
            )
        }

        val content = assistant.content.ifBlank {
            latestToolResult.ifBlank { "Tool call completed." }
        }
        return Result(content = content, executedToolCalls = executedToolCalls)
    }

    private fun postChatCompletion(
        normalizedBaseUrl: String,
        modelName: String,
        sessionId: String,
        messages: JSONArray,
        includeTools: Boolean,
    ): AssistantMessage {
        val payload = JSONObject()
            .put("model", modelName)
            .put("stream", false)
            .put("timeout_ms", NATIVE_TOOL_GENERATION_TIMEOUT_MS)
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

        val state = HermesLinuxSubsystemBridge.ensureInstalled(appContext)
        val homeDir = File(state.getString("home_path")).apply { mkdirs() }.canonicalFile
        val appFilesDir = appContext.filesDir.canonicalFile
        val target = if (File(rawPath).isAbsolute) {
            File(rawPath)
        } else {
            File(homeDir, rawPath)
        }.canonicalFile

        val allowedRoots = listOf(homeDir, appFilesDir)
        if (allowedRoots.none { root -> target == root || target.path.startsWith(root.path + File.separator) }) {
            return JSONObject()
                .put("exit_code", 13)
                .put("error", "file_write_tool can only write inside the Hermes app workspace")
                .put("path", target.absolutePath)
                .put("cwd", homeDir.absolutePath)
                .toString()
        }

        target.parentFile?.mkdirs()
        if (append) {
            target.appendText(content, Charsets.UTF_8)
        } else {
            target.writeText(content, Charsets.UTF_8)
        }

        return JSONObject()
            .put("exit_code", 0)
            .put("path", target.absolutePath)
            .put("bytes", target.length())
            .put("append", append)
            .put("cwd", homeDir.absolutePath)
            .toString()
    }

    private fun executeAndroidSystemTool(toolCall: ToolCall): String {
        val action = listOf("action", "operation", "name")
            .firstNotNullOfOrNull { key -> toolCall.arguments.optString(key).takeIf { it.isNotBlank() } }
            ?.trim()
            .orEmpty()
        return if (action.isBlank() || action == "status" || action == "read_status") {
            HermesSystemControlBridge.statusJson()
        } else {
            HermesSystemControlBridge.performActionJson(action)
        }
    }

    private fun systemMessage(): JSONObject {
        return JSONObject()
            .put("role", "system")
            .put(
                "content",
                "You are Hermes running inside the native Android app. " +
                    "You have functions named terminal_tool, file_write_tool, and android_system_tool. " +
                    "When the user asks to write or replace a text file, prefer file_write_tool so multiline content is written exactly. " +
                    "When the user asks to run a command, inspect the filesystem, read a file, or use a device command, call terminal_tool instead of simulating the result. " +
                    "terminal_tool runs through /system/bin/sh in the Hermes app workspace. " +
                    "file_write_tool writes UTF-8 text files in the Hermes app workspace. " +
                    "When the user asks about Android settings, phone connectivity, permissions, background runtime, or safe system panels, call android_system_tool. " +
                    "Protected Android settings require user-granted permissions or an opened settings panel.",
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
                                "Read Hermes Android phone/device status or perform a safe system action such as opening settings panels or starting/stopping the background runtime.",
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
                                                        "Use status to read device state, or one of the available system actions returned in status.",
                                                    ),
                                            ),
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
        private const val MAX_TOOL_ROUNDS = 2
        private const val TOOL_TIMEOUT_SECONDS = 60
        private const val NATIVE_TOOL_GENERATION_TIMEOUT_MS = 300_000L
        private const val MAX_TOOL_RESULT_CHARS = 12_000
    }
}
