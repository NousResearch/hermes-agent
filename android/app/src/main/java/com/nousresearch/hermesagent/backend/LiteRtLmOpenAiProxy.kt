package com.nousresearch.hermesagent.backend

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.tool
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

object LiteRtLmOpenAiProxy {
    @Volatile private var server: LiteRtLmServer? = null
    @Volatile private var activeModelPath: String = ""

    @Synchronized
    fun ensureRunning(
        context: Context,
        modelPath: String,
        requestedModelName: String,
        port: Int,
    ): LocalBackendStatus {
        val current = server
        if (current != null && current.isAlive() && activeModelPath == modelPath) {
            return LocalBackendStatus(
                backendKind = BackendKind.LITERT_LM,
                started = true,
                baseUrl = "http://127.0.0.1:$port/v1",
                modelName = current.modelName,
                sourceModelPath = modelPath,
                statusMessage = "LiteRT-LM is serving locally through the in-app proxy",
            )
        }

        stop()
        return try {
            val newServer = LiteRtLmServer(
                context = context.applicationContext,
                modelPath = modelPath,
                requestedModelName = requestedModelName,
                port = port,
            )
            newServer.start(SOCKET_READ_TIMEOUT, false)
            server = newServer
            activeModelPath = modelPath
            LocalBackendStatus(
                backendKind = BackendKind.LITERT_LM,
                started = true,
                baseUrl = "http://127.0.0.1:$port/v1",
                modelName = newServer.modelName,
                sourceModelPath = modelPath,
                statusMessage = "LiteRT-LM is serving locally through the in-app proxy",
            )
        } catch (error: Throwable) {
            stop()
            LocalBackendStatus(
                backendKind = BackendKind.LITERT_LM,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = error.message ?: error.javaClass.simpleName,
            )
        }
    }

    @Synchronized
    fun stop() {
        server?.shutdown()
        server = null
        activeModelPath = ""
    }

    private class LiteRtLmServer(
        context: Context,
        modelPath: String,
        requestedModelName: String,
        port: Int,
    ) : NanoHTTPD("127.0.0.1", port) {
        private val initializedEngine = initializeEngine(context, modelPath)
        private val engine = initializedEngine.first
        private val runtimeBackendLabel = initializedEngine.second

        val modelName: String = requestedModelName.ifBlank { File(modelPath).name }

        override fun serve(session: IHTTPSession): Response {
            return try {
                when {
                    session.method == Method.GET && session.uri == "/health" -> jsonResponse(
                        JSONObject().apply {
                            put("status", "ok")
                            put("backend", "litert-lm")
                            put("accelerator", runtimeBackendLabel)
                            put("model", modelName)
                        }
                    )
                    session.method == Method.GET && session.uri == "/v1/models" -> jsonResponse(modelsPayload())
                    session.method == Method.POST && session.uri == "/v1/chat/completions" -> handleChatCompletions(session)
                    else -> jsonResponse(
                        JSONObject().put("error", "Not found"),
                        status = Response.Status.NOT_FOUND,
                    )
                }
            } catch (error: Throwable) {
                jsonResponse(
                    JSONObject().apply {
                        put("error", error.message ?: error.javaClass.simpleName)
                    },
                    status = Response.Status.INTERNAL_ERROR,
                )
            }
        }

        fun shutdown() {
            kotlin.runCatching { stop() }
            kotlin.runCatching { engine.close() }
        }

        private fun initializeEngine(context: Context, modelPath: String): Pair<Engine, String> {
            var lastError: Throwable? = null
            val backends = listOf(
                Backend.GPU() to "gpu",
                Backend.CPU() to "cpu",
            )
            for ((backend, label) in backends) {
                val candidate = Engine(
                    EngineConfig(
                        modelPath = modelPath,
                        backend = backend,
                        cacheDir = context.cacheDir.absolutePath,
                    )
                )
                try {
                    candidate.initialize()
                    return candidate to label
                } catch (error: Throwable) {
                    lastError = error
                    kotlin.runCatching { candidate.close() }
                }
            }
            throw lastError ?: IllegalStateException("LiteRT-LM engine initialization failed")
        }

        private fun handleChatCompletions(session: IHTTPSession): Response {
            val requestJson = readRequestJson(session)
            val requestMessages = requestJson.optJSONArray("messages") ?: JSONArray()
            if (requestMessages.length() == 0) {
                return jsonResponse(
                    JSONObject().put("error", "messages are required"),
                    status = Response.Status.BAD_REQUEST,
                )
            }

            val systemInstruction = buildSystemInstruction(requestMessages)
            val mappedMessages = mapMessages(requestMessages)
            val promptMessage = mappedMessages.lastOrNull()
                ?: return jsonResponse(
                    JSONObject().put("error", "no prompt message could be constructed"),
                    status = Response.Status.BAD_REQUEST,
                )
            val initialMessages = if (mappedMessages.size > 1) mappedMessages.dropLast(1) else emptyList()
            val toolProviders = buildToolProviders(requestJson.optJSONArray("tools"))
            val conversation = engine.createConversation(
                ConversationConfig(
                    systemInstruction = systemInstruction,
                    initialMessages = initialMessages,
                    tools = toolProviders,
                    automaticToolCalling = false,
                )
            )
            conversation.use { convo ->
                val responseMessage = convo.sendMessage(promptMessage, emptyMap())
                val payload = completionPayload(responseMessage)
                return if (requestJson.optBoolean("stream", false)) {
                    sseResponse(payload)
                } else {
                    jsonResponse(payload)
                }
            }
        }

        private fun buildSystemInstruction(messages: JSONArray): com.google.ai.edge.litertlm.Contents? {
            val systemText = buildString {
                for (index in 0 until messages.length()) {
                    val message = messages.optJSONObject(index) ?: continue
                    if (message.optString("role") == "system") {
                        val text = extractTextContent(message)
                        if (text.isNotBlank()) {
                            if (isNotBlank()) {
                                append("\n\n")
                            }
                            append(text)
                        }
                    }
                }
            }
            return systemText.ifBlank { null }?.let { com.google.ai.edge.litertlm.Contents.of(it) }
        }

        private fun mapMessages(messages: JSONArray): List<Message> {
            val toolIdToName = mutableMapOf<String, String>()
            val mapped = mutableListOf<Message>()
            for (index in 0 until messages.length()) {
                val message = messages.optJSONObject(index) ?: continue
                when (message.optString("role")) {
                    "system" -> Unit
                    "user" -> mapped += Message.user(extractTextContent(message))
                    "assistant" -> {
                        val content = extractTextContent(message)
                        val toolCalls = mutableListOf<ToolCall>()
                        val rawToolCalls = message.optJSONArray("tool_calls") ?: JSONArray()
                        for (toolIndex in 0 until rawToolCalls.length()) {
                            val toolCallJson = rawToolCalls.optJSONObject(toolIndex) ?: continue
                            val toolId = toolCallJson.optString("id")
                            val function = toolCallJson.optJSONObject("function") ?: JSONObject()
                            val name = function.optString("name").ifBlank { "tool" }
                            val arguments = jsonObjectToMap(parseJsonObject(function.optString("arguments", "{}")))
                            if (toolId.isNotBlank()) {
                                toolIdToName[toolId] = name
                            }
                            toolCalls += ToolCall(name, arguments)
                        }
                        mapped += Message.model(
                            contents = com.google.ai.edge.litertlm.Contents.of(
                                if (content.isBlank()) emptyList() else listOf(Content.Text(content))
                            ),
                            toolCalls = toolCalls,
                        )
                    }
                    "tool" -> {
                        val toolName = message.optString("name").ifBlank {
                            toolIdToName[message.optString("tool_call_id")] ?: "tool"
                        }
                        mapped += Message.tool(
                            com.google.ai.edge.litertlm.Contents.of(
                                Content.ToolResponse(toolName, parseJsonValue(message.optString("content")))
                            )
                        )
                    }
                }
            }
            return mapped
        }

        private fun buildToolProviders(rawTools: JSONArray?): List<com.google.ai.edge.litertlm.ToolProvider> {
            if (rawTools == null) {
                return emptyList()
            }
            val providers = mutableListOf<com.google.ai.edge.litertlm.ToolProvider>()
            for (index in 0 until rawTools.length()) {
                val toolJson = rawTools.optJSONObject(index) ?: continue
                val function = toolJson.optJSONObject("function") ?: continue
                val spec = JSONObject().apply {
                    put("name", function.optString("name"))
                    put("description", function.optString("description"))
                    put("parameters", function.optJSONObject("parameters") ?: JSONObject().put("type", "object"))
                }
                providers += tool(JsonSchemaTool(spec.toString()))
            }
            return providers
        }

        private fun completionPayload(responseMessage: Message): JSONObject {
            val toolCallsJson = JSONArray()
            responseMessage.toolCalls.forEachIndexed { index, toolCall ->
                toolCallsJson.put(
                    JSONObject().apply {
                        put("id", "call_${UUID.randomUUID()}_$index")
                        put("type", "function")
                        put(
                            "function",
                            JSONObject().apply {
                                put("name", toolCall.name)
                                put("arguments", mapToJsonObject(toolCall.arguments).toString())
                            }
                        )
                    }
                )
            }
            val content = responseMessage.toString()
            val finishReason = if (responseMessage.toolCalls.isNotEmpty()) "tool_calls" else "stop"
            return JSONObject().apply {
                put("id", "chatcmpl-${UUID.randomUUID()}")
                put("object", "chat.completion")
                put("created", System.currentTimeMillis() / 1000)
                put("model", modelName)
                put(
                    "choices",
                    JSONArray().put(
                        JSONObject().apply {
                            put("index", 0)
                            put(
                                "message",
                                JSONObject().apply {
                                    put("role", "assistant")
                                    put("content", if (content.isBlank()) JSONObject.NULL else content)
                                    if (toolCallsJson.length() > 0) {
                                        put("tool_calls", toolCallsJson)
                                    }
                                }
                            )
                            put("finish_reason", finishReason)
                        }
                    )
                )
                put(
                    "usage",
                    JSONObject().apply {
                        put("prompt_tokens", 0)
                        put("completion_tokens", 0)
                        put("total_tokens", 0)
                    }
                )
            }
        }

        private fun modelsPayload(): JSONObject {
            return JSONObject().apply {
                put(
                    "data",
                    JSONArray().put(
                        JSONObject().apply {
                            put("id", modelName)
                            put("object", "model")
                            put("owned_by", "litert-lm")
                        }
                    )
                )
                put("object", "list")
            }
        }

        private fun readRequestJson(session: IHTTPSession): JSONObject {
            val files = HashMap<String, String>()
            session.parseBody(files)
            val body = files["postData"].orEmpty()
            return JSONObject(body)
        }

        private fun jsonResponse(payload: JSONObject, status: Response.Status = Response.Status.OK): Response {
            return newFixedLengthResponse(status, "application/json", payload.toString())
        }

        private fun sseResponse(payload: JSONObject): Response {
            val delta = JSONObject().apply {
                put("id", "chatcmpl-${UUID.randomUUID()}")
                put("object", "chat.completion.chunk")
                put("created", System.currentTimeMillis() / 1000)
                put("model", modelName)
                put(
                    "choices",
                    JSONArray().put(
                        JSONObject().apply {
                            put("index", 0)
                            put(
                                "delta",
                                JSONObject().apply {
                                    put("role", "assistant")
                                    val message = payload.getJSONArray("choices").getJSONObject(0).getJSONObject("message")
                                    if (!message.isNull("content")) {
                                        put("content", message.optString("content"))
                                    }
                                    if (message.has("tool_calls")) {
                                        put("tool_calls", message.getJSONArray("tool_calls"))
                                    }
                                }
                            )
                            put("finish_reason", payload.getJSONArray("choices").getJSONObject(0).optString("finish_reason"))
                        }
                    )
                )
            }
            val body = buildString {
                append("data: ")
                append(delta.toString())
                append("\n\n")
                append("data: [DONE]\n\n")
            }
            return newFixedLengthResponse(Response.Status.OK, "text/event-stream", body)
        }

        private fun extractTextContent(message: JSONObject): String {
            val content = message.opt("content")
            return when (content) {
                is JSONArray -> buildString {
                    for (index in 0 until content.length()) {
                        val part = content.optJSONObject(index) ?: continue
                        if (part.optString("type") == "text") {
                            append(part.optString("text"))
                        }
                    }
                }
                is JSONObject -> content.optString("text")
                JSONObject.NULL, null -> ""
                else -> content.toString()
            }
        }

        private fun parseJsonValue(raw: String): Any? {
            val trimmed = raw.trim()
            if (trimmed.isBlank()) {
                return ""
            }
            return kotlin.runCatching {
                when {
                    trimmed.startsWith("{") -> jsonObjectToMap(JSONObject(trimmed))
                    trimmed.startsWith("[") -> jsonArrayToList(JSONArray(trimmed))
                    else -> raw
                }
            }.getOrDefault(raw)
        }

        private fun parseJsonObject(raw: String): JSONObject {
            return kotlin.runCatching { JSONObject(raw) }.getOrDefault(JSONObject())
        }

        private fun jsonObjectToMap(jsonObject: JSONObject): Map<String, Any?> {
            val result = linkedMapOf<String, Any?>()
            val keys = jsonObject.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                result[key] = jsonValueToAny(jsonObject.opt(key))
            }
            return result
        }

        private fun jsonArrayToList(jsonArray: JSONArray): List<Any?> {
            return buildList {
                for (index in 0 until jsonArray.length()) {
                    add(jsonValueToAny(jsonArray.opt(index)))
                }
            }
        }

        private fun jsonValueToAny(value: Any?): Any? {
            return when (value) {
                is JSONObject -> jsonObjectToMap(value)
                is JSONArray -> jsonArrayToList(value)
                JSONObject.NULL -> null
                else -> value
            }
        }

        private fun mapToJsonObject(value: Map<String, Any?>): JSONObject {
            val jsonObject = JSONObject()
            value.forEach { (key, item) ->
                jsonObject.put(key, anyToJson(item))
            }
            return jsonObject
        }

        private fun anyToJson(value: Any?): Any? {
            return when (value) {
                null -> JSONObject.NULL
                is Map<*, *> -> {
                    val jsonObject = JSONObject()
                    value.forEach { (key, item) ->
                        if (key != null) {
                            jsonObject.put(key.toString(), anyToJson(item))
                        }
                    }
                    jsonObject
                }
                is Iterable<*> -> JSONArray().apply { value.forEach { put(anyToJson(it)) } }
                else -> value
            }
        }

        private class JsonSchemaTool(private val spec: String) : OpenApiTool {
            override fun getToolDescriptionJsonString(): String = spec

            override fun execute(paramsJsonString: String): String {
                throw IllegalStateException("LiteRT-LM proxy uses manual tool-calling mode")
            }
        }
    }

    private const val SOCKET_READ_TIMEOUT = 0
}
