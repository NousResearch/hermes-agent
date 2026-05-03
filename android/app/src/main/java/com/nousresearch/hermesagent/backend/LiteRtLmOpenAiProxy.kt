package com.nousresearch.hermesagent.backend

import android.content.Context
import android.os.Build
import android.util.Base64
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.tool
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.UUID

object LiteRtLmOpenAiProxy {
    @Volatile private var server: LiteRtLmServer? = null
    @Volatile private var activeModelPath: String = ""

    /** LiteRT-LM inference configuration from catalog entry or defaults */
    data class InferenceConfig(
        val topK: Int = 40,              // Edge Gallery default
        val topP: Float = 0.95f,         // Edge Gallery default
        val temperature: Float = 1.0f,   // Edge Gallery default
        val maxTokens: Int = -1,         // -1 = backend default
        val maxContextLength: Int = -1,  // -1 = backend default
        val supportImage: Boolean = false,
        val supportAudio: Boolean = false,
    )

    @Synchronized
    fun ensureRunning(
        context: Context,
        modelPath: String,
        requestedModelName: String,
        port: Int,
        inferenceConfig: InferenceConfig = InferenceConfig(),
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
        val artifactError = validateModelArtifact(modelPath)
        if (artifactError != null) {
            return LocalBackendStatus(
                backendKind = BackendKind.LITERT_LM,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = artifactError,
            )
        }
        return try {
            val newServer = LiteRtLmServer(
                context = context.applicationContext,
                modelPath = modelPath,
                requestedModelName = requestedModelName,
                port = port,
                inferenceConfig = inferenceConfig,
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

    private fun validateModelArtifact(modelPath: String): String? {
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            return "Preferred local model is missing on disk: $modelPath"
        }
        val header = ByteArray(8)
        val bytesRead = runCatching {
            modelFile.inputStream().use { it.read(header) }
        }.getOrElse { error ->
            return "Unable to inspect local LiteRT-LM model file: ${error.message ?: error.javaClass.simpleName}"
        }
        if (bytesRead <= 0) {
            return "Local LiteRT-LM model file is empty: ${modelFile.name}"
        }

        val lowerName = modelFile.name.lowercase(Locale.US)
        val startsWithLiteRtLm = bytesRead >= 8 &&
            header[0] == 'L'.code.toByte() &&
            header[1] == 'I'.code.toByte() &&
            header[2] == 'T'.code.toByte() &&
            header[3] == 'E'.code.toByte() &&
            header[4] == 'R'.code.toByte() &&
            header[5] == 'T'.code.toByte() &&
            header[6] == 'L'.code.toByte() &&
            header[7] == 'M'.code.toByte()
        val startsWithZip = bytesRead >= 4 &&
            header[0] == 'P'.code.toByte() &&
            header[1] == 'K'.code.toByte()
        val containsTfl3Magic = bytesRead >= 8 &&
            header[4] == 'T'.code.toByte() &&
            header[5] == 'F'.code.toByte() &&
            header[6] == 'L'.code.toByte() &&
            header[7] == '3'.code.toByte()

        return when {
            lowerName.endsWith(".litertlm") && !startsWithLiteRtLm ->
                "${modelFile.name} is not a valid LiteRT-LM bundle. Download the .litertlm artifact from the LiteRT-LM repo."
            lowerName.endsWith(".task") && containsTfl3Magic ->
                "${modelFile.name} is a web/browser .task FlatBuffer, not an Android LiteRT-LM zip bundle. Remove it and download the .litertlm artifact instead."
            lowerName.endsWith(".task") && !startsWithZip ->
                "${modelFile.name} is not an Android LiteRT-LM .task zip bundle. Download the .litertlm artifact instead."
            else -> null
        }
    }

    private class LiteRtLmServer(
        context: Context,
        modelPath: String,
        requestedModelName: String,
        port: Int,
        inferenceConfig: InferenceConfig = InferenceConfig(),
    ) : NanoHTTPD("127.0.0.1", port) {
        /** Engine initialization result with accelerator labels for each modality */
        data class EngineInitResult(
            val engine: Engine,
            val backend: String,
            val visionBackend: String,
            val audioBackend: String,
        )

        private val engineInitResult = initializeEngine(
            context = context,
            modelPath = modelPath,
            supportImage = inferenceConfig.supportImage,
            supportAudio = inferenceConfig.supportAudio,
        )
        private val engine = engineInitResult.engine
        private val runtimeBackendLabel = engineInitResult.backend
        private val visionBackendLabel = engineInitResult.visionBackend
        private val audioBackendLabel = engineInitResult.audioBackend
        private val supportsImageInput = inferenceConfig.supportImage

        val modelName: String = requestedModelName.ifBlank { File(modelPath).name }
        private val samplerConfig = SamplerConfig(
            topK = inferenceConfig.topK,
            topP = inferenceConfig.topP.toDouble(),
            temperature = inferenceConfig.temperature.toDouble(),
        )

        override fun serve(session: IHTTPSession): Response {
            return try {
                when {
                    session.method == Method.GET && session.uri == "/health" -> jsonResponse(
                        JSONObject().apply {
                            put("status", "ok")
                            put("backend", "litert-lm")
                            put("accelerator", runtimeBackendLabel)
                            put("vision_accelerator", visionBackendLabel)
                            put("audio_accelerator", audioBackendLabel)
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

        /**
         * Initialize LiteRT-LM engine with GPU-first strategy and multimodal backends.
         * Follows Edge Gallery pattern: GPU primary, CPU fallback.
         * For multimodal models: vision uses GPU, audio uses CPU.
         */
        private fun initializeEngine(
            context: Context,
            modelPath: String,
            supportImage: Boolean,
            supportAudio: Boolean,
        ): EngineInitResult {
            var lastError: Throwable? = null
            val backends = if (isTranslatedArm64OnX86(context)) {
                listOf(Backend.CPU() to "cpu")
            } else {
                listOf(
                    Backend.GPU() to "gpu",
                    Backend.CPU() to "cpu",
                )
            }
            for ((backend, label) in backends) {
                val candidate = Engine(
                    EngineConfig(
                        modelPath = modelPath,
                        backend = backend,
                        visionBackend = if (supportImage) Backend.GPU() else null,
                        audioBackend = if (supportAudio) Backend.CPU() else null,
                        maxNumImages = if (supportImage) 1 else null,
                        cacheDir = context.cacheDir.absolutePath,
                    )
                )
                try {
                    candidate.initialize()
                    return EngineInitResult(
                        engine = candidate,
                        backend = label,
                        visionBackend = if (supportImage) "gpu" else "none",
                        audioBackend = if (supportAudio) "cpu" else "none",
                    )
                } catch (error: Throwable) {
                    lastError = error
                    kotlin.runCatching { candidate.close() }
                }
            }
            throw lastError ?: IllegalStateException("LiteRT-LM engine initialization failed")
        }

        private fun isTranslatedArm64OnX86(context: Context): Boolean {
            val nativeLibraryDir = context.applicationInfo.nativeLibraryDir.orEmpty()
            val packageUsesArm64 = nativeLibraryDir.contains("/arm64") ||
                nativeLibraryDir.contains("\\arm64")
            val deviceSupportsX86 = Build.SUPPORTED_ABIS.any { it.startsWith("x86") }
            return packageUsesArm64 && deviceSupportsX86
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
            if (requestContainsImage(requestMessages) && !supportsImageInput) {
                return jsonResponse(
                    JSONObject().put(
                        "error",
                        "image input requires a LiteRT-LM model started with image support, such as Gemma 3n or Gemma 3 vision models",
                    ),
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
                    samplerConfig = samplerConfig,
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
                    "user" -> mapped += Message.user(extractMessageContents(message))
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

        private fun extractMessageContents(message: JSONObject): com.google.ai.edge.litertlm.Contents {
            val content = message.opt("content")
            val parts = when (content) {
                is JSONArray -> extractContentParts(content)
                is JSONObject -> listOfNotNull(content.optString("text").takeIf { it.isNotBlank() }?.let { Content.Text(it) })
                JSONObject.NULL, null -> emptyList()
                else -> listOf(Content.Text(content.toString()))
            }
            return com.google.ai.edge.litertlm.Contents.of(parts)
        }

        private fun extractContentParts(content: JSONArray): List<Content> {
            val parts = mutableListOf<Content>()
            for (index in 0 until content.length()) {
                val part = content.optJSONObject(index) ?: continue
                when (part.optString("type")) {
                    "text" -> {
                        val text = part.optString("text")
                        if (text.isNotBlank()) {
                            parts += Content.Text(text)
                        }
                    }
                    "image_url", "input_image" -> {
                        val imageUrl = part.optJSONObject("image_url")?.optString("url").orEmpty()
                            .ifBlank { part.optString("image_url") }
                            .ifBlank { part.optString("url") }
                        contentFromImageUrl(imageUrl)?.let { parts += it }
                    }
                }
            }
            return parts
        }

        private fun contentFromImageUrl(imageUrl: String): Content? {
            val url = imageUrl.trim()
            if (url.isBlank()) {
                return null
            }
            if (url.startsWith("data:", ignoreCase = true)) {
                val base64Payload = url.substringAfter("base64,", missingDelimiterValue = "")
                require(base64Payload.isNotBlank()) { "image_url data URI must include base64 data" }
                return Content.ImageBytes(Base64.decode(base64Payload, Base64.DEFAULT))
            }
            if (url.startsWith("file://", ignoreCase = true)) {
                return Content.ImageFile(url.removePrefix("file://"))
            }
            if (url.startsWith("/")) {
                return Content.ImageFile(url)
            }
            throw IllegalArgumentException("LiteRT-LM local vision only supports data: image URLs or app-local file paths")
        }

        private fun requestContainsImage(messages: JSONArray): Boolean {
            for (index in 0 until messages.length()) {
                val content = messages.optJSONObject(index)?.opt("content")
                if (content is JSONArray) {
                    for (partIndex in 0 until content.length()) {
                        val part = content.optJSONObject(partIndex) ?: continue
                        val type = part.optString("type")
                        if (type == "image_url" || type == "input_image") {
                            return true
                        }
                    }
                }
            }
            return false
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
