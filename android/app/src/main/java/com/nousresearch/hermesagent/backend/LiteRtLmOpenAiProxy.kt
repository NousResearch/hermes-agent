package com.nousresearch.hermesagent.backend

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Base64
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Capabilities
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
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
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

object LiteRtLmOpenAiProxy {
    @Volatile private var server: LiteRtLmServer? = null
    @Volatile private var activeModelPath: String = ""
    @Volatile private var activeRuntimeConfigKey: String = ""

    /** LiteRT-LM inference configuration from catalog entry or defaults */
    data class InferenceConfig(
        val topK: Int = 40,              // Edge Gallery default
        val topP: Float = 0.95f,         // Edge Gallery default
        val temperature: Float = 1.0f,   // Edge Gallery default
        val maxTokens: Int = -1,         // -1 = backend default
        val maxContextLength: Int = -1,  // -1 = backend default
        val supportImage: Boolean = false,
        val supportAudio: Boolean = false,
        val speculativeDecodingMode: SpeculativeDecodingMode = SpeculativeDecodingMode.AUTO,
    )

    enum class SpeculativeDecodingMode {
        AUTO,
        ENABLED,
        DISABLED,
    }

    internal data class ModalityDecision(
        val supportImage: Boolean,
        val supportAudio: Boolean,
        val policy: String,
    )

    internal data class SpeculativeDecodingDecision(
        val supported: Boolean,
        val enabled: Boolean,
        val policy: String,
    )

    internal data class EngineTokenBudget(
        val value: Int?,
        val policy: String,
    )

    private const val DEFAULT_GENERATION_TIMEOUT_MS = 300_000L
    private const val MIN_GENERATION_TIMEOUT_MS = 5_000L
    private const val MAX_GENERATION_TIMEOUT_MS = 300_000L

    @Synchronized
    fun ensureRunning(
        context: Context,
        modelPath: String,
        requestedModelName: String,
        port: Int,
        inferenceConfig: InferenceConfig = InferenceConfig(),
    ): LocalBackendStatus {
        val current = server
        val requestedRuntimeConfigKey = inferenceConfig.runtimeConfigKey()
        if (
            current != null &&
            current.isAlive() &&
            activeModelPath == modelPath &&
            activeRuntimeConfigKey == requestedRuntimeConfigKey
        ) {
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
            activeRuntimeConfigKey = requestedRuntimeConfigKey
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
        activeRuntimeConfigKey = ""
    }

    private fun InferenceConfig.runtimeConfigKey(): String {
        return listOf(
            topK,
            topP,
            temperature,
            maxTokens,
            maxContextLength,
            supportImage,
            supportAudio,
            speculativeDecodingMode,
        ).joinToString("|")
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
            val supportsImageInput: Boolean,
            val supportsAudioInput: Boolean,
            val modalityPolicy: String,
            val speculativeDecoding: Boolean,
            val speculativeDecodingSupported: Boolean,
            val speculativeDecodingPolicy: String,
            val gpuPolicy: String,
            val maxNumTokens: Int?,
            val contextWindowPolicy: String,
        )

        private val engineMaxNumTokens = resolveEngineMaxNumTokens(
            context = context,
            modelPath = modelPath,
            requestedMaxTokens = inferenceConfig.maxTokens,
            requestedMaxContextLength = inferenceConfig.maxContextLength,
        )

        private val engineInitResult = initializeEngine(
            context = context,
            modelPath = modelPath,
            supportImage = inferenceConfig.supportImage,
            supportAudio = inferenceConfig.supportAudio,
            speculativeDecodingMode = inferenceConfig.speculativeDecodingMode,
            maxNumTokens = engineMaxNumTokens.value,
            contextWindowPolicy = engineMaxNumTokens.policy,
        )
        private val engine = engineInitResult.engine
        private val runtimeBackendLabel = engineInitResult.backend
        private val visionBackendLabel = engineInitResult.visionBackend
        private val audioBackendLabel = engineInitResult.audioBackend
        private val supportsImageInput = engineInitResult.supportsImageInput

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
                            put("image_input_supported", engineInitResult.supportsImageInput)
                            put("audio_input_supported", engineInitResult.supportsAudioInput)
                            put("modality_policy", engineInitResult.modalityPolicy)
                            put(
                                "multimodal_fallback",
                                engineInitResult.modalityPolicy.startsWith("text-only fallback") ||
                                    engineInitResult.modalityPolicy.startsWith("text-only memory guard"),
                            )
                            put("speculative_decoding", engineInitResult.speculativeDecoding)
                            put("speculative_decoding_supported", engineInitResult.speculativeDecodingSupported)
                            put("mtp_policy", engineInitResult.speculativeDecodingPolicy)
                            put("gpu_policy", engineInitResult.gpuPolicy)
                            put("max_num_tokens", engineInitResult.maxNumTokens ?: JSONObject.NULL)
                            put("context_window_policy", engineInitResult.contextWindowPolicy)
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
        @OptIn(ExperimentalApi::class)
        private fun initializeEngine(
            context: Context,
            modelPath: String,
            supportImage: Boolean,
            supportAudio: Boolean,
            speculativeDecodingMode: SpeculativeDecodingMode,
            maxNumTokens: Int?,
            contextWindowPolicy: String,
        ): EngineInitResult {
            var lastError: Throwable? = null
            val openClAvailable = hasLoadableOpenClLibrary()
            val gpuPolicy = gpuBackendPolicy(context, openClAvailable)
            val speculativeDecoding = speculativeDecodingDecision(context, modelPath, speculativeDecodingMode)
            val modalityDecision = memorySafeModalityDecision(
                totalRamBytes = totalDeviceRamBytes(context),
                modelBytes = runCatching { File(modelPath).length() }.getOrDefault(0L),
                requestedImage = supportImage,
                requestedAudio = supportAudio,
            )
            val backends = if (gpuPolicy.enabled) {
                listOf(
                    Backend.GPU() to "gpu",
                    Backend.CPU() to "cpu",
                )
            } else {
                listOf(Backend.CPU() to "cpu")
            }

            fun tryInitialize(
                requestedSupportImage: Boolean,
                requestedSupportAudio: Boolean,
                modalityPolicy: String,
            ): EngineInitResult? {
                for ((backend, label) in backends) {
                    val visionBackend = when {
                        !requestedSupportImage -> null
                        label == "gpu" -> Backend.GPU()
                        else -> Backend.CPU()
                    }
                    val visionBackendLabel = when {
                        !requestedSupportImage -> "none"
                        label == "gpu" -> "gpu"
                        else -> "cpu"
                    }
                    val attempts = if (speculativeDecoding.enabled) {
                        listOf(
                            true to speculativeDecoding.policy,
                            false to "disabled: Gemma 4 MTP failed during $label engine initialization; retried without MTP",
                        )
                    } else {
                        listOf(false to speculativeDecoding.policy)
                    }
                    for ((enableMtp, mtpPolicy) in attempts) {
                        var candidate: Engine? = null
                        try {
                            ExperimentalFlags.enableSpeculativeDecoding = enableMtp
                            candidate = Engine(
                                EngineConfig(
                                    modelPath = modelPath,
                                    backend = backend,
                                    visionBackend = visionBackend,
                                    audioBackend = if (requestedSupportAudio) Backend.CPU() else null,
                                    maxNumImages = if (requestedSupportImage) 1 else null,
                                    maxNumTokens = maxNumTokens,
                                    cacheDir = context.cacheDir.absolutePath,
                                )
                            )
                            candidate.initialize()
                            ExperimentalFlags.enableSpeculativeDecoding = false
                            return EngineInitResult(
                                engine = candidate,
                                backend = label,
                                visionBackend = visionBackendLabel,
                                audioBackend = if (requestedSupportAudio) "cpu" else "none",
                                supportsImageInput = requestedSupportImage,
                                supportsAudioInput = requestedSupportAudio,
                                modalityPolicy = modalityPolicy,
                                speculativeDecoding = enableMtp,
                                speculativeDecodingSupported = speculativeDecoding.supported,
                                speculativeDecodingPolicy = mtpPolicy,
                                gpuPolicy = gpuPolicy.description,
                                maxNumTokens = maxNumTokens,
                                contextWindowPolicy = contextWindowPolicy,
                            )
                        } catch (error: Throwable) {
                            lastError = error
                            ExperimentalFlags.enableSpeculativeDecoding = false
                            kotlin.runCatching { candidate?.close() }
                        }
                    }
                }
                return null
            }

            tryInitialize(
                modalityDecision.supportImage,
                modalityDecision.supportAudio,
                modalityDecision.policy,
            )?.let { return it }

            val multimodalError = lastError
            if (modalityDecision.supportImage || modalityDecision.supportAudio) {
                val fallbackPolicy =
                    "text-only fallback: multimodal adapter initialization failed on this device (${shortError(multimodalError)})"
                tryInitialize(
                    requestedSupportImage = false,
                    requestedSupportAudio = false,
                    modalityPolicy = fallbackPolicy,
                )?.let { return it }
                throw IllegalStateException(
                    "LiteRT-LM text-only fallback also failed after multimodal adapter initialization failed. " +
                        "Multimodal: ${shortError(multimodalError)}; text-only: ${shortError(lastError)}",
                    lastError ?: multimodalError,
                )
            }
            throw lastError ?: IllegalStateException("LiteRT-LM engine initialization failed")
        }

        private fun shortError(error: Throwable?): String {
            return error?.message
                ?.lineSequence()
                ?.firstOrNull { it.isNotBlank() }
                ?.trim()
                ?.take(180)
                ?: error?.javaClass?.simpleName
                ?: "unknown error"
        }

        private data class GpuBackendPolicy(
            val enabled: Boolean,
            val description: String,
        )

        private fun resolveEngineMaxNumTokens(
            context: Context,
            modelPath: String,
            requestedMaxTokens: Int,
            requestedMaxContextLength: Int,
        ): EngineTokenBudget {
            return decideEngineTokenBudget(
                requestedMaxTokens = requestedMaxTokens,
                requestedMaxContextLength = requestedMaxContextLength,
                totalRamBytes = totalDeviceRamBytes(context),
                modelBytes = runCatching { File(modelPath).length() }.getOrDefault(0L),
                isX86Device = Build.SUPPORTED_ABIS.any { it.startsWith("x86") },
            )
        }

        private fun totalDeviceRamBytes(context: Context): Long {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
                ?: return 0L
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            return memoryInfo.totalMem
        }

        private fun gpuBackendPolicy(context: Context, openClAvailable: Boolean): GpuBackendPolicy {
            if (isTranslatedArm64OnX86(context)) {
                return GpuBackendPolicy(
                    enabled = false,
                    description = "disabled: translated arm64 package on x86 emulator/device",
                )
            }
            if (Build.SUPPORTED_ABIS.any { it.startsWith("x86") }) {
                return GpuBackendPolicy(
                    enabled = false,
                    description = "disabled: x86 emulator/device build",
                )
            }
            if (openClAvailable) {
                return GpuBackendPolicy(
                    enabled = true,
                    description = "enabled: OpenCL library was loadable",
                )
            }
            if (Build.SUPPORTED_ABIS.any { it.startsWith("arm") }) {
                val identity = androidHardwareIdentity()
                val deviceLabel = if (listOf("qualcomm", "qcom", "snapdragon", "adreno").any { it in identity }) {
                    "ARM Qualcomm/Adreno"
                } else {
                    "ARM Android"
                }
                return GpuBackendPolicy(
                    enabled = true,
                    description = "enabled: $deviceLabel device; attempting LiteRT-LM GPU with CPU fallback even though OpenCL probe was not loadable",
                )
            }
            return GpuBackendPolicy(
                enabled = false,
                description = "disabled: no ARM ABI or loadable OpenCL GPU path detected",
            )
        }

        private fun androidHardwareIdentity(): String {
            val socManufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MANUFACTURER else ""
            val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else ""
            return listOf(
                Build.MANUFACTURER,
                Build.BRAND,
                Build.DEVICE,
                Build.HARDWARE,
                Build.BOARD,
                socManufacturer,
                socModel,
            ).joinToString(" ").lowercase(Locale.US)
        }

        private fun speculativeDecodingDecision(
            context: Context,
            modelPath: String,
            mode: SpeculativeDecodingMode,
        ): SpeculativeDecodingDecision {
            val supported = runCatching {
                Capabilities(modelPath).use { capabilities ->
                    capabilities.hasSpeculativeDecodingSupport()
                }
            }.getOrDefault(false)
            val modelFile = File(modelPath)
            return decideSpeculativeDecoding(
                capabilitiesSupported = supported,
                modelName = modelFile.name,
                modelBytes = runCatching { modelFile.length() }.getOrDefault(0L),
                totalRamBytes = totalDeviceRamBytes(context),
                isX86Device = Build.SUPPORTED_ABIS.any { it.startsWith("x86") },
                mode = mode,
            )
        }

        private fun isTranslatedArm64OnX86(context: Context): Boolean {
            val nativeLibraryDir = context.applicationInfo.nativeLibraryDir.orEmpty()
            val packageUsesArm64 = nativeLibraryDir.contains("/arm64") ||
                nativeLibraryDir.contains("\\arm64")
            val deviceSupportsX86 = Build.SUPPORTED_ABIS.any { it.startsWith("x86") }
            return packageUsesArm64 && deviceSupportsX86
        }

        private fun hasLoadableOpenClLibrary(): Boolean {
            if (runCatching { System.loadLibrary("OpenCL") }.isSuccess) {
                return true
            }
            return listOf(
                "/vendor/lib64/libOpenCL.so",
                "/system/vendor/lib64/libOpenCL.so",
                "/system/lib64/libOpenCL.so",
                "/odm/lib64/libOpenCL.so",
                "/vendor/lib/libOpenCL.so",
                "/system/vendor/lib/libOpenCL.so",
                "/system/lib/libOpenCL.so",
                "/odm/lib/libOpenCL.so",
            ).any { path ->
                val file = File(path)
                file.isFile && runCatching { System.load(file.absolutePath) }.isSuccess
            }
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
                val errorMessage = if (engineInitResult.modalityPolicy.startsWith("text-only fallback")) {
                    "image input is unavailable because LiteRT-LM fell back to text-only after multimodal adapter initialization failed on this device. Check /health modality_policy for details."
                } else if (engineInitResult.modalityPolicy.startsWith("text-only memory guard")) {
                    "image input is unavailable because Hermes started this large local LiteRT-LM model in text-only mode to avoid an out-of-memory crash on this device. Check /health modality_policy for details."
                } else {
                    "image input requires a LiteRT-LM model started with image support, such as Gemma 4, Gemma 3n, or Gemma 3 vision models"
                }
                return jsonResponse(
                    JSONObject().put("error", errorMessage),
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
            val extraContext = chatTemplateExtraContext(requestJson)
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
                val payload = runInferenceWithTimeout(
                    conversation = convo,
                    promptMessage = promptMessage,
                    timeoutMs = generationTimeoutMs(requestJson),
                    extraContext = extraContext,
                )
                return if (requestJson.optBoolean("stream", false)) {
                    sseResponse(payload)
                } else {
                    jsonResponse(payload)
                }
            }
        }

        private fun runInferenceWithTimeout(
            conversation: Conversation,
            promptMessage: Message,
            timeoutMs: Long,
            extraContext: Map<String, Any>,
        ): JSONObject {
            val executor = Executors.newSingleThreadExecutor()
            return try {
                val future = executor.submit<Message> {
                    conversation.sendMessage(promptMessage, extraContext)
                }
                val responseMessage = try {
                    future.get(timeoutMs, TimeUnit.MILLISECONDS)
                } catch (timeout: TimeoutException) {
                    future.cancel(true)
                    throw timeout
                }
                completionPayload(responseMessage)
            } catch (_: TimeoutException) {
                kotlin.runCatching { conversation.cancelProcess() }
                throw IllegalStateException(
                    "LiteRT-LM generation timed out after ${timeoutMs / 1000} seconds before producing a response"
                )
            } finally {
                executor.shutdownNow()
            }
        }

        private fun chatTemplateExtraContext(requestJson: JSONObject): Map<String, Any> {
            val contextJson = requestJson.optJSONObject("chat_template_kwargs")
                ?: requestJson.optJSONObject("extra_context")
                ?: return emptyMap()
            return buildMap {
                jsonObjectToMap(contextJson).forEach { (key, value) ->
                    if (value != null) {
                        put(key, value)
                    }
                }
            }
        }

        private fun generationTimeoutMs(requestJson: JSONObject): Long {
            val requested = requestJson.optLong("timeout_ms", DEFAULT_GENERATION_TIMEOUT_MS)
            return requested.coerceIn(MIN_GENERATION_TIMEOUT_MS, MAX_GENERATION_TIMEOUT_MS)
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

        private fun completionPayload(responseMessage: Message, finishReasonOverride: String? = null): JSONObject {
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
            val finishReason = finishReasonOverride ?: if (responseMessage.toolCalls.isNotEmpty()) "tool_calls" else "stop"
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
    internal fun memorySafeModalityDecision(
        totalRamBytes: Long,
        modelBytes: Long,
        requestedImage: Boolean,
        requestedAudio: Boolean,
    ): ModalityDecision {
        if (!requestedImage && !requestedAudio) {
            return ModalityDecision(
                supportImage = false,
                supportAudio = false,
                policy = "text-only",
            )
        }

        val requestedLabel = buildString {
            append("requested")
            if (requestedImage) append(" image")
            if (requestedImage && requestedAudio) append(" and")
            if (requestedAudio) append(" audio")
            append(" adapter support")
        }
        val minimumRamBytes = minimumRamForLargeModelExtras(modelBytes)
        if (minimumRamBytes > 0L) {
            val shouldGuard = if (totalRamBytes > 0L) {
                totalRamBytes < minimumRamBytes
            } else {
                true
            }
            if (shouldGuard) {
                val currentRam = if (totalRamBytes > 0L) {
                    "${formatRamGb(totalRamBytes)}GB RAM"
                } else {
                    "unknown RAM"
                }
                return ModalityDecision(
                    supportImage = false,
                    supportAudio = false,
                    policy = "text-only memory guard: skipped $requestedLabel for a ${formatRamGb(modelBytes)}GB model on $currentRam; ${formatRamGb(minimumRamBytes)}GB RAM recommended",
                )
            }
        }
        return ModalityDecision(
            supportImage = requestedImage,
            supportAudio = requestedAudio,
            policy = requestedLabel,
        )
    }

    internal fun decideSpeculativeDecoding(
        capabilitiesSupported: Boolean,
        modelName: String,
        modelBytes: Long,
        totalRamBytes: Long,
        isX86Device: Boolean,
        mode: SpeculativeDecodingMode,
    ): SpeculativeDecodingDecision {
        val gemma4Fallback = !capabilitiesSupported && modelName.lowercase(Locale.US).contains("gemma-4")
        val capabilitySupported = capabilitiesSupported || gemma4Fallback
        if (mode == SpeculativeDecodingMode.DISABLED) {
            return SpeculativeDecodingDecision(
                supported = capabilitySupported,
                enabled = false,
                policy = if (capabilitySupported) {
                    "disabled: runtime setting disabled Gemma 4 MTP"
                } else {
                    "disabled: runtime setting disabled speculative decoding"
                },
            )
        }
        if (!capabilitySupported) {
            return SpeculativeDecodingDecision(
                supported = false,
                enabled = false,
                policy = if (mode == SpeculativeDecodingMode.ENABLED) {
                    "disabled: runtime setting requested Gemma 4 MTP but model does not advertise support"
                } else {
                    "disabled: model does not advertise Gemma 4 MTP support"
                },
            )
        }
        if (isX86Device) {
            return SpeculativeDecodingDecision(
                supported = true,
                enabled = false,
                policy = "disabled: x86 emulator/device build",
            )
        }
        val minimumRamBytes = minimumRamForLargeModelExtras(modelBytes)
        if (minimumRamBytes > 0L && totalRamBytes > 0L && totalRamBytes < minimumRamBytes) {
            return SpeculativeDecodingDecision(
                supported = true,
                enabled = false,
                policy = "disabled: memory guard for Gemma 4 MTP on ${formatRamGb(totalRamBytes)}GB RAM device; ${formatRamGb(minimumRamBytes)}GB RAM recommended",
            )
        }
        return SpeculativeDecodingDecision(
            supported = true,
            enabled = true,
            policy = when {
                mode == SpeculativeDecodingMode.ENABLED && capabilitiesSupported ->
                    "enabled: runtime setting requested Gemma 4 MTP and LiteRT-LM capabilities advertise support"
                mode == SpeculativeDecodingMode.ENABLED ->
                    "enabled: runtime setting requested Gemma 4 MTP with Gemma 4 filename fallback after capabilities probe failed"
                capabilitiesSupported ->
                    "enabled: LiteRT-LM capabilities advertise Gemma 4 MTP support"
                else ->
                    "enabled: Gemma 4 filename fallback after capabilities probe failed"
            },
        )
    }

    internal fun decideEngineTokenBudget(
        requestedMaxTokens: Int,
        requestedMaxContextLength: Int,
        totalRamBytes: Long,
        modelBytes: Long,
        isX86Device: Boolean,
    ): EngineTokenBudget {
        val x86Limit = if (isX86Device) X86_LITERT_LM_TOKEN_BUDGET else Int.MAX_VALUE
        val requested = when {
            requestedMaxContextLength > 0 -> requestedMaxContextLength
            requestedMaxTokens > 0 -> requestedMaxTokens
            isX86Device -> return EngineTokenBudget(
                value = X86_LITERT_LM_TOKEN_BUDGET,
                policy = "using x86 emulator/device LiteRT-LM token budget $X86_LITERT_LM_TOKEN_BUDGET",
            )
            else -> return EngineTokenBudget(null, "backend default")
        }

        if (requestedMaxContextLength <= 0) {
            val selected = requested.coerceAtMost(x86Limit)
            return EngineTokenBudget(
                value = selected,
                policy = if (selected == requested) {
                    "using requested max token budget"
                } else {
                    "clamped requested max token budget $requested to $selected on x86 emulator/device"
                },
            )
        }

        val safeLimit = memorySafeContextWindowLimit(totalRamBytes, modelBytes).coerceAtMost(x86Limit)
        val selected = requested.coerceAtMost(safeLimit)
        val totalRamGb = if (totalRamBytes > 0L) "${formatRamGb(totalRamBytes)}GB RAM" else "unknown RAM"
        return EngineTokenBudget(
            value = selected,
            policy = when {
                selected == requested ->
                    "using requested context window $requested tokens on $totalRamGb device"
                isX86Device ->
                    "clamped requested context window $requested to $selected on x86 emulator/device"
                else ->
                    "clamped requested context window $requested to $selected on $totalRamGb device"
            },
        )
    }

    private fun memorySafeContextWindowLimit(totalRamBytes: Long, modelBytes: Long): Int {
        if (totalRamBytes <= 0L) {
            return if (modelBytes >= GEMMA4_E4B_SIZE_FLOOR_BYTES) 8_192 else 16_384
        }
        return when {
            modelBytes >= GEMMA4_E4B_SIZE_FLOOR_BYTES && totalRamBytes < 12_000_000_000L -> 4_096
            modelBytes >= GEMMA4_E4B_SIZE_FLOOR_BYTES && totalRamBytes < 16_000_000_000L -> 8_192
            totalRamBytes < 6_000_000_000L -> 4_096
            totalRamBytes < 10_000_000_000L -> 8_192
            totalRamBytes < 14_000_000_000L -> 16_384
            else -> 32_000
        }
    }

    private fun minimumRamForLargeModelExtras(modelBytes: Long): Long {
        return when {
            modelBytes >= GEMMA4_E4B_SIZE_FLOOR_BYTES -> 12_000_000_000L
            modelBytes >= LARGE_MULTIMODAL_MODEL_SIZE_FLOOR_BYTES -> 8_000_000_000L
            modelBytes >= MEDIUM_MULTIMODAL_MODEL_SIZE_FLOOR_BYTES -> 6_000_000_000L
            else -> 0L
        }
    }

    private fun formatRamGb(bytes: Long): String {
        return "%.1f".format(Locale.US, bytes / 1_000_000_000.0)
    }

    private const val GEMMA4_E4B_SIZE_FLOOR_BYTES = 3_000_000_000L
    private const val LARGE_MULTIMODAL_MODEL_SIZE_FLOOR_BYTES = 2_000_000_000L
    private const val MEDIUM_MULTIMODAL_MODEL_SIZE_FLOOR_BYTES = 1_500_000_000L
    private const val X86_LITERT_LM_TOKEN_BUDGET = 1_024
}
