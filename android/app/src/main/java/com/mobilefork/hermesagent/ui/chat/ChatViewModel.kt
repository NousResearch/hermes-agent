package com.mobilefork.hermesagent.ui.chat

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import android.text.format.DateFormat
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilefork.hermesagent.api.ChatCompletionRequest
import com.mobilefork.hermesagent.api.ChatContentPart
import com.mobilefork.hermesagent.api.ChatMessage
import com.mobilefork.hermesagent.api.HermesEndpointUrl
import com.mobilefork.hermesagent.api.HermesApiClient
import com.mobilefork.hermesagent.api.HermesSseClient
import com.mobilefork.hermesagent.backend.BackendKind
import com.mobilefork.hermesagent.backend.HermesRuntimeManager
import com.mobilefork.hermesagent.backend.OnDeviceBackendManager
import com.mobilefork.hermesagent.data.ConversationStore
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.HermesNetworkPolicy
import com.mobilefork.hermesagent.data.ProviderPresets
import com.mobilefork.hermesagent.data.SecureSecretsStore
import com.mobilefork.hermesagent.data.StoredConversationAttachment
import com.mobilefork.hermesagent.data.StoredConversationMessage
import com.mobilefork.hermesagent.device.HermesDeviceDiagnosticsBridge
import com.mobilefork.hermesagent.device.HermesHindsightMemoryBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.URI
import java.util.UUID

private val DIRECT_OPENAI_COMPATIBLE_PROVIDERS = setOf(
    "openrouter",
    "openai",
    "codex",
    "gemini",
    "alibaba",
    "alibaba-coding-plan",
    "qwen-oauth",
    "zai",
    "zai-coding-plan",
    "groq",
    "mistral",
    "perplexity",
    "cerebras",
    "together",
    "fireworks",
    "deepinfra",
)
private val RESPONSES_API_PROVIDERS = setOf("openai", "codex")

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val conversationStore = ConversationStore(application)
    private val _uiState = MutableStateFlow(buildState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    fun updateInput(value: String) {
        _uiState.update { it.copy(input = value) }
    }

    fun attachImage(uriString: String) {
        val uri = Uri.parse(uriString)
        val details = queryAttachmentDetails(uri)
        _uiState.update { state ->
            if (state.attachments.any { it.uri == uriString }) {
                state
            } else {
                state.copy(
                    attachments = state.attachments + ChatAttachment(
                        uri = uriString,
                        displayName = details.displayName,
                        mimeType = details.mimeType,
                        sizeBytes = details.sizeBytes,
                    ),
                    status = "Image attached for multimodal Gemma requests",
                    error = "",
                )
            }
        }
    }

    fun removeAttachment(uriString: String) {
        _uiState.update { state ->
            state.copy(attachments = state.attachments.filterNot { it.uri == uriString })
        }
    }

    fun applyVoiceInput(text: String) {
        _uiState.update { state ->
            val merged = listOf(state.input.trim(), text.trim()).filter { it.isNotBlank() }.joinToString(" ")
            state.copy(input = merged, isListening = false, status = "Voice input captured", error = "")
        }
    }

    fun setListening(active: Boolean) {
        _uiState.update { it.copy(isListening = active, status = if (active) "Listening…" else it.status) }
    }

    fun setStatus(message: String) {
        _uiState.update { it.copy(status = message) }
    }

    fun clearStatus() {
        _uiState.update { it.copy(status = "") }
    }

    fun startNewConversation() {
        val conversation = conversationStore.createNewConversation()
        _uiState.value = buildState(
            activeConversationId = conversation.sessionId,
            messages = emptyList(),
            status = "Started a new chat",
        )
    }

    fun clearCurrentConversation() {
        val nextConversation = conversationStore.clearCurrentConversation()
        _uiState.value = buildState(
            activeConversationId = nextConversation.sessionId,
            messages = nextConversation.messages.toUiMessages(),
            status = "Cleared the previous conversation",
        )
    }

    fun showHistory() {
        _uiState.update {
            it.copy(
                isShowingHistory = true,
                conversationSummaries = loadSummaries(),
                status = "",
                error = "",
            )
        }
    }

    fun hideHistory() {
        _uiState.update { it.copy(isShowingHistory = false) }
    }

    fun openConversation(sessionId: String) {
        val conversation = conversationStore.switchConversation(sessionId) ?: return
        _uiState.value = buildState(
            activeConversationId = conversation.sessionId,
            messages = conversation.messages.toUiMessages(),
            isShowingHistory = false,
            status = "Opened ${conversation.title}",
        )
    }

    fun consumeCommandResult(commandText: String, feedback: String?) {
        if (feedback.isNullOrBlank()) {
            _uiState.update { it.copy(input = "", error = "", isSending = false, status = "") }
            return
        }
        val now = System.currentTimeMillis()
        val sessionId = conversationStore.currentSessionId()
        val userMessage = ChatUiMessage(UUID.randomUUID().toString(), "user", commandText, now)
        val assistantMessage = ChatUiMessage(UUID.randomUUID().toString(), "assistant", feedback, now + 1)
        persistMessages(sessionId, userMessage, assistantMessage)
        _uiState.update {
            it.copy(
                activeConversationId = sessionId,
                activeConversationTitle = conversationStore.currentConversation().title,
                conversationSummaries = loadSummaries(),
                messages = conversationStore.currentConversationMessages().toUiMessages(),
                input = "",
                isSending = false,
                error = "",
                status = "",
            )
        }
    }

    fun latestAssistantReply(): String {
        return _uiState.value.messages.lastOrNull { it.role == "assistant" && it.content.isNotBlank() }?.content.orEmpty()
    }

    fun sendMessage() {
        val snapshot = _uiState.value
        sendPreparedMessage(text = snapshot.input.trim(), attachments = snapshot.attachments)
    }

    fun sendQuickPrompt(prompt: String) {
        val normalized = prompt.trim()
        val snapshot = _uiState.value
        if (normalized.isEmpty() || snapshot.isSending) {
            return
        }
        if (snapshot.input.isNotBlank() || snapshot.attachments.isNotEmpty()) {
            _uiState.update {
                it.copy(status = "Send or clear the current draft before running a signal quick action.")
            }
            return
        }
        sendPreparedMessage(text = normalized, attachments = emptyList())
    }

    private fun sendPreparedMessage(text: String, attachments: List<ChatAttachment>) {
        val snapshot = _uiState.value
        if ((text.isEmpty() && attachments.isEmpty()) || snapshot.isSending) {
            return
        }

        _uiState.update {
            it.copy(
                isSending = true,
                error = "",
                status = "Starting Hermes runtime…",
                isShowingHistory = false,
            )
        }

        val sessionId = conversationStore.currentSessionId()
        val priorConversationMessages = buildPriorChatRequestMessages(snapshot.messages)
        val now = System.currentTimeMillis()
        val userMessage = ChatUiMessage(UUID.randomUUID().toString(), "user", text, now, attachments)
        val assistantMessageId = UUID.randomUUID().toString()
        val assistantPlaceholder = ChatUiMessage(assistantMessageId, "assistant", "", now + 1)

        viewModelScope.launch(Dispatchers.IO) {
            val directDiagnosticArguments = if (attachments.isEmpty()) directNativeDiagnosticArgumentsForPrompt(text) else null
            if (directDiagnosticArguments != null) {
                persistMessages(sessionId, userMessage, assistantPlaceholder)
                _uiState.update {
                    it.copy(
                        activeConversationId = sessionId,
                        activeConversationTitle = conversationStore.currentConversation().title,
                        conversationSummaries = loadSummaries(),
                        messages = conversationStore.currentConversationMessages().toUiMessages(),
                        input = "",
                        attachments = emptyList(),
                        isSending = true,
                        error = "",
                        status = "Running native Android diagnostics…",
                        isShowingHistory = false,
                    )
                }
                val content = runCatching {
                    val action = directDiagnosticArguments.optString("action").ifBlank { "agent_native_tool_self_test_report" }
                    HermesDeviceDiagnosticsBridge.performActionJson(
                        context = getApplication<Application>(),
                        action = action,
                        arguments = directDiagnosticArguments,
                    )
                }.fold(
                    onSuccess = { formatDirectNativeDiagnosticsReply(it) },
                    onFailure = { error ->
                        "Native Android diagnostics failed: ${error.message ?: error.javaClass.simpleName}"
                    },
                )
                conversationStore.updateMessageContent(
                    sessionId = sessionId,
                    messageId = assistantMessageId,
                    newContent = content,
                )
                retainConversationMemory(sessionId, text, content)
                _uiState.update { state ->
                    state.copy(
                        activeConversationTitle = conversationStore.currentConversation().title,
                        conversationSummaries = loadSummaries(),
                        messages = state.messages.map { message ->
                            if (message.id == assistantMessageId) {
                                message.copy(content = content)
                            } else {
                                message
                            }
                        },
                        isSending = false,
                        error = "",
                        status = "",
                    )
                }
                return@launch
            }

            val directEndpoint = resolveDirectProviderEndpoint()
            val runtime = if (directEndpoint == null) {
                ensureRuntimeReady()
            } else {
                HermesRuntimeManager.RuntimeState(started = true)
            }
            val endpoint = directEndpoint ?: resolveChatEndpoint(runtime)
            if (!runtime.started || endpoint == null) {
                _uiState.update {
                    it.copy(
                        isSending = false,
                        error = runtime.error ?: "Hermes runtime is not ready",
                        status = "",
                    )
                }
                return@launch
            }
            _uiState.update {
                it.copy(
                    status = "Checking ${endpoint.debugLabel()} before sending…",
                    error = "",
                )
            }

            val userContentParts = runCatching { buildUserContentParts(text, attachments) }.getOrElse { error ->
                _uiState.update {
                    it.copy(
                        isSending = false,
                        error = error.message ?: error.javaClass.simpleName,
                        status = "",
                    )
                }
                return@launch
            }
            val memoryContext = recallConversationMemoryContext(text)

            persistMessages(sessionId, userMessage, assistantPlaceholder)

            _uiState.update {
                it.copy(
                    activeConversationId = sessionId,
                    activeConversationTitle = conversationStore.currentConversation().title,
                    conversationSummaries = loadSummaries(),
                    messages = conversationStore.currentConversationMessages().toUiMessages(),
                    input = "",
                    attachments = emptyList(),
                    isSending = true,
                    error = "",
                    status = endpoint.streamingStatus(attachments.isNotEmpty()),
                    isShowingHistory = false,
                )
            }

            if (endpoint.nativeToolCalling) {
                runCatching {
                    val result = NativeToolCallingChatClient(getApplication<Application>()).send(
                        baseUrl = endpoint.baseUrl,
                        modelName = endpoint.modelName,
                        sessionId = sessionId,
                        userText = text,
                        userContentParts = userContentParts,
                        priorMessages = priorConversationMessages,
                        relevantMemoryContext = memoryContext,
                    )
                    conversationStore.updateMessageContent(
                        sessionId = sessionId,
                        messageId = assistantMessageId,
                        newContent = result.content,
                    )
                    retainConversationMemory(sessionId, text, result.content)
                    _uiState.update { state ->
                        state.copy(
                            activeConversationTitle = conversationStore.currentConversation().title,
                            conversationSummaries = loadSummaries(),
                            messages = state.messages.map { message ->
                                if (message.id == assistantMessageId) {
                                    message.copy(content = result.content)
                                } else {
                                    message
                                }
                            },
                            isSending = false,
                            status = "",
                        )
                    }
                }.onFailure { error ->
                    _uiState.update {
                        it.copy(
                            isSending = false,
                            error = endpoint.failureMessage(error.message ?: error.javaClass.simpleName),
                            status = "",
                        )
                    }
                }
                return@launch
            }

            val client = HermesSseClient(
                baseUrl = endpoint.baseUrl,
                apiKey = endpoint.apiKey,
                networkGuard = { url ->
                    HermesNetworkPolicy.requireExternalNetworkAllowed(
                        getApplication<Application>(),
                        url,
                        actionLabel = "chat request",
                    )
                },
            )
            val customSystemPrompt = AppSettingsStore(getApplication<Application>()).load().customSystemPrompt
            val request = ChatCompletionRequest(
                model = endpoint.modelName,
                messages = buildChatRequestMessages(
                    userText = text,
                    userContentParts = userContentParts,
                    customSystemPrompt = customSystemPrompt,
                    priorMessages = priorConversationMessages,
                    memoryContext = memoryContext,
                ),
                stream = true,
                sessionId = sessionId,
            )
            runCatching {
                val onDelta: (String) -> Unit = { delta ->
                    val persistedPrefix = conversationStore.loadConversation(sessionId)
                        ?.messages
                        ?.firstOrNull { it.id == assistantMessageId }
                        ?.content
                        .orEmpty()
                    conversationStore.updateMessageContent(
                        sessionId = sessionId,
                        messageId = assistantMessageId,
                        newContent = persistedPrefix + delta,
                    )
                    _uiState.update { state ->
                        state.copy(
                            activeConversationTitle = conversationStore.currentConversation().title,
                            conversationSummaries = loadSummaries(),
                            messages = state.messages.map { message ->
                                if (message.id == assistantMessageId) {
                                    message.copy(content = message.content + delta)
                                } else {
                                    message
                                }
                            },
                        )
                    }
                }
                val onComplete: () -> Unit = {
                    val assistantContent = conversationStore.loadConversation(sessionId)
                        ?.messages
                        ?.firstOrNull { it.id == assistantMessageId }
                        ?.content
                        .orEmpty()
                    retainConversationMemory(sessionId, text, assistantContent)
                    _uiState.update {
                        it.copy(
                            isSending = false,
                            status = "",
                            conversationSummaries = loadSummaries(),
                        )
                    }
                }
                val onError: (String) -> Unit = { error ->
                    tryNonStreamingEndpointFallback(
                        endpoint = endpoint,
                        request = request,
                        sessionId = sessionId,
                        assistantMessageId = assistantMessageId,
                        streamError = error,
                    )
                }
                val onStatus: (String) -> Unit = { status ->
                    _uiState.update {
                        it.copy(status = "${endpoint.debugLabel()}: $status")
                    }
                }
                if (endpoint.apiMode == EndpointApiMode.RESPONSES) {
                    client.streamResponse(
                        request = request,
                        onDelta = onDelta,
                        onComplete = onComplete,
                        onError = onError,
                        onStatus = onStatus,
                    )
                } else {
                    client.streamChatCompletion(
                        request = request,
                        onDelta = onDelta,
                        onComplete = onComplete,
                        onError = onError,
                        onStatus = onStatus,
                    )
                }
            }.onFailure { error ->
                val message = error.message ?: error.javaClass.simpleName
                tryNonStreamingEndpointFallback(
                    endpoint = endpoint,
                    request = request,
                    sessionId = sessionId,
                    assistantMessageId = assistantMessageId,
                    streamError = message,
                )
            }
        }
    }

    private fun tryNonStreamingEndpointFallback(
        endpoint: ChatEndpoint,
        request: ChatCompletionRequest,
        sessionId: String,
        assistantMessageId: String,
        streamError: String,
    ): Boolean {
        if (endpoint.nativeToolCalling) {
            return false
        }
        _uiState.update {
            it.copy(
                status = "${endpoint.debugLabel()}: stream issue detected; retrying non-stream request…",
                error = "",
            )
        }
        return runCatching {
            val fallbackClient = HermesApiClient(
                baseUrl = endpoint.baseUrl,
                apiKey = endpoint.apiKey,
                networkGuard = { url ->
                    HermesNetworkPolicy.requireExternalNetworkAllowed(
                        getApplication<Application>(),
                        url,
                        actionLabel = "chat fallback request",
                    )
                },
            )
            val result = if (endpoint.apiMode == EndpointApiMode.RESPONSES) {
                fallbackClient.createResponse(request.copy(stream = false))
            } else {
                fallbackClient.createChatCompletion(request.copy(stream = false))
            }
            val content = if (endpoint.apiMode == EndpointApiMode.RESPONSES) {
                extractAssistantContentFromResponse(result.rawBody)
            } else {
                extractAssistantContentFromChatCompletion(result.rawBody)
            }
            require(content.isNotBlank()) {
                "Non-stream endpoint returned no assistant text"
            }
            conversationStore.updateMessageContent(
                sessionId = sessionId,
                messageId = assistantMessageId,
                newContent = content,
            )
            retainConversationMemory(sessionId, request.messages.lastOrNull { it.role == "user" }?.content.orEmpty(), content)
            _uiState.update { state ->
                state.copy(
                    activeConversationTitle = conversationStore.currentConversation().title,
                    conversationSummaries = loadSummaries(),
                    messages = state.messages.map { message ->
                        if (message.id == assistantMessageId) {
                            message.copy(content = content)
                        } else {
                            message
                        }
                    },
                    isSending = false,
                    error = "",
                    status = "${endpoint.debugLabel()}: recovered with non-stream request after SSE failed.",
                )
            }
            true
        }.getOrElse { fallbackError ->
            _uiState.update {
                it.copy(
                    isSending = false,
                    error = endpoint.failureMessage(
                        "Streaming failed: $streamError. Non-stream fallback also failed: " +
                            (fallbackError.message ?: fallbackError.javaClass.simpleName),
                    ),
                    status = "",
                )
            }
            false
        }
    }

    private data class ChatEndpoint(
        val baseUrl: String,
        val apiKey: String?,
        val modelName: String,
        val nativeToolCalling: Boolean = false,
        val apiMode: EndpointApiMode = EndpointApiMode.CHAT_COMPLETIONS,
        val directProvider: Boolean = false,
    )

    private enum class EndpointApiMode {
        CHAT_COMPLETIONS,
        RESPONSES,
    }

    private fun ChatEndpoint.streamingStatus(hasAttachments: Boolean): String {
        val action = if (hasAttachments) "Hermes is reading the image" else "Hermes is replying"
        return if (nativeToolCalling) {
            "$action on-device…"
        } else {
            "$action via ${endpointHostLabel(baseUrl)}…"
        }
    }

    private fun ChatEndpoint.failureMessage(message: String): String {
        val clean = message.ifBlank { "Endpoint request failed" }
        if (nativeToolCalling || !clean.looksLikeEndpointDisconnect()) {
            return clean
        }
        return "$clean Hermes normalizes raw hosts, /v1 URLs, and /v1/chat/completions URLs, but the host must still be reachable, the model name must match the server exactly, and streaming endpoints must stay open until [DONE]."
    }

    private fun ChatEndpoint.debugLabel(): String {
        val mode = when {
            nativeToolCalling -> "on-device"
            apiMode == EndpointApiMode.RESPONSES -> "responses"
            directProvider -> "provider"
            else -> "endpoint"
        }
        return "$mode ${endpointHostLabel(baseUrl)} · $modelName"
    }

    private fun String.looksLikeEndpointDisconnect(): Boolean {
        val lower = lowercase()
        return listOf("timeout", "closed", "reset", "disconnect", "unexpected end", "[done]", "sse", "stream").any { token ->
            lower.contains(token)
        }
    }

    private fun endpointHostLabel(baseUrl: String): String {
        val normalizedBaseUrl = runCatching { HermesEndpointUrl.normalizeBaseUrl(baseUrl) }.getOrDefault(baseUrl)
        return runCatching {
            val uri = URI(normalizedBaseUrl)
            val host = uri.host.orEmpty().ifBlank { normalizedBaseUrl }
            val port = uri.port.takeIf { it > 0 }?.let { ":$it" }.orEmpty()
            "$host$port"
        }.getOrDefault(normalizedBaseUrl)
            .replace("https://", "")
            .replace("http://", "")
            .take(64)
    }

    private fun resolveChatEndpoint(runtime: HermesRuntimeManager.RuntimeState): ChatEndpoint? {
        val localBackend = OnDeviceBackendManager.currentStatus()
        if (localBackend.started && localBackend.baseUrl.isNotBlank() && localBackend.modelName.isNotBlank()) {
            return ChatEndpoint(
                baseUrl = HermesEndpointUrl.normalizeBaseUrl(localBackend.baseUrl),
                apiKey = null,
                modelName = localBackend.modelName,
                nativeToolCalling = true,
            )
        }
        val runtimeBaseUrl = runtime.baseUrl?.takeIf { it.isNotBlank() } ?: return null
        val normalizedRuntimeBaseUrl = runCatching {
            HermesEndpointUrl.normalizeBaseUrl(runtimeBaseUrl)
        }.getOrNull() ?: return null
        return ChatEndpoint(
            baseUrl = normalizedRuntimeBaseUrl,
            apiKey = runtime.apiKey,
            modelName = runtime.modelName ?: "hermes-agent-android",
        )
    }

    private fun resolveDirectProviderEndpoint(): ChatEndpoint? {
        val settings = AppSettingsStore(getApplication<Application>()).load()
        if (settings.offlineAirplaneMode || BackendKind.fromPersistedValue(settings.onDeviceBackend) != BackendKind.NONE) {
            return null
        }
        val provider = settings.provider.trim().lowercase()
        if (provider !in DIRECT_OPENAI_COMPATIBLE_PROVIDERS) {
            return null
        }
        val preset = ProviderPresets.find(provider)
        val baseUrl = settings.baseUrl.ifBlank { preset?.baseUrl.orEmpty() }
        val modelName = settings.model.ifBlank { preset?.modelHint.orEmpty() }
        if (baseUrl.isBlank() || modelName.isBlank()) {
            return null
        }
        val apiKey = SecureSecretsStore(getApplication<Application>()).loadApiKey(provider)
        if (apiKey.isBlank()) {
            return null
        }
        return ChatEndpoint(
            baseUrl = HermesEndpointUrl.normalizeBaseUrl(baseUrl),
            apiKey = apiKey,
            modelName = modelName,
            apiMode = if (provider in RESPONSES_API_PROVIDERS) EndpointApiMode.RESPONSES else EndpointApiMode.CHAT_COMPLETIONS,
            directProvider = true,
        )
    }

    private fun ensureRuntimeReady(): HermesRuntimeManager.RuntimeState {
        val current = HermesRuntimeManager.currentState()
        if (current.started && resolveChatEndpoint(current) != null) {
            return current
        }
        return HermesRuntimeManager.ensureStarted(getApplication())
    }

    private fun retainConversationMemory(sessionId: String, userText: String, assistantText: String) {
        val fact = conversationMemoryFact(sessionId, userText, assistantText)
        if (fact.isBlank()) {
            return
        }
        runCatching {
            HermesHindsightMemoryBridge.performActionJson(
                context = getApplication<Application>(),
                rawAction = "retain",
                arguments = JSONObject()
                    .put("content", fact)
                    .put("source", "chat")
                    .put("category", "conversation")
                    .put("tags", JSONArray().put("conversation").put("auto_recall")),
            )
        }
    }

    private fun recallConversationMemoryContext(userText: String): String {
        return runCatching {
            JSONObject(
                HermesHindsightMemoryBridge.performActionJson(
                    context = getApplication<Application>(),
                    rawAction = "relevant_context",
                    arguments = JSONObject()
                        .put("query", userText)
                        .put("limit", 6)
                        .put("max_chars", 1600),
                ),
            ).optString("system_prompt_context")
        }.getOrDefault("").trim()
    }

    private fun buildState(
        activeConversationId: String = conversationStore.currentSessionId(),
        messages: List<ChatUiMessage> = conversationStore.currentConversationMessages().toUiMessages(),
        isShowingHistory: Boolean = false,
        status: String = "",
    ): ChatUiState {
        val conversation = conversationStore.loadConversation(activeConversationId) ?: conversationStore.currentConversation()
        return ChatUiState(
            activeConversationId = conversation.sessionId,
            activeConversationTitle = conversation.title,
            conversationSummaries = loadSummaries(),
            isShowingHistory = isShowingHistory,
            messages = messages,
            status = status,
        )
    }

    private fun loadSummaries(): List<ChatConversationSummary> {
        return conversationStore.listConversationSummaries().map { summary ->
            ChatConversationSummary(
                id = summary.sessionId,
                title = summary.title,
                preview = summary.preview,
                updatedLabel = DateFormat.format("MMM d, HH:mm", summary.updatedAtEpochMs).toString(),
                messageCount = summary.messageCount,
            )
        }
    }

    private fun persistMessages(sessionId: String, vararg messages: ChatUiMessage) {
        messages.forEach { message ->
            conversationStore.upsertMessage(
                sessionId = sessionId,
                message = StoredConversationMessage(
                    id = message.id,
                    role = message.role,
                    content = message.content,
                    createdAtEpochMs = message.createdAtEpochMs,
                    attachments = message.attachments.map { attachment ->
                        StoredConversationAttachment(
                            uri = attachment.uri,
                            displayName = attachment.displayName,
                            mimeType = attachment.mimeType,
                            sizeBytes = attachment.sizeBytes,
                        )
                    },
                ),
            )
        }
    }

    private data class AttachmentDetails(
        val displayName: String,
        val mimeType: String,
        val sizeBytes: Long,
    )

    private fun queryAttachmentDetails(uri: Uri): AttachmentDetails {
        val app = getApplication<Application>()
        var displayName = uri.lastPathSegment ?: "image"
        var sizeBytes = 0L
        if (uri.scheme == "file") {
            uri.path?.let(::File)?.let { file ->
                displayName = file.name.ifBlank { displayName }
                sizeBytes = file.length().coerceAtLeast(0L)
            }
        }
        runCatching { app.contentResolver.query(uri, null, null, null, null) }.getOrNull()?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (cursor.moveToFirst()) {
                if (nameIndex >= 0) {
                    displayName = cursor.getString(nameIndex) ?: displayName
                }
                if (sizeIndex >= 0) {
                    sizeBytes = cursor.getLong(sizeIndex).coerceAtLeast(0L)
                }
            }
        }
        val mimeType = runCatching { app.contentResolver.getType(uri) }.getOrNull().orEmpty().ifBlank {
            when (displayName.substringAfterLast('.', "").lowercase()) {
                "jpg", "jpeg" -> "image/jpeg"
                "png" -> "image/png"
                "webp" -> "image/webp"
                else -> "image/*"
            }
        }
        return AttachmentDetails(displayName = displayName, mimeType = mimeType, sizeBytes = sizeBytes)
    }

    private fun buildUserContentParts(text: String, attachments: List<ChatAttachment>): List<ChatContentPart> {
        if (attachments.isEmpty()) {
            return emptyList()
        }
        val parts = mutableListOf<ChatContentPart>()
        if (text.isNotBlank()) {
            parts += ChatContentPart(type = "text", text = text)
        }
        attachments.forEach { attachment ->
            parts += ChatContentPart(
                type = "image_url",
                imageUrl = readAttachmentAsDataUrl(attachment),
            )
        }
        return parts
    }

    private fun readAttachmentAsDataUrl(attachment: ChatAttachment): String {
        val app = getApplication<Application>()
        val uri = Uri.parse(attachment.uri)
        val mimeType = attachment.mimeType.ifBlank {
            app.contentResolver.getType(uri).orEmpty().ifBlank { "application/octet-stream" }
        }
        val bytes = app.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw IllegalArgumentException("Unable to read ${attachment.displayName}")
        require(bytes.isNotEmpty()) { "Selected image ${attachment.displayName} is empty" }
        return "data:$mimeType;base64," + Base64.encodeToString(bytes, Base64.NO_WRAP)
    }

    private fun List<StoredConversationMessage>.toUiMessages(): List<ChatUiMessage> {
        return map { message ->
            ChatUiMessage(
                id = message.id,
                role = message.role,
                content = message.content,
                createdAtEpochMs = message.createdAtEpochMs,
                attachments = message.attachments.map { attachment ->
                    ChatAttachment(
                        uri = attachment.uri,
                        displayName = attachment.displayName,
                        mimeType = attachment.mimeType,
                        sizeBytes = attachment.sizeBytes,
                    )
                },
            )
        }
    }
}

internal fun extractAssistantContentFromChatCompletion(rawBody: String): String {
    val root = JSONObject(rawBody)
    val choices = root.optJSONArray("choices") ?: return ""
    if (choices.length() == 0) {
        return ""
    }
    val choice = choices.optJSONObject(0) ?: return ""
    val messageContent = choice.optJSONObject("message")?.opt("content")
    val deltaContent = choice.optJSONObject("delta")?.opt("content")
    return chatCompletionContentToText(messageContent ?: deltaContent).trim()
}

internal fun extractAssistantContentFromResponse(rawBody: String): String {
    val root = JSONObject(rawBody)
    val directOutput = root.optString("output_text").trim()
    if (directOutput.isNotBlank()) {
        return directOutput
    }
    val output = root.optJSONArray("output") ?: return ""
    val chunks = mutableListOf<String>()
    for (outputIndex in 0 until output.length()) {
        val item = output.optJSONObject(outputIndex) ?: continue
        val content = item.opt("content")
        val text = when (content) {
            is JSONArray -> responseContentArrayToText(content)
            else -> chatCompletionContentToText(content)
        }.trim()
        if (text.isNotBlank()) {
            chunks += text
        }
    }
    return chunks.joinToString("\n").trim()
}

internal fun conversationMemoryFact(sessionId: String, userText: String, assistantText: String): String {
    val user = userText.compactForMemory()
    val assistant = assistantText.compactForMemory()
    if (user.isBlank() && assistant.isBlank()) {
        return ""
    }
    return buildString {
        append("Conversation ")
        append(sessionId.take(36))
        append(": ")
        if (user.isNotBlank()) {
            append("user asked: ")
            append(user.take(420))
        }
        if (assistant.isNotBlank()) {
            if (user.isNotBlank()) append(" | ")
            append("assistant answered: ")
            append(assistant.take(700))
        }
    }.take(1_200)
}

internal fun buildChatRequestMessages(
    userText: String,
    userContentParts: List<ChatContentPart> = emptyList(),
    customSystemPrompt: String = "",
    priorMessages: List<ChatMessage> = emptyList(),
    memoryContext: String = "",
): List<ChatMessage> {
    val userMessage = ChatMessage(role = "user", content = userText, contentParts = userContentParts)
    val persona = NativeToolContextCompressor.compactCustomSystemPrompt(
        AppSettings.normalizeCustomSystemPrompt(customSystemPrompt),
    )
    val relevantMemory = NativeToolContextCompressor.compactPromotedMemoryContext(memoryContext)
    val requestMessages = mutableListOf<ChatMessage>()
    if (persona.isBlank() && relevantMemory.isBlank()) {
        requestMessages += NativeToolContextCompressor.compactPriorChatRequestMessages(priorMessages)
        requestMessages += userMessage
        return requestMessages
    }
    requestMessages += ChatMessage(
        role = "system",
        content = buildString {
            if (persona.isNotBlank()) {
                append("User-configured agent persona/system instructions. Apply them unless they conflict ")
                append("with the current user request, Android permissions, tool truthfulness, or safety constraints:\n")
                append(persona)
            }
            if (relevantMemory.isNotBlank()) {
                if (isNotEmpty()) append("\n\n")
                append("Relevant local memory context recalled from prior conversations. Use it when it helps the current request, and ignore stale or unrelated rows:\n")
                append(relevantMemory)
            }
        },
    )
    requestMessages += NativeToolContextCompressor.compactPriorChatRequestMessages(priorMessages)
    requestMessages += userMessage
    return requestMessages
}

internal fun buildPriorChatRequestMessages(messages: List<ChatUiMessage>): List<ChatMessage> {
    return NativeToolContextCompressor.compactPriorChatRequestMessages(
        messages.mapNotNull { message ->
            if (message.role != "user" && message.role != "assistant") {
                return@mapNotNull null
            }
            val content = buildString {
                val text = message.content.trim()
                if (text.isNotBlank()) {
                    append(text)
                }
                val attachmentLabels = message.attachments.map { attachment ->
                    attachment.displayName
                        .ifBlank { attachment.mimeType }
                        .ifBlank { "attachment" }
                }
                if (attachmentLabels.isNotEmpty()) {
                    if (isNotEmpty()) append('\n')
                    append(
                        attachmentLabels.joinToString("\n") { label ->
                            "[prior turn attachment omitted: $label]"
                        },
                    )
                }
            }.trim()
            if (content.isBlank()) {
                null
            } else {
                ChatMessage(
                    role = message.role,
                    content = content,
                )
            }
        },
    )
}

internal fun directNativeDiagnosticArgumentsForPrompt(text: String): JSONObject? {
    val prompt = text.trim()
    if (prompt.isBlank()) {
        return null
    }
    return NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(prompt)
        ?: NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(prompt)
}

internal fun formatDirectNativeDiagnosticsReply(rawJson: String): String {
    val json = runCatching { JSONObject(rawJson) }.getOrNull()
        ?: return rawJson.take(4_000)
    val output = json.optString("output")
    if (output.isNotBlank()) {
        return output
    }
    val cards = json.optJSONArray("cards")
    if (cards != null && cards.length() > 0) {
        val lines = mutableListOf<String>()
        for (index in 0 until cards.length()) {
            val card = cards.optJSONObject(index) ?: continue
            val title = card.optString("title").ifBlank { "Diagnostic" }
            val body = card.optString("body").ifBlank { card.optString("subtitle") }
            lines.add(listOf(title, body).filter { it.isNotBlank() }.joinToString(": "))
        }
        if (lines.isNotEmpty()) {
            return lines.joinToString("\n")
        }
    }
    return json.toString(2).take(4_000)
}

private fun String.compactForMemory(): String {
    return replace(Regex("\\s+"), " ").trim()
}

private fun chatCompletionContentToText(value: Any?): String {
    if (value == null || value == JSONObject.NULL) {
        return ""
    }
    return when (value) {
        is String -> value
        is JSONArray -> buildString {
            for (index in 0 until value.length()) {
                val item = value.opt(index)
                val text = when (item) {
                    is JSONObject -> item.optString("text")
                        .ifBlank { item.optString("content") }
                    is String -> item
                    else -> item?.toString().orEmpty()
                }
                if (text.isNotBlank()) {
                    if (isNotEmpty()) append('\n')
                    append(text)
                }
            }
        }
        is JSONObject -> value.optString("text").ifBlank { value.optString("content") }
        else -> value.toString()
    }
}

private fun responseContentArrayToText(value: JSONArray): String {
    return buildString {
        for (index in 0 until value.length()) {
            val item = value.opt(index)
            val text = when (item) {
                is JSONObject -> when (item.optString("type")) {
                    "output_text", "input_text", "summary_text" -> item.optString("text")
                    else -> item.optString("text")
                        .ifBlank { item.optString("content") }
                }
                is String -> item
                else -> item?.toString().orEmpty()
            }
            if (text.isNotBlank()) {
                if (isNotEmpty()) append('\n')
                append(text)
            }
        }
    }
}
