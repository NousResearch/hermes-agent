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
import com.mobilefork.hermesagent.backend.HermesRuntimeManager
import com.mobilefork.hermesagent.backend.OnDeviceBackendManager
import com.mobilefork.hermesagent.data.ConversationStore
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.HermesNetworkPolicy
import com.mobilefork.hermesagent.data.StoredConversationAttachment
import com.mobilefork.hermesagent.data.StoredConversationMessage
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
        val now = System.currentTimeMillis()
        val userMessage = ChatUiMessage(UUID.randomUUID().toString(), "user", text, now, attachments)
        val assistantMessageId = UUID.randomUUID().toString()
        val assistantPlaceholder = ChatUiMessage(assistantMessageId, "assistant", "", now + 1)

        viewModelScope.launch(Dispatchers.IO) {
            val runtime = ensureRuntimeReady()
            val endpoint = resolveChatEndpoint(runtime)
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
                    )
                    conversationStore.updateMessageContent(
                        sessionId = sessionId,
                        messageId = assistantMessageId,
                        newContent = result.content,
                    )
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
                ),
                stream = true,
                sessionId = sessionId,
            )
            runCatching {
                client.streamChatCompletion(
                    request = request,
                    onDelta = { delta ->
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
                    },
                    onComplete = {
                        _uiState.update {
                            it.copy(
                                isSending = false,
                                status = "",
                                conversationSummaries = loadSummaries(),
                            )
                        }
                    },
                    onError = { error ->
                        tryNonStreamingEndpointFallback(
                            endpoint = endpoint,
                            request = request,
                            sessionId = sessionId,
                            assistantMessageId = assistantMessageId,
                            streamError = error,
                        )
                    },
                    onStatus = { status ->
                        _uiState.update {
                            it.copy(status = "${endpoint.debugLabel()}: $status")
                        }
                    },
                )
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
                status = "${endpoint.debugLabel()}: stream issue detected; retrying non-stream chat…",
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
            val result = fallbackClient.createChatCompletion(request.copy(stream = false))
            val content = extractAssistantContentFromChatCompletion(result.rawBody)
            require(content.isNotBlank()) {
                "Non-stream endpoint returned no assistant text"
            }
            conversationStore.updateMessageContent(
                sessionId = sessionId,
                messageId = assistantMessageId,
                newContent = content,
            )
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
                    status = "${endpoint.debugLabel()}: recovered with non-stream chat after SSE failed.",
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
    )

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
        val mode = if (nativeToolCalling) "on-device" else "endpoint"
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

    private fun ensureRuntimeReady(): HermesRuntimeManager.RuntimeState {
        val current = HermesRuntimeManager.currentState()
        if (current.started && resolveChatEndpoint(current) != null) {
            return current
        }
        return HermesRuntimeManager.ensureStarted(getApplication())
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

internal fun buildChatRequestMessages(
    userText: String,
    userContentParts: List<ChatContentPart> = emptyList(),
    customSystemPrompt: String = "",
): List<ChatMessage> {
    val userMessage = ChatMessage(role = "user", content = userText, contentParts = userContentParts)
    val persona = NativeToolContextCompressor.compactCustomSystemPrompt(
        AppSettings.normalizeCustomSystemPrompt(customSystemPrompt),
    )
    if (persona.isBlank()) {
        return listOf(userMessage)
    }
    return listOf(
        ChatMessage(
            role = "system",
            content = "User-configured agent persona/system instructions. Apply them unless they conflict " +
                "with the current user request, Android permissions, tool truthfulness, or safety constraints:\n" +
                persona,
        ),
        userMessage,
    )
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
