package com.nousresearch.hermesagent.ui.chat

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import android.text.format.DateFormat
import android.util.Base64
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.api.ChatCompletionRequest
import com.nousresearch.hermesagent.api.ChatContentPart
import com.nousresearch.hermesagent.api.ChatMessage
import com.nousresearch.hermesagent.api.HermesSseClient
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.data.ConversationStore
import com.nousresearch.hermesagent.data.HermesNetworkPolicy
import com.nousresearch.hermesagent.data.StoredConversationAttachment
import com.nousresearch.hermesagent.data.StoredConversationMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
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
        val text = snapshot.input.trim()
        val attachments = snapshot.attachments
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
                    status = if (attachments.isEmpty()) "Hermes is replying…" else "Hermes is reading the image…",
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
                            error = error.message ?: error.javaClass.simpleName,
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
            val request = ChatCompletionRequest(
                model = endpoint.modelName,
                messages = listOf(ChatMessage(role = "user", content = text, contentParts = userContentParts)),
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
                        _uiState.update { it.copy(isSending = false, error = error, status = "") }
                    },
                )
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        isSending = false,
                        error = error.message ?: error.javaClass.simpleName,
                        status = "",
                    )
                }
            }
        }
    }

    private data class ChatEndpoint(
        val baseUrl: String,
        val apiKey: String?,
        val modelName: String,
        val nativeToolCalling: Boolean = false,
    )

    private fun resolveChatEndpoint(runtime: HermesRuntimeManager.RuntimeState): ChatEndpoint? {
        val localBackend = OnDeviceBackendManager.currentStatus()
        if (localBackend.started && localBackend.baseUrl.isNotBlank() && localBackend.modelName.isNotBlank()) {
            return ChatEndpoint(
                baseUrl = localBackend.baseUrl.removeSuffix("/v1"),
                apiKey = null,
                modelName = localBackend.modelName,
                nativeToolCalling = true,
            )
        }
        val runtimeBaseUrl = runtime.baseUrl?.takeIf { it.isNotBlank() } ?: return null
        return ChatEndpoint(
            baseUrl = runtimeBaseUrl,
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
        app.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
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
        val mimeType = app.contentResolver.getType(uri).orEmpty().ifBlank { "image/*" }
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
