package com.nousresearch.hermesagent.ui.chat

import android.app.Application
import android.text.format.DateFormat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.api.ChatCompletionRequest
import com.nousresearch.hermesagent.api.ChatMessage
import com.nousresearch.hermesagent.api.HermesSseClient
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.data.ConversationStore
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
        val text = _uiState.value.input.trim()
        if (text.isEmpty() || _uiState.value.isSending) {
            return
        }

        val runtime = HermesRuntimeManager.currentState()
        val baseUrl = runtime.baseUrl
        val modelName = runtime.modelName ?: "hermes-agent-android"
        if (!runtime.started || baseUrl.isNullOrBlank()) {
            _uiState.update { it.copy(error = runtime.error ?: "Hermes runtime is not ready") }
            return
        }

        val sessionId = conversationStore.currentSessionId()
        val now = System.currentTimeMillis()
        val userMessage = ChatUiMessage(UUID.randomUUID().toString(), "user", text, now)
        val assistantMessageId = UUID.randomUUID().toString()
        val assistantPlaceholder = ChatUiMessage(assistantMessageId, "assistant", "", now + 1)
        persistMessages(sessionId, userMessage, assistantPlaceholder)

        _uiState.update {
            it.copy(
                activeConversationId = sessionId,
                activeConversationTitle = conversationStore.currentConversation().title,
                conversationSummaries = loadSummaries(),
                messages = conversationStore.currentConversationMessages().toUiMessages(),
                input = "",
                isSending = true,
                error = "",
                status = "Hermes is replying…",
                isShowingHistory = false,
            )
        }

        viewModelScope.launch(Dispatchers.IO) {
            val client = HermesSseClient(baseUrl = baseUrl, apiKey = runtime.apiKey)
            val request = ChatCompletionRequest(
                model = modelName,
                messages = listOf(ChatMessage(role = "user", content = text)),
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
                ),
            )
        }
    }

    private fun List<StoredConversationMessage>.toUiMessages(): List<ChatUiMessage> {
        return map { message ->
            ChatUiMessage(
                id = message.id,
                role = message.role,
                content = message.content,
                createdAtEpochMs = message.createdAtEpochMs,
            )
        }
    }
}
