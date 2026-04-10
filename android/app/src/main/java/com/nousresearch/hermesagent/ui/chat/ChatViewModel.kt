package com.nousresearch.hermesagent.ui.chat

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.api.ChatCompletionRequest
import com.nousresearch.hermesagent.api.ChatMessage
import com.nousresearch.hermesagent.api.HermesSseClient
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.data.ConversationStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    fun updateInput(value: String) {
        _uiState.update { it.copy(input = value) }
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

        val conversationStore = ConversationStore(getApplication())
        val sessionId = conversationStore.currentSessionId()
        val userMessage = ChatUiMessage(UUID.randomUUID().toString(), "user", text)
        val assistantMessageId = UUID.randomUUID().toString()

        _uiState.update {
            it.copy(
                messages = it.messages + userMessage + ChatUiMessage(assistantMessageId, "assistant", ""),
                input = "",
                isSending = true,
                error = "",
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
            client.streamChatCompletion(
                request = request,
                onDelta = { delta ->
                    _uiState.update { state ->
                        state.copy(
                            messages = state.messages.map { message ->
                                if (message.id == assistantMessageId) {
                                    message.copy(content = message.content + delta)
                                } else {
                                    message
                                }
                            }
                        )
                    }
                },
                onComplete = {
                    _uiState.update { it.copy(isSending = false) }
                },
                onError = { error ->
                    _uiState.update { it.copy(isSending = false, error = error) }
                },
            )
        }
    }
}
