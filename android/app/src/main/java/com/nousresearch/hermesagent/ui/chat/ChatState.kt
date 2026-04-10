package com.nousresearch.hermesagent.ui.chat

data class ChatUiMessage(
    val id: String,
    val role: String,
    val content: String,
)

data class ChatUiState(
    val messages: List<ChatUiMessage> = emptyList(),
    val input: String = "",
    val isSending: Boolean = false,
    val error: String = "",
)
