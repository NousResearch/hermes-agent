package com.nousresearch.hermesagent.ui.chat

data class ChatUiMessage(
    val id: String,
    val role: String,
    val content: String,
    val createdAtEpochMs: Long,
)

data class ChatConversationSummary(
    val id: String,
    val title: String,
    val preview: String,
    val updatedLabel: String,
    val messageCount: Int,
)

data class ChatUiState(
    val activeConversationId: String = "",
    val activeConversationTitle: String = "New chat",
    val conversationSummaries: List<ChatConversationSummary> = emptyList(),
    val isShowingHistory: Boolean = false,
    val messages: List<ChatUiMessage> = emptyList(),
    val input: String = "",
    val isSending: Boolean = false,
    val isListening: Boolean = false,
    val status: String = "",
    val error: String = "",
)
