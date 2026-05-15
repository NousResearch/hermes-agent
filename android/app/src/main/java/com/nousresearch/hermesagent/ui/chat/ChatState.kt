package com.nousresearch.hermesagent.ui.chat

data class ChatUiMessage(
    val id: String,
    val role: String,
    val content: String,
    val createdAtEpochMs: Long,
    val attachments: List<ChatAttachment> = emptyList(),
)

data class ChatTurn(
    val id: String,
    val userMessage: ChatUiMessage?,
    val assistantMessages: List<ChatUiMessage>,
)

data class ChatConversationSummary(
    val id: String,
    val title: String,
    val preview: String,
    val updatedLabel: String,
    val messageCount: Int,
)

data class ChatAttachment(
    val uri: String,
    val displayName: String,
    val mimeType: String,
    val sizeBytes: Long = 0L,
)

data class ChatUiState(
    val activeConversationId: String = "",
    val activeConversationTitle: String = "New chat",
    val conversationSummaries: List<ChatConversationSummary> = emptyList(),
    val isShowingHistory: Boolean = false,
    val messages: List<ChatUiMessage> = emptyList(),
    val input: String = "",
    val attachments: List<ChatAttachment> = emptyList(),
    val isSending: Boolean = false,
    val isListening: Boolean = false,
    val status: String = "",
    val error: String = "",
)

fun buildChatTurns(messages: List<ChatUiMessage>): List<ChatTurn> {
    val turns = mutableListOf<ChatTurn>()
    var pendingUser: ChatUiMessage? = null
    var assistantMessages = mutableListOf<ChatUiMessage>()

    fun flush() {
        if (pendingUser == null && assistantMessages.isEmpty()) return
        val id = pendingUser?.id ?: assistantMessages.first().id
        turns += ChatTurn(
            id = id,
            userMessage = pendingUser,
            assistantMessages = assistantMessages.toList(),
        )
        pendingUser = null
        assistantMessages = mutableListOf()
    }

    messages.forEach { message ->
        if (message.role == "user") {
            flush()
            pendingUser = message
        } else {
            assistantMessages += message
        }
    }
    flush()
    return turns
}

fun minuteBucket(epochMs: Long): Long = epochMs / 60_000L

fun shortPromptPreview(text: String, maxLength: Int = 96): String {
    val singleLine = text
        .lineSequence()
        .firstOrNull { it.isNotBlank() }
        .orEmpty()
        .replace(Regex("\\s+"), " ")
        .trim()
    return when {
        singleLine.isBlank() -> "Attachment"
        singleLine.length <= maxLength -> singleLine
        else -> singleLine.take(maxLength - 1).trimEnd() + "…"
    }
}
