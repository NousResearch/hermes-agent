package com.nousresearch.hermesagent.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class ChatViewModelTest {
    @Test
    fun chatUiState_defaultsAreEmptyAndIdle() {
        val state = ChatUiState()
        assertEquals(emptyList<ChatUiMessage>(), state.messages)
        assertEquals("", state.input)
        assertFalse(state.isSending)
        assertEquals("", state.error)
    }

    @Test
    fun buildChatTurnsPairsPromptWithAssistantReply() {
        val user = ChatUiMessage("u1", "user", "Please use your back camera", 60_000L)
        val assistant = ChatUiMessage("a1", "assistant", "Taking a photo now.", 60_001L)

        val turns = buildChatTurns(listOf(user, assistant))

        assertEquals(1, turns.size)
        assertEquals(user, turns[0].userMessage)
        assertEquals(listOf(assistant), turns[0].assistantMessages)
    }

    @Test
    fun shortPromptPreviewUsesFirstLineAndCompactsWhitespace() {
        val preview = shortPromptPreview(
            """
              Please   use camera
            with a long second line
            """.trimIndent(),
            maxLength = 40,
        )

        assertEquals("Please use camera", preview)
    }
}
