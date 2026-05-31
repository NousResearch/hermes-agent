package com.mobilefork.hermesagent.ui.chat

import com.mobilefork.hermesagent.api.ChatContentPart
import com.mobilefork.hermesagent.api.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
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

    @Test
    fun extractAssistantContentFromChatCompletionReadsStringMessageContent() {
        val content = extractAssistantContentFromChatCompletion(
            """{"choices":[{"message":{"role":"assistant","content":"Endpoint recovered"}}]}""",
        )

        assertEquals("Endpoint recovered", content)
    }

    @Test
    fun extractAssistantContentFromChatCompletionReadsArrayMessageContent() {
        val content = extractAssistantContentFromChatCompletion(
            """{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"First"},{"type":"text","text":"Second"}]}}]}""",
        )

        assertEquals("First\nSecond", content)
    }

    @Test
    fun extractAssistantContentFromResponseReadsOutputTextHelperAndItems() {
        assertEquals(
            "Endpoint recovered",
            extractAssistantContentFromResponse("""{"output_text":"Endpoint recovered"}"""),
        )
        assertEquals(
            "First\nSecond",
            extractAssistantContentFromResponse(
                """{"output":[{"type":"message","content":[{"type":"output_text","text":"First"},{"type":"output_text","text":"Second"}]}]}""",
            ),
        )
    }

    @Test
    fun buildChatRequestMessagesAddsSavedPersonaBeforeEndpointUserMessage() {
        val messages = buildChatRequestMessages(
            userText = "Check the local model",
            customSystemPrompt = "Prefer local tools and keep replies short.",
        )

        assertEquals(2, messages.size)
        assertEquals("system", messages[0].role)
        assertTrue(messages[0].content.contains("User-configured agent persona"))
        assertTrue(messages[0].content.contains("Prefer local tools"))
        assertEquals("user", messages[1].role)
        assertEquals("Check the local model", messages[1].content)
    }

    @Test
    fun conversationMemoryFactBoundsCompletedChatTurnsForRecall() {
        val fact = conversationMemoryFact(
            sessionId = "session-123",
            userText = "  How should the keyboard behave?\n\nKeep it away from the composer. ",
            assistantText = "The composer should slide with IME insets and return to the bottom after send.",
        )

        assertTrue(fact.contains("session-123"))
        assertTrue(fact.contains("user asked: How should the keyboard behave? Keep it away from the composer."))
        assertTrue(fact.contains("assistant answered: The composer should slide with IME insets"))
        assertTrue(fact.length <= 1_200)
    }

    @Test
    fun buildChatRequestMessagesAddsRelevantMemoryContextBeforeEndpointUserMessage() {
        val messages = buildChatRequestMessages(
            userText = "What should I check on the phone?",
            memoryContext = "1. User cares about physical overlay validation on the home screen.",
        )

        assertEquals(2, messages.size)
        assertEquals("system", messages[0].role)
        assertTrue(messages[0].content.contains("Relevant local memory context recalled from prior conversations"))
        assertTrue(messages[0].content.contains("physical overlay validation"))
        assertEquals("user", messages[1].role)
        assertEquals("What should I check on the phone?", messages[1].content)
    }

    @Test
    fun buildChatRequestMessagesPreservesAttachmentPartsAndBoundsPersona() {
        val messages = buildChatRequestMessages(
            userText = "Describe this",
            userContentParts = listOf(ChatContentPart(type = "text", text = "Describe this")),
            customSystemPrompt = "x".repeat(2_000),
        )

        assertEquals(2, messages.size)
        assertTrue(messages[0].content.contains("hermes context compressed"))
        assertTrue(messages[0].content.length < 1_200)
        assertEquals(1, messages[1].contentParts.size)
        assertEquals("text", messages[1].contentParts.single().type)
    }

    @Test
    fun buildChatRequestMessagesIncludesBoundedPriorConversationBeforeCurrentUser() {
        val messages = buildChatRequestMessages(
            userText = "What did I just send?",
            priorMessages = listOf(
                ChatMessage(role = "user", content = "hello"),
                ChatMessage(role = "assistant", content = "You sent hello."),
            ),
        )

        assertEquals(3, messages.size)
        assertEquals("user", messages[0].role)
        assertEquals("hello", messages[0].content)
        assertEquals("assistant", messages[1].role)
        assertEquals("You sent hello.", messages[1].content)
        assertEquals("What did I just send?", messages[2].content)
    }

    @Test
    fun buildChatRequestMessagesUsesLargerPriorContextWhenCacheResendIsEnabled() {
        val priorMessages = (1..20).map { index ->
            ChatMessage(
                role = if (index % 2 == 0) "assistant" else "user",
                content = "turn-$index " + "x".repeat(2_400),
            )
        }

        val compact = buildChatRequestMessages(
            userText = "continue",
            priorMessages = priorMessages,
        )
        val cacheFriendly = buildChatRequestMessages(
            userText = "continue",
            priorMessages = priorMessages,
            cacheResendEnabled = true,
        )

        assertEquals(13, compact.size)
        assertEquals(21, cacheFriendly.size)
        assertTrue(compact.first().content.length < 1_400)
        assertTrue(cacheFriendly.first().content.length > 2_000)
    }

    @Test
    fun buildPriorChatRequestMessagesDropsBlankAssistantPlaceholdersAndAttachmentPayloads() {
        val prior = buildPriorChatRequestMessages(
            listOf(
                ChatUiMessage("u1", "user", "hello", 1L),
                ChatUiMessage("a1", "assistant", "", 2L),
                ChatUiMessage(
                    id = "u2",
                    role = "user",
                    content = "",
                    createdAtEpochMs = 3L,
                    attachments = listOf(ChatAttachment("content://image", "photo.jpg", "image/jpeg")),
                ),
            ),
        )

        assertEquals(2, prior.size)
        assertEquals("hello", prior[0].content)
        assertTrue(prior[1].content.contains("[prior turn attachment omitted: photo.jpg]"))
    }

    @Test
    fun allFeaturesPromptRoutesDirectlyToNativeSelfTestDiagnostics() {
        val arguments = directNativeDiagnosticArgumentsForPrompt("Run a full all features test for Hermes native tools")

        requireNotNull(arguments)
        assertEquals("agent_native_tool_self_test_report", arguments.getString("action"))
    }

    @Test
    fun chineseAllFeaturesPromptRoutesDirectlyToNativeSelfTestDiagnostics() {
        val arguments = directNativeDiagnosticArgumentsForPrompt("全部功能全测试")

        requireNotNull(arguments)
        assertEquals("agent_native_tool_self_test_report", arguments.getString("action"))
    }

    @Test
    fun ordinaryChatPromptDoesNotBypassConfiguredEndpoint() {
        val arguments = directNativeDiagnosticArgumentsForPrompt("Write a short welcome message")

        assertEquals(null, arguments)
    }

    @Test
    fun directNativeDiagnosticsReplyPrefersBridgeOutputText() {
        val reply = formatDirectNativeDiagnosticsReply(
            """{"success":true,"output":"Hermes native tool self-test\nterminal_tool: ready"}""",
        )

        assertEquals("Hermes native tool self-test\nterminal_tool: ready", reply)
    }
}
