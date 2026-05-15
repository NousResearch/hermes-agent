package com.nousresearch.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.InterruptedIOException

class NativeToolCallingChatClientTest {
    @Test
    fun skipsLocalFollowUpAfterExternalActivityHandoff() {
        val result = JSONObject()
            .put("success", true)
            .put("action", "open_uri")
            .put("external_activity_handoff", true)
            .put("message", "Started Android intent")

        assertTrue(NativeToolCallingChatClient.shouldSkipNativeFollowUpAfterToolResult(result.toString()))
    }

    @Test
    fun continuesLocalFollowUpAfterOrdinaryToolResult() {
        val result = JSONObject()
            .put("success", true)
            .put("path", "hermes-output.txt")

        assertFalse(NativeToolCallingChatClient.shouldSkipNativeFollowUpAfterToolResult(result.toString()))
    }

    @Test
    fun recoversHtmlGenerationTimeoutsWithFallbackDocument() {
        assertTrue(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                IllegalStateException("LiteRT-LM generation timed out after 45 seconds before producing a response"),
            )
        )
        assertTrue(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                InterruptedIOException("timeout"),
            )
        )
    }

    @Test
    fun doesNotHideNonTimeoutHtmlGenerationErrors() {
        assertFalse(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                IllegalStateException("image input is unavailable"),
            )
        )
    }

    @Test
    fun compactsLargeJsonToolResultForLocalModelContext() {
        val largeOutput = "scroll-item\n".repeat(700)
        val result = JSONObject()
            .put("exit_code", 0)
            .put("output", largeOutput)
            .put("cwd", "/data/data/com.nousresearch.hermesagent/files")
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertTrue(parsed.getInt("_original_chars") > compacted.length)
        assertEquals(0, parsed.getInt("exit_code"))
        assertTrue(parsed.getString("output").contains("scroll-item"))
        assertTrue(parsed.getString("output").contains("hermes context compressed"))
        assertTrue(compacted.length < result.length)
    }

    @Test
    fun compactsCompletedNativeToolRoundsButKeepsLatestAssistantBlock() {
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", "tools enabled"))
            .put(JSONObject().put("role", "user").put("content", "Scroll TikTok, reply to DMs, then draft email"))
        repeat(6) { index ->
            messages
                .put(
                    assistantToolCall(
                        id = "call_$index",
                        name = "android_ui_tool",
                    ),
                )
                .put(
                    JSONObject()
                        .put("role", "tool")
                        .put("tool_call_id", "call_$index")
                        .put("name", "android_ui_tool")
                        .put(
                            "content",
                            NativeToolContextCompressor.compactToolResult(
                                "Scrolled feed $index\n" + "visible row $index ".repeat(1_800),
                            ),
                        ),
                )
        }

        val compacted = NativeToolContextCompressor.compactMessages(messages)

        assertTrue(compacted.length() < messages.length())
        assertEquals("system", compacted.getJSONObject(0).getString("role"))
        assertEquals("user", compacted.getJSONObject(1).getString("role"))
        assertTrue(compacted.getJSONObject(2).getString("content").contains("compacted prior native tool context"))
        val latestAssistant = compacted.getJSONObject(compacted.length() - 2)
        assertEquals("assistant", latestAssistant.getString("role"))
        assertEquals("call_5", latestAssistant.getJSONArray("tool_calls").getJSONObject(0).getString("id"))
        assertEquals("tool", compacted.getJSONObject(compacted.length() - 1).getString("role"))
    }

    private fun assistantToolCall(id: String, name: String): JSONObject {
        return JSONObject()
            .put("role", "assistant")
            .put("content", JSONObject.NULL)
            .put(
                "tool_calls",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("id", id)
                            .put("type", "function")
                            .put(
                                "function",
                                JSONObject()
                                    .put("name", name)
                                    .put("arguments", JSONObject().put("action", "scroll_forward").toString()),
                            ),
                    ),
            )
    }
}
