package com.mobilefork.hermesagent.api

import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException

class HermesSseClientTest {
    @Test
    fun streamChatCompletion_reports_transport_failures_via_onError() {
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = OkHttpClient.Builder()
                .addInterceptor(Interceptor { throw IOException("socket boom") })
                .build(),
        )

        var error: String? = null
        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = {},
            onComplete = {},
            onError = { error = it },
        )

        assertEquals("socket boom", error)
    }

    @Test
    fun streamChatCompletion_reports_malformed_sse_payload_instead_of_throwing() {
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient("data: not-json\n\ndata: [DONE]\n\n"),
        )

        val deltas = mutableListOf<String>()
        var completed = false
        var error: String? = null

        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = { deltas += it },
            onComplete = { completed = true },
            onError = { error = it },
        )

        assertTrue(deltas.isEmpty())
        assertFalse(completed)
        assertNotNull(error)
        assertTrue(error!!.isNotBlank())
    }

    @Test
    fun streamChatCompletion_emits_delta_and_completion_for_valid_sse_payload() {
        val body = """
            data: {"choices":[{"delta":{"content":"hello"}}]}

            data: [DONE]

        """.trimIndent() + "\n"
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient(body),
        )

        val deltas = mutableListOf<String>()
        var completed = false
        var error: String? = null

        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = { deltas += it },
            onComplete = { completed = true },
            onError = { error = it },
        )

        assertEquals(listOf("hello"), deltas)
        assertTrue(completed)
        assertNull(error)
    }

    @Test
    fun streamChatCompletion_reports_endpoint_status_steps() {
        val body = """
            data: {"choices":[{"delta":{"content":"hello"}}]}

            data: [DONE]

        """.trimIndent() + "\n"
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient(body),
        )

        val statuses = mutableListOf<String>()
        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = {},
            onComplete = {},
            onError = {},
            onStatus = { statuses += it },
        )

        assertTrue(statuses.any { it.contains("Opening endpoint stream") })
        assertTrue(statuses.any { it.contains("Endpoint responded HTTP 200") })
        assertTrue(statuses.any { it.contains("Endpoint stream is live") })
    }

    @Test
    fun streamChatCompletion_reports_http_error_body_snippet() {
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient(
                body = """{"error":{"message":"model not found"}}""",
                code = 404,
                message = "Not Found",
            ),
        )

        var error: String? = null
        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = {},
            onComplete = {},
            onError = { error = it },
        )

        assertEquals("""SSE request failed: 404 Not Found {"error":{"message":"model not found"}}""", error)
    }

    @Test
    fun streamChatCompletion_reports_endpoint_hint_when_sse_stream_closes_before_done() {
        val body = """
            data: {"choices":[{"delta":{"content":"partial"}}]}

        """.trimIndent() + "\n"
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient(body),
        )

        val deltas = mutableListOf<String>()
        var completed = false
        var error: String? = null

        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = { deltas += it },
            onComplete = { completed = true },
            onError = { error = it },
        )

        assertEquals(listOf("partial"), deltas)
        assertFalse(completed)
        assertNotNull(error)
        assertTrue(error!!.contains("closed before"))
        assertTrue(error!!.contains("[DONE]"))
        assertTrue(error!!.contains("Base URL"))
    }

    @Test
    fun streamChatCompletion_accepts_finishReasonAsCompletionWhenDoneFrameIsMissing() {
        val body = """
            data: {"choices":[{"delta":{"content":"hello"}}]}

            data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

        """.trimIndent() + "\n"
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient(body),
        )

        val deltas = mutableListOf<String>()
        var completed = false
        var error: String? = null

        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = { deltas += it },
            onComplete = { completed = true },
            onError = { error = it },
        )

        assertEquals(listOf("hello"), deltas)
        assertTrue(completed)
        assertNull(error)
    }

    @Test
    fun streamChatCompletion_accepts_dataFramesWithoutSpaceAndKeepAliveLines() {
        val body = """
            : keep-alive
            event: message
            data:{"choices":[{"delta":{"content":"hello"}}]}

            : keep-alive
            data:[DONE]

        """.trimIndent() + "\n"
        val client = HermesSseClient(
            baseUrl = "http://127.0.0.1:15436",
            httpClient = singleResponseClient(body),
        )

        val deltas = mutableListOf<String>()
        var completed = false
        var error: String? = null

        client.streamChatCompletion(
            request = sampleRequest(),
            onDelta = { deltas += it },
            onComplete = { completed = true },
            onError = { error = it },
        )

        assertEquals(listOf("hello"), deltas)
        assertTrue(completed)
        assertNull(error)
    }

    private fun sampleRequest(): ChatCompletionRequest {
        return ChatCompletionRequest(
            model = "gemma-4-local",
            messages = listOf(ChatMessage(role = "user", content = "hello")),
            stream = true,
            sessionId = "session-123",
        )
    }

    private fun singleResponseClient(body: String, code: Int = 200, message: String = "OK"): OkHttpClient {
        return OkHttpClient.Builder()
            .addInterceptor { chain ->
                Response.Builder()
                    .request(chain.request())
                    .protocol(Protocol.HTTP_1_1)
                    .code(code)
                    .message(message)
                    .body(body.toResponseBody("text/event-stream".toMediaType()))
                    .build()
            }
            .build()
    }
}
