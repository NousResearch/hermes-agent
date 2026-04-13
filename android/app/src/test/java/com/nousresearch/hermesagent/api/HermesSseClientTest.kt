package com.nousresearch.hermesagent.api

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

    private fun sampleRequest(): ChatCompletionRequest {
        return ChatCompletionRequest(
            model = "gemma-4-local",
            messages = listOf(ChatMessage(role = "user", content = "hello")),
            stream = true,
            sessionId = "session-123",
        )
    }

    private fun singleResponseClient(body: String): OkHttpClient {
        return OkHttpClient.Builder()
            .addInterceptor { chain ->
                Response.Builder()
                    .request(chain.request())
                    .protocol(Protocol.HTTP_1_1)
                    .code(200)
                    .message("OK")
                    .body(body.toResponseBody("text/event-stream".toMediaType()))
                    .build()
            }
            .build()
    }
}
