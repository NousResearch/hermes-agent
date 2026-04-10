package com.nousresearch.hermesagent.api

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class HermesApiClientTest {
    private lateinit var server: MockWebServer

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun getHealth_parsesResponse() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""
            {"status":"ok","platform":"hermes-agent"}
        """.trimIndent()))

        val client = HermesApiClient(server.url("/").toString(), apiKey = "secret")
        val response = client.getHealth()

        val recorded = server.takeRequest()
        assertEquals("/health", recorded.path)
        assertEquals("Bearer secret", recorded.getHeader("Authorization"))
        assertEquals("ok", response.status)
        assertEquals("hermes-agent", response.platform)
    }

    @Test
    fun listModels_parsesIds() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""
            {"data":[{"id":"hermes-agent-android"},{"id":"backup-model"}]}
        """.trimIndent()))

        val client = HermesApiClient(server.url("/").toString())
        val response = client.listModels()

        assertEquals(listOf("hermes-agent-android", "backup-model"), response.data.map { it.id })
    }

    @Test
    fun createChatCompletion_sendsSessionHeaderAndBody() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("{" + "\"ok\":true}"))

        val client = HermesApiClient(server.url("/").toString(), apiKey = "secret")
        val result = client.createChatCompletion(
            ChatCompletionRequest(
                model = "hermes-agent-android",
                messages = listOf(ChatMessage(role = "user", content = "hello")),
                stream = false,
                sessionId = "session-123",
            )
        )

        val recorded = server.takeRequest()
        assertEquals("/v1/chat/completions", recorded.path)
        assertEquals("Bearer secret", recorded.getHeader("Authorization"))
        assertEquals("session-123", recorded.getHeader(HermesApiClient.SESSION_HEADER))
        assertTrue(recorded.body.readUtf8().contains("\"hello\""))
        assertEquals("{\"ok\":true}", result.rawBody)
    }
}
