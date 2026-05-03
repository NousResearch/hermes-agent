package com.nousresearch.hermesagent.api

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.json.JSONObject

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
        val recordedBody = recorded.body.readUtf8()
        assertTrue(recordedBody.contains("\"hello\""))
        assertEquals("{\"ok\":true}", result.rawBody)
    }

    @Test
    fun createChatCompletion_sendsOpenAiMultimodalContentParts() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("{" + "\"ok\":true}"))

        val client = HermesApiClient(server.url("/").toString())
        client.createChatCompletion(
            ChatCompletionRequest(
                model = "gemma-3n-local",
                messages = listOf(
                    ChatMessage(
                        role = "user",
                        content = "describe this",
                        contentParts = listOf(
                            ChatContentPart(type = "text", text = "describe this"),
                            ChatContentPart(type = "image_url", imageUrl = "data:image/png;base64,AA=="),
                        ),
                    )
                ),
            )
        )

        val body = JSONObject(server.takeRequest().body.readUtf8())
        val content = body
            .getJSONArray("messages")
            .getJSONObject(0)
            .getJSONArray("content")
        assertEquals("text", content.getJSONObject(0).getString("type"))
        assertEquals("image_url", content.getJSONObject(1).getString("type"))
        assertEquals("data:image/png;base64,AA==", content.getJSONObject(1).getJSONObject("image_url").getString("url"))
    }
}
