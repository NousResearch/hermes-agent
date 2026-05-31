package com.mobilefork.hermesagent.api

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test
    fun createChatCompletion_normalizesPastedFullEndpointUrl() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("{" + "\"ok\":true}"))

        val client = HermesApiClient(server.url("/proxy/v1/chat/completions").toString())
        client.createChatCompletion(
            ChatCompletionRequest(
                model = "custom-model",
                messages = listOf(ChatMessage(role = "user", content = "hello")),
            )
        )

        assertEquals("/proxy/v1/chat/completions", server.takeRequest().path)
    }

    @Test
    fun createResponse_sendsResponsesPayloadWithStoreDisabled() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"output_text":"ok"}"""))

        val client = HermesApiClient(server.url("/").toString(), apiKey = "secret")
        val result = client.createResponse(
            ChatCompletionRequest(
                model = "gpt-5",
                messages = listOf(ChatMessage(role = "user", content = "hello")),
                stream = false,
                sessionId = "session-123",
            ),
        )

        val recorded = server.takeRequest()
        assertEquals("/v1/responses", recorded.path)
        assertEquals("Bearer secret", recorded.getHeader("Authorization"))
        assertEquals("session-123", recorded.getHeader(HermesApiClient.SESSION_HEADER))
        val body = JSONObject(recorded.body.readUtf8())
        assertEquals("gpt-5", body.getString("model"))
        assertEquals(false, body.getBoolean("stream"))
        assertEquals(false, body.getBoolean("store"))
        assertEquals("hello", body.getJSONArray("input").getJSONObject(0).getString("content"))
        assertEquals("""{"output_text":"ok"}""", result.rawBody)
    }

    @Test
    fun createResponse_convertsImagePartsToResponsesInputContent() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"output_text":"ok"}"""))

        val client = HermesApiClient(server.url("/").toString())
        client.createResponse(
            ChatCompletionRequest(
                model = "gpt-5",
                messages = listOf(
                    ChatMessage(
                        role = "user",
                        content = "describe this",
                        contentParts = listOf(
                            ChatContentPart(type = "text", text = "describe this"),
                            ChatContentPart(type = "image_url", imageUrl = "data:image/png;base64,AA=="),
                        ),
                    ),
                ),
            ),
        )

        val content = JSONObject(server.takeRequest().body.readUtf8())
            .getJSONArray("input")
            .getJSONObject(0)
            .getJSONArray("content")
        assertEquals("input_text", content.getJSONObject(0).getString("type"))
        assertEquals("input_image", content.getJSONObject(1).getString("type"))
        assertEquals("data:image/png;base64,AA==", content.getJSONObject(1).getString("image_url"))
    }

    @Test
    fun chatCompletionPayloadIncludesOptionalGenerationKnobs() {
        val payload = ChatCompletionRequest(
            model = "model",
            messages = listOf(ChatMessage("user", "hello")),
            stream = true,
            maxTokens = 4096,
            topP = 0.9f,
            temperature = 0.7f,
        ).toChatCompletionPayload()

        assertEquals(4096, payload.getInt("max_tokens"))
        assertEquals(0.9, payload.getDouble("top_p"), 0.0001)
        assertEquals(0.7, payload.getDouble("temperature"), 0.0001)
    }

    @Test
    fun responsesPayloadMapsMaxTokensToMaxOutputTokens() {
        val payload = ChatCompletionRequest(
            model = "model",
            messages = listOf(ChatMessage("user", "hello")),
            maxTokens = 2048,
        ).toResponsesPayload()

        assertEquals(2048, payload.getInt("max_output_tokens"))
        assertFalse(payload.has("max_tokens"))
    }
}
