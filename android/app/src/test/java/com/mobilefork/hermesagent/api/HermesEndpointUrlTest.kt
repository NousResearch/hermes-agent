package com.mobilefork.hermesagent.api

import org.junit.Assert.assertEquals
import org.junit.Test

class HermesEndpointUrlTest {
    @Test
    fun normalizeBaseUrl_acceptsRawHttpsHostWithOpenAiPath() {
        assertEquals(
            "https://api.example.com",
            HermesEndpointUrl.normalizeBaseUrl("api.example.com/v1/chat/completions"),
        )
        assertEquals(
            "https://api.example.com/v1/chat/completions",
            HermesEndpointUrl.chatCompletionsUrl("api.example.com/v1/chat/completions"),
        )
    }

    @Test
    fun normalizeBaseUrl_usesHttpForLoopbackAndLanWithoutScheme() {
        assertEquals(
            "http://localhost:8000",
            HermesEndpointUrl.normalizeBaseUrl("localhost:8000/v1"),
        )
        assertEquals(
            "http://192.168.1.24:8080",
            HermesEndpointUrl.normalizeBaseUrl("192.168.1.24:8080/v1/models"),
        )
        assertEquals(
            "http://[::1]:9000",
            HermesEndpointUrl.normalizeBaseUrl("[::1]:9000/v1"),
        )
    }

    @Test
    fun normalizeBaseUrl_preservesProxyPrefixWhenFullEndpointIsPasted() {
        assertEquals(
            "https://gateway.example.com/openai",
            HermesEndpointUrl.normalizeBaseUrl("https://gateway.example.com/openai/v1/chat/completions?debug=1"),
        )
    }

    @Test
    fun normalizeBaseUrl_convertsWebSocketSchemesForHttpCompatibleRequests() {
        assertEquals(
            "https://stream.example.com",
            HermesEndpointUrl.normalizeBaseUrl("wss://stream.example.com/v1/chat/completions"),
        )
        assertEquals(
            "http://127.0.0.1:9000",
            HermesEndpointUrl.normalizeBaseUrl("ws://127.0.0.1:9000/v1"),
        )
    }

    @Test
    fun openAiRuntimeBaseUrl_preservesProxyPrefixAndKeepsV1ForSdkCalls() {
        assertEquals(
            "https://gateway.example.com/openai/v1",
            HermesEndpointUrl.openAiRuntimeBaseUrl(
                "gateway.example.com/openai/v1/chat/completions?ignored=true",
            ),
        )
        assertEquals(
            "http://localhost:11434/v1",
            HermesEndpointUrl.openAiRuntimeBaseUrl("localhost:11434"),
        )
    }
}
