package com.nousresearch.hermesagent.auth

import android.net.Uri
import com.nousresearch.hermesagent.data.AuthSessionStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException
import java.util.Collections

@RunWith(RobolectricTestRunner::class)
class OpenRouterOAuthClientTest {
    @Test
    fun createStartRequestBuildsPkceAuthUrlWithHermesCallback() {
        val request = OpenRouterOAuthClient.createStartRequest(
            state = "state-123",
            verifier = "test-verifier",
        )
        val startUri = request.startUri
        val callbackUri = Uri.parse(startUri.getQueryParameter("callback_url"))

        assertEquals("https", startUri.scheme)
        assertEquals("openrouter.ai", startUri.host)
        assertEquals("/auth", startUri.path)
        assertEquals("S256", startUri.getQueryParameter("code_challenge_method"))
        assertEquals(
            OpenRouterOAuthClient.codeChallenge("test-verifier"),
            startUri.getQueryParameter("code_challenge"),
        )
        assertEquals("hermesagent", callbackUri.scheme)
        assertEquals("auth", callbackUri.host)
        assertEquals("/callback", callbackUri.path)
        assertEquals("openrouter", callbackUri.getQueryParameter("method"))
        assertEquals("openrouter", callbackUri.getQueryParameter("provider"))
        assertEquals("state-123", callbackUri.getQueryParameter("state"))
        assertEquals("openrouter", request.pendingRequest.methodId)
        assertEquals("openrouter-oauth", request.pendingRequest.authProvider)
        assertEquals("test-verifier", request.pendingRequest.codeVerifier)
        assertEquals(request.startUri.toString(), request.pendingRequest.startUrl)
    }

    @Test
    fun createStartRequestAcceptsLoopbackCallbackUrlForMobileBrowserOAuth() {
        val callbackUrl = OpenRouterLoopbackOAuthServer.callbackUrlForState("state-123")

        val request = OpenRouterOAuthClient.createStartRequest(
            state = "state-123",
            verifier = "test-verifier",
            callbackUrl = callbackUrl,
        )
        val callbackUri = Uri.parse(request.startUri.getQueryParameter("callback_url"))

        assertEquals("http", callbackUri.scheme)
        assertEquals("localhost", callbackUri.host)
        assertEquals(3000, callbackUri.port)
        assertEquals("/hermes/openrouter/callback", callbackUri.path)
        assertEquals("openrouter", callbackUri.getQueryParameter("method"))
        assertEquals("openrouter", callbackUri.getQueryParameter("provider"))
        assertEquals("state-123", callbackUri.getQueryParameter("state"))
    }

    @Test
    fun exchangeCodeForApiKeyPostsPkcePayloadAndReturnsKey() {
        val server = TestHttpServer { target, body ->
            assertEquals("/api/v1/auth/keys", target)
            assertTrue(body.contains("\"code\":\"callback-code\""))
            assertTrue(body.contains("\"code_verifier\":\"verifier-123\""))
            """{"key":"sk-or-v1-test"}"""
        }
        try {
            val result = OpenRouterOAuthClient.exchangeCodeForApiKey(
                code = "callback-code",
                codeVerifier = "verifier-123",
                exchangeUrl = "http://127.0.0.1:${server.port}/api/v1/auth/keys",
                timeoutMs = 1_000,
            )

            assertTrue(result.success)
            assertEquals("sk-or-v1-test", result.apiKey)
        } finally {
            server.close()
        }
    }

    @Test
    fun exchangeCallbackForSessionRejectsStateMismatchBeforeNetwork() {
        val request = OpenRouterOAuthClient.createStartRequest(
            state = "state-123",
            verifier = "verifier-123",
        ).pendingRequest

        val session = OpenRouterOAuthClient.exchangeCallbackForSession(
            uri = Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=openrouter&provider=openrouter&state=wrong&code=callback-code"),
            pending = request,
            nowEpochMs = 100L,
            exchangeUrl = "http://127.0.0.1:1",
        )

        assertFalse(session.signedIn)
        assertEquals("OpenRouter sign-in rejected: state mismatch", session.status)
    }

    private class TestHttpServer(
        private val responseBodyForRequest: (String, String) -> String,
    ) : AutoCloseable {
        private val serverSocket = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))
        private val requests = Collections.synchronizedList(mutableListOf<String>())
        private val thread = Thread {
            try {
                while (!serverSocket.isClosed) {
                    val socket = serverSocket.accept()
                    socket.use { client ->
                        val reader = BufferedReader(InputStreamReader(client.getInputStream()))
                        val requestLine = reader.readLine().orEmpty()
                        val target = requestLine.split(" ").getOrNull(1).orEmpty()
                        var contentLength = 0
                        while (true) {
                            val line = reader.readLine() ?: break
                            if (line.isBlank()) break
                            if (line.startsWith("Content-Length:", ignoreCase = true)) {
                                contentLength = line.substringAfter(":").trim().toIntOrNull() ?: 0
                            }
                        }
                        val bodyChars = CharArray(contentLength)
                        if (contentLength > 0) {
                            reader.read(bodyChars, 0, contentLength)
                        }
                        val body = String(bodyChars)
                        requests.add(target)
                        val responseBody = responseBodyForRequest(target, body)
                        client.getOutputStream().write(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${responseBody.toByteArray().size}\r\nConnection: close\r\n\r\n$responseBody"
                                .toByteArray(Charsets.UTF_8)
                        )
                    }
                }
            } catch (_: SocketException) {
                // Closing the server socket stops the test server.
            }
        }.apply {
            isDaemon = true
            start()
        }

        val port: Int
            get() = serverSocket.localPort

        override fun close() {
            serverSocket.close()
            thread.join(1_000)
        }
    }
}
