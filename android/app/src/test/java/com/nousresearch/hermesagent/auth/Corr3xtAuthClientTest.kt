package com.nousresearch.hermesagent.auth

import com.nousresearch.hermesagent.data.AuthCatalog
import com.nousresearch.hermesagent.data.AuthScope
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertNull
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
class Corr3xtAuthClientTest {
    @Test
    fun normalizeConfiguredBaseUrl_stripsQueryFragmentAndTrailingSlash() {
        assertEquals(
            "https://auth.corr3xt.com/base",
            Corr3xtAuthClient.normalizeConfiguredBaseUrl("https://auth.corr3xt.com/base/?foo=bar#frag"),
        )
    }

    @Test
    fun normalizeConfiguredBaseUrl_rejectsUnsupportedSchemes() {
        assertNull(Corr3xtAuthClient.normalizeConfiguredBaseUrl("javascript:alert(1)"))
    }

    @Test
    fun normalizeConfiguredBaseUrl_keepsBlankUrlsUnconfigured() {
        assertNull(Corr3xtAuthClient.normalizeConfiguredBaseUrl(""))
        assertEquals("", Corr3xtAuthClient.normalizedBaseUrl(""))
    }

    @Test(expected = IllegalArgumentException::class)
    fun buildStartUri_rejectsBlankBaseUrl() {
        val option = requireNotNull(AuthCatalog.find("google"))
        Corr3xtAuthClient.buildStartUri("", option, "state-123")
    }

    @Test
    fun buildStartUri_includesCallbackContractAndRedirectUri() {
        val option = requireNotNull(AuthCatalog.find("chatgpt"))
        val uri = Corr3xtAuthClient.buildStartUri("https://auth.corr3xt.com/", option, "state-123")

        assertEquals("https", uri.scheme)
        assertEquals("auth.corr3xt.com", uri.host)
        assertEquals("/oauth/start", uri.path)
        assertEquals("v1", uri.getQueryParameter("callback_contract"))
        assertEquals("hermes-android", uri.getQueryParameter("client"))
        assertEquals("hermesagent://auth/callback", uri.getQueryParameter("redirect_uri"))
        assertEquals("state-123", uri.getQueryParameter("state"))
    }

    @Test
    fun probeStartUri_checksOAuthStartRouteWithoutTriggeringQuerySideEffects() {
        val option = requireNotNull(AuthCatalog.find("google"))
        val server = TestHttpServer { target -> if (target == "/oauth/start") 400 else 404 }
        try {
            val uri = Corr3xtAuthClient.buildStartUri(
                "http://127.0.0.1:${server.port}",
                option,
                "state-123",
            )

            val result = Corr3xtAuthClient.probeStartUri(uri, timeoutMs = 1_000)

            assertTrue(result.reachable)
            assertEquals(listOf("/oauth/start"), server.seenRequests())
        } finally {
            server.close()
        }
    }

    @Test
    fun probeStartUri_acceptsOauthBackendsThatRequireQueryParameters() {
        val option = requireNotNull(AuthCatalog.find("google"))
        val server = TestHttpServer { target ->
            when {
                target == "/oauth/start" -> 404
                target.startsWith("/oauth/start?") -> 200
                else -> 404
            }
        }
        try {
            val uri = Corr3xtAuthClient.buildStartUri(
                "http://127.0.0.1:${server.port}",
                option,
                "state-123",
            )

            val result = Corr3xtAuthClient.probeStartUri(uri, timeoutMs = 1_000)

            assertTrue(result.reachable)
            assertEquals("query_required", result.status)
            assertEquals(2, server.seenRequests().size)
            assertEquals("/oauth/start", server.seenRequests().first())
            assertTrue(server.seenRequests().last().startsWith("/oauth/start?"))
        } finally {
            server.close()
        }
    }

    @Test
    fun probeStartUri_rejectsReachableHostsWithoutOAuthStartRoute() {
        val option = requireNotNull(AuthCatalog.find("google"))
        val server = TestHttpServer { target -> if (target == "/") 200 else 404 }
        try {
            val uri = Corr3xtAuthClient.buildStartUri(
                "http://127.0.0.1:${server.port}",
                option,
                "state-123",
            )

            val result = Corr3xtAuthClient.probeStartUri(uri, timeoutMs = 1_000)

            assertEquals(false, result.reachable)
            assertEquals("network_error", result.status)
            assertEquals("HTTP 404", result.errorName)
        } finally {
            server.close()
        }
    }

    private class TestHttpServer(private val responseCodeForTarget: (String) -> Int) : AutoCloseable {
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
                        requests.add(target)
                        val code = responseCodeForTarget(target)
                        val reason = when (code) {
                            200 -> "OK"
                            400 -> "Bad Request"
                            404 -> "Not Found"
                            else -> "Status"
                        }
                        client.getOutputStream().write(
                            "HTTP/1.1 $code $reason\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
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

        fun seenRequests(): List<String> = requests.toList()

        override fun close() {
            serverSocket.close()
            thread.join(1_000)
        }
    }

    @Test
    fun authCatalog_includesOpenRouterForApiKeySetup() {
        val option = requireNotNull(AuthCatalog.find("openrouter"))

        assertEquals(AuthScope.RuntimeProvider, option.scope)
        assertEquals("openrouter", option.runtimeProvider)
        assertEquals("https://openrouter.ai/api/v1", option.defaultBaseUrl)
        assertTrue(option.defaultModel.isNotBlank())
    }

    @Test
    fun runtimeProviderAuthOptionsDeclareApiKeyFallbackTargets() {
        val runtimeOptions = AuthCatalog.options.filter { it.scope == AuthScope.RuntimeProvider }

        assertTrue(runtimeOptions.isNotEmpty())
        runtimeOptions.forEach { option ->
            assertTrue("${option.id} should declare runtimeProvider", option.runtimeProvider.isNotBlank())
            assertTrue("${option.id} should declare defaultBaseUrl", option.defaultBaseUrl.isNotBlank())
            assertTrue("${option.id} should declare defaultModel", option.defaultModel.isNotBlank())
            if (option.id == "openrouter") {
                assertTrue("${option.id} should support native OpenRouter PKCE", option.browserSignInSupported)
            } else {
                assertTrue("${option.id} should prefer local key setup over unavailable Corr3xt", !option.browserSignInSupported)
            }
        }
    }
}
