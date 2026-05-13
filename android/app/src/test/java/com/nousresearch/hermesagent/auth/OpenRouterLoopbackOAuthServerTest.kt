package com.nousresearch.hermesagent.auth

import android.net.Uri
import com.nousresearch.hermesagent.data.AuthScope
import com.nousresearch.hermesagent.data.AuthSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.net.HttpURLConnection
import java.net.URL

@RunWith(RobolectricTestRunner::class)
class OpenRouterLoopbackOAuthServerTest {
    @Test
    fun callbackUrlUsesDeviceLoopbackPort3000WithStateAndProviderMetadata() {
        val url = OpenRouterLoopbackOAuthServer.callbackUrlForState("state-123")
        val uri = Uri.parse(url)

        assertEquals("http", uri.scheme)
        assertEquals("127.0.0.1", uri.host)
        assertEquals(3000, uri.port)
        assertEquals("/hermes/openrouter/callback", uri.path)
        assertEquals("openrouter", uri.getQueryParameter("method"))
        assertEquals("openrouter", uri.getQueryParameter("provider"))
        assertEquals("state-123", uri.getQueryParameter("state"))
    }

    @Test
    fun serverAcceptsCallbackAndReturnsCompletionPage() {
        val seenCallbacks = mutableListOf<Uri>()
        val start = OpenRouterLoopbackOAuthServer.start(
            port = 0,
            state = "state-123",
            callbackHandler = { callbackUri ->
                seenCallbacks += callbackUri
                AuthSession(
                    methodId = "openrouter",
                    label = "OpenRouter",
                    scope = AuthScope.RuntimeProvider,
                    runtimeProvider = "openrouter",
                    signedIn = true,
                    status = "Signed in with OpenRouter OAuth and saved the API key securely.",
                )
            },
        )
        try {
            assertTrue(start.started)
            val callback = Uri.parse(start.callbackUrl).buildUpon()
                .appendQueryParameter("code", "callback-code")
                .build()
                .toString()

            val connection = URL(callback).openConnection() as HttpURLConnection
            connection.connectTimeout = 2_000
            connection.readTimeout = 2_000
            val body = connection.inputStream.bufferedReader().use { it.readText() }

            assertEquals(200, connection.responseCode)
            assertEquals("callback-code", seenCallbacks.single().getQueryParameter("code"))
            assertTrue(body.contains("OpenRouter is connected"))
        } finally {
            start.handle?.stop()
        }
    }
}
