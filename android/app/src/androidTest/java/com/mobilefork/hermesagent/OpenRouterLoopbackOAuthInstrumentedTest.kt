package com.mobilefork.hermesagent

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.mobilefork.hermesagent.auth.OpenRouterLoopbackOAuthServer
import com.mobilefork.hermesagent.data.AuthScope
import com.mobilefork.hermesagent.data.AuthSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.net.HttpURLConnection
import java.net.URL

@RunWith(AndroidJUnit4::class)
class OpenRouterLoopbackOAuthInstrumentedTest {
    @Test
    fun loopbackServerReceivesOpenRouterCallbackOnAndroid() {
        val seenCallbacks = mutableListOf<Uri>()
        val start = OpenRouterLoopbackOAuthServer.start(
            port = 0,
            state = "state-device",
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
            assertTrue(start.errorName, start.started)
            val callback = Uri.parse(start.callbackUrl).buildUpon()
                .appendQueryParameter("code", "device-code")
                .build()
                .toString()
            val connection = URL(callback).openConnection() as HttpURLConnection
            connection.connectTimeout = 2_000
            connection.readTimeout = 2_000

            val body = connection.inputStream.bufferedReader().use { it.readText() }

            assertEquals(200, connection.responseCode)
            assertEquals("device-code", seenCallbacks.single().getQueryParameter("code"))
            assertTrue(body.contains("OpenRouter is connected"))
        } finally {
            start.handle?.stop()
        }
    }
}
