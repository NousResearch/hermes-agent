package com.mobilefork.hermesagent.auth

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class ProviderSetupUrlProbeTest {
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
    fun probeAcceptsReachableSetupPage() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("<html>API keys</html>"))

        val result = ProviderSetupUrlProbe.probe(server.url("/keys").toString())

        assertTrue(result.reachable)
        assertEquals("HTTP 200", result.statusLabel)
    }

    @Test
    fun probeRejectsDesktopOnlyMobileUnsupportedPage() {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody(
                    """
                    <html>
                      <body>
                        This page is currently not available on mobile devices.
                        Please copy the link below and open it on a desktop for the best experience.
                      </body>
                    </html>
                    """.trimIndent(),
                ),
        )

        val result = ProviderSetupUrlProbe.probe(server.url("/api-keys").toString())

        assertFalse(result.reachable)
        assertEquals("HTTP 200; mobile unsupported page", result.statusLabel)
    }

    @Test
    fun probeKeepsHttpClientErrorsVisibleAsReachable() {
        server.enqueue(MockResponse().setResponseCode(401).setBody("Sign in required"))

        val result = ProviderSetupUrlProbe.probe(server.url("/sign-in").toString())

        assertTrue(result.reachable)
        assertEquals("HTTP 401", result.statusLabel)
    }
}
