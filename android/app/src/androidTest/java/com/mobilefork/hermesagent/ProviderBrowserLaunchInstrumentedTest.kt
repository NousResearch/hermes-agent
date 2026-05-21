package com.mobilefork.hermesagent

import android.app.Activity
import android.app.Application
import android.app.Instrumentation
import android.content.Intent
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.Intents.intending
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.mobilefork.hermesagent.auth.OpenRouterLoopbackOAuthServer
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.AuthSessionStore
import com.mobilefork.hermesagent.device.HermesExternalBrowserLauncher
import com.mobilefork.hermesagent.device.HermesProviderSetupWebActivity
import com.mobilefork.hermesagent.ui.auth.AuthViewModel
import com.mobilefork.hermesagent.ui.settings.SettingsViewModel
import org.hamcrest.Description
import org.hamcrest.TypeSafeMatcher
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.atomic.AtomicBoolean

@RunWith(AndroidJUnit4::class)
class ProviderBrowserLaunchInstrumentedTest {
    private val app: Application = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        OpenRouterLoopbackOAuthServer.stopCurrent()
        AuthSessionStore(app).clearPendingRequest()
    }

    @Test
    fun openRouterOAuthLaunchesExternalBrowserWithLoopbackCallback() {
        val expectedHost = "openrouter.ai"
        assumeTrue(
            "No browser is installed on this test device",
            HermesExternalBrowserLauncher.createBrowserIntent(
                app,
                Uri.parse("https://$expectedHost/auth"),
            ).resolveActivity(app.packageManager) != null,
        )
        val opened = AtomicBoolean(false)
        val matcher = externalBrowserIntentFor(
            onMatch = { opened.set(true) },
        ) { uri ->
            val callbackUrl = Uri.parse(uri.getQueryParameter("callback_url").orEmpty())
            uri.scheme == "https" &&
                uri.host == expectedHost &&
                uri.path == "/auth" &&
                uri.getQueryParameter("code_challenge_method") == "S256" &&
                callbackUrl.scheme == "http" &&
                callbackUrl.host == "localhost" &&
                callbackUrl.port == OpenRouterLoopbackOAuthServer.DEFAULT_PORT &&
                callbackUrl.path == "/hermes/openrouter/callback"
        }

        Intents.init()
        try {
            intending(matcher).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, null))

            assertTrue(AuthViewModel(app).startAuth("openrouter"))

            val pending = AuthSessionStore(app).loadPendingRequest()
            assertEquals("openrouter", pending?.methodId)
            assertEquals("openrouter-oauth", pending?.authProvider)
            assertEquals("S256", pending?.codeChallengeMethod)
            assertTrue(pending?.startUrl.orEmpty().startsWith("https://openrouter.ai/auth"))
        } finally {
            Intents.release()
        }
        assertTrue(opened.get())
    }

    @Test
    fun accountsRuntimeProviderSetupLaunchesExternalBrowser() {
        val setupUri = Uri.parse("https://docs.qwencloud.com/api-reference/preparation/api-key")
        assumeTrue(
            "No browser is installed on this test device",
            HermesExternalBrowserLauncher.createBrowserIntent(app, setupUri)
                .resolveActivity(app.packageManager) != null,
        )
        val opened = AtomicBoolean(false)

        Intents.init()
        try {
            intending(
                externalBrowserIntentFor(onMatch = { opened.set(true) }) { it == setupUri },
            ).respondWith(
                Instrumentation.ActivityResult(Activity.RESULT_OK, null),
            )

            assertTrue(AuthViewModel(app).startAuth("qwen"))

            assertEquals("alibaba", AppSettingsStore(app).load().provider)
        } finally {
            Intents.release()
        }
        assertTrue(opened.get())
    }

    @Test
    fun settingsProviderSetupLaunchesExternalBrowser() {
        val setupUri = Uri.parse("https://docs.qwencloud.com/api-reference/preparation/api-key")
        assumeTrue(
            "No browser is installed on this test device",
            HermesExternalBrowserLauncher.createBrowserIntent(app, setupUri)
                .resolveActivity(app.packageManager) != null,
        )
        val opened = AtomicBoolean(false)

        Intents.init()
        try {
            intending(
                externalBrowserIntentFor(onMatch = { opened.set(true) }) { it == setupUri },
            ).respondWith(
                Instrumentation.ActivityResult(Activity.RESULT_OK, null),
            )

            SettingsViewModel(app).openProviderKeyPage(setupUri.toString())

        } finally {
            Intents.release()
        }
        assertTrue(opened.get())
    }

    private fun externalBrowserIntentFor(
        onMatch: () -> Unit = {},
        matchesUri: (Uri) -> Boolean,
    ): TypeSafeMatcher<Intent> {
        return object : TypeSafeMatcher<Intent>() {
            override fun describeTo(description: Description) {
                description.appendText("external browser intent")
            }

            override fun matchesSafely(intent: Intent): Boolean {
                if (intent.component?.className == HermesProviderSetupWebActivity::class.java.name) {
                    return false
                }
                val chooserTarget = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
                val uri = intent.data ?: chooserTarget?.data ?: return false
                val matches = intent.action in setOf(Intent.ACTION_VIEW, Intent.ACTION_CHOOSER) &&
                    matchesUri(uri)
                if (matches) {
                    onMatch()
                }
                return matches
            }
        }
    }
}
