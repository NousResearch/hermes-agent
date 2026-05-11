package com.nousresearch.hermesagent

import android.app.Activity
import android.app.Application
import android.app.Instrumentation
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.Intents.intended
import androidx.test.espresso.intent.Intents.intending
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nousresearch.hermesagent.backend.BackendKind
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import com.nousresearch.hermesagent.ui.boot.BootUiState
import com.nousresearch.hermesagent.ui.shell.AppShellScreen
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.hamcrest.Description
import org.hamcrest.Matcher
import org.hamcrest.TypeSafeMatcher
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException
import java.util.Collections

@RunWith(AndroidJUnit4::class)
class DeepAppUiVisualInstrumentedTest {
    @get:Rule
    val composeRule = createComposeRule()

    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesRuntimeManager.stop()
        OnDeviceBackendManager.stopAll()
    }

    @Test
    fun settingsOneTapModelNavigationAndTranslationFlowCapturesScreenshots() {
        LocalModelDownloadStore(app).apply {
            saveDownloads(emptyList())
            setPreferredDownloadId("")
        }
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "visual-ui-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithText("Hermes Chat").assertIsDisplayed()
        capture("01-hermes-chat")
        composeRule.onNodeWithTag("HermesChatInput").performTextInput("Describe the attached image and then summarize the phone status.")
        capture("02-hermes-typing")

        composeRule.onNodeWithTag("HermesNavSettings").performClick()
        composeRule.onAllNodesWithText("Settings")[0].assertIsDisplayed()
        capture("03-settings")
        composeRule.onNodeWithText("One-tap local models").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Qwen3.5 0.8B Q4_K_M (GGUF)").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Gemma 4 E2B (LiteRT-LM)").performScrollTo().assertIsDisplayed()
        capture("04-one-tap-models")

        composeRule.onNodeWithText("🇪🇸 Español").performScrollTo().performClick()
        assertTrue(composeRule.onAllNodesWithText("Idioma de la app").fetchSemanticsNodes().isNotEmpty())
        composeRule.onNodeWithText(
            "Aún no hay un modelo local compatible seleccionado. Descárgalo y márcalo como preferido primero."
        ).performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Modelos locales con un toque").performScrollTo()
        assertTrue(composeRule.onAllNodesWithText("Descargar e iniciar").fetchSemanticsNodes().isNotEmpty())
        capture("05-settings-spanish")

        composeRule.onNodeWithTag("HermesNavAccounts").performClick()
        composeRule.onAllNodesWithText("Cuentas")[0].assertIsDisplayed()
        capture("06-accounts-spanish")

        composeRule.onNodeWithTag("HermesNavDevice").performClick()
        composeRule.onAllNodesWithText("Dispositivo")[0].assertIsDisplayed()
        capture("07-device-spanish")

        composeRule.onNodeWithTag("HermesNavNousPortal").performClick()
        composeRule.onAllNodesWithText("Nous Portal")[0].assertIsDisplayed()
        capture("08-portal-spanish")
    }

    @Test
    fun signinQwenCommandOpensSetupPageAndReloadsSettingsProviderProfile() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "signin-qwen-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        val qwenSetupIntent = browserChooserFor(Uri.parse("https://home.qwencloud.com/api-keys"))
        Intents.init()
        try {
            intending(qwenSetupIntent).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, null))

            composeRule.onNodeWithTag("HermesChatInput").performTextInput("/signin qwen")
            composeRule.onNodeWithText("Send").performClick()

            intended(qwenSetupIntent)
            composeRule.onAllNodesWithText("Settings")[0].assertIsDisplayed()
            assertTrue(
                composeRule.onAllNodesWithText("Current provider profile: Qwen Cloud / DashScope API key")
                    .fetchSemanticsNodes()
                    .isNotEmpty()
            )
        } finally {
            Intents.release()
        }
    }

    @Test
    fun accountsRuntimeProvidersExposeDirectSetupUrls() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "accounts-provider-setup-url-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithTag("HermesNavAccounts").performClick()
        composeRule.onAllNodesWithText("Accounts")[0].assertIsDisplayed()
        composeRule.onNodeWithText("Qwen OAuth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Set up Qwen OAuth API key").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderOpenSetup-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderCopySetup-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderCopySetup-qwen-oauth").performClick()
        composeRule.onNodeWithText("Copied Qwen OAuth setup URL and 1 alternate official page.").performScrollTo().assertIsDisplayed()
    }

    @Test
    fun corr3xtSignInRejectsReachableHostWithoutOAuthStartRoute() {
        val server = TestHttpServer { target -> if (target == "/") 200 else 404 }
        try {
            AppSettingsStore(app).save(
                AppSettings(
                    provider = "openrouter",
                    baseUrl = "https://openrouter.ai/api/v1",
                    model = "anthropic/claude-sonnet-4",
                    corr3xtBaseUrl = "http://127.0.0.1:${server.port}",
                    onDeviceBackend = BackendKind.NONE.persistedValue,
                    languageTag = "en",
                )
            )

            composeRule.setContent {
                AppShellScreen(
                    bootUiState = BootUiState(
                        status = "Hermes backend is ready",
                        ready = true,
                        probeResult = "corr3xt-route-test",
                        baseUrl = "http://127.0.0.1:15436/v1",
                    ),
                    onRetryHermes = {},
                )
            }

            composeRule.onNodeWithTag("HermesNavAccounts").performClick()
            composeRule.onAllNodesWithText("Accounts")[0].assertIsDisplayed()
            composeRule.onAllNodesWithText("Sign in")[1].performClick()
            composeRule.waitUntil(timeoutMillis = 5_000) {
                composeRule.onAllNodesWithText(
                    "Corr3xt app sign-in page could not be reached: HTTP 404",
                    substring = true,
                ).fetchSemanticsNodes().isNotEmpty()
            }
            assertTrue(server.seenRequests().contains("/oauth/start"))
        } finally {
            server.close()
        }
    }

    private fun capture(name: String) {
        composeRule.waitForIdle()
        val outputDir = File(app.filesDir, "hermes-ui-visuals").apply { mkdirs() }
        val bitmap = InstrumentationRegistry.getInstrumentation().uiAutomation.takeScreenshot()
        FileOutputStream(File(outputDir, "$name.png")).use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
    }

    private fun browserChooserFor(uri: Uri): Matcher<Intent> {
        return object : TypeSafeMatcher<Intent>() {
            override fun describeTo(description: Description) {
                description.appendText("browser chooser for ").appendValue(uri)
            }

            override fun matchesSafely(intent: Intent): Boolean {
                if (intent.action != Intent.ACTION_CHOOSER) {
                    return false
                }
                @Suppress("DEPRECATION")
                val target = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
                return target?.action == Intent.ACTION_VIEW && target.data == uri
            }
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
}
