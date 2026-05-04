package com.nousresearch.hermesagent

import android.app.Application
import android.graphics.Bitmap
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
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
import java.io.File
import java.io.FileOutputStream

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
        assertTrue(
            composeRule.onAllNodesWithText(
                "Aún no hay un modelo local compatible seleccionado. Descárgalo y márcalo como preferido primero."
            ).fetchSemanticsNodes().isNotEmpty()
        )
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

    private fun capture(name: String) {
        composeRule.waitForIdle()
        val outputDir = File(app.filesDir, "hermes-ui-visuals").apply { mkdirs() }
        val bitmap = InstrumentationRegistry.getInstrumentation().uiAutomation.takeScreenshot()
        FileOutputStream(File(outputDir, "$name.png")).use { output ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
        }
    }
}
