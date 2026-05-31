package com.mobilefork.hermesagent.settings

import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class AppSettingsStorePersistenceTest {
    @Test
    fun offlineAirplaneModeAndPortalEnabledPersist() {
        val store = AppSettingsStore(RuntimeEnvironment.getApplication())
        store.save(AppSettings())

        assertFalse(store.load().offlineAirplaneMode)
        assertTrue(store.load().portalEnabled)

        store.save(
            store.load().copy(
                offlineAirplaneMode = true,
                portalEnabled = false,
            )
        )

        val reloaded = store.load()
        assertTrue(reloaded.offlineAirplaneMode)
        assertFalse(reloaded.portalEnabled)
    }

    @Test
    fun appearanceSettingsPersist() {
        val store = AppSettingsStore(RuntimeEnvironment.getApplication())
        store.save(AppSettings())

        store.save(
            store.load().copy(
                chatDisplayMode = "expanded",
                keywordHighlightingEnabled = false,
                themePrimaryHex = "#D2B35E",
                themeSecondaryHex = "#72D6C9",
                themeBackgroundHex = "#000000",
                themeSurfaceHex = "#101014",
                themeSurfaceVariantHex = "#20242C",
                themeCardShape = "square",
            )
        )

        val reloaded = store.load()
        assertEquals("expanded", reloaded.chatDisplayMode)
        assertFalse(reloaded.keywordHighlightingEnabled)
        assertEquals("#D2B35E", reloaded.themePrimaryHex)
        assertEquals("#72D6C9", reloaded.themeSecondaryHex)
        assertEquals("#000000", reloaded.themeBackgroundHex)
        assertEquals("#101014", reloaded.themeSurfaceHex)
        assertEquals("#20242C", reloaded.themeSurfaceVariantHex)
        assertEquals("square", reloaded.themeCardShape)
    }

    @Test
    fun appSettingsExportImportRoundTripsWithoutSecrets() {
        val store = AppSettingsStore(RuntimeEnvironment.getApplication())
        store.save(
            AppSettings(
                provider = "gemini",
                baseUrl = "https://example.test/v1",
                model = "gemini-test",
                dataSaverMode = true,
                offlineAirplaneMode = true,
                portalEnabled = false,
                onDeviceBackend = "litert_lm",
                liteRtLmSpeculativeDecodingMode = "disabled",
                localModelMaxTokens = 2048,
                localModelTopK = 64,
                localModelTopP = 0.9f,
                localModelTemperature = 0.7f,
                localModelAccelerator = "gpu",
                apiGenerationKnobsEnabled = true,
                languageTag = "es",
                customSystemPrompt = "Stay concise and ask before external sends.",
                chatDisplayMode = "expanded",
                keywordHighlightingEnabled = false,
                themePrimaryHex = "#112233",
                themeSecondaryHex = "#445566",
                themeBackgroundHex = "#000000",
                themeSurfaceHex = "#101010",
                themeSurfaceVariantHex = "#202020",
                themeCardShape = "square",
            ),
        )

        val exported = store.exportBundleJson()
        assertEquals(AppSettings.EXPORT_KIND, exported.getString("kind"))
        assertFalse(exported.getBoolean("secrets_included"))
        assertTrue(exported.getJSONArray("redacted_secret_fields").toString().contains("api_key"))
        assertFalse(exported.toString().contains("sk-"))
        assertEquals("gemini", exported.getJSONObject("settings").getString("provider"))
        assertEquals(
            "Stay concise and ask before external sends.",
            exported.getJSONObject("settings").getString("custom_system_prompt"),
        )

        store.save(AppSettings())
        val imported = store.importBundleJson(exported)

        assertEquals("gemini", imported.provider)
        assertEquals("https://example.test/v1", imported.baseUrl)
        assertEquals("gemini-test", imported.model)
        assertTrue(imported.dataSaverMode)
        assertTrue(imported.offlineAirplaneMode)
        assertFalse(imported.portalEnabled)
        assertEquals("litert_lm", imported.onDeviceBackend)
        assertEquals("disabled", imported.liteRtLmSpeculativeDecodingMode)
        assertEquals(2048, imported.localModelMaxTokens)
        assertEquals(64, imported.localModelTopK)
        assertEquals(0.9f, imported.localModelTopP, 0.0001f)
        assertEquals(0.7f, imported.localModelTemperature, 0.0001f)
        assertEquals("gpu", imported.localModelAccelerator)
        assertTrue(imported.apiGenerationKnobsEnabled)
        assertEquals("es", imported.languageTag)
        assertEquals("Stay concise and ask before external sends.", imported.customSystemPrompt)
        assertEquals("expanded", imported.chatDisplayMode)
        assertFalse(imported.keywordHighlightingEnabled)
        assertEquals("#112233", store.load().themePrimaryHex)
        assertEquals("square", store.load().themeCardShape)
    }

    @Test
    fun customSystemPromptIsNormalizedAndBoundedForMobileContext() {
        val longPrompt = "x".repeat(AppSettings.MAX_CUSTOM_SYSTEM_PROMPT_CHARS + 50)
        val normalized = AppSettings.normalizeCustomSystemPrompt("\r\n$longPrompt\u0000")

        assertEquals(AppSettings.MAX_CUSTOM_SYSTEM_PROMPT_CHARS, normalized.length)
        assertFalse(normalized.contains("\u0000"))
    }

    @Test
    fun modelGenerationSettingsPersistWithBoundedDefaults() {
        val store = AppSettingsStore(RuntimeEnvironment.getApplication())
        store.save(
            AppSettings(
                localModelMaxTokens = 99_999,
                localModelTopK = 999,
                localModelTopP = 9.5f,
                localModelTemperature = -1.0f,
                localModelAccelerator = "tpu",
                apiGenerationKnobsEnabled = true,
            )
        )

        val reloaded = store.load()
        assertEquals(AppSettings.MAX_LOCAL_MODEL_MAX_TOKENS, reloaded.localModelMaxTokens)
        assertEquals(AppSettings.MAX_LOCAL_MODEL_TOP_K, reloaded.localModelTopK)
        assertEquals(AppSettings.MAX_LOCAL_MODEL_TOP_P, reloaded.localModelTopP, 0.0001f)
        assertEquals(AppSettings.MIN_LOCAL_MODEL_TEMPERATURE, reloaded.localModelTemperature, 0.0001f)
        assertEquals(AppSettings.DEFAULT_LOCAL_MODEL_ACCELERATOR, reloaded.localModelAccelerator)
        assertTrue(reloaded.apiGenerationKnobsEnabled)
    }
}
