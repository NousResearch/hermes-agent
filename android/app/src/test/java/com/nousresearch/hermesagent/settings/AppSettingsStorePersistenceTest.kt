package com.nousresearch.hermesagent.settings

import com.nousresearch.hermesagent.data.AppSettingsStore
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
}
