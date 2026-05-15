package com.nousresearch.hermesagent.settings

import com.nousresearch.hermesagent.data.AppSettingsStore
import org.junit.Assert.assertFalse
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
}
