package com.nousresearch.hermesagent.settings

import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.HermesNetworkPolicy
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class HermesNetworkPolicyTest {
    @Test
    fun offlineAirplaneModeBlocksExternalUrlsButAllowsLocalhost() {
        val app = RuntimeEnvironment.getApplication()
        val store = AppSettingsStore(app)
        store.save(store.load().copy(offlineAirplaneMode = true))

        assertTrue(HermesNetworkPolicy.isExternalNetworkBlocked(app, "https://portal.nousresearch.com"))
        assertFalse(HermesNetworkPolicy.isExternalNetworkBlocked(app, "http://127.0.0.1:15436/v1/models"))
        assertFalse(HermesNetworkPolicy.isExternalNetworkBlocked(app, "http://localhost:15435/v1/chat/completions"))
    }

    @Test
    fun turningOfflineAirplaneModeOffRestoresExternalNetworkAccess() {
        val app = RuntimeEnvironment.getApplication()
        val store = AppSettingsStore(app)
        store.save(store.load().copy(offlineAirplaneMode = true))
        assertTrue(HermesNetworkPolicy.isExternalNetworkBlocked(app, "https://portal.nousresearch.com"))

        store.save(store.load().copy(offlineAirplaneMode = false))

        assertFalse(HermesNetworkPolicy.isOfflineAirplaneModeEnabled(app))
        assertFalse(HermesNetworkPolicy.isExternalNetworkBlocked(app, "https://portal.nousresearch.com"))
    }
}
