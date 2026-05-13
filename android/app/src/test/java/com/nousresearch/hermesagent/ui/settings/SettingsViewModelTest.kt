package com.nousresearch.hermesagent.ui.settings

import com.nousresearch.hermesagent.device.HermesProviderSetupWebActivity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class SettingsViewModelTest {
    @Test
    fun openProviderKeyPageUsesHermesViewerForProviderSetupUrls() {
        val application = RuntimeEnvironment.getApplication()
        val viewModel = SettingsViewModel(application)

        viewModel.openProviderKeyPage("https://docs.qwencloud.com/api-reference/preparation/api-key")

        val started = Shadows.shadowOf(application).nextStartedActivity
        assertEquals(HermesProviderSetupWebActivity::class.java.name, started.component?.className)
        assertEquals(
            "https://docs.qwencloud.com/api-reference/preparation/api-key",
            started.getStringExtra("com.nousresearch.hermesagent.PROVIDER_SETUP_URL"),
        )
        assertEquals(
            "Open Qwen Cloud / DashScope API key setup page",
            started.getStringExtra("com.nousresearch.hermesagent.PROVIDER_SETUP_TITLE"),
        )
        assertTrue(viewModel.uiState.value.status.contains("in Hermes"))
    }
}
