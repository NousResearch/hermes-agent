package com.nousresearch.hermesagent.device

import android.content.Intent
import android.net.Uri
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class HermesProviderSetupWebActivityTest {
    @Test
    fun createIntentTargetsInternalProviderSetupViewer() {
        val context = RuntimeEnvironment.getApplication()
        val uri = Uri.parse("https://openrouter.ai/settings/keys")

        val intent = HermesProviderSetupWebActivity.createIntent(context, uri, "Open OpenRouter setup")

        assertEquals(HermesProviderSetupWebActivity::class.java.name, intent.component?.className)
        assertTrue(intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)
        assertEquals(uri.toString(), intent.getStringExtra(HermesProviderSetupWebActivity.EXTRA_URL))
        assertEquals(
            "Open OpenRouter setup",
            intent.getStringExtra(HermesProviderSetupWebActivity.EXTRA_TITLE),
        )
    }

    @Test
    fun openStartsExternalBrowserForHttpProviderSetupUrl() {
        val context = RuntimeEnvironment.getApplication()
        val uri = Uri.parse("https://docs.qwencloud.com/api-reference/preparation/api-key")

        val result = HermesProviderSetupWebActivity.open(context, uri, "Open Qwen setup")
        val started = Shadows.shadowOf(context).nextStartedActivity

        assertTrue(result.success)
        assertEquals(Intent.ACTION_VIEW, started.action)
        assertEquals(uri, started.data)
        assertTrue(started.categories.orEmpty().contains(Intent.CATEGORY_BROWSABLE))
    }

    @Test
    fun openInAppStartsInternalProviderSetupViewerOnlyWhenExplicitlyRequested() {
        val context = RuntimeEnvironment.getApplication()
        val uri = Uri.parse("https://openrouter.ai/auth")

        val result = HermesProviderSetupWebActivity.openInApp(context, uri, "Open OpenRouter sign-in")
        val started = Shadows.shadowOf(context).nextStartedActivity

        assertTrue(result.success)
        assertEquals(HermesProviderSetupWebActivity::class.java.name, started.component?.className)
        assertEquals(uri.toString(), started.getStringExtra(HermesProviderSetupWebActivity.EXTRA_URL))
    }

    @Test
    fun openRejectsUnsupportedOrHostlessUris() {
        val context = RuntimeEnvironment.getApplication()

        val fileResult = HermesProviderSetupWebActivity.open(
            context = context,
            uri = Uri.parse("file:///sdcard/token.txt"),
            title = "Open file",
        )
        val hostlessResult = HermesProviderSetupWebActivity.open(
            context = context,
            uri = Uri.parse("https:///missing-host"),
            title = "Open broken URL",
        )

        assertFalse(fileResult.success)
        assertFalse(hostlessResult.success)
        assertEquals("UnsupportedScheme", fileResult.errorName)
        assertEquals("UnsupportedScheme", hostlessResult.errorName)
    }
}
