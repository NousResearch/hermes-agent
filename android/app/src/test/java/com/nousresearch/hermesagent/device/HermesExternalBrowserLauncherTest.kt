package com.nousresearch.hermesagent.device

import android.content.Intent
import android.net.Uri
import android.provider.Browser
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
class HermesExternalBrowserLauncherTest {
    @Test
    fun createBrowserIntentBuildsDirectBrowsableActionViewIntent() {
        val context = RuntimeEnvironment.getApplication()
        val uri = Uri.parse("https://openrouter.ai/settings/keys")

        val intent = HermesExternalBrowserLauncher.createBrowserIntent(context, uri)

        assertEquals(Intent.ACTION_VIEW, intent.action)
        assertEquals(uri, intent.data)
        assertTrue(intent.categories.orEmpty().contains(Intent.CATEGORY_BROWSABLE))
        assertTrue(intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)
        assertEquals(context.packageName, intent.getStringExtra(Browser.EXTRA_APPLICATION_ID))
    }

    @Test
    @Suppress("DEPRECATION")
    fun createChooserIntentRemainsAvailableAsFallback() {
        val context = RuntimeEnvironment.getApplication()
        val uri = Uri.parse("https://docs.qwencloud.com/api-reference/preparation/api-key")

        val intent = HermesExternalBrowserLauncher.createChooserIntent(context, uri, "Open Qwen setup")
        val wrapped = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)

        assertEquals(Intent.ACTION_CHOOSER, intent.action)
        assertTrue(intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)
        assertEquals(Intent.ACTION_VIEW, wrapped?.action)
        assertEquals(uri, wrapped?.data)
    }

    @Test
    fun openStartsDirectBrowserIntentWithoutForcedChooser() {
        val context = RuntimeEnvironment.getApplication()
        val uri = Uri.parse("https://z.ai/manage-apikey/apikey-list")

        val result = HermesExternalBrowserLauncher.open(context, uri, "Open Z.AI setup")
        val started = Shadows.shadowOf(context).nextStartedActivity

        assertTrue(result.success)
        assertEquals(Intent.ACTION_VIEW, started.action)
        assertFalse(started.action == Intent.ACTION_CHOOSER)
        assertEquals(uri, started.data)
    }

    @Test
    fun openRejectsUnsupportedSchemes() {
        val context = RuntimeEnvironment.getApplication()

        val result = HermesExternalBrowserLauncher.open(
            context = context,
            uri = Uri.parse("file:///sdcard/Download/token.txt"),
            title = "Open file",
        )

        assertFalse(result.success)
        assertEquals("UnsupportedScheme", result.errorName)
    }
}
