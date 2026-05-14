package com.nousresearch.hermesagent

import android.app.Activity
import android.app.Instrumentation
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.widget.Button
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.Intents.intending
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.device.HermesExternalBrowserLauncher
import com.nousresearch.hermesagent.device.HermesProviderSetupWebActivity
import org.hamcrest.Description
import org.hamcrest.TypeSafeMatcher
import org.junit.After
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.util.concurrent.atomic.AtomicBoolean

@RunWith(AndroidJUnit4::class)
class ProviderSetupWebActivityInstrumentedTest {
    @After
    fun tearDown() {
        shellOutput("input keyevent KEYCODE_HOME")
    }

    @Test
    fun providerSetupOpenUsesExternalBrowserForQwenCloudWhenAvailable() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val target = requireNotNull(ProviderPresets.setupTarget("alibaba", 0))
        val uri = Uri.parse(target.url)
        val browserIntent = HermesExternalBrowserLauncher.createBrowserIntent(context, uri)
        val resolved = browserIntent.resolveActivity(context.packageManager)
        assumeTrue("No browser is installed on this test device", resolved != null)
        assumeTrue(
            "Provider setup should not resolve back to Hermes",
            resolved?.packageName != context.packageName,
        )

        val qwenDocsOpened = AtomicBoolean(false)
        val qwenDocsIntent = object : TypeSafeMatcher<Intent>() {
            override fun describeTo(description: Description) {
                description.appendText("Qwen Cloud setup browser intent")
            }

            override fun matchesSafely(intent: Intent): Boolean {
                val targetIntent = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
                val targetUri = intent.data ?: targetIntent?.data ?: return false
                val matches = intent.action in setOf(Intent.ACTION_VIEW, Intent.ACTION_CHOOSER) &&
                    targetUri == uri
                if (matches) {
                    qwenDocsOpened.set(true)
                }
                return matches
            }
        }

        Intents.init()
        try {
            intending(qwenDocsIntent).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, null))

            val result = HermesProviderSetupWebActivity.open(context, uri, "Open Qwen setup")

            assertTrue(result.toString(), result.success)
            assertTrue("Expected provider setup to launch the Qwen docs browser intent", qwenDocsOpened.get())
        } finally {
            Intents.release()
        }
    }

    @Test
    fun providerSetupOpenHandsOffCurrentQwenSetupTargetToRealBrowser() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val target = requireNotNull(ProviderPresets.setupTarget("alibaba", 0))
        val uri = Uri.parse(target.url)
        val browserIntent = HermesExternalBrowserLauncher.createBrowserIntent(context, uri)
        val resolved = browserIntent.resolveActivity(context.packageManager)
        assumeTrue("No browser is installed on this test device", resolved != null)
        assumeTrue(
            "Provider setup should not resolve back to Hermes",
            resolved?.packageName != context.packageName,
        )

        val result = HermesProviderSetupWebActivity.open(context, uri, "Open Qwen setup")

        assertTrue(result.toString(), result.success)
        val foregroundPackage = waitForForegroundPackage(expectedPackage = resolved!!.packageName)
        assertTrue(
            "Expected provider setup to focus the external browser ${resolved.packageName}, got '$foregroundPackage'",
            foregroundPackage == resolved.packageName,
        )
    }

    @Test
    fun providerSetupViewerStartsForApiKeyAndTokenProviders() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        listOf("openrouter", "alibaba", "alibaba-coding-plan", "qwen-oauth", "zai", "zai-coding-plan").forEach { providerId ->
            val target = requireNotNull(ProviderPresets.setupTarget(providerId, 0)) {
                "Expected setup target for $providerId"
            }
            val intent = HermesProviderSetupWebActivity.createIntent(
                context = context,
                uri = Uri.parse(target.url),
                title = "Open $providerId setup",
            )

            ActivityScenario.launch<HermesProviderSetupWebActivity>(intent).use { scenario ->
                scenario.onActivity { activity ->
                    val root = activity.window.decorView
                    val webView = root.findFirstWebView()
                    val toolbarLabels = root.findButtons().map { it.text.toString() }.toSet()
                    if (webView != null) {
                        val currentUrl = webView.url.orEmpty().ifBlank { webView.originalUrl.orEmpty() }
                        assertTrue(
                            "Expected $providerId setup WebView to start loading ${target.url}, got '$currentUrl'",
                            currentUrl.startsWith("http://") || currentUrl.startsWith("https://"),
                        )

                        assertTrue("Missing Back button for $providerId: $toolbarLabels", "Back" in toolbarLabels)
                        assertTrue("Missing Browser button for $providerId: $toolbarLabels", "Browser" in toolbarLabels)
                        assertTrue("Missing Copy button for $providerId: $toolbarLabels", "Copy" in toolbarLabels)
                        assertTrue("Missing Close button for $providerId: $toolbarLabels", "Close" in toolbarLabels)
                    } else {
                        assertTrue("Missing browser fallback button for $providerId: $toolbarLabels", "Open in browser" in toolbarLabels)
                        assertTrue("Missing copy fallback button for $providerId: $toolbarLabels", "Copy URL" in toolbarLabels)
                        assertTrue("Missing close fallback button for $providerId: $toolbarLabels", "Close" in toolbarLabels)
                    }

                    webView?.stopLoading()
                    activity.finish()
                }
            }
        }
    }

    @Test
    fun providerSetupViewerShowsCopyableFallbackForInvalidSetupUrl() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val intent = HermesProviderSetupWebActivity.createIntent(
            context = context,
            uri = Uri.parse("https:///missing-host"),
            title = "Open broken provider setup",
        )

        ActivityScenario.launch<HermesProviderSetupWebActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val root = activity.window.decorView
                assertNull(root.findFirstWebView())
                val toolbarLabels = root.findButtons().map { it.text.toString() }.toSet()
                assertTrue("Missing browser fallback button: $toolbarLabels", "Open in browser" in toolbarLabels)
                assertTrue("Missing copy fallback button: $toolbarLabels", "Copy URL" in toolbarLabels)
                assertTrue("Missing close fallback button: $toolbarLabels", "Close" in toolbarLabels)
            }
        }
    }

    private fun View.findFirstWebView(): WebView? {
        if (this is WebView) {
            return this
        }
        if (this is ViewGroup) {
            for (index in 0 until childCount) {
                val match = getChildAt(index).findFirstWebView()
                if (match != null) {
                    return match
                }
            }
        }
        return null
    }

    private fun View.findButtons(): List<Button> {
        val matches = mutableListOf<Button>()
        collectButtons(matches)
        return matches
    }

    private fun View.collectButtons(matches: MutableList<Button>) {
        if (this is Button) {
            matches.add(this)
        }
        if (this is ViewGroup) {
            for (index in 0 until childCount) {
                getChildAt(index).collectButtons(matches)
            }
        }
    }

    private fun shellOutput(command: String): String {
        val descriptor = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand(command)
        return descriptor.use { fd ->
            FileInputStream(fd.fileDescriptor).bufferedReader().use { it.readText() }
        }
    }

    private fun waitForForegroundPackage(expectedPackage: String): String {
        val deadline = SystemClock.elapsedRealtime() + 15_000L
        var lastPackage = currentForegroundPackage()
        while (SystemClock.elapsedRealtime() < deadline) {
            if (lastPackage == expectedPackage) {
                return lastPackage
            }
            SystemClock.sleep(250L)
            lastPackage = currentForegroundPackage()
        }
        return lastPackage
    }

    private fun currentForegroundPackage(): String {
        val output = shellOutput("dumpsys window")
        return FOCUS_PACKAGE_REGEX.find(output)?.groupValues?.getOrNull(1).orEmpty()
            .ifBlank {
                FOCUSED_APP_PACKAGE_REGEX.find(output)?.groupValues?.getOrNull(1).orEmpty()
            }
    }

    companion object {
        private val FOCUS_PACKAGE_REGEX = Regex("""mCurrentFocus=Window\{[^ ]+ u\d+ ([^/\s]+)/""")
        private val FOCUSED_APP_PACKAGE_REGEX = Regex("""mFocusedApp=ActivityRecord\{[^ ]+ u\d+ ([^/\s]+)/""")
    }

}
