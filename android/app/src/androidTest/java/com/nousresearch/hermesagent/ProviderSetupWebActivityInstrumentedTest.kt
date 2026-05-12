package com.nousresearch.hermesagent

import android.content.Context
import android.net.Uri
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.widget.Button
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.device.HermesProviderSetupWebActivity
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ProviderSetupWebActivityInstrumentedTest {
    @Test
    fun providerSetupViewerStartsForApiKeyAndTokenProviders() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        listOf("openrouter", "alibaba", "qwen-oauth", "zai").forEach { providerId ->
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
}
