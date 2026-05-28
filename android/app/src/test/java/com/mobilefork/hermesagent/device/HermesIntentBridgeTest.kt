package com.mobilefork.hermesagent.device

import org.junit.Assert.assertEquals
import org.junit.Test

class HermesIntentBridgeTest {
    @Test
    fun selectPreferredBrowserPackageUsesResolvedBrowserWhenItIsSpecific() {
        assertEquals(
            "org.mozilla.firefox",
            HermesIntentBridge.selectPreferredBrowserPackage(
                resolvedPackage = "org.mozilla.firefox",
                candidatePackages = listOf("com.android.chrome"),
            ),
        )
    }

    @Test
    fun selectPreferredBrowserPackageAvoidsResolverAndPrefersChrome() {
        assertEquals(
            "com.android.chrome",
            HermesIntentBridge.selectPreferredBrowserPackage(
                resolvedPackage = "android",
                candidatePackages = listOf("com.example.viewer", "com.android.chrome"),
            ),
        )
    }

    @Test
    fun selectPreferredBrowserPackageRecognizesBraveNightly() {
        assertEquals(
            "com.brave.browser_nightly",
            HermesIntentBridge.selectPreferredBrowserPackage(
                resolvedPackage = "android",
                candidatePackages = listOf("com.example.viewer", "com.brave.browser_nightly"),
            ),
        )
    }
}
