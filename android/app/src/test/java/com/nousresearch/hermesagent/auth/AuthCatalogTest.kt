package com.nousresearch.hermesagent.auth

import com.nousresearch.hermesagent.data.AuthCatalog
import com.nousresearch.hermesagent.data.AuthScope
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AuthCatalogTest {
    @Test
    fun qwenZaiAndOtherApiKeyProvidersDoNotExposeBrowserOAuthSignIn() {
        listOf(
            "openai",
            "chatgpt",
            "claude",
            "gemini",
            "qwen",
            "qwen-coding-plan",
            "qwen-oauth",
            "zai",
            "zai-coding-plan",
        ).forEach { optionId ->
            val option = requireNotNull(AuthCatalog.find(optionId)) {
                "Missing auth option $optionId"
            }

            assertEquals("Expected $optionId to configure a runtime provider", AuthScope.RuntimeProvider, option.scope)
            assertFalse("$optionId must use secure API-key/token setup, not browser OAuth", option.browserSignInSupported)
            assertTrue("$optionId must map to a provider preset", option.runtimeProvider.isNotBlank())
        }
    }

    @Test
    fun openRouterRemainsTheBrowserOAuthRuntimeProvider() {
        val option = requireNotNull(AuthCatalog.find("openrouter"))

        assertEquals(AuthScope.RuntimeProvider, option.scope)
        assertEquals("openrouter", option.runtimeProvider)
        assertTrue(option.browserSignInSupported)
    }
}
