package com.nousresearch.hermesagent.auth

import com.nousresearch.hermesagent.data.AuthCatalog
import com.nousresearch.hermesagent.data.AuthScope
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class Corr3xtAuthClientTest {
    @Test
    fun normalizeConfiguredBaseUrl_stripsQueryFragmentAndTrailingSlash() {
        assertEquals(
            "https://auth.corr3xt.com/base",
            Corr3xtAuthClient.normalizeConfiguredBaseUrl("https://auth.corr3xt.com/base/?foo=bar#frag"),
        )
    }

    @Test
    fun normalizeConfiguredBaseUrl_rejectsUnsupportedSchemes() {
        assertNull(Corr3xtAuthClient.normalizeConfiguredBaseUrl("javascript:alert(1)"))
    }

    @Test
    fun buildStartUri_includesCallbackContractAndRedirectUri() {
        val option = requireNotNull(AuthCatalog.find("chatgpt"))
        val uri = Corr3xtAuthClient.buildStartUri("https://auth.corr3xt.com/", option, "state-123")

        assertEquals("https", uri.scheme)
        assertEquals("auth.corr3xt.com", uri.host)
        assertEquals("/oauth/start", uri.path)
        assertEquals("v1", uri.getQueryParameter("callback_contract"))
        assertEquals("hermes-android", uri.getQueryParameter("client"))
        assertEquals("hermesagent://auth/callback", uri.getQueryParameter("redirect_uri"))
        assertEquals("state-123", uri.getQueryParameter("state"))
    }

    @Test
    fun authCatalog_includesOpenRouterForApiKeySetup() {
        val option = requireNotNull(AuthCatalog.find("openrouter"))

        assertEquals(AuthScope.RuntimeProvider, option.scope)
        assertEquals("openrouter", option.runtimeProvider)
        assertEquals("https://openrouter.ai/api/v1", option.defaultBaseUrl)
        assertTrue(option.defaultModel.isNotBlank())
    }

    @Test
    fun runtimeProviderAuthOptionsDeclareApiKeyFallbackTargets() {
        val runtimeOptions = AuthCatalog.options.filter { it.scope == AuthScope.RuntimeProvider }

        assertTrue(runtimeOptions.isNotEmpty())
        runtimeOptions.forEach { option ->
            assertTrue("${option.id} should declare runtimeProvider", option.runtimeProvider.isNotBlank())
            assertTrue("${option.id} should declare defaultBaseUrl", option.defaultBaseUrl.isNotBlank())
            assertTrue("${option.id} should declare defaultModel", option.defaultModel.isNotBlank())
            assertTrue("${option.id} should prefer local key setup over unavailable Corr3xt", !option.browserSignInSupported)
        }
    }
}
