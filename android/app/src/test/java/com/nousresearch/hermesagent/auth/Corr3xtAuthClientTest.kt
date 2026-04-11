package com.nousresearch.hermesagent.auth

import com.nousresearch.hermesagent.data.AuthCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

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
}
