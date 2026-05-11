package com.nousresearch.hermesagent.ui.settings

import com.nousresearch.hermesagent.data.ProviderPresets
import org.junit.Assert.assertEquals
import org.junit.Test

class ProviderPresetsTest {
    @Test
    fun zaiDefaultBaseUrlDoesNotOverrideCliEndpointDetection() {
        val preset = requireNotNull(ProviderPresets.find("zai"))

        val configBaseUrl = ProviderPresets.runtimeConfigBaseUrl("zai", preset.baseUrl)

        assertEquals("", configBaseUrl)
    }

    @Test
    fun customZaiBaseUrlIsPreserved() {
        val configBaseUrl = ProviderPresets.runtimeConfigBaseUrl(
            providerId = "zai",
            baseUrl = "https://open.bigmodel.cn/api/coding/paas/v4/",
        )

        assertEquals("https://open.bigmodel.cn/api/coding/paas/v4", configBaseUrl)
    }

    @Test
    fun nonZaiProviderDefaultBaseUrlIsPreserved() {
        val preset = requireNotNull(ProviderPresets.find("openrouter"))

        val configBaseUrl = ProviderPresets.runtimeConfigBaseUrl("openrouter", preset.baseUrl)

        assertEquals("https://openrouter.ai/api/v1", configBaseUrl)
    }

    @Test
    fun setupTargetsCycleThroughOfficialFallbacks() {
        val first = requireNotNull(ProviderPresets.setupTarget("qwen-oauth", 0))
        val second = requireNotNull(ProviderPresets.setupTarget("qwen-oauth", 1))
        val wrapped = requireNotNull(ProviderPresets.setupTarget("qwen-oauth", 6))

        assertEquals("https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/", first.url)
        assertEquals("https://home.qwencloud.com/api-keys", second.url)
        assertEquals(first.url, wrapped.url)
        assertEquals(6, first.total)
        assertEquals(1, first.nextIndex)
    }
}
