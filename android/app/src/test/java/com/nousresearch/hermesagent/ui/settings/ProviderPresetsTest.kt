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
    fun zaiCodingPlanDefaultBaseUrlDoesNotOverrideCliEndpointDetection() {
        val preset = requireNotNull(ProviderPresets.find("zai-coding-plan"))

        val configBaseUrl = ProviderPresets.runtimeConfigBaseUrl("zai-coding-plan", preset.baseUrl)

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
        assertEquals("https://docs.qwencloud.com/api-reference/preparation/api-key", second.url)
        assertEquals(first.url, wrapped.url)
        assertEquals(6, first.total)
        assertEquals(1, first.nextIndex)
    }

    @Test
    fun qwenCloudOpensMobileSafeDocsBeforeConsoleFallbacks() {
        val first = requireNotNull(ProviderPresets.setupTarget("alibaba", 0))
        val second = requireNotNull(ProviderPresets.setupTarget("alibaba", 1))
        val third = requireNotNull(ProviderPresets.setupTarget("alibaba", 2))

        assertEquals("https://docs.qwencloud.com/developer-guides/administration/api-keys", first.url)
        assertEquals("https://modelstudio.console.alibabacloud.com/?tab=playground", second.url)
        assertEquals("https://www.alibabacloud.com/help/en/model-studio/get-api-key", third.url)
        assertEquals(6, first.total)
    }

    @Test
    fun qwenCodingPlanUsesDedicatedEndpointAndCliEnvAliases() {
        val preset = requireNotNull(ProviderPresets.find("alibaba-coding-plan"))
        val firstTarget = requireNotNull(ProviderPresets.setupTarget("alibaba-coding-plan", 0))
        val secondTarget = requireNotNull(ProviderPresets.setupTarget("alibaba-coding-plan", 1))
        val envHelp = ProviderPresets.credentialInputHelp("alibaba-coding-plan")

        assertEquals("https://coding-intl.dashscope.aliyuncs.com/v1", preset.baseUrl)
        assertEquals("qwen3.6-plus", preset.modelHint)
        assertEquals("https://docs.qwencloud.com/coding-plan/overview", firstTarget.url)
        assertEquals(
            "https://modelstudio.console.alibabacloud.com/?tab=playground",
            secondTarget.url,
        )
        assertEquals(
            "sk-bailian-test",
            ProviderPresets.parseCredentialInput(
                "alibaba-coding-plan",
                "BAILIAN_CODING_PLAN_API_KEY=sk-bailian-test",
            ).apiKey,
        )
        assertEquals(
            "sk-alibaba-plan-test",
            ProviderPresets.parseCredentialInput(
                "alibaba-coding-plan",
                "export ALIBABA_CODING_PLAN_API_KEY='sk-alibaba-plan-test'",
            ).apiKey,
        )
        assertEquals(true, envHelp.contains("BAILIAN_CODING_PLAN_API_KEY"))
        assertEquals(true, envHelp.contains("ALIBABA_CODING_PLAN_API_KEY"))
    }

    @Test
    fun providerIdForSetupUrlHonorsPreferredProviderForSharedSetupPages() {
        val qwenAccountUrl = "https://modelstudio.console.alibabacloud.com/?tab=playground"
        val zaiAccountUrl = "https://z.ai/manage-apikey/apikey-list"

        assertEquals("alibaba", ProviderPresets.providerIdForSetupUrl(qwenAccountUrl))
        assertEquals(
            "alibaba-coding-plan",
            ProviderPresets.providerIdForSetupUrl(qwenAccountUrl, "alibaba-coding-plan"),
        )
        assertEquals("zai", ProviderPresets.providerIdForSetupUrl(zaiAccountUrl))
        assertEquals(
            "zai-coding-plan",
            ProviderPresets.providerIdForSetupUrl(zaiAccountUrl, "zai-coding-plan"),
        )
    }

    @Test
    fun parsesProviderEnvStyleCredentialInput() {
        assertEquals(
            "sk-or-v1-test",
            ProviderPresets.parseCredentialInput("openrouter", "OPENROUTER_API_KEY=sk-or-v1-test").apiKey,
        )
        assertEquals(
            "sk-or-v1-setx",
            ProviderPresets.parseCredentialInput("openrouter", "setx OPENROUTER_API_KEY sk-or-v1-setx").apiKey,
        )
        assertEquals(
            "sk-qwen-test",
            ProviderPresets.parseCredentialInput("alibaba", "export DASHSCOPE_API_KEY='sk-qwen-test'").apiKey,
        )
        assertEquals(
            "sk-qwen-yaml",
            ProviderPresets.parseCredentialInput("alibaba", "DASHSCOPE_API_KEY: sk-qwen-yaml").apiKey,
        )
        assertEquals(
            "glm-test",
            ProviderPresets.parseCredentialInput("zai", "\$env:ZAI_API_KEY=\"glm-test\"").apiKey,
        )
        assertEquals(
            "glm-plan-test",
            ProviderPresets.parseCredentialInput("zai-coding-plan", "ZAI_CODING_PLAN_API_KEY=glm-plan-test").apiKey,
        )
        assertEquals(
            "google-test",
            ProviderPresets.parseCredentialInput("gemini", "{\"GOOGLE_API_KEY\":\"google-test\"}").apiKey,
        )
    }

    @Test
    fun parsesGenericCliCredentialBlocks() {
        val parsedJson = ProviderPresets.parseCredentialInput("openrouter", """{"api_key":"sk-generic-json"}""")
        assertEquals("sk-generic-json", parsedJson.apiKey)
        assertEquals("credential block", parsedJson.sourceLabel)
        assertEquals(
            "qwen-generic-token",
            ProviderPresets.parseCredentialInput("qwen-oauth", "access_token: qwen-generic-token").apiKey,
        )
    }

    @Test
    fun parsesBearerTokenCredentialInput() {
        assertEquals(
            "qwen-oauth-token",
            ProviderPresets.parseCredentialInput("qwen-oauth", "Bearer qwen-oauth-token").apiKey,
        )
        assertEquals(
            "zai-oauth-token",
            ProviderPresets.parseCredentialInput("zai", "Authorization: Bearer zai-oauth-token").apiKey,
        )
        assertEquals(
            "openrouter-token",
            ProviderPresets.parseCredentialInput("openrouter", """{"Authorization":"Bearer openrouter-token"}""").apiKey,
        )
    }

    @Test
    fun rawProviderCredentialInputIsPreserved() {
        assertEquals(
            "sk-raw-test",
            ProviderPresets.parseCredentialInput("openrouter", "sk-raw-test").apiKey,
        )
    }
}
