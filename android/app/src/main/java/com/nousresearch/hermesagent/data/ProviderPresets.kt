package com.nousresearch.hermesagent.data

data class ProviderPreset(
    val id: String,
    val label: String,
    val baseUrl: String,
    val modelHint: String,
    val apiKeyUrl: String = "",
    val fallbackSetupUrls: List<String> = emptyList(),
)

data class ProviderSetupTarget(
    val providerId: String,
    val url: String,
    val index: Int,
    val total: Int,
) {
    val displayIndex: Int
        get() = index + 1

    val nextIndex: Int
        get() = if (total <= 0) 0 else (index + 1) % total
}

data class ModelSelectionPreset(
    val id: String,
    val label: String,
    val description: String,
)

object ProviderPresets {
    val firstClassLocalModels = listOf(
        ModelSelectionPreset(
            id = "gemma-4-E2B-it",
            label = "Gemma 4 E2B (LiteRT-LM)",
            description = "Fast Gemma 4 local chat and Android tool-calling model.",
        ),
        ModelSelectionPreset(
            id = "gemma-4-E4B-it",
            label = "Gemma 4 E4B (LiteRT-LM)",
            description = "Larger Gemma 4 local model under the 5 GB mobile test ceiling.",
        ),
        ModelSelectionPreset(
            id = "gemma3-1b-it-int4",
            label = "Gemma 3 1B IT INT4 (LiteRT-LM)",
            description = "Small Gemma 3 text model for low-memory local checks.",
        ),
        ModelSelectionPreset(
            id = "gemma3-4b-it-int4-web",
            label = "Gemma 3 4B IT Vision (.task)",
            description = "Gemma 3 image-text model for LiteRT-LM vision requests.",
        ),
        ModelSelectionPreset(
            id = "gemma-3n-E2B-it-int4",
            label = "Gemma 3n E2B IT Vision (LiteRT-LM)",
            description = "Gemma 3n multimodal model with image input support.",
        ),
        ModelSelectionPreset(
            id = "gemma-3n-E4B-it-int4",
            label = "Gemma 3n E4B IT Vision (LiteRT-LM)",
            description = "Larger Gemma 3n multimodal model with image input support.",
        ),
    )

    val defaults = listOf(
        ProviderPreset(
            id = "openrouter",
            label = "OpenRouter",
            baseUrl = "https://openrouter.ai/api/v1",
            modelHint = "anthropic/claude-sonnet-4",
            apiKeyUrl = "https://openrouter.ai/keys",
            fallbackSetupUrls = listOf("https://openrouter.ai/docs/quickstart"),
        ),
        ProviderPreset(
            id = "openai",
            label = "OpenAI",
            baseUrl = "https://api.openai.com/v1",
            modelHint = "gpt-4.1",
            apiKeyUrl = "https://platform.openai.com/settings/organization/api-keys",
            fallbackSetupUrls = listOf("https://platform.openai.com/docs/quickstart"),
        ),
        ProviderPreset(
            id = "chatgpt-web",
            label = "ChatGPT Web",
            baseUrl = "https://chatgpt.com/backend-api/f",
            modelHint = "gpt-5-thinking",
            apiKeyUrl = "https://chatgpt.com/",
            fallbackSetupUrls = listOf("https://chatgpt.com/#settings"),
        ),
        ProviderPreset(
            id = "anthropic",
            label = "Claude / Anthropic",
            baseUrl = "https://api.anthropic.com",
            modelHint = "claude-sonnet-4",
            apiKeyUrl = "https://console.anthropic.com/settings/keys",
            fallbackSetupUrls = listOf("https://docs.anthropic.com/claude/docs/quickstart-guide"),
        ),
        ProviderPreset(
            id = "gemini",
            label = "Gemini / Google AI Studio",
            baseUrl = "https://generativelanguage.googleapis.com/v1beta/openai",
            modelHint = "gemini-2.5-pro",
            apiKeyUrl = "https://aistudio.google.com/apikey",
            fallbackSetupUrls = listOf("https://ai.google.dev/gemini-api/docs/api-key"),
        ),
        ProviderPreset(
            id = "alibaba",
            label = "Qwen Cloud / DashScope API key",
            baseUrl = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            modelHint = "qwen3.6-plus",
            apiKeyUrl = "https://home.qwencloud.com/api-keys",
            fallbackSetupUrls = listOf(
                "https://docs.qwencloud.com/api-reference/preparation/api-key",
                "https://docs.qwencloud.com/developer-guides/administration/api-keys",
            ),
        ),
        ProviderPreset(
            id = "qwen-oauth",
            label = "Qwen OAuth / Qwen Chat token (legacy)",
            baseUrl = "https://portal.qwen.ai/v1",
            modelHint = "qwen3-coder-plus",
            apiKeyUrl = "https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/",
            fallbackSetupUrls = listOf(
                "https://home.qwencloud.com/api-keys",
                "https://docs.qwencloud.com/api-reference/preparation/api-key",
                "https://docs.qwencloud.com/developer-guides/getting-started/first-api-call",
                "https://qwen.ai/apiplatform",
                "https://chat.qwen.ai/",
            ),
        ),
        ProviderPreset(
            id = "zai",
            label = "Z.AI / GLM",
            baseUrl = "https://api.z.ai/api/paas/v4",
            modelHint = "glm-5",
            apiKeyUrl = "https://z.ai/manage-apikey/apikey-list",
            fallbackSetupUrls = listOf("https://docs.z.ai/guides/"),
        ),
        ProviderPreset(
            id = "nous",
            label = "Nous",
            baseUrl = "",
            modelHint = "",
        ),
        ProviderPreset(
            id = "custom",
            label = "Custom OpenAI-compatible",
            baseUrl = "",
            modelHint = "",
        ),
    )

    val androidSettingsDefaults = defaults

    fun find(id: String): ProviderPreset? = defaults.firstOrNull { it.id == id }

    fun setupUrls(providerId: String): List<String> {
        val preset = find(providerId) ?: return emptyList()
        return (listOf(preset.apiKeyUrl) + preset.fallbackSetupUrls)
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .distinct()
    }

    fun setupTarget(providerId: String, requestedIndex: Int): ProviderSetupTarget? {
        val urls = setupUrls(providerId)
        if (urls.isEmpty()) {
            return null
        }
        val index = requestedIndex.floorMod(urls.size)
        return ProviderSetupTarget(
            providerId = providerId,
            url = urls[index],
            index = index,
            total = urls.size,
        )
    }

    fun setupClipboardText(providerId: String): String {
        return setupUrls(providerId).joinToString(separator = "\n")
    }

    fun providerIdForSetupUrl(url: String): String? {
        val normalized = url.trim()
        return defaults.firstOrNull { preset ->
            setupUrls(preset.id).any { it == normalized }
        }?.id
    }

    fun runtimeConfigBaseUrl(providerId: String, baseUrl: String): String {
        val normalized = baseUrl.trim().trimEnd('/')
        val presetDefault = find(providerId)?.baseUrl.orEmpty().trim().trimEnd('/')
        return when {
            providerId == "zai" && normalized == presetDefault -> ""
            else -> normalized
        }
    }

    fun modelSelections(providerId: String): List<ModelSelectionPreset> {
        val providerHint = find(providerId)?.modelHint.orEmpty().takeIf { it.isNotBlank() }?.let {
            ModelSelectionPreset(
                id = it,
                label = it,
                description = "Provider suggested model",
            )
        }
        return listOfNotNull(providerHint) + firstClassLocalModels
    }

    private fun Int.floorMod(divisor: Int): Int {
        return ((this % divisor) + divisor) % divisor
    }
}
