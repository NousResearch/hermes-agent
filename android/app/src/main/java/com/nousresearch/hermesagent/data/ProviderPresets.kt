package com.nousresearch.hermesagent.data

data class ProviderPreset(
    val id: String,
    val label: String,
    val baseUrl: String,
    val modelHint: String,
    val apiKeyUrl: String = "",
    val fallbackSetupUrls: List<String> = emptyList(),
)

data class ParsedProviderCredential(
    val apiKey: String,
    val sourceLabel: String = "",
) {
    val importedFromEnvLine: Boolean
        get() = sourceLabel.isNotBlank()
}

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
            apiKeyUrl = "https://openrouter.ai/settings/keys",
            fallbackSetupUrls = listOf(
                "https://openrouter.ai/keys",
                "https://openrouter.ai/docs/api-keys",
                "https://openrouter.ai/docs/quickstart",
            ),
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
            id = "alibaba-coding-plan",
            label = "Qwen Coding Plan",
            baseUrl = "https://coding-intl.dashscope.aliyuncs.com/v1",
            modelHint = "qwen3-coder-plus",
            apiKeyUrl = "https://docs.qwencloud.com/coding-plan/tools/cline",
            fallbackSetupUrls = listOf(
                "https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/",
                "https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/",
                "https://home.qwencloud.com/api-keys",
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

    fun apiKeyEnvVars(providerId: String): List<String> {
        return when (providerId.trim().lowercase()) {
            "openrouter" -> listOf("OPENROUTER_API_KEY")
            "openai", "custom" -> listOf("OPENAI_API_KEY")
            "anthropic" -> listOf("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN")
            "gemini" -> listOf("GOOGLE_API_KEY", "GEMINI_API_KEY")
            "chatgpt-web" -> listOf("CHATGPT_WEB_ACCESS_TOKEN")
            "alibaba", "dashscope" -> listOf("DASHSCOPE_API_KEY", "QWEN_API_KEY")
            "alibaba-coding-plan" -> listOf(
                "BAILIAN_CODING_PLAN_API_KEY",
                "ALIBABA_CODING_PLAN_API_KEY",
                "DASHSCOPE_API_KEY",
            )
            "qwen-oauth" -> listOf("QWEN_ACCESS_TOKEN", "QWEN_API_KEY", "DASHSCOPE_API_KEY")
            "zai" -> listOf("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")
            "nous" -> listOf("NOUS_API_KEY")
            else -> listOf(providerId.trim().uppercase().replace('-', '_') + "_API_KEY")
        }.distinct()
    }

    fun credentialInputHelp(providerId: String): String {
        val envVars = apiKeyEnvVars(providerId)
        val primary = envVars.firstOrNull().orEmpty()
        val aliases = envVars.drop(1).joinToString(separator = ", ")
        return if (aliases.isBlank()) {
            "Paste a raw key or a CLI env line such as $primary=..."
        } else {
            "Paste a raw key or a CLI env line such as $primary=...; also accepts $aliases."
        }
    }

    fun parseCredentialInput(providerId: String, input: String): ParsedProviderCredential {
        val trimmed = input.trim()
        if (trimmed.isBlank()) {
            return ParsedProviderCredential("")
        }
        val envVars = apiKeyEnvVars(providerId)
        envVars.forEach { envVar ->
            extractEnvValue(trimmed, envVar)?.let { value ->
                return ParsedProviderCredential(value, envVar)
            }
        }
        extractBearerCredential(trimmed)?.let { value ->
            return ParsedProviderCredential(value, "Bearer")
        }
        extractGenericCredential(trimmed)?.let { value ->
            return ParsedProviderCredential(value, "credential block")
        }
        extractAnyLikelyCredential(trimmed)?.let { value ->
            return ParsedProviderCredential(value, "env")
        }
        return ParsedProviderCredential(unquote(trimmed))
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

    private fun extractEnvValue(input: String, envVar: String): String? {
        val escapedEnvVar = Regex.escape(envVar)
        val patterns = listOf(
            Regex("""(?im)^\s*(?:export\s+|set\s+|setx\s+)?$escapedEnvVar\s*=\s*(.+?)\s*$"""),
            Regex("""(?im)^\s*setx\s+$escapedEnvVar\s+(.+?)\s*$"""),
            Regex("""(?im)^\s*$escapedEnvVar\s*:\s*(.+?)\s*$"""),
            Regex("""(?im)^\s*\${'$'}env:$escapedEnvVar\s*=\s*(.+?)\s*$"""),
            Regex("""(?im)["']$escapedEnvVar["']\s*:\s*["']([^"']+)["']"""),
        )
        return patterns.firstNotNullOfOrNull { pattern ->
            pattern.find(input)?.groupValues?.getOrNull(1)?.let(::cleanCredentialValue)
        }
    }

    private fun extractAnyLikelyCredential(input: String): String? {
        val assignment = Regex("""(?im)^\s*(?:export\s+|set\s+|setx\s+|\${'$'}env:)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$""")
            .findAll(input)
            .firstOrNull { match ->
                val key = match.groupValues[1].uppercase()
                key.endsWith("_API_KEY") || key.endsWith("_ACCESS_TOKEN") || key.endsWith("_TOKEN")
            }
        return assignment?.groupValues?.getOrNull(2)?.let(::cleanCredentialValue)
    }

    private fun extractBearerCredential(input: String): String? {
        val inline = Regex("""(?im)^\s*(?:authorization\s*:\s*)?bearer\s+(.+?)\s*$""")
            .find(input)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::cleanCredentialValue)
        if (!inline.isNullOrBlank()) {
            return inline
        }
        return Regex("""(?i)["']?authorization["']?\s*[:=]\s*["']bearer\s+([^"'\s]+)""")
            .find(input)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::cleanCredentialValue)
    }

    private fun extractGenericCredential(input: String): String? {
        val names = """(?:api[_-]?key|access[_-]?token|auth[_-]?token|session[_-]?token|token|secret)"""
        val quoted = Regex("""(?i)["']$names["']\s*:\s*["']([^"']+)["']""")
            .find(input)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::cleanCredentialValue)
        if (!quoted.isNullOrBlank()) {
            return quoted
        }
        return Regex("""(?im)^\s*$names\s*[:=]\s*(.+?)\s*$""")
            .find(input)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::cleanCredentialValue)
    }

    private fun cleanCredentialValue(value: String): String {
        return unquote(
            value.trim()
                .substringBefore(" #")
                .substringBefore(" //")
                .trim()
                .trimEnd(';'),
        )
    }

    private fun unquote(value: String): String {
        val trimmed = value.trim()
        if (trimmed.length < 2) {
            return trimmed
        }
        val first = trimmed.first()
        val last = trimmed.last()
        return if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            trimmed.substring(1, trimmed.length - 1).trim()
        } else {
            trimmed
        }
    }
}
