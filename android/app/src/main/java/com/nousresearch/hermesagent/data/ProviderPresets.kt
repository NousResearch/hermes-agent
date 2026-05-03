package com.nousresearch.hermesagent.data

data class ProviderPreset(
    val id: String,
    val label: String,
    val baseUrl: String,
    val modelHint: String,
)

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
        ),
        ProviderPreset(
            id = "openai",
            label = "OpenAI",
            baseUrl = "https://api.openai.com/v1",
            modelHint = "gpt-4.1",
        ),
        ProviderPreset(
            id = "chatgpt-web",
            label = "ChatGPT Web",
            baseUrl = "https://chatgpt.com/backend-api/f",
            modelHint = "gpt-5-thinking",
        ),
        ProviderPreset(
            id = "anthropic",
            label = "Claude / Anthropic",
            baseUrl = "https://api.anthropic.com",
            modelHint = "claude-sonnet-4",
        ),
        ProviderPreset(
            id = "gemini",
            label = "Gemini / Google AI Studio",
            baseUrl = "https://generativelanguage.googleapis.com/v1beta/openai",
            modelHint = "gemini-2.5-pro",
        ),
        ProviderPreset(
            id = "qwen-oauth",
            label = "Qwen OAuth",
            baseUrl = "https://portal.qwen.ai/v1",
            modelHint = "qwen3-coder-plus",
        ),
        ProviderPreset(
            id = "zai",
            label = "Z.AI / GLM",
            baseUrl = "https://api.z.ai/api/paas/v4",
            modelHint = "glm-5",
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

    fun find(id: String): ProviderPreset? = defaults.firstOrNull { it.id == id }

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
}
