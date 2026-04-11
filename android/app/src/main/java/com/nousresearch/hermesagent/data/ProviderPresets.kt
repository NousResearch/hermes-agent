package com.nousresearch.hermesagent.data

data class ProviderPreset(
    val id: String,
    val label: String,
    val baseUrl: String,
    val modelHint: String,
)

object ProviderPresets {
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
}
