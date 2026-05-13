package com.nousresearch.hermesagent.ui.chat

import com.nousresearch.hermesagent.ui.shell.AppSection

data class ChatCommandResult(
    val handled: Boolean,
    val feedback: String? = null,
)

data class ChatCommandHost(
    val openHistory: () -> Unit,
    val newConversation: () -> Unit,
    val clearConversation: () -> Unit,
    val navigateToSection: (AppSection) -> Unit,
    val applyProvider: (String) -> Boolean,
    val applyModel: (String) -> Boolean,
    val startAuthMethod: (String) -> Boolean,
    val speakLastReply: () -> Boolean,
)

object ChatCommandRouter {
    private val runtimeProviderAuthMethods = setOf("openrouter", "openai", "chatgpt", "claude", "gemini", "qwen", "qwen-coding-plan", "qwen-oauth", "zai")

    fun execute(rawInput: String, host: ChatCommandHost): ChatCommandResult {
        val input = rawInput.trim()
        if (!input.startsWith("/")) {
            return ChatCommandResult(handled = false)
        }

        val parts = input.split(Regex("\\s+"), limit = 2)
        val command = parts.firstOrNull().orEmpty().lowercase()
        val remainder = parts.getOrNull(1).orEmpty().trim()

        return when (command) {
            "/help" -> ChatCommandResult(
                handled = true,
                feedback = "Available app commands: /new, /history, /clear, /accounts, /settings, /device, /portal, /auth, /signin <openrouter|openai|chatgpt|claude|gemini|qwen|qwen-coding-plan|qwen-oauth|zai|google|email|phone>, /provider <id>, /model <name>, /speak last.",
            )

            "/new" -> {
                host.newConversation()
                ChatCommandResult(handled = true)
            }

            "/history" -> {
                host.openHistory()
                ChatCommandResult(handled = true)
            }

            "/clear" -> {
                host.clearConversation()
                ChatCommandResult(handled = true)
            }

            "/accounts", "/auth" -> {
                host.navigateToSection(AppSection.Accounts)
                ChatCommandResult(handled = true, feedback = "Opened Accounts so you can manage sign-ins and provider auth.")
            }

            "/settings" -> {
                host.navigateToSection(AppSection.Settings)
                ChatCommandResult(handled = true, feedback = "Opened Settings for provider, base URL, model, and API key controls.")
            }

            "/device" -> {
                host.navigateToSection(AppSection.Device)
                ChatCommandResult(handled = true, feedback = "Opened Device for Linux commands, shared folders, and accessibility controls.")
            }

            "/portal" -> {
                host.navigateToSection(AppSection.NousPortal)
                ChatCommandResult(handled = true, feedback = "Opened the Nous Portal page.")
            }

            "/provider" -> {
                if (remainder.isBlank()) {
                    ChatCommandResult(handled = true, feedback = "Usage: /provider <provider-id>")
                } else if (host.applyProvider(remainder.lowercase())) {
                    ChatCommandResult(handled = true, feedback = "Applied provider ${remainder.lowercase()} and restarted the Hermes backend.")
                } else {
                    ChatCommandResult(handled = true, feedback = "Unknown provider '${remainder}'. Open Settings for the available provider profiles.")
                }
            }

            "/model" -> {
                if (remainder.isBlank()) {
                    ChatCommandResult(handled = true, feedback = "Usage: /model <model-name>")
                } else if (host.applyModel(remainder)) {
                    ChatCommandResult(handled = true, feedback = "Updated the active Hermes model to '$remainder' and restarted the backend.")
                } else {
                    ChatCommandResult(handled = true, feedback = "Could not apply model '$remainder'. Open Settings to edit the model directly.")
                }
            }

            "/signin" -> {
                val method = normalizeAuthMethod(remainder)
                if (method == null) {
                    ChatCommandResult(handled = true, feedback = "Usage: /signin <openrouter|openai|chatgpt|claude|gemini|qwen|qwen-coding-plan|qwen-oauth|zai|google|email|phone>")
                } else if (host.startAuthMethod(method)) {
                    if (method in runtimeProviderAuthMethods) {
                        val feedback = when (method) {
                            "openrouter" -> {
                                host.navigateToSection(AppSection.Accounts)
                                "Opened OpenRouter OAuth in Hermes. Approve Hermes to save a user-controlled API key, or paste an OpenRouter API key in Settings."
                            }
                            "qwen-oauth" -> {
                                host.navigateToSection(AppSection.Settings)
                                "Prepared legacy qwen-oauth token setup in Settings and opened the provider setup page in Hermes. Qwen OAuth sign-ins were discontinued on 2026-04-15; use /signin qwen for new Qwen Cloud API-key setup."
                            }
                            else -> {
                                host.navigateToSection(AppSection.Settings)
                                "Prepared $method API-key/token setup in Settings and opened the provider setup page in Hermes. Paste the provider credential there to power Hermes."
                            }
                        }
                        ChatCommandResult(handled = true, feedback = feedback)
                    } else {
                        host.navigateToSection(AppSection.Accounts)
                        ChatCommandResult(handled = true, feedback = "Opened Corr3xt app sign-in for $method. Complete it in your browser, then come back to Hermes.")
                    }
                } else {
                    ChatCommandResult(handled = true, feedback = "Could not start sign-in for '$remainder'. Configure a reachable Corr3xt URL in Accounts, or use provider API keys in Settings.")
                }
            }

            "/speak" -> {
                val normalized = remainder.lowercase()
                if (normalized == "last") {
                    if (host.speakLastReply()) {
                        ChatCommandResult(handled = true, feedback = "Speaking the latest Hermes reply.")
                    } else {
                        ChatCommandResult(handled = true, feedback = "There is no assistant reply available to speak yet.")
                    }
                } else {
                    ChatCommandResult(handled = true, feedback = "Usage: /speak last")
                }
            }

            else -> ChatCommandResult(handled = false)
        }
    }

    private fun normalizeAuthMethod(value: String): String? {
        return when (value.lowercase()) {
            "openrouter" -> "openrouter"
            "openai", "openai-api", "gpt" -> "openai"
            "chatgpt", "chatgpt-web", "chatgpt-token" -> "chatgpt"
            "claude", "anthropic" -> "claude"
            "gemini", "google-ai", "googleai" -> "gemini"
            "qwen", "dashscope", "alibaba" -> "qwen"
            "qwen-coding-plan", "qwen-coding", "coding-plan", "alibaba-coding", "alibaba-coding-plan",
            "bailian", "bailian-coding-plan" -> "qwen-coding-plan"
            "qwen-oauth", "qwen-portal", "qwen-cli", "qwen-chat" -> "qwen-oauth"
            "zai", "z.ai", "glm" -> "zai"
            "google" -> "google"
            "email" -> "email"
            "phone", "sms" -> "phone"
            else -> null
        }
    }
}
