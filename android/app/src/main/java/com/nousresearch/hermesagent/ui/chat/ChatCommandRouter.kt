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
                feedback = "Available app commands: /new, /history, /clear, /accounts, /settings, /device, /portal, /auth, /signin <chatgpt|claude|gemini|google|email|phone>, /provider <id>, /model <name>, /speak last.",
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
                    ChatCommandResult(handled = true, feedback = "Usage: /signin <chatgpt|claude|gemini|google|email|phone>")
                } else if (host.startAuthMethod(method)) {
                    host.navigateToSection(AppSection.Accounts)
                    ChatCommandResult(handled = true, feedback = "Opened Corr3xt sign-in for $method. Complete it in your browser, then come back to Hermes.")
                } else {
                    ChatCommandResult(handled = true, feedback = "Could not start sign-in for '$remainder'. Open Accounts and try again.")
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
            "chatgpt", "chatgpt-web", "openai" -> "chatgpt"
            "claude", "anthropic" -> "claude"
            "gemini", "google-ai", "googleai" -> "gemini"
            "google" -> "google"
            "email" -> "email"
            "phone", "sms" -> "phone"
            else -> null
        }
    }
}
