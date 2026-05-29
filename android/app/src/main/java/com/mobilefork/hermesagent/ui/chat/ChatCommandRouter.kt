package com.mobilefork.hermesagent.ui.chat

import com.mobilefork.hermesagent.ui.i18n.AppLanguage
import com.mobilefork.hermesagent.ui.i18n.HermesStrings
import com.mobilefork.hermesagent.ui.i18n.hermesStringsFor
import com.mobilefork.hermesagent.ui.shell.AppSection

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
    private val runtimeProviderAuthMethods = setOf("openrouter", "openai", "codex", "chatgpt", "claude", "gemini", "qwen", "qwen-coding-plan", "qwen-oauth", "zai")

    fun execute(
        rawInput: String,
        host: ChatCommandHost,
        strings: HermesStrings = hermesStringsFor(AppLanguage.ENGLISH),
    ): ChatCommandResult {
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
                feedback = strings.chatCommandHelp(),
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
                ChatCommandResult(handled = true, feedback = strings.chatCommandOpenedAccounts())
            }

            "/settings" -> {
                host.navigateToSection(AppSection.Settings)
                ChatCommandResult(handled = true, feedback = strings.chatCommandOpenedSettings())
            }

            "/device" -> {
                host.navigateToSection(AppSection.Device)
                ChatCommandResult(handled = true, feedback = strings.chatCommandOpenedDevice())
            }

            "/portal" -> {
                host.navigateToSection(AppSection.NousPortal)
                ChatCommandResult(handled = true, feedback = strings.chatCommandOpenedPortal())
            }

            "/provider" -> {
                if (remainder.isBlank()) {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandProviderUsage())
                } else if (host.applyProvider(remainder.lowercase())) {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandProviderApplied(remainder.lowercase()))
                } else {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandUnknownProvider(remainder))
                }
            }

            "/model" -> {
                if (remainder.isBlank()) {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandModelUsage())
                } else if (host.applyModel(remainder)) {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandModelUpdated(remainder))
                } else {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandModelFailed(remainder))
                }
            }

            "/signin" -> {
                val method = normalizeAuthMethod(remainder)
                if (method == null) {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandSignInUsage())
                } else if (host.startAuthMethod(method)) {
                    if (method in runtimeProviderAuthMethods) {
                        val feedback = when (method) {
                            "openrouter" -> {
                                host.navigateToSection(AppSection.Accounts)
                                strings.chatCommandOpenRouterOAuth()
                            }
                            "qwen-oauth" -> {
                                host.navigateToSection(AppSection.Settings)
                                strings.chatCommandLegacyQwenOAuth()
                            }
                            else -> {
                                host.navigateToSection(AppSection.Settings)
                                strings.chatCommandProviderTokenSetup(method)
                            }
                        }
                        ChatCommandResult(handled = true, feedback = feedback)
                    } else {
                        host.navigateToSection(AppSection.Accounts)
                        ChatCommandResult(handled = true, feedback = strings.chatCommandCorr3xtSignIn(method))
                    }
                } else {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandSignInFailed(remainder))
                }
            }

            "/speak" -> {
                val normalized = remainder.lowercase()
                if (normalized == "last") {
                    if (host.speakLastReply()) {
                        ChatCommandResult(handled = true, feedback = strings.chatCommandSpeakingLatest())
                    } else {
                        ChatCommandResult(handled = true, feedback = strings.chatCommandNoReplyToSpeak())
                    }
                } else {
                    ChatCommandResult(handled = true, feedback = strings.chatCommandSpeakUsage())
                }
            }

            else -> ChatCommandResult(handled = false)
        }
    }

    private fun normalizeAuthMethod(value: String): String? {
        return when (value.lowercase()) {
            "openrouter" -> "openrouter"
            "openai", "openai-api", "gpt" -> "openai"
            "codex", "openai-codex", "codex-api" -> "codex"
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
