package com.nousresearch.hermesagent.data

enum class AuthScope {
    AppAccount,
    RuntimeProvider,
}

data class AuthOption(
    val id: String,
    val label: String,
    val description: String,
    val scope: AuthScope,
    val runtimeProvider: String = "",
    val defaultBaseUrl: String = "",
    val defaultModel: String = "",
    val browserSignInSupported: Boolean = true,
)

data class AuthSession(
    val methodId: String,
    val label: String,
    val scope: AuthScope,
    val runtimeProvider: String = "",
    val signedIn: Boolean = false,
    val status: String = "Not signed in",
    val email: String = "",
    val phone: String = "",
    val displayName: String = "",
    val accessToken: String = "",
    val refreshToken: String = "",
    val sessionToken: String = "",
    val apiKey: String = "",
    val baseUrl: String = "",
    val model: String = "",
    val updatedAtEpochMs: Long = System.currentTimeMillis(),
)

data class PendingAuthRequest(
    val state: String,
    val methodId: String,
    val startUrl: String,
    val authProvider: String = "corr3xt",
    val codeVerifier: String = "",
    val codeChallengeMethod: String = "",
    val createdAtEpochMs: Long = System.currentTimeMillis(),
)

object AuthCatalog {
    val options = listOf(
        AuthOption(
            id = "email",
            label = "Email",
            description = "Sign in to the app through Corr3xt using an email link or password flow.",
            scope = AuthScope.AppAccount,
        ),
        AuthOption(
            id = "google",
            label = "Google",
            description = "Sign in to the app with a Google account via Corr3xt.",
            scope = AuthScope.AppAccount,
        ),
        AuthOption(
            id = "phone",
            label = "Phone",
            description = "Sign in to the app with an SMS / phone verification flow via Corr3xt.",
            scope = AuthScope.AppAccount,
        ),
        AuthOption(
            id = "openrouter",
            label = "OpenRouter",
            description = "Use an OpenRouter API key for Hermes Android remote model calls.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "openrouter",
            defaultBaseUrl = "https://openrouter.ai/api/v1",
            defaultModel = "anthropic/claude-sonnet-4",
            browserSignInSupported = true,
        ),
        AuthOption(
            id = "openai",
            label = "OpenAI",
            description = "Use an OpenAI API key for Hermes Android remote model calls.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "openai",
            defaultBaseUrl = "https://api.openai.com/v1",
            defaultModel = "gpt-4.1",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "chatgpt",
            label = "ChatGPT",
            description = "Paste a ChatGPT Web access token and sync it into Hermes Android.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "chatgpt-web",
            defaultBaseUrl = "https://chatgpt.com/backend-api/f",
            defaultModel = "gpt-5-thinking",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "claude",
            label = "Claude",
            description = "Use an Anthropic / Claude API key for Hermes Android remote model calls.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "anthropic",
            defaultBaseUrl = "https://api.anthropic.com",
            defaultModel = "claude-sonnet-4",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "gemini",
            label = "Gemini",
            description = "Use a Google AI Studio / Gemini API key for Hermes Android remote model calls.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "gemini",
            defaultBaseUrl = "https://generativelanguage.googleapis.com/v1beta/openai",
            defaultModel = "gemini-2.5-pro",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "qwen",
            label = "Qwen Cloud",
            description = "Use a Qwen Cloud / DashScope API key for Hermes Android remote model calls.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "alibaba",
            defaultBaseUrl = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            defaultModel = "qwen3.6-plus",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "qwen-coding-plan",
            label = "Qwen Coding Plan",
            description = "Use a Qwen Coding Plan API key with the dedicated DashScope coding endpoint.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "alibaba-coding-plan",
            defaultBaseUrl = "https://coding-intl.dashscope.aliyuncs.com/v1",
            defaultModel = "qwen3-coder-plus",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "qwen-oauth",
            label = "Qwen OAuth (legacy)",
            description = "Reuse an existing Qwen OAuth / Qwen Chat token in Hermes Android; new Qwen OAuth sign-ins were discontinued on 2026-04-15, so use Qwen Cloud for new setup.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "qwen-oauth",
            defaultBaseUrl = "https://portal.qwen.ai/v1",
            defaultModel = "qwen3-coder-plus",
            browserSignInSupported = false,
        ),
        AuthOption(
            id = "zai",
            label = "Z.AI",
            description = "Use a Z.AI / GLM API key for Hermes Android remote model calls.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "zai",
            defaultBaseUrl = "https://api.z.ai/api/paas/v4",
            defaultModel = "glm-5",
            browserSignInSupported = false,
        ),
    )

    fun find(id: String): AuthOption? = options.firstOrNull { it.id == id }
}
