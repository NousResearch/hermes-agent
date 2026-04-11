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
            id = "chatgpt",
            label = "ChatGPT",
            description = "Authenticate ChatGPT Web access and sync it into Hermes Android automatically.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "chatgpt-web",
            defaultBaseUrl = "https://chatgpt.com/backend-api/f",
            defaultModel = "gpt-5-thinking",
        ),
        AuthOption(
            id = "claude",
            label = "Claude",
            description = "Authenticate Anthropic / Claude credentials and apply them to Hermes Android.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "anthropic",
            defaultBaseUrl = "https://api.anthropic.com",
            defaultModel = "claude-sonnet-4",
        ),
        AuthOption(
            id = "gemini",
            label = "Gemini",
            description = "Authenticate Google AI Studio / Gemini access and apply it to Hermes Android.",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "gemini",
            defaultBaseUrl = "https://generativelanguage.googleapis.com/v1beta/openai",
            defaultModel = "gemini-2.5-pro",
        ),
    )

    fun find(id: String): AuthOption? = options.firstOrNull { it.id == id }
}
