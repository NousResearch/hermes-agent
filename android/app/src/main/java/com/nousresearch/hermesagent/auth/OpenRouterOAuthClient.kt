package com.nousresearch.hermesagent.auth

import android.net.Uri
import android.util.Base64
import com.nousresearch.hermesagent.data.AuthCatalog
import com.nousresearch.hermesagent.data.AuthScope
import com.nousresearch.hermesagent.data.AuthSession
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.data.PendingAuthRequest
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.security.SecureRandom

data class OpenRouterOAuthStartRequest(
    val pendingRequest: PendingAuthRequest,
    val startUri: Uri,
)

data class OpenRouterOAuthExchangeResult(
    val apiKey: String = "",
    val errorStatus: String = "",
) {
    val success: Boolean
        get() = apiKey.isNotBlank()
}

object OpenRouterOAuthClient {
    private const val AUTH_PROVIDER = "openrouter-oauth"
    private const val AUTH_URL = "https://openrouter.ai/auth"
    private const val DEFAULT_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
    private const val CODE_CHALLENGE_METHOD = "S256"
    private const val MAX_STATUS_LENGTH = 240
    private val secureRandom = SecureRandom()

    fun createStartRequest(state: String, verifier: String = createCodeVerifier()): OpenRouterOAuthStartRequest {
        val callbackUrl = Uri.parse(AuthSessionStore.CALLBACK_URI).buildUpon()
            .appendQueryParameter("method", "openrouter")
            .appendQueryParameter("provider", "openrouter")
            .appendQueryParameter("state", state)
            .build()
            .toString()
        val startUri = Uri.parse(AUTH_URL).buildUpon()
            .appendQueryParameter("callback_url", callbackUrl)
            .appendQueryParameter("code_challenge", codeChallenge(verifier))
            .appendQueryParameter("code_challenge_method", CODE_CHALLENGE_METHOD)
            .build()
        val pendingRequest = PendingAuthRequest(
            state = state,
            methodId = "openrouter",
            startUrl = startUri.toString(),
            authProvider = AUTH_PROVIDER,
            codeVerifier = verifier,
            codeChallengeMethod = CODE_CHALLENGE_METHOD,
        )
        return OpenRouterOAuthStartRequest(pendingRequest, startUri)
    }

    fun isOpenRouterCallback(uri: Uri?, pending: PendingAuthRequest?): Boolean {
        return AuthSessionStore.isAuthCallback(uri) &&
            pending?.methodId == "openrouter" &&
            pending.authProvider == AUTH_PROVIDER
    }

    fun exchangeCallbackForSession(
        uri: Uri,
        pending: PendingAuthRequest,
        nowEpochMs: Long = System.currentTimeMillis(),
        exchangeUrl: String = DEFAULT_EXCHANGE_URL,
    ): AuthSession {
        val option = requireNotNull(AuthCatalog.find("openrouter"))
        val callbackState = uri.getQueryParameter("state").orEmpty().trim()
        if (AuthSessionStore.isPendingRequestExpired(pending, nowEpochMs)) {
            return failureSession("OpenRouter sign-in expired. Start sign-in again.", nowEpochMs)
        }
        if (callbackState.isBlank() || callbackState != pending.state) {
            return failureSession("OpenRouter sign-in rejected: state mismatch", nowEpochMs)
        }
        val callbackError = uri.getQueryParameter("error_description").orEmpty()
            .ifBlank { uri.getQueryParameter("error").orEmpty() }
            .sanitizeStatus()
        if (callbackError.isNotBlank()) {
            return failureSession("OpenRouter sign-in failed: $callbackError", nowEpochMs)
        }
        val code = uri.getQueryParameter("code").orEmpty().trim()
        if (code.isBlank()) {
            return failureSession("OpenRouter sign-in rejected: no authorization code returned", nowEpochMs)
        }
        if (pending.codeVerifier.isBlank()) {
            return failureSession("OpenRouter sign-in rejected: missing PKCE verifier", nowEpochMs)
        }

        val exchange = exchangeCodeForApiKey(
            code = code,
            codeVerifier = pending.codeVerifier,
            exchangeUrl = exchangeUrl,
        )
        if (!exchange.success) {
            return failureSession(exchange.errorStatus.ifBlank { "OpenRouter sign-in failed during API-key exchange" }, nowEpochMs)
        }
        return AuthSession(
            methodId = option.id,
            label = option.label,
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = option.runtimeProvider,
            signedIn = true,
            status = "Signed in with OpenRouter OAuth and saved the API key securely.",
            apiKey = exchange.apiKey,
            baseUrl = option.defaultBaseUrl,
            model = option.defaultModel,
            updatedAtEpochMs = nowEpochMs,
        )
    }

    fun exchangeCodeForApiKey(
        code: String,
        codeVerifier: String,
        exchangeUrl: String = DEFAULT_EXCHANGE_URL,
        timeoutMs: Int = 15_000,
    ): OpenRouterOAuthExchangeResult {
        return try {
            val body = JSONObject()
                .put("code", code)
                .put("code_verifier", codeVerifier)
                .put("code_challenge_method", CODE_CHALLENGE_METHOD)
                .toString()
            val connection = (URL(exchangeUrl).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = timeoutMs
                readTimeout = timeoutMs
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Accept", "application/json")
            }
            connection.use {
                OutputStreamWriter(outputStream, Charsets.UTF_8).use { writer -> writer.write(body) }
                val responseText = if (responseCode in 200..299) {
                    inputStream.bufferedReader().use { it.readText() }
                } else {
                    errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                }
                if (responseCode !in 200..299) {
                    return OpenRouterOAuthExchangeResult(
                        errorStatus = "OpenRouter sign-in failed during API-key exchange: HTTP $responseCode",
                    )
                }
                val key = JSONObject(responseText).optString("key").trim()
                if (key.isBlank()) {
                    OpenRouterOAuthExchangeResult(
                        errorStatus = "OpenRouter sign-in failed: API-key exchange returned no key",
                    )
                } else {
                    OpenRouterOAuthExchangeResult(apiKey = key)
                }
            }
        } catch (error: Exception) {
            OpenRouterOAuthExchangeResult(
                errorStatus = "OpenRouter sign-in failed during API-key exchange: ${error.javaClass.simpleName}",
            )
        }
    }

    internal fun codeChallenge(verifier: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray(Charsets.US_ASCII))
        return Base64.encodeToString(digest, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }

    private fun createCodeVerifier(): String {
        val bytes = ByteArray(32)
        secureRandom.nextBytes(bytes)
        return Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }

    private fun failureSession(status: String, nowEpochMs: Long): AuthSession {
        val option = requireNotNull(AuthCatalog.find("openrouter"))
        return AuthSession(
            methodId = option.id,
            label = option.label,
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = option.runtimeProvider,
            signedIn = false,
            status = status.sanitizeStatus(),
            baseUrl = option.defaultBaseUrl,
            model = option.defaultModel,
            updatedAtEpochMs = nowEpochMs,
        )
    }

    private fun String.sanitizeStatus(): String {
        return replace(Regex("[\\u0000-\\u001F]"), " ")
            .trim()
            .take(MAX_STATUS_LENGTH)
    }

    private inline fun <T> HttpURLConnection.use(block: HttpURLConnection.() -> T): T {
        return try {
            block()
        } finally {
            disconnect()
        }
    }
}
