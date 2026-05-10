package com.nousresearch.hermesagent.auth

import android.net.Uri
import com.nousresearch.hermesagent.data.AuthOption
import com.nousresearch.hermesagent.data.AuthSessionStore
import java.net.HttpURLConnection
import java.net.URL
import java.net.UnknownHostException

data class AuthStartProbeResult(
    val reachable: Boolean,
    val status: String = "",
    val host: String = "",
    val errorName: String = "",
)

object Corr3xtAuthClient {
    const val DEFAULT_BASE_URL = ""

    fun normalizeConfiguredBaseUrl(baseUrl: String): String? {
        val candidate = baseUrl.trim()
        if (candidate.isBlank()) {
            return null
        }

        val parsed = runCatching { Uri.parse(candidate) }.getOrNull() ?: return null
        val scheme = parsed.scheme?.lowercase().orEmpty()
        val authority = parsed.encodedAuthority.orEmpty()
        if (scheme !in setOf("http", "https") || authority.isBlank()) {
            return null
        }

        val normalizedPath = parsed.encodedPath
            ?.trim()
            ?.trimEnd('/')
            ?.takeIf { it.isNotBlank() && it != "/" }

        return Uri.Builder()
            .scheme(scheme)
            .encodedAuthority(authority)
            .apply {
                if (!normalizedPath.isNullOrBlank()) {
                    encodedPath(normalizedPath)
                }
            }
            .build()
            .toString()
            .trimEnd('/')
    }

    fun normalizedBaseUrl(baseUrl: String): String {
        return normalizeConfiguredBaseUrl(baseUrl).orEmpty()
    }

    fun buildStartUri(
        baseUrl: String,
        option: AuthOption,
        state: String,
        languageTag: String = "en",
    ): Uri {
        val normalizedBaseUrl = normalizeConfiguredBaseUrl(baseUrl)
            ?: throw IllegalArgumentException("Corr3xt base URL is not configured")
        val normalizedLanguageTag = languageTag.trim().ifBlank { "en" }
        return Uri.parse("$normalizedBaseUrl/oauth/start").buildUpon()
            .appendQueryParameter("method", option.id)
            .appendQueryParameter("provider", option.runtimeProvider.ifBlank { option.id })
            .appendQueryParameter("client", "hermes-android")
            .appendQueryParameter("callback_contract", "v1")
            .appendQueryParameter("redirect_uri", AuthSessionStore.CALLBACK_URI)
            .appendQueryParameter("state", state)
            .appendQueryParameter("lang", normalizedLanguageTag)
            .appendQueryParameter("locale", normalizedLanguageTag)
            .appendQueryParameter("ui_locales", normalizedLanguageTag)
            .build()
    }

    fun probeStartUri(uri: Uri, timeoutMs: Int = 4_000): AuthStartProbeResult {
        val host = uri.host.orEmpty()
        val oauthStartSuffix = "/oauth/start"
        val basePath = uri.encodedPath
            .orEmpty()
            .removeSuffix(oauthStartSuffix)
            .ifBlank { "/" }
        val probeUri = uri.buildUpon()
            .encodedPath(basePath)
            .encodedQuery(null)
            .fragment(null)
            .build()
        return try {
            val connection = (URL(probeUri.toString()).openConnection() as HttpURLConnection).apply {
                connectTimeout = timeoutMs
                readTimeout = timeoutMs
                instanceFollowRedirects = false
                requestMethod = "GET"
            }
            connection.use {
                responseCode
                AuthStartProbeResult(reachable = true)
            }
        } catch (_: UnknownHostException) {
            AuthStartProbeResult(
                reachable = false,
                status = "unknown_host",
                host = host,
            )
        } catch (error: Exception) {
            AuthStartProbeResult(
                reachable = false,
                status = "network_error",
                errorName = error.javaClass.simpleName,
            )
        }
    }

    private inline fun <T> HttpURLConnection.use(block: HttpURLConnection.() -> T): T {
        return try {
            block()
        } finally {
            disconnect()
        }
    }
}
