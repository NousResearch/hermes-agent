package com.nousresearch.hermesagent.auth

import android.net.Uri
import com.nousresearch.hermesagent.data.AuthOption
import com.nousresearch.hermesagent.data.AuthSessionStore

object Corr3xtAuthClient {
    const val DEFAULT_BASE_URL = "https://auth.corr3xt.com"

    fun normalizeConfiguredBaseUrl(baseUrl: String): String? {
        val candidate = baseUrl.trim()
        if (candidate.isBlank()) {
            return DEFAULT_BASE_URL
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
        return normalizeConfiguredBaseUrl(baseUrl) ?: DEFAULT_BASE_URL
    }

    fun buildStartUri(
        baseUrl: String,
        option: AuthOption,
        state: String,
    ): Uri {
        val normalizedBaseUrl = normalizedBaseUrl(baseUrl)
        return Uri.parse("$normalizedBaseUrl/oauth/start").buildUpon()
            .appendQueryParameter("method", option.id)
            .appendQueryParameter("provider", option.runtimeProvider.ifBlank { option.id })
            .appendQueryParameter("client", "hermes-android")
            .appendQueryParameter("callback_contract", "v1")
            .appendQueryParameter("redirect_uri", AuthSessionStore.CALLBACK_URI)
            .appendQueryParameter("state", state)
            .build()
    }
}
