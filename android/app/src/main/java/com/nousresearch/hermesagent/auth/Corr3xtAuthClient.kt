package com.nousresearch.hermesagent.auth

import android.net.Uri
import com.nousresearch.hermesagent.data.AuthOption
import com.nousresearch.hermesagent.data.AuthSessionStore

object Corr3xtAuthClient {
    const val DEFAULT_BASE_URL = "https://auth.corr3xt.com"

    fun normalizedBaseUrl(baseUrl: String): String {
        return baseUrl.trim().trimEnd('/').ifBlank { DEFAULT_BASE_URL }
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
            .appendQueryParameter("redirect_uri", AuthSessionStore.CALLBACK_URI)
            .appendQueryParameter("state", state)
            .build()
    }
}
