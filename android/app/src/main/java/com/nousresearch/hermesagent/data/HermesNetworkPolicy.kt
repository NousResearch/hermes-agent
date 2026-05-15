package com.nousresearch.hermesagent.data

import android.content.Context
import android.net.Uri
import java.io.IOException
import java.util.Locale

object HermesNetworkPolicy {
    fun isOfflineAirplaneModeEnabled(context: Context): Boolean {
        return AppSettingsStore(context.applicationContext).load().offlineAirplaneMode
    }

    fun isExternalNetworkBlocked(context: Context, url: String): Boolean {
        return isOfflineAirplaneModeEnabled(context) && !isLocalNetworkUrl(url)
    }

    @Throws(IOException::class)
    fun requireExternalNetworkAllowed(
        context: Context,
        url: String,
        actionLabel: String = "network request",
    ) {
        if (isExternalNetworkBlocked(context, url)) {
            throw IOException(offlineBlockedMessage(actionLabel))
        }
    }

    fun offlineBlockedMessage(actionLabel: String): String {
        return "Offline airplane mode is on; Hermes blocked this $actionLabel so the app stays phone-local."
    }

    fun isLocalNetworkUrl(url: String): Boolean {
        val target = url.trim()
        val parsed = runCatching { Uri.parse(target) }.getOrNull() ?: return false
        val scheme = parsed.scheme?.lowercase(Locale.US).orEmpty()
        if (scheme !in setOf("http", "https")) {
            return true
        }
        val host = parsed.host?.trim('[', ']')?.lowercase(Locale.US).orEmpty()
        return host in LOCAL_HOSTS
    }

    private val LOCAL_HOSTS = setOf(
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    )
}
