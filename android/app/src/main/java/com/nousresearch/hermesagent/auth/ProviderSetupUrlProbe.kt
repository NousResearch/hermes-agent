package com.nousresearch.hermesagent.auth

import android.content.Context
import android.net.Uri
import com.nousresearch.hermesagent.data.HermesNetworkPolicy
import java.net.HttpURLConnection
import java.net.URL
import java.net.UnknownHostException

data class ProviderSetupProbeResult(
    val url: String,
    val reachable: Boolean,
    val statusLabel: String,
)

object ProviderSetupUrlProbe {
    const val DEFAULT_TIMEOUT_MS = 6_000
    const val MAX_STATUS_LENGTH = 900
    private const val MAX_BODY_PROBE_CHARS = 8_192

    private val mobileUnsupportedPhrases = listOf(
        "not available on mobile devices",
        "does not currently support mobile access",
        "currently not support mobile access",
        "please copy the link below and open it on a desktop",
        "use a desktop browser",
    )

    fun probe(url: String, timeoutMs: Int = DEFAULT_TIMEOUT_MS, context: Context? = null): ProviderSetupProbeResult {
        val target = url.trim()
        val parsed = runCatching { Uri.parse(target) }.getOrNull()
        val scheme = parsed?.scheme?.lowercase().orEmpty()
        if (target.isBlank() || scheme !in setOf("http", "https") || parsed?.host.isNullOrBlank()) {
            return ProviderSetupProbeResult(target, reachable = false, statusLabel = "invalid URL")
        }
        if (context != null && HermesNetworkPolicy.isExternalNetworkBlocked(context, target)) {
            return ProviderSetupProbeResult(
                target,
                reachable = false,
                statusLabel = HermesNetworkPolicy.offlineBlockedMessage("provider setup probe"),
            )
        }
        return try {
            val connection = (URL(target).openConnection() as HttpURLConnection).apply {
                connectTimeout = timeoutMs
                readTimeout = timeoutMs
                instanceFollowRedirects = true
                requestMethod = "GET"
                setRequestProperty("User-Agent", "HermesAgentAndroidProviderSetup/1.0")
                setRequestProperty("Accept", "text/html,application/json,*/*")
            }
            connection.use {
                val code = responseCode
                val statusLabel = "HTTP $code"
                if (code in 200..499) {
                    val body = readProbeBody(connection)
                    val normalizedBody = body.lowercase()
                    val mobileUnsupported = mobileUnsupportedPhrases.any { phrase ->
                        normalizedBody.contains(phrase)
                    }
                    if (mobileUnsupported) {
                        return@use ProviderSetupProbeResult(
                            url = target,
                            reachable = false,
                            statusLabel = "$statusLabel; mobile unsupported page",
                        )
                    }
                }
                ProviderSetupProbeResult(
                    url = target,
                    reachable = code in 200..499,
                    statusLabel = statusLabel,
                )
            }
        } catch (_: UnknownHostException) {
            ProviderSetupProbeResult(target, reachable = false, statusLabel = "unknown host")
        } catch (error: Exception) {
            ProviderSetupProbeResult(target, reachable = false, statusLabel = error.javaClass.simpleName)
        }
    }

    private inline fun <T> HttpURLConnection.use(block: HttpURLConnection.() -> T): T {
        return try {
            block()
        } finally {
            disconnect()
        }
    }

    private fun readProbeBody(connection: HttpURLConnection): String {
        val stream = runCatching { connection.inputStream }.getOrNull()
            ?: connection.errorStream
            ?: return ""
        return stream.use { input ->
            input.reader(Charsets.UTF_8).use { reader ->
                val buffer = CharArray(1_024)
                val builder = StringBuilder()
                while (builder.length < MAX_BODY_PROBE_CHARS) {
                    val read = reader.read(buffer, 0, minOf(buffer.size, MAX_BODY_PROBE_CHARS - builder.length))
                    if (read <= 0) {
                        break
                    }
                    builder.append(buffer, 0, read)
                }
                builder.toString()
            }
        }
    }
}
