package com.mobilefork.hermesagent.api

import java.util.Locale

object HermesEndpointUrl {
    private val SCHEME_PATTERN = Regex("^[A-Za-z][A-Za-z0-9+.-]*://")
    private val LAN_172_PATTERN = Regex("^172\\.(1[6-9]|2\\d|3[0-1])\\.")
    private val KNOWN_ENDPOINT_SUFFIXES = listOf(
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/models",
        "/models",
        "/v1/responses",
        "/responses",
        "/v1",
    )

    fun normalizeBaseUrl(rawBaseUrl: String): String {
        var candidate = rawBaseUrl.trim()
        require(candidate.isNotBlank()) { "Endpoint Base URL is not configured" }
        candidate = candidate.replace(Regex("\\s+"), "")
        candidate = when {
            candidate.startsWith("wss://", ignoreCase = true) -> "https://${candidate.drop(6)}"
            candidate.startsWith("ws://", ignoreCase = true) -> "http://${candidate.drop(5)}"
            SCHEME_PATTERN.containsMatchIn(candidate) -> candidate
            else -> "${defaultSchemeFor(candidate)}://$candidate"
        }
        candidate = candidate.substringBefore('#').substringBefore('?').trimEnd('/')
        return stripKnownEndpointSuffixes(candidate).trimEnd('/')
    }

    fun healthUrl(baseUrl: String): String = "${normalizeBaseUrl(baseUrl)}/health"

    fun modelsUrl(baseUrl: String): String = "${normalizeBaseUrl(baseUrl)}/v1/models"

    fun chatCompletionsUrl(baseUrl: String): String = "${normalizeBaseUrl(baseUrl)}/v1/chat/completions"

    fun openAiRuntimeBaseUrl(baseUrl: String): String = "${normalizeBaseUrl(baseUrl)}/v1"

    private fun defaultSchemeFor(candidate: String): String {
        val authority = candidate.substringBefore('/').trim()
        val host = if (authority.startsWith("[")) {
            authority.substringAfter('[').substringBefore(']')
        } else {
            authority.substringBefore(':')
        }.lowercase(Locale.US)
        return if (
            host == "localhost" ||
            host == "::1" ||
            host == "0.0.0.0" ||
            host == "127.0.0.1" ||
            host.startsWith("10.") ||
            host.startsWith("192.168.") ||
            LAN_172_PATTERN.containsMatchIn(host)
        ) {
            "http"
        } else {
            "https"
        }
    }

    private fun stripKnownEndpointSuffixes(input: String): String {
        var output = input
        var changed: Boolean
        do {
            changed = false
            val lower = output.lowercase(Locale.US)
            for (suffix in KNOWN_ENDPOINT_SUFFIXES) {
                if (lower.endsWith(suffix)) {
                    output = output.dropLast(suffix.length).trimEnd('/')
                    changed = true
                    break
                }
            }
        } while (changed)
        return output
    }
}
