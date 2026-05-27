package com.mobilefork.hermesagent.api

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okio.BufferedSource
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class HermesSseClient(
    baseUrl: String,
    private val apiKey: String? = null,
    private val httpClient: OkHttpClient = DEFAULT_HTTP_CLIENT,
    private val networkGuard: (String) -> Unit = {},
) {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')

    fun streamChatCompletion(
        request: ChatCompletionRequest,
        onDelta: (String) -> Unit,
        onComplete: () -> Unit,
        onError: (String) -> Unit,
        onStatus: (String) -> Unit = {},
    ) {
        try {
            val payload = JSONObject().apply {
                put("model", request.model)
                put("stream", true)
                put(
                    "messages",
                    JSONArray().apply {
                        request.messages.forEach { msg ->
                            put(msg.toJsonObject())
                        }
                    }
                )
            }
            val chatUrl = "$normalizedBaseUrl/v1/chat/completions"
            onStatus("Opening endpoint stream at ${endpointLabel(chatUrl)}")
            networkGuard(chatUrl)
            val builder = Request.Builder()
                .url(chatUrl)
                .header("Accept", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
            if (!apiKey.isNullOrBlank()) {
                builder.header("Authorization", "Bearer $apiKey")
            }
            if (!request.sessionId.isNullOrBlank()) {
                builder.header(HermesApiClient.SESSION_HEADER, request.sessionId)
            }

            httpClient.newCall(builder.build()).execute().use { response ->
                onStatus("Endpoint responded HTTP ${response.code}; reading SSE frames")
                val body = response.body
                if (!response.isSuccessful) {
                    onError("SSE request failed: ${response.code} ${response.message} ${body?.string().orEmpty().takeBodySnippet()}")
                    return
                }
                val source = body?.source()
                if (source == null) {
                    onError("SSE response body was empty")
                    return
                }
                parseStream(source, onDelta, onComplete, onError, onStatus)
            }
        } catch (error: Exception) {
            onError(endpointTransportErrorMessage(error))
        }
    }

    internal fun parseStream(
        source: BufferedSource,
        onDelta: (String) -> Unit,
        onComplete: () -> Unit,
        onError: (String) -> Unit,
        onStatus: (String) -> Unit = {},
    ) {
        var sawDataFrame = false
        var sawFinishReason = false
        while (!source.exhausted()) {
            val line = source.readUtf8Line() ?: break
            val payload = sseDataPayload(line) ?: continue
            if (payload.isBlank()) {
                continue
            }
            if (!sawDataFrame) {
                sawDataFrame = true
                onStatus("Endpoint stream is live; waiting for assistant text")
            }
            if (payload == "[DONE]") {
                onComplete()
                return
            }
            val event = runCatching { extractStreamEvent(payload) }.getOrElse { error ->
                onError(error.message ?: error.javaClass.simpleName)
                return
            }
            if (!event.finishReason.isNullOrBlank() && event.finishReason != "null") {
                sawFinishReason = true
            }
            if (!event.delta.isNullOrEmpty()) {
                onDelta(event.delta)
            }
        }
        if (sawFinishReason) {
            onComplete()
        } else {
            onError(
                if (sawDataFrame) {
                    EARLY_CLOSE_ERROR
                } else {
                    "Custom endpoint stream closed before any SSE data arrived. $CUSTOM_ENDPOINT_HINT"
                },
            )
        }
    }

    private fun sseDataPayload(line: String): String? {
        if (!line.startsWith("data:")) {
            return null
        }
        return line.removePrefix("data:").trim()
    }

    private data class StreamEvent(
        val delta: String?,
        val finishReason: String?,
    )

    private fun extractStreamEvent(payload: String): StreamEvent {
        val root = JSONObject(payload)
        val choices = root.optJSONArray("choices") ?: return StreamEvent(delta = null, finishReason = null)
        if (choices.length() == 0) {
            return StreamEvent(delta = null, finishReason = null)
        }
        val choice = choices.optJSONObject(0) ?: return StreamEvent(delta = null, finishReason = null)
        val delta = choice.optJSONObject("delta")
        return StreamEvent(
            delta = delta?.optString("content")?.ifBlank { null },
            finishReason = choice.optString("finish_reason").ifBlank { null },
        )
    }

    private fun endpointTransportErrorMessage(error: Exception): String {
        val raw = error.message?.takeIf { it.isNotBlank() } ?: error.javaClass.simpleName
        return when {
            raw.contains("timeout", ignoreCase = true) ->
                "Custom endpoint stream timed out while waiting for data. $CUSTOM_ENDPOINT_HINT"
            raw.contains("closed", ignoreCase = true) ||
                raw.contains("reset", ignoreCase = true) ||
                raw.contains("disconnect", ignoreCase = true) ||
                raw.contains("unexpected end", ignoreCase = true) ->
                "Custom endpoint stream disconnected: $raw. $CUSTOM_ENDPOINT_HINT"
            else -> raw
        }
    }

    private fun endpointLabel(url: String): String {
        return url
            .removePrefix("https://")
            .removePrefix("http://")
            .take(96)
    }

    private fun String.takeBodySnippet(limit: Int = 240): String {
        val compact = replace(Regex("\\s+"), " ").trim()
        return when {
            compact.isBlank() -> ""
            compact.length <= limit -> compact
            else -> compact.take(limit).trimEnd() + "..."
        }
    }

    companion object {
        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
        private const val CUSTOM_ENDPOINT_HINT =
            "Check the Base URL, exact model name, mobile network, server timeout, and that the OpenAI-compatible endpoint keeps SSE open until [DONE]."
        private val EARLY_CLOSE_ERROR =
            "Custom endpoint stream closed before the endpoint sent [DONE]. $CUSTOM_ENDPOINT_HINT"
        private val DEFAULT_HTTP_CLIENT = OkHttpClient.Builder()
            .connectTimeout(20, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build()
    }
}
