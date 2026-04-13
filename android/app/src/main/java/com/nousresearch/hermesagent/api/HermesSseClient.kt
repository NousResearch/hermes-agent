package com.nousresearch.hermesagent.api

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okio.BufferedSource
import org.json.JSONArray
import org.json.JSONObject

class HermesSseClient(
    baseUrl: String,
    private val apiKey: String? = null,
    private val httpClient: OkHttpClient = OkHttpClient(),
) {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')

    fun streamChatCompletion(
        request: ChatCompletionRequest,
        onDelta: (String) -> Unit,
        onComplete: () -> Unit,
        onError: (String) -> Unit,
    ) {
        try {
            val payload = JSONObject().apply {
                put("model", request.model)
                put("stream", true)
                put(
                    "messages",
                    JSONArray().apply {
                        request.messages.forEach { msg ->
                            put(
                                JSONObject().apply {
                                    put("role", msg.role)
                                    put("content", msg.content)
                                }
                            )
                        }
                    }
                )
            }
            val builder = Request.Builder()
                .url("$normalizedBaseUrl/v1/chat/completions")
                .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
            if (!apiKey.isNullOrBlank()) {
                builder.header("Authorization", "Bearer $apiKey")
            }
            if (!request.sessionId.isNullOrBlank()) {
                builder.header(HermesApiClient.SESSION_HEADER, request.sessionId)
            }

            httpClient.newCall(builder.build()).execute().use { response ->
                if (!response.isSuccessful) {
                    onError("SSE request failed: ${response.code}")
                    return
                }
                val source = response.body?.source()
                if (source == null) {
                    onError("SSE response body was empty")
                    return
                }
                parseStream(source, onDelta, onComplete, onError)
            }
        } catch (error: Exception) {
            onError(error.message ?: error.javaClass.simpleName)
        }
    }

    internal fun parseStream(
        source: BufferedSource,
        onDelta: (String) -> Unit,
        onComplete: () -> Unit,
        onError: (String) -> Unit,
    ) {
        while (!source.exhausted()) {
            val line = source.readUtf8Line() ?: break
            if (!line.startsWith("data: ")) {
                continue
            }
            val payload = line.removePrefix("data: ").trim()
            if (payload == "[DONE]") {
                onComplete()
                return
            }
            val delta = runCatching { extractDelta(payload) }.getOrElse { error ->
                onError(error.message ?: error.javaClass.simpleName)
                return
            }
            if (!delta.isNullOrEmpty()) {
                onDelta(delta)
            }
        }
    }

    private fun extractDelta(payload: String): String? {
        val root = JSONObject(payload)
        val choices = root.optJSONArray("choices") ?: return null
        if (choices.length() == 0) {
            return null
        }
        val choice = choices.optJSONObject(0) ?: return null
        val delta = choice.optJSONObject("delta") ?: return null
        return delta.optString("content").ifBlank { null }
    }

    companion object {
        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
    }
}
