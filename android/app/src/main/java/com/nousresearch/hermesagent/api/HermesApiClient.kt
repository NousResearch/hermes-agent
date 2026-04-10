package com.nousresearch.hermesagent.api

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

class HermesApiClient(
    baseUrl: String,
    private val apiKey: String? = null,
    private val httpClient: OkHttpClient = OkHttpClient(),
) {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')

    fun getHealth(): HealthResponse {
        val request = requestBuilder("$normalizedBaseUrl/health").get().build()
        httpClient.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            require(response.isSuccessful) { "Health request failed: ${response.code} $body" }
            val json = JSONObject(body)
            return HealthResponse(
                status = json.optString("status"),
                platform = json.optString("platform"),
            )
        }
    }

    fun listModels(): ModelsResponse {
        val request = requestBuilder("$normalizedBaseUrl/v1/models").get().build()
        httpClient.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            require(response.isSuccessful) { "Models request failed: ${response.code} $body" }
            val json = JSONObject(body)
            val models = mutableListOf<ModelInfo>()
            val data = json.optJSONArray("data") ?: JSONArray()
            for (index in 0 until data.length()) {
                val item = data.optJSONObject(index) ?: continue
                models += ModelInfo(id = item.optString("id"))
            }
            return ModelsResponse(data = models)
        }
    }

    fun createChatCompletion(request: ChatCompletionRequest): ChatCompletionResult {
        val payload = JSONObject().apply {
            put("model", request.model)
            put("stream", request.stream)
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
        val builder = requestBuilder("$normalizedBaseUrl/v1/chat/completions")
            .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
        if (!request.sessionId.isNullOrBlank()) {
            builder.header(SESSION_HEADER, request.sessionId)
        }
        httpClient.newCall(builder.build()).execute().use { response ->
            val body = response.body?.string().orEmpty()
            require(response.isSuccessful) { "Chat request failed: ${response.code} $body" }
            return ChatCompletionResult(rawBody = body)
        }
    }

    private fun requestBuilder(url: String): Request.Builder {
        val builder = Request.Builder().url(url)
        if (!apiKey.isNullOrBlank()) {
            builder.header("Authorization", "Bearer $apiKey")
        }
        return builder
    }

    companion object {
        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
        const val SESSION_HEADER = "X-Hermes-Session-Id"
    }
}
