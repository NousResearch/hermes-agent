package com.nousresearch.hermesagent.api

import org.json.JSONArray
import org.json.JSONObject

data class HealthResponse(
    val status: String,
    val platform: String,
)

data class ModelInfo(
    val id: String,
)

data class ModelsResponse(
    val data: List<ModelInfo>,
)

data class ChatMessage(
    val role: String,
    val content: String,
    val contentParts: List<ChatContentPart> = emptyList(),
)

data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val stream: Boolean = false,
    val sessionId: String? = null,
)

data class ChatCompletionResult(
    val rawBody: String,
)

data class ChatContentPart(
    val type: String,
    val text: String = "",
    val imageUrl: String = "",
    val detail: String = "auto",
)

fun ChatMessage.toJsonObject(): JSONObject {
    return JSONObject().apply {
        put("role", role)
        put("content", toJsonContent())
    }
}

fun ChatMessage.toJsonContent(): Any {
    if (contentParts.isEmpty()) {
        return content
    }
    return JSONArray().apply {
        contentParts.forEach { part ->
            when (part.type) {
                "text" -> put(
                    JSONObject()
                        .put("type", "text")
                        .put("text", part.text),
                )
                "image_url" -> put(
                    JSONObject()
                        .put("type", "image_url")
                        .put(
                            "image_url",
                            JSONObject()
                                .put("url", part.imageUrl)
                                .put("detail", part.detail),
                        ),
                )
            }
        }
    }
}
