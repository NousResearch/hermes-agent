package com.nousresearch.hermesagent.api

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
