package com.mobilefork.hermesagent.ui.chat

import android.content.Context
import com.mobilefork.hermesagent.api.ChatContentPart
import com.mobilefork.hermesagent.api.ChatMessage
import org.json.JSONObject

internal data class NativeToolChatSendResult(
    val content: String,
)

internal object NativeToolChatSender {
    fun extractDirectDiagnosticsArguments(prompt: String): JSONObject? {
        return NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(prompt)
            ?: NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(prompt)
    }

    fun send(
        context: Context,
        baseUrl: String,
        modelName: String,
        sessionId: String,
        userText: String,
        userContentParts: List<ChatContentPart>,
        priorMessages: List<ChatMessage>,
        relevantMemoryContext: String,
    ): NativeToolChatSendResult {
        val result = NativeToolCallingChatClient(context.applicationContext).send(
            baseUrl = baseUrl,
            modelName = modelName,
            sessionId = sessionId,
            userText = userText,
            userContentParts = userContentParts,
            priorMessages = priorMessages,
            relevantMemoryContext = relevantMemoryContext,
        )
        return NativeToolChatSendResult(content = result.content)
    }
}
