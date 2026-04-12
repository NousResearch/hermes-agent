package com.nousresearch.hermesagent.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class StoredConversationMessage(
    val id: String,
    val role: String,
    val content: String,
    val createdAtEpochMs: Long,
)

data class StoredConversation(
    val sessionId: String,
    val title: String,
    val updatedAtEpochMs: Long,
    val messages: List<StoredConversationMessage>,
)

data class ConversationSummary(
    val sessionId: String,
    val title: String,
    val preview: String,
    val updatedAtEpochMs: Long,
    val messageCount: Int,
)

class ConversationStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun currentSessionId(): String = ensureCurrentConversation().sessionId

    fun currentConversation(): StoredConversation = ensureCurrentConversation()

    fun currentConversationMessages(): List<StoredConversationMessage> = ensureCurrentConversation().messages

    fun listConversationSummaries(): List<ConversationSummary> {
        return readConversations()
            .map { conversation ->
                ConversationSummary(
                    sessionId = conversation.sessionId,
                    title = conversation.title,
                    preview = conversation.messages.lastOrNull()?.content?.trim().orEmpty(),
                    updatedAtEpochMs = conversation.updatedAtEpochMs,
                    messageCount = conversation.messages.size,
                )
            }
            .sortedByDescending { it.updatedAtEpochMs }
    }

    fun loadConversation(sessionId: String): StoredConversation? {
        return readConversations().firstOrNull { it.sessionId == sessionId }
    }

    fun switchConversation(sessionId: String): StoredConversation? {
        val conversation = loadConversation(sessionId) ?: return null
        preferences.edit().putString(KEY_SESSION_ID, sessionId).apply()
        return conversation
    }

    fun createNewConversation(title: String = DEFAULT_TITLE): StoredConversation {
        val now = System.currentTimeMillis()
        val conversation = StoredConversation(
            sessionId = UUID.randomUUID().toString(),
            title = title,
            updatedAtEpochMs = now,
            messages = emptyList(),
        )
        val updated = (readConversations() + conversation).sortedByDescending { it.updatedAtEpochMs }
        writeConversations(updated)
        preferences.edit().putString(KEY_SESSION_ID, conversation.sessionId).apply()
        return conversation
    }

    fun upsertMessage(sessionId: String, message: StoredConversationMessage) {
        val conversations = readConversations().toMutableList()
        val index = conversations.indexOfFirst { it.sessionId == sessionId }
        val base = if (index >= 0) conversations[index] else createShellConversation(sessionId)
        val updatedMessages = base.messages.toMutableList()
        val existingIndex = updatedMessages.indexOfFirst { it.id == message.id }
        if (existingIndex >= 0) {
            updatedMessages[existingIndex] = message
        } else {
            updatedMessages += message
        }
        val updatedConversation = base.copy(
            title = deriveTitle(base.title, updatedMessages),
            updatedAtEpochMs = maxOf(base.updatedAtEpochMs, message.createdAtEpochMs, System.currentTimeMillis()),
            messages = updatedMessages,
        )
        if (index >= 0) {
            conversations[index] = updatedConversation
        } else {
            conversations += updatedConversation
        }
        writeConversations(conversations.sortedByDescending { it.updatedAtEpochMs })
        preferences.edit().putString(KEY_SESSION_ID, sessionId).apply()
    }

    fun updateMessageContent(sessionId: String, messageId: String, newContent: String) {
        val conversation = loadConversation(sessionId) ?: return
        val updatedMessages = conversation.messages.map { message ->
            if (message.id == messageId) message.copy(content = newContent) else message
        }
        replaceConversation(
            conversation.copy(
                title = deriveTitle(conversation.title, updatedMessages),
                updatedAtEpochMs = System.currentTimeMillis(),
                messages = updatedMessages,
            )
        )
    }

    fun clearCurrentConversation(): StoredConversation {
        val currentId = preferences.getString(KEY_SESSION_ID, null)
        if (currentId.isNullOrBlank()) {
            return createNewConversation()
        }
        return clearConversation(currentId)
    }

    fun clearConversation(sessionId: String): StoredConversation {
        val remaining = readConversations().filterNot { it.sessionId == sessionId }
        writeConversations(remaining)
        val next = remaining.firstOrNull()?.let {
            preferences.edit().putString(KEY_SESSION_ID, it.sessionId).apply()
            it
        } ?: createNewConversation()
        return next
    }

    fun clearAll() {
        preferences.edit().remove(KEY_CONVERSATIONS).remove(KEY_SESSION_ID).apply()
    }

    fun clearSession() {
        preferences.edit().remove(KEY_SESSION_ID).apply()
    }

    private fun ensureCurrentConversation(): StoredConversation {
        val conversations = readConversations()
        val currentId = preferences.getString(KEY_SESSION_ID, null)
        val current = conversations.firstOrNull { it.sessionId == currentId }
        if (current != null) {
            return current
        }
        if (conversations.isNotEmpty()) {
            val latest = conversations.maxByOrNull { it.updatedAtEpochMs } ?: conversations.first()
            preferences.edit().putString(KEY_SESSION_ID, latest.sessionId).apply()
            return latest
        }
        return createNewConversation()
    }

    private fun replaceConversation(updatedConversation: StoredConversation) {
        val conversations = readConversations().toMutableList()
        val index = conversations.indexOfFirst { it.sessionId == updatedConversation.sessionId }
        if (index >= 0) {
            conversations[index] = updatedConversation
        } else {
            conversations += updatedConversation
        }
        writeConversations(conversations.sortedByDescending { it.updatedAtEpochMs })
        preferences.edit().putString(KEY_SESSION_ID, updatedConversation.sessionId).apply()
    }

    private fun createShellConversation(sessionId: String): StoredConversation {
        return StoredConversation(
            sessionId = sessionId,
            title = DEFAULT_TITLE,
            updatedAtEpochMs = System.currentTimeMillis(),
            messages = emptyList(),
        )
    }

    private fun readConversations(): List<StoredConversation> {
        val raw = preferences.getString(KEY_CONVERSATIONS, null).orEmpty()
        if (raw.isBlank()) {
            return emptyList()
        }
        return runCatching {
            val array = JSONArray(raw)
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.optJSONObject(index) ?: continue
                    add(item.toConversation())
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeConversations(conversations: List<StoredConversation>) {
        val array = JSONArray()
        conversations.forEach { array.put(it.toJson()) }
        preferences.edit().putString(KEY_CONVERSATIONS, array.toString()).apply()
    }

    private fun deriveTitle(existingTitle: String, messages: List<StoredConversationMessage>): String {
        val firstUserText = messages.firstOrNull { it.role == "user" }
            ?.content
            ?.trim()
            .orEmpty()
            .removePrefix("/")
            .ifBlank { existingTitle }
        if (firstUserText.isBlank()) {
            return existingTitle.ifBlank { DEFAULT_TITLE }
        }
        val collapsed = firstUserText.replace(Regex("\\s+"), " ")
        return if (collapsed.length > 48) collapsed.take(45) + "..." else collapsed
    }

    private fun StoredConversation.toJson(): JSONObject {
        return JSONObject().apply {
            put("sessionId", sessionId)
            put("title", title)
            put("updatedAtEpochMs", updatedAtEpochMs)
            put(
                "messages",
                JSONArray().apply {
                    messages.forEach { message -> put(message.toJson()) }
                },
            )
        }
    }

    private fun JSONObject.toConversation(): StoredConversation {
        val messages = optJSONArray("messages")?.let { array ->
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.optJSONObject(index) ?: continue
                    add(item.toMessage())
                }
            }
        }.orEmpty()
        return StoredConversation(
            sessionId = optString("sessionId", UUID.randomUUID().toString()),
            title = optString("title", DEFAULT_TITLE),
            updatedAtEpochMs = optLong("updatedAtEpochMs", System.currentTimeMillis()),
            messages = messages,
        )
    }

    private fun StoredConversationMessage.toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("role", role)
            put("content", content)
            put("createdAtEpochMs", createdAtEpochMs)
        }
    }

    private fun JSONObject.toMessage(): StoredConversationMessage {
        return StoredConversationMessage(
            id = optString("id", UUID.randomUUID().toString()),
            role = optString("role", "assistant"),
            content = optString("content", ""),
            createdAtEpochMs = optLong("createdAtEpochMs", System.currentTimeMillis()),
        )
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_conversation"
        private const val KEY_CONVERSATIONS = "conversations_json"
        private const val KEY_SESSION_ID = "session_id"
        private const val DEFAULT_TITLE = "New chat"
    }
}
