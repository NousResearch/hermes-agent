package com.nousresearch.hermesagent.data

import android.content.Context
import java.util.UUID

class ConversationStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun currentSessionId(): String {
        val existing = preferences.getString(KEY_SESSION_ID, null)
        if (!existing.isNullOrBlank()) {
            return existing
        }
        val generated = UUID.randomUUID().toString()
        preferences.edit().putString(KEY_SESSION_ID, generated).apply()
        return generated
    }

    fun clearSession() {
        preferences.edit().remove(KEY_SESSION_ID).apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_conversation"
        private const val KEY_SESSION_ID = "session_id"
    }
}
