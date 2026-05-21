package com.mobilefork.hermesagent.device

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import org.json.JSONObject

object HermesClipboardActionBridge {
    fun setClipboardJson(context: Context, text: String, label: String = DEFAULT_LABEL): JSONObject {
        if (text.indexOf('\u0000') >= 0 || label.indexOf('\u0000') >= 0) {
            return errorJson("Clipboard text and label must not contain NUL bytes")
        }
        val clipboard = context.applicationContext.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            ?: return errorJson("Android clipboard service is unavailable")
        clipboard.setPrimaryClip(ClipData.newPlainText(label.ifBlank { DEFAULT_LABEL }, text))
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", "set_clipboard")
            .put("message", "Clipboard text updated")
            .put("text_length", text.length)
    }

    private fun errorJson(message: String): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("action", "set_clipboard")
            .put("error", message)
    }

    private const val DEFAULT_LABEL = "Hermes"
}
