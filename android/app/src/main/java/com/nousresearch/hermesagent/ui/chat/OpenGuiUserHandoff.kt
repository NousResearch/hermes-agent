package com.nousresearch.hermesagent.ui.chat

import android.content.Context
import com.nousresearch.hermesagent.device.HermesNotificationActionBridge
import com.nousresearch.hermesagent.device.HermesToastActionBridge
import com.nousresearch.hermesagent.device.HermesVibrationActionBridge
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.absoluteValue

internal object OpenGuiUserHandoff {
    private const val MAX_MESSAGE_CHARS = 500
    private const val MAX_TOAST_CHARS = 180
    private const val CHANNEL_ID = "hermes_opengui_handoff"
    private const val NOTIFICATION_TAG = "opengui_call_user"

    fun execute(context: Context, parsed: ParsedOpenGuiAction, sessionId: String): JSONObject {
        val message = userMessage(parsed)
        val notification = runCatching {
            JSONObject(HermesNotificationActionBridge.performNotificationJson(context, notificationPayload(parsed, message, sessionId)))
        }.getOrElse { error ->
            bridgeError("notification_post", error)
        }
        val toast = runCatching {
            HermesToastActionBridge.showToastJson(
                context,
                JSONObject()
                    .put("text", message.take(MAX_TOAST_CHARS))
                    .put("long", true),
            )
        }.getOrElse { error ->
            bridgeError("show_toast", error)
        }
        val vibration = runCatching {
            HermesVibrationActionBridge.vibrateJson(
                context,
                JSONObject().put("pattern_ms", JSONArray(listOf(0, 200, 100, 200))),
            )
        }.getOrElse { error ->
            bridgeError("vibrate", error)
        }
        return resultJson(message, notification, toast, vibration)
    }

    internal fun notificationPayload(parsed: ParsedOpenGuiAction, message: String, sessionId: String): JSONObject {
        val executionId = executionId(sessionId)
        return JSONObject()
            .put("notification_action", "post")
            .put("title", "Hermes needs you")
            .put("text", message)
            .put("status_text", "OpenGUI user handoff")
            .put("channel_id", CHANNEL_ID)
            .put("channel_name", "Hermes OpenGUI handoff")
            .put("priority", "high")
            .put("importance", "high")
            .put("ongoing", true)
            .put("auto_cancel", true)
            .put("notification_tag", NOTIFICATION_TAG)
            .put("notification_id", notificationId(executionId, parsed.rawText))
            .put(
                "notification_buttons",
                JSONArray(
                    listOf(
                        JSONObject()
                            .put("title", "Open Hermes")
                            .put("action", "open_app")
                            .put("dismiss_on_tap", false),
                        JSONObject()
                            .put("title", "Dismiss")
                            .put("action", "cancel")
                            .put("dismiss_on_tap", true),
                    ),
                ),
            )
    }

    internal fun userMessage(parsed: ParsedOpenGuiAction): String {
        return parsed.content
            .ifBlank { "OpenGUI action requested user intervention." }
            .replace('\u0000', ' ')
            .trim()
            .take(MAX_MESSAGE_CHARS)
    }

    internal fun resultJson(
        message: String,
        notification: JSONObject,
        toast: JSONObject,
        vibration: JSONObject,
    ): JSONObject {
        val surfaced = notification.optBoolean("success", false) ||
            toast.optBoolean("success", false) ||
            vibration.optBoolean("success", false)
        return JSONObject()
            .put("success", true)
            .put("action", "call_user")
            .put("status", "call_user")
            .put("paused", true)
            .put("terminal", false)
            .put("requires_user_intervention", true)
            .put("opengui_user_handoff", true)
            .put("surfaced_on_device", surfaced)
            .put("message", message)
            .put("resume_hint", "After the user finishes the phone-side step, continue the Hermes chat or send an OpenGUI-compatible resume command.")
            .put("compatible_commands", JSONArray(listOf("/opengui resume <execution_id> <feedback>", "/opengui cancel <execution_id>")))
            .put("notification_result", notification)
            .put("toast_result", toast)
            .put("vibration_result", vibration)
    }

    private fun executionId(sessionId: String): String {
        return sessionId.trim().ifBlank { "local-chat" }.take(64)
    }

    private fun notificationId(executionId: String, rawAction: String): Int {
        return "$executionId|$rawAction".hashCode()
            .toLong()
            .absoluteValue
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()
            .takeIf { it > 0 }
            ?: 4117
    }

    private fun bridgeError(action: String, error: Throwable): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("action", action)
            .put("error", error.message ?: error.javaClass.simpleName)
    }
}
