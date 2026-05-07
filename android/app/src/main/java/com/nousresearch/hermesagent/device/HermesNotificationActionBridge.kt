package com.nousresearch.hermesagent.device

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.nousresearch.hermesagent.MainActivity
import com.nousresearch.hermesagent.R
import org.json.JSONObject
import kotlin.math.absoluteValue

object HermesNotificationActionBridge {
    fun performNotificationJson(context: Context, payload: JSONObject): String {
        val action = normalizeAction(payload.optString("notification_action").ifBlank { "post" })
        if (action == "cancel") {
            val id = notificationId(payload)
            val tag = payload.optString("notification_tag").ifBlank { null }
            val cancelResult = runCatching {
                NotificationManagerCompat.from(context).cancel(tag, id)
            }
            if (cancelResult.isFailure) {
                return notificationErrorJson(
                    action = "notification_cancel",
                    message = "Unable to cancel Hermes notification",
                    throwable = cancelResult.exceptionOrNull(),
                )
            }
            return JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "notification_cancel")
                .put("notification_id", id)
                .put("notification_tag", tag ?: "")
                .put("message", "Cancelled Hermes notification")
                .toString()
        }
        if (action != "post") {
            return errorJson("Unsupported notification action: $action")
        }
        val title = payload.optString("title").take(MAX_NOTIFICATION_FIELD_CHARS)
        val text = payload.optString("text").take(MAX_NOTIFICATION_TEXT_CHARS)
        if (title.isBlank() && text.isBlank()) {
            return errorJson("notification task requires title or text")
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            return JSONObject()
                .put("success", false)
                .put("exit_code", 1)
                .put("action", "notification_post")
                .put("requires_permission", Manifest.permission.POST_NOTIFICATIONS)
                .put("message", "Grant notification permission before posting Hermes notifications.")
                .toString()
        }
        val channelId = payload.optString("channel_id").ifBlank { DEFAULT_CHANNEL_ID }.take(MAX_NOTIFICATION_FIELD_CHARS)
        ensureChannel(context, channelId, payload)
        val notificationId = notificationId(payload)
        val tag = payload.optString("notification_tag").ifBlank { null }?.take(MAX_NOTIFICATION_FIELD_CHARS)
        val builder = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle(title.ifBlank { context.getString(R.string.app_name) })
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(openAppPendingIntent(context))
            .setPriority(priority(payload.optString("priority")))
            .setOngoing(payload.optBoolean("ongoing", false))
            .setAutoCancel(payload.optBoolean("auto_cancel", true))
            .setOnlyAlertOnce(payload.optBoolean("only_alert_once", false))
            .setShowWhen(payload.optBoolean("show_when", true))
        val groupKey = payload.optString("group_key").ifBlank { "" }.take(MAX_NOTIFICATION_FIELD_CHARS)
        if (groupKey.isNotBlank()) {
            builder.setGroup(groupKey)
            builder.setGroupSummary(payload.optBoolean("group_summary", false))
        }
        val notifyResult = runCatching {
            NotificationManagerCompat.from(context).notify(tag, notificationId, builder.build())
        }
        if (notifyResult.isFailure) {
            return notificationErrorJson(
                action = "notification_post",
                message = "Unable to post Hermes notification",
                throwable = notifyResult.exceptionOrNull(),
            )
        }
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", "notification_post")
            .put("notification_id", notificationId)
            .put("notification_tag", tag ?: "")
            .put("channel_id", channelId)
            .put("group_key", groupKey)
            .put("message", "Posted Hermes notification")
            .toString()
    }

    fun normalizeAction(action: String): String {
        return when (action.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "", "post", "show", "notify", "update", "live_update" -> "post"
            "cancel", "clear", "dismiss", "remove" -> "cancel"
            else -> action.trim().lowercase().replace("-", "_").replace(" ", "_")
        }
    }

    private fun ensureChannel(context: Context, channelId: String, payload: JSONObject) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val manager = context.getSystemService(NotificationManager::class.java) ?: return
        if (manager.getNotificationChannel(channelId) != null) {
            return
        }
        val channelName = payload.optString("channel_name").ifBlank { "Hermes automation" }
            .take(MAX_NOTIFICATION_FIELD_CHARS)
        val channel = NotificationChannel(channelId, channelName, importance(payload.optString("importance")))
        manager.createNotificationChannel(channel)
    }

    private fun notificationId(payload: JSONObject): Int {
        val raw = payload.optString("notification_id").ifBlank { payload.optString("notify_id") }
        val numeric = raw.trim().toIntOrNull()
        if (numeric != null) {
            return numeric.toLong().absoluteValue
                .coerceAtMost(Int.MAX_VALUE.toLong())
                .toInt()
                .takeIf { it > 0 }
                ?: DEFAULT_NOTIFICATION_ID
        }
        val seed = listOf(
            payload.optString("notification_tag"),
            payload.optString("title"),
            payload.optString("text"),
        ).joinToString("|").ifBlank { DEFAULT_CHANNEL_ID }
        return seed.hashCode().toLong().absoluteValue
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()
            .takeIf { it > 0 }
            ?: DEFAULT_NOTIFICATION_ID
    }

    private fun openAppPendingIntent(context: Context): PendingIntent {
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun priority(raw: String): Int {
        return when (raw.trim().lowercase()) {
            "min" -> NotificationCompat.PRIORITY_MIN
            "low" -> NotificationCompat.PRIORITY_LOW
            "high" -> NotificationCompat.PRIORITY_HIGH
            "max", "critical" -> NotificationCompat.PRIORITY_MAX
            else -> NotificationCompat.PRIORITY_DEFAULT
        }
    }

    private fun importance(raw: String): Int {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return 0
        }
        return when (raw.trim().lowercase()) {
            "min" -> NotificationManager.IMPORTANCE_MIN
            "low" -> NotificationManager.IMPORTANCE_LOW
            "high" -> NotificationManager.IMPORTANCE_HIGH
            "max", "critical" -> NotificationManager.IMPORTANCE_MAX
            else -> NotificationManager.IMPORTANCE_DEFAULT
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("error", message)
            .toString()
    }

    private fun notificationErrorJson(action: String, message: String, throwable: Throwable?): String {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("action", action)
            .put("error", throwable?.message ?: message)
            .put("message", message)
            .toString()
    }

    private const val DEFAULT_CHANNEL_ID = "hermes_automation"
    private const val DEFAULT_NOTIFICATION_ID = 1001
    private const val MAX_NOTIFICATION_FIELD_CHARS = 120
    private const val MAX_NOTIFICATION_TEXT_CHARS = 2_000
}
