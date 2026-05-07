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
    const val ACTION_NOTIFICATION_BUTTON = "com.nousresearch.hermesagent.NOTIFICATION_BUTTON"
    const val EXTRA_BUTTON_ACTION = "button_action"
    const val EXTRA_AUTOMATION_ID = "automation_id"
    const val EXTRA_DISMISS_ON_TAP = "dismiss_on_tap"
    const val EXTRA_NOTIFICATION_ID = "notification_id"
    const val EXTRA_NOTIFICATION_TAG = "notification_tag"

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
        val buttonResult = parseNotificationButtons(payload)
        if (buttonResult.error != null) {
            return errorJson(buttonResult.error)
        }
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
        buttonResult.buttons.forEachIndexed { index, button ->
            builder.addAction(
                R.drawable.ic_nav_hermes,
                button.title,
                pendingIntentForButton(context, notificationId, tag, index, button),
            )
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
            .put("notification_button_count", buttonResult.buttons.size)
            .put("message", "Posted Hermes notification")
            .toString()
    }

    fun handleNotificationButtonIntentJson(context: Context, intent: Intent): String {
        if (intent.action != ACTION_NOTIFICATION_BUTTON) {
            return errorJson("Unsupported notification button intent: ${intent.action.orEmpty()}")
        }
        val buttonAction = normalizeButtonAction(intent.getStringExtra(EXTRA_BUTTON_ACTION).orEmpty())
        val notificationId = intent.getIntExtra(EXTRA_NOTIFICATION_ID, DEFAULT_NOTIFICATION_ID)
        val tag = intent.getStringExtra(EXTRA_NOTIFICATION_TAG)?.ifBlank { null }
        val dismissOnTap = intent.getBooleanExtra(EXTRA_DISMISS_ON_TAP, buttonAction == BUTTON_ACTION_CANCEL)
        val appContext = context.applicationContext
        val result = when (buttonAction) {
            BUTTON_ACTION_OPEN_APP -> openHermesFromButton(appContext)
            BUTTON_ACTION_CANCEL -> JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "notification_button_cancel")
                .put("message", "Dismissed Hermes notification")
            BUTTON_ACTION_RUN_AUTOMATION -> {
                val automationId = intent.getStringExtra(EXTRA_AUTOMATION_ID).orEmpty()
                if (automationId.isBlank()) {
                    return errorJson("notification run_automation button requires automation_id")
                }
                JSONObject(HermesAutomationBridge.runAutomationJson(appContext, automationId, "notification_button"))
                    .put("automation_id", automationId)
            }
            else -> return errorJson("Unsupported notification button action: $buttonAction")
        }
        if (dismissOnTap || buttonAction == BUTTON_ACTION_CANCEL) {
            dismissNotification(appContext, notificationId, tag)
        }
        return result
            .put("notification_button_action", buttonAction)
            .put("notification_id", notificationId)
            .put("notification_tag", tag ?: "")
            .toString()
    }

    fun normalizeAction(action: String): String {
        return when (action.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "", "post", "show", "notify", "update", "live_update" -> "post"
            "cancel", "clear", "dismiss", "remove" -> "cancel"
            else -> action.trim().lowercase().replace("-", "_").replace(" ", "_")
        }
    }

    fun normalizeButtonAction(action: String): String {
        val normalized = action.trim().lowercase().replace("-", "_").replace(" ", "_")
        return when (normalized) {
            "", "open", "open_app", "launch", "launch_app", "hermes", "open_hermes" -> BUTTON_ACTION_OPEN_APP
            "cancel", "clear", "dismiss", "remove" -> BUTTON_ACTION_CANCEL
            "run", "run_task", "run_automation", "automation", "perform_task", "run_saved_automation" -> BUTTON_ACTION_RUN_AUTOMATION
            else -> normalized
        }
    }

    fun isSupportedButtonAction(action: String): Boolean {
        return normalizeButtonAction(action) in SUPPORTED_BUTTON_ACTIONS
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

    private fun parseNotificationButtons(payload: JSONObject): NotificationButtonParseResult {
        val rawButtons = payload.optJSONArray("notification_buttons")
            ?: payload.optJSONArray("buttons")
            ?: return NotificationButtonParseResult(emptyList())
        if (rawButtons.length() > MAX_NOTIFICATION_BUTTONS) {
            return NotificationButtonParseResult(error = "notification_buttons supports at most $MAX_NOTIFICATION_BUTTONS buttons")
        }
        val buttons = mutableListOf<NotificationButtonSpec>()
        for (index in 0 until rawButtons.length()) {
            val raw = rawButtons.optJSONObject(index)
                ?: return NotificationButtonParseResult(error = "notification_buttons[$index] must be a JSON object")
            val title = raw.optString("title")
                .ifBlank { raw.optString("label") }
                .ifBlank { raw.optString("text") }
                .take(MAX_NOTIFICATION_FIELD_CHARS)
            if (title.isBlank()) {
                return NotificationButtonParseResult(error = "notification_buttons[$index] requires title")
            }
            val buttonAction = normalizeButtonAction(
                raw.optString("action")
                    .ifBlank { raw.optString("button_action") }
                    .ifBlank { raw.optString("type") }
                    .ifBlank { BUTTON_ACTION_OPEN_APP },
            )
            if (buttonAction !in SUPPORTED_BUTTON_ACTIONS) {
                return NotificationButtonParseResult(error = "Unsupported notification button action: $buttonAction")
            }
            val automationId = raw.optString("automation_id")
                .ifBlank { raw.optString("automation") }
                .ifBlank { raw.optString("id") }
                .take(MAX_NOTIFICATION_AUTOMATION_ID_CHARS)
            if (buttonAction == BUTTON_ACTION_RUN_AUTOMATION && automationId.isBlank()) {
                return NotificationButtonParseResult(error = "notification run_automation button requires automation_id")
            }
            if (listOf(title, buttonAction, automationId).any { it.indexOf('\u0000') >= 0 }) {
                return NotificationButtonParseResult(error = "notification button fields must not contain NUL bytes")
            }
            buttons += NotificationButtonSpec(
                title = title,
                action = buttonAction,
                automationId = automationId,
                dismissOnTap = raw.optBoolean("dismiss_on_tap", raw.optBoolean("cancel_notification", false)),
            )
        }
        return NotificationButtonParseResult(buttons)
    }

    private fun pendingIntentForButton(
        context: Context,
        notificationId: Int,
        tag: String?,
        index: Int,
        button: NotificationButtonSpec,
    ): PendingIntent {
        if (button.action == BUTTON_ACTION_OPEN_APP) {
            return openAppPendingIntent(context, notificationButtonRequestCode(notificationId, tag, index, button))
        }
        val intent = Intent(context, HermesAutomationReceiver::class.java).apply {
            action = ACTION_NOTIFICATION_BUTTON
            putExtra(EXTRA_BUTTON_ACTION, button.action)
            putExtra(EXTRA_NOTIFICATION_ID, notificationId)
            putExtra(EXTRA_NOTIFICATION_TAG, tag.orEmpty())
            putExtra(EXTRA_AUTOMATION_ID, button.automationId)
            putExtra(EXTRA_DISMISS_ON_TAP, button.dismissOnTap)
        }
        return PendingIntent.getBroadcast(
            context,
            notificationButtonRequestCode(notificationId, tag, index, button),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun notificationButtonRequestCode(
        notificationId: Int,
        tag: String?,
        index: Int,
        button: NotificationButtonSpec,
    ): Int {
        val seed = listOf(notificationId.toString(), tag.orEmpty(), index.toString(), button.title, button.action, button.automationId)
            .joinToString("|")
        return seed.hashCode().toLong().absoluteValue
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()
            .takeIf { it > 0 }
            ?: (DEFAULT_NOTIFICATION_ID + index + 1)
    }

    private fun openAppPendingIntent(context: Context, requestCode: Int = 0): PendingIntent {
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            context,
            requestCode,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun openHermesFromButton(context: Context): JSONObject {
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val started = runCatching { context.startActivity(intent) }
        if (started.isFailure) {
            return JSONObject()
                .put("success", false)
                .put("exit_code", 1)
                .put("action", "notification_button_open_app")
                .put("error", started.exceptionOrNull()?.message ?: "Unable to open Hermes")
        }
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", "notification_button_open_app")
            .put("message", "Opened Hermes")
    }

    private fun dismissNotification(context: Context, id: Int, tag: String?) {
        runCatching { NotificationManagerCompat.from(context).cancel(tag, id) }
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
    private const val BUTTON_ACTION_OPEN_APP = "open_app"
    private const val BUTTON_ACTION_CANCEL = "cancel"
    private const val BUTTON_ACTION_RUN_AUTOMATION = "run_automation"
    private const val MAX_NOTIFICATION_BUTTONS = 3
    private const val MAX_NOTIFICATION_FIELD_CHARS = 120
    private const val MAX_NOTIFICATION_TEXT_CHARS = 2_000
    private const val MAX_NOTIFICATION_AUTOMATION_ID_CHARS = 120
    private val SUPPORTED_BUTTON_ACTIONS = setOf(
        BUTTON_ACTION_OPEN_APP,
        BUTTON_ACTION_CANCEL,
        BUTTON_ACTION_RUN_AUTOMATION,
    )

    private data class NotificationButtonSpec(
        val title: String,
        val action: String,
        val automationId: String,
        val dismissOnTap: Boolean,
    )

    private data class NotificationButtonParseResult(
        val buttons: List<NotificationButtonSpec> = emptyList(),
        val error: String? = null,
    )
}
