package com.mobilefork.hermesagent.device

import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Bundle
import org.json.JSONObject
import java.security.SecureRandom
import java.util.UUID

object HermesTaskerEventBridge {
    const val ACTION_EDIT_EVENT = "net.dinglisch.android.tasker.ACTION_EDIT_EVENT"
    const val ACTION_REQUEST_QUERY = "com.twofortyfouram.locale.intent.action.REQUEST_QUERY"
    const val EXTRA_ACTIVITY = "com.twofortyfouram.locale.intent.extra.ACTIVITY"
    const val EXTRA_BUNDLE = HermesTaskerPluginBridge.EXTRA_BUNDLE
    const val EXTRA_STRING_BLURB = HermesTaskerPluginBridge.EXTRA_STRING_BLURB
    const val EXTRA_VARIABLES = HermesTaskerConditionBridge.EXTRA_VARIABLES
    const val EXTRA_REQUEST_QUERY_PASS_THROUGH_DATA = "net.dinglisch.android.tasker.extras.PASS_THROUGH_DATA"
    const val PASS_THROUGH_BUNDLE_MESSAGE_ID_KEY = "net.dinglisch.android.tasker.MESSAGE_ID"

    const val KEY_EVENT_TYPE = "com.mobilefork.hermesagent.tasker.EVENT_TYPE"
    const val KEY_AUTOMATION_ID = "com.mobilefork.hermesagent.tasker.EVENT_AUTOMATION_ID"
    const val KEY_TOKEN = "com.mobilefork.hermesagent.tasker.EVENT_TOKEN"
    const val KEY_LABEL = "com.mobilefork.hermesagent.tasker.EVENT_LABEL"

    const val EVENT_AUTOMATION_FINISHED = "automation_finished"
    const val EVENT_AUTOMATION_SUCCEEDED = "automation_succeeded"
    const val EVENT_AUTOMATION_FAILED = "automation_failed"
    const val EVENT_SHIZUKU_AVAILABLE = "shizuku_available"
    const val EVENT_SHIZUKU_UNAVAILABLE = "shizuku_unavailable"

    const val PAYLOAD_EVENT_TYPE = "event_type"
    const val PAYLOAD_AUTOMATION_ID = "automation_id"
    const val PAYLOAD_AUTOMATION_LABEL = "automation_label"
    const val PAYLOAD_TRIGGER = "trigger"
    const val PAYLOAD_SUCCESS = "success"
    const val PAYLOAD_RESULT = "result"
    const val PAYLOAD_ERROR = "error"
    const val PAYLOAD_SHIZUKU_AVAILABLE = "shizuku_available"
    const val PAYLOAD_SHIZUKU_RUNNING = "shizuku_running"
    const val PAYLOAD_SHIZUKU_PERMISSION = "shizuku_permission"
    const val PAYLOAD_SHIZUKU_UID = "shizuku_uid"
    const val PAYLOAD_MESSAGE_ID = "message_id"
    const val PAYLOAD_EVENT_EPOCH_MS = "event_epoch_ms"

    data class EventChoice(val id: String, val label: String) {
        override fun toString(): String = label
    }

    data class EventResult(
        val resultCode: Int,
        val variables: Bundle,
        val error: String = "",
    )

    fun eventChoices(): List<EventChoice> = listOf(
        EventChoice(EVENT_AUTOMATION_FINISHED, "Automation finished"),
        EventChoice(EVENT_AUTOMATION_SUCCEEDED, "Automation succeeded"),
        EventChoice(EVENT_AUTOMATION_FAILED, "Automation failed"),
        EventChoice(EVENT_SHIZUKU_AVAILABLE, "Shizuku available"),
        EventChoice(EVENT_SHIZUKU_UNAVAILABLE, "Shizuku unavailable"),
    )

    fun bundleFromIntent(intent: Intent): Bundle? = intent.getBundleExtra(EXTRA_BUNDLE)

    fun isEventBundle(bundle: Bundle?): Boolean = bundle?.containsKey(KEY_EVENT_TYPE) == true

    fun buildResultIntent(
        context: Context,
        eventType: String,
        automationId: String = "",
        label: String = "",
        existingToken: String = "",
    ): Intent {
        val normalizedType = normalizeEventType(eventType)
            ?: throw IllegalArgumentException("Choose a supported Hermes event")
        val normalizedAutomationId = normalizeOptionalAutomationId(automationId)
        val signature = eventSignature(normalizedType, normalizedAutomationId)
        val token = ensureAuthorizedToken(context, signature, existingToken)
        val blurb = blurbFor(normalizedType, normalizedAutomationId, label)
        val bundle = Bundle().apply {
            putString(KEY_EVENT_TYPE, normalizedType)
            putString(KEY_AUTOMATION_ID, normalizedAutomationId)
            putString(KEY_TOKEN, token)
            putString(KEY_LABEL, label.take(MAX_LABEL_CHARS))
        }
        return Intent()
            .putExtra(EXTRA_BUNDLE, bundle)
            .putExtra(EXTRA_STRING_BLURB, blurb)
    }

    fun notifyAutomationFinished(
        context: Context,
        record: HermesAutomationRecord,
        trigger: String,
        success: Boolean,
        resultText: String,
    ) {
        val payload = JSONObject()
            .put(PAYLOAD_EVENT_TYPE, EVENT_AUTOMATION_FINISHED)
            .put(PAYLOAD_AUTOMATION_ID, record.id.take(MAX_EVENT_VALUE_CHARS))
            .put(PAYLOAD_AUTOMATION_LABEL, record.label.take(MAX_EVENT_VALUE_CHARS))
            .put(PAYLOAD_TRIGGER, trigger.take(MAX_EVENT_VALUE_CHARS))
            .put(PAYLOAD_SUCCESS, success)
            .put(PAYLOAD_RESULT, resultText.take(MAX_VARIABLE_RETURN_CHARS))
        requestQuery(context.applicationContext, payload)
    }

    fun notifyShizukuState(
        context: Context,
        available: Boolean,
        status: HermesPrivilegedAccessStatus,
    ) {
        val payload = JSONObject()
            .put(PAYLOAD_EVENT_TYPE, if (available) EVENT_SHIZUKU_AVAILABLE else EVENT_SHIZUKU_UNAVAILABLE)
            .put(PAYLOAD_SHIZUKU_AVAILABLE, available)
            .put(PAYLOAD_SHIZUKU_RUNNING, status.shizukuBinderAlive)
            .put(PAYLOAD_SHIZUKU_PERMISSION, status.shizukuPermissionGranted)
            .put(PAYLOAD_SHIZUKU_UID, status.shizukuUid ?: JSONObject.NULL)
        requestQuery(context.applicationContext, payload)
    }

    fun buildRequestQueryIntent(context: Context, payload: JSONObject): Intent {
        val now = System.currentTimeMillis()
        val messageId = nextMessageId()
        val storedPayload = JSONObject(payload.toString())
            .put(PAYLOAD_MESSAGE_ID, messageId)
            .put(PAYLOAD_EVENT_EPOCH_MS, now)
        storeEventPayload(context.applicationContext, messageId, storedPayload, now)
        return Intent(ACTION_REQUEST_QUERY)
            .addFlags(Intent.FLAG_RECEIVER_FOREGROUND)
            .putExtra(EXTRA_ACTIVITY, HermesTaskerEventEditActivity::class.java.name)
            .putExtra(EXTRA_REQUEST_QUERY_PASS_THROUGH_DATA, bundleForPayload(storedPayload))
    }

    fun queryEvent(context: Context, hostIntent: Intent?, bundle: Bundle?): EventResult {
        if (bundle == null) {
            return unknown("Hermes event query is missing Locale EXTRA_BUNDLE")
        }
        val eventType = normalizeEventType(bundle.getString(KEY_EVENT_TYPE).orEmpty())
            ?: return unknown("Hermes event query has an unsupported event type")
        val automationId = runCatching {
            normalizeOptionalAutomationId(bundle.getString(KEY_AUTOMATION_ID).orEmpty())
        }.getOrElse {
            return unknown("Hermes event query has an invalid automation ID")
        }
        val token = bundle.getString(KEY_TOKEN).orEmpty().trim()
        if (!isAuthorizedToken(context, eventSignature(eventType, automationId), token)) {
            return unknown("Hermes event token is missing or invalid")
        }
        val messageId = retrieveMessageId(hostIntent) ?: return unknown("Hermes event update is missing a verified message id")
        val payload = readEventPayload(context.applicationContext, messageId)
            ?: return unknown("Hermes event update is unknown or expired")
        val variables = variablesForPayload(eventType, automationId, payload)
        val satisfied = eventMatches(eventType, automationId, payload)
        variables.putString("%hermes_satisfied", satisfied.toString())
        return EventResult(
            resultCode = if (satisfied) {
                HermesTaskerConditionBridge.RESULT_CONDITION_SATISFIED
            } else {
                HermesTaskerConditionBridge.RESULT_CONDITION_UNSATISFIED
            },
            variables = variables,
        )
    }

    fun blurbFor(eventType: String, automationId: String, label: String): String {
        val custom = label.trim()
        if (custom.isNotBlank()) {
            return custom.take(MAX_LABEL_CHARS)
        }
        val suffix = automationId.takeIf { it.isNotBlank() }?.let { ": $it" }.orEmpty()
        return when (eventType) {
            EVENT_AUTOMATION_FINISHED -> "Hermes event: automation finished$suffix"
            EVENT_AUTOMATION_SUCCEEDED -> "Hermes event: automation succeeded$suffix"
            EVENT_AUTOMATION_FAILED -> "Hermes event: automation failed$suffix"
            EVENT_SHIZUKU_AVAILABLE -> "Hermes event: Shizuku available"
            EVENT_SHIZUKU_UNAVAILABLE -> "Hermes event: Shizuku unavailable"
            else -> "Hermes event"
        }.take(MAX_LABEL_CHARS)
    }

    private fun requestQuery(context: Context, payload: JSONObject) {
        val intentRequest = buildRequestQueryIntent(context, payload)
        val handledPackages = runCatching {
            requestQueryThroughServices(context, intentRequest)
        }.getOrDefault(emptyList())
        requestQueryThroughBroadcasts(context, intentRequest, handledPackages)
    }

    private fun requestQueryThroughServices(context: Context, intentRequest: Intent): List<String> {
        val result = mutableListOf<String>()
        val packageManager = context.packageManager
        val services = packageManager.queryIntentServices(Intent(ACTION_REQUEST_QUERY), 0)
        services.forEach { resolveInfo ->
            val serviceInfo = resolveInfo.serviceInfo ?: return@forEach
            val component = ComponentName(serviceInfo.packageName, serviceInfo.name)
            val serviceIntent = Intent(intentRequest).setComponent(component)
            if (runCatching { context.startService(serviceIntent) }.isSuccess) {
                result.add(serviceInfo.packageName)
            }
        }
        return result
    }

    private fun requestQueryThroughBroadcasts(context: Context, intentRequest: Intent, ignorePackages: List<String>) {
        if (ignorePackages.isEmpty()) {
            context.sendBroadcast(Intent(intentRequest).setComponent(null))
            return
        }
        val packageManager = context.packageManager
        val receivers = packageManager.queryBroadcastReceivers(Intent(ACTION_REQUEST_QUERY), 0)
        receivers.forEach { resolveInfo ->
            val activityInfo = resolveInfo.activityInfo ?: return@forEach
            if (activityInfo.packageName in ignorePackages) {
                return@forEach
            }
            val component = ComponentName(activityInfo.packageName, activityInfo.name)
            context.sendBroadcast(Intent(intentRequest).setComponent(component))
        }
    }

    private fun variablesForPayload(eventType: String, automationId: String, payload: JSONObject): Bundle {
        val payloadEventType = payload.optString(PAYLOAD_EVENT_TYPE)
        val payloadAutomationId = payload.optString(PAYLOAD_AUTOMATION_ID)
        val success = payload.optBoolean(PAYLOAD_SUCCESS, false)
        return Bundle().apply {
            putString("%hermes_event", eventType)
            putString("%hermes_update_event", payloadEventType)
            putString("%hermes_automation_id", automationId.ifBlank { payloadAutomationId })
            putString("%hermes_update_automation_id", payloadAutomationId)
            putString("%hermes_automation_label", payload.optString(PAYLOAD_AUTOMATION_LABEL).take(MAX_VARIABLE_RETURN_CHARS))
            putString("%hermes_trigger", payload.optString(PAYLOAD_TRIGGER).take(MAX_VARIABLE_RETURN_CHARS))
            putString("%hermes_success", success.toString())
            putString("%hermes_result", payload.optString(PAYLOAD_RESULT).take(MAX_VARIABLE_RETURN_CHARS))
            putString("%hermes_error", payload.optString(PAYLOAD_ERROR).take(MAX_VARIABLE_RETURN_CHARS))
            putString("%hermes_shizuku_available", payload.optString(PAYLOAD_SHIZUKU_AVAILABLE))
            putString("%hermes_shizuku_running", payload.optString(PAYLOAD_SHIZUKU_RUNNING))
            putString("%hermes_shizuku_permission", payload.optString(PAYLOAD_SHIZUKU_PERMISSION))
            putString("%hermes_shizuku_uid", payload.optString(PAYLOAD_SHIZUKU_UID))
            putString("%hermes_event_epoch_ms", payload.optLong(PAYLOAD_EVENT_EPOCH_MS, 0L).toString())
        }
    }

    private fun eventMatches(eventType: String, automationId: String, payload: JSONObject): Boolean {
        val payloadEventType = payload.optString(PAYLOAD_EVENT_TYPE)
        val payloadAutomationId = payload.optString(PAYLOAD_AUTOMATION_ID)
        if (automationId.isNotBlank() && automationId != payloadAutomationId) {
            return false
        }
        return when (eventType) {
            EVENT_AUTOMATION_FINISHED -> payloadEventType == EVENT_AUTOMATION_FINISHED
            EVENT_AUTOMATION_SUCCEEDED -> payloadEventType == EVENT_AUTOMATION_FINISHED &&
                payload.optBoolean(PAYLOAD_SUCCESS, false)
            EVENT_AUTOMATION_FAILED -> payloadEventType == EVENT_AUTOMATION_FINISHED &&
                !payload.optBoolean(PAYLOAD_SUCCESS, false)
            EVENT_SHIZUKU_AVAILABLE -> payloadEventType == EVENT_SHIZUKU_AVAILABLE
            EVENT_SHIZUKU_UNAVAILABLE -> payloadEventType == EVENT_SHIZUKU_UNAVAILABLE
            else -> false
        }
    }

    private fun bundleForPayload(payload: JSONObject): Bundle {
        return Bundle().apply {
            putInt(PASS_THROUGH_BUNDLE_MESSAGE_ID_KEY, payload.optInt(PAYLOAD_MESSAGE_ID))
            putString(PAYLOAD_EVENT_TYPE, payload.optString(PAYLOAD_EVENT_TYPE))
            putString(PAYLOAD_AUTOMATION_ID, payload.optString(PAYLOAD_AUTOMATION_ID))
            putString(PAYLOAD_TRIGGER, payload.optString(PAYLOAD_TRIGGER))
            putString(PAYLOAD_SUCCESS, payload.optString(PAYLOAD_SUCCESS))
            putString(PAYLOAD_EVENT_EPOCH_MS, payload.optString(PAYLOAD_EVENT_EPOCH_MS))
        }
    }

    private fun retrieveMessageId(intent: Intent?): Int? {
        val passThrough = intent?.getBundleExtra(EXTRA_REQUEST_QUERY_PASS_THROUGH_DATA) ?: return null
        val id = passThrough.getInt(PASS_THROUGH_BUNDLE_MESSAGE_ID_KEY, -1)
        return id.takeIf { it > 0 }
    }

    private fun storeEventPayload(context: Context, messageId: Int, payload: JSONObject, now: Long) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val editor = prefs.edit().putString(messageKey(messageId), payload.toString())
        prefs.all.forEach { (key, value) ->
            if (!key.startsWith(MESSAGE_PREFIX) || value !is String) {
                return@forEach
            }
            val eventTime = runCatching { JSONObject(value).optLong(PAYLOAD_EVENT_EPOCH_MS, 0L) }.getOrDefault(0L)
            if (eventTime <= 0L || now - eventTime > EVENT_TTL_MS) {
                editor.remove(key)
            }
        }
        editor.apply()
    }

    private fun readEventPayload(context: Context, messageId: Int): JSONObject? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(messageKey(messageId), null) ?: return null
        val payload = runCatching { JSONObject(raw) }.getOrNull() ?: return null
        val eventTime = payload.optLong(PAYLOAD_EVENT_EPOCH_MS, 0L)
        if (eventTime <= 0L || System.currentTimeMillis() - eventTime > EVENT_TTL_MS) {
            prefs.edit().remove(messageKey(messageId)).apply()
            return null
        }
        return payload
    }

    private fun ensureAuthorizedToken(context: Context, signature: String, existingToken: String = ""): String {
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val reusable = existingToken.trim().takeIf { token ->
            token.length in 16..128 && prefs.getString(tokenKey(token), "") == signature
        }
        val token = reusable ?: UUID.randomUUID().toString()
        prefs.edit()
            .putString(tokenKey(token), signature)
            .apply()
        return token
    }

    private fun isAuthorizedToken(context: Context, signature: String, token: String): Boolean {
        if (token.length !in 16..128) {
            return false
        }
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(tokenKey(token), "") == signature
    }

    private fun normalizeEventType(raw: String): String? {
        val normalized = raw.trim().lowercase().replace('-', '_').replace(' ', '_')
        return EVENT_SYNONYMS[normalized] ?: normalized.takeIf { it in SUPPORTED_EVENTS }
    }

    private fun normalizeOptionalAutomationId(raw: String): String {
        val id = raw.trim()
        if (id.isBlank()) {
            return ""
        }
        require(id.length <= MAX_AUTOMATION_ID_CHARS && id.indexOf('\u0000') < 0) {
            "Hermes automation ID is too long or invalid"
        }
        return id
    }

    private fun eventSignature(eventType: String, automationId: String): String {
        return JSONObject()
            .put("event_type", eventType)
            .put("automation_id", automationId)
            .toString()
    }

    private fun unknown(message: String): EventResult {
        return EventResult(
            resultCode = HermesTaskerConditionBridge.RESULT_CONDITION_UNKNOWN,
            variables = Bundle().apply {
                putString("%hermes_satisfied", "unknown")
                putString("%hermes_error", message.take(MAX_VARIABLE_RETURN_CHARS))
            },
            error = message,
        )
    }

    private fun nextMessageId(): Int = RANDOM.nextInt(Int.MAX_VALUE - 1) + 1

    private fun tokenKey(token: String): String = "tasker_event_token:$token"

    private fun messageKey(messageId: Int): String = "$MESSAGE_PREFIX$messageId"

    private const val PREFS_NAME = "hermes_tasker_event_tokens"
    private const val MESSAGE_PREFIX = "tasker_event_message:"
    private const val MAX_AUTOMATION_ID_CHARS = 128
    private const val MAX_LABEL_CHARS = 80
    private const val MAX_EVENT_VALUE_CHARS = 256
    private const val MAX_VARIABLE_RETURN_CHARS = 512
    private const val EVENT_TTL_MS = 10 * 60 * 1000L

    private val RANDOM = SecureRandom()
    private val SUPPORTED_EVENTS = setOf(
        EVENT_AUTOMATION_FINISHED,
        EVENT_AUTOMATION_SUCCEEDED,
        EVENT_AUTOMATION_FAILED,
        EVENT_SHIZUKU_AVAILABLE,
        EVENT_SHIZUKU_UNAVAILABLE,
    )
    private val EVENT_SYNONYMS = mapOf(
        "finished" to EVENT_AUTOMATION_FINISHED,
        "automation_complete" to EVENT_AUTOMATION_FINISHED,
        "automation_completed" to EVENT_AUTOMATION_FINISHED,
        "success" to EVENT_AUTOMATION_SUCCEEDED,
        "succeeded" to EVENT_AUTOMATION_SUCCEEDED,
        "automation_success" to EVENT_AUTOMATION_SUCCEEDED,
        "failure" to EVENT_AUTOMATION_FAILED,
        "failed" to EVENT_AUTOMATION_FAILED,
        "automation_failure" to EVENT_AUTOMATION_FAILED,
        "shizuku" to EVENT_SHIZUKU_AVAILABLE,
        "shizuku_ready" to EVENT_SHIZUKU_AVAILABLE,
        "shizuku_running" to EVENT_SHIZUKU_AVAILABLE,
        "shizuku_missing" to EVENT_SHIZUKU_UNAVAILABLE,
        "shizuku_not_running" to EVENT_SHIZUKU_UNAVAILABLE,
    )
}
