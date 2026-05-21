package com.mobilefork.hermesagent.device

import android.content.Context
import android.content.Intent
import android.os.Bundle
import org.json.JSONObject
import java.util.UUID

object HermesTaskerPluginBridge {
    const val ACTION_EDIT_SETTING = "com.twofortyfouram.locale.intent.action.EDIT_SETTING"
    const val ACTION_FIRE_SETTING = "com.twofortyfouram.locale.intent.action.FIRE_SETTING"
    const val EXTRA_BUNDLE = "com.twofortyfouram.locale.intent.extra.BUNDLE"
    const val EXTRA_STRING_BLURB = "com.twofortyfouram.locale.intent.extra.BLURB"
    const val KEY_AUTOMATION_ID = "com.mobilefork.hermesagent.tasker.AUTOMATION_ID"
    const val KEY_TOKEN = "com.mobilefork.hermesagent.tasker.TOKEN"
    const val KEY_LABEL = "com.mobilefork.hermesagent.tasker.LABEL"
    const val TRIGGER_TASKER_PLUGIN = "tasker_plugin"

    fun bundleFromIntent(intent: Intent): Bundle? = intent.getBundleExtra(EXTRA_BUNDLE)

    fun buildResultIntent(
        context: Context,
        automationId: String,
        label: String,
        existingToken: String = "",
    ): Intent {
        val normalizedId = normalizeAutomationId(automationId)
            ?: throw IllegalArgumentException("Tasker plugin automation id is required")
        val token = ensureAuthorizedToken(context, normalizedId, existingToken)
        val blurb = blurbFor(normalizedId, label)
        val bundle = Bundle().apply {
            putString(KEY_AUTOMATION_ID, normalizedId)
            putString(KEY_TOKEN, token)
            putString(KEY_LABEL, label.take(MAX_LABEL_CHARS))
        }
        return Intent()
            .putExtra(EXTRA_BUNDLE, bundle)
            .putExtra(EXTRA_STRING_BLURB, blurb)
    }

    fun runPluginBundleJson(context: Context, bundle: Bundle?, senderPackage: String = ""): String {
        if (bundle == null) {
            return errorJson("Tasker plugin fire request is missing Locale EXTRA_BUNDLE")
        }
        val automationId = normalizeAutomationId(bundle.getString(KEY_AUTOMATION_ID).orEmpty())
            ?: return errorJson("Tasker plugin fire request is missing a Hermes automation id")
        val token = bundle.getString(KEY_TOKEN).orEmpty().trim()
        if (!isAuthorizedToken(context, automationId, token)) {
            return errorJson("Tasker plugin token is missing or invalid for automation $automationId")
        }
        val appContext = context.applicationContext
        val store = HermesAutomationStore(appContext)
        store.setVariable("TASKER_PLUGIN_AUTOMATION_ID", automationId.take(MAX_EVENT_VALUE_CHARS))
        store.setVariable("TASKER_PLUGIN_CALLER_PACKAGE", senderPackage.take(MAX_EVENT_VALUE_CHARS))
        return JSONObject(HermesAutomationBridge.runAutomationJson(appContext, automationId, TRIGGER_TASKER_PLUGIN))
            .put("tasker_plugin", true)
            .put("trigger", TRIGGER_TASKER_PLUGIN)
            .toString()
    }

    fun blurbFor(automationId: String, label: String): String {
        val display = label.trim().ifBlank { automationId.trim() }.take(MAX_LABEL_CHARS)
        return "Run Hermes automation: $display"
    }

    fun tokenForAutomation(context: Context, automationId: String): String {
        val normalizedId = normalizeAutomationId(automationId) ?: return ""
        return ensureAuthorizedToken(context, normalizedId)
    }

    private fun ensureAuthorizedToken(context: Context, automationId: String, existingToken: String = ""): String {
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val reusable = existingToken.trim().takeIf { token ->
            token.length in 16..128 && prefs.getString(tokenKey(token), "") == automationId
        }
        val token = reusable ?: UUID.randomUUID().toString()
        prefs.edit()
            .putString(tokenKey(token), automationId)
            .apply()
        return token
    }

    private fun isAuthorizedToken(context: Context, automationId: String, token: String): Boolean {
        if (token.length !in 16..128) {
            return false
        }
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(tokenKey(token), "") == automationId
    }

    private fun normalizeAutomationId(raw: String): String? {
        val id = raw.trim()
        if (id.isBlank() || id.length > MAX_AUTOMATION_ID_CHARS || id.indexOf('\u0000') >= 0) {
            return null
        }
        return id
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .toString()
    }

    private fun tokenKey(token: String): String = "tasker_token:$token"

    private const val PREFS_NAME = "hermes_tasker_plugin_tokens"
    private const val MAX_AUTOMATION_ID_CHARS = 128
    private const val MAX_LABEL_CHARS = 80
    private const val MAX_EVENT_VALUE_CHARS = 256
}
