package com.nousresearch.hermesagent.device

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONObject

class HermesExternalTriggerReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_EXTERNAL_TRIGGER) {
            return
        }
        val arguments = argumentsFromIntent(intent)
        if (arguments.optString("trigger_id").isBlank() || arguments.optString("external_token").isBlank()) {
            return
        }
        val pendingResult = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                HermesAutomationBridge.runExternalTriggerJson(context.applicationContext, arguments)
            } finally {
                pendingResult.finish()
            }
        }
    }

    private fun argumentsFromIntent(intent: Intent): JSONObject {
        val extrasJson = JSONObject()
        intent.extras?.keySet()?.sorted()?.forEach { key ->
            if (key !in RESERVED_EXTRA_KEYS) {
                putPrimitiveExtra(extrasJson, key, intent.extras?.get(key))
            }
        }
        val senderPackage = if (Build.VERSION.SDK_INT >= 34) sentFromPackage.orEmpty() else ""
        return JSONObject()
            .put("trigger_id", firstExtra(intent, "trigger_id", EXTRA_TRIGGER_ID, "external_trigger_id", "event_id"))
            .put("external_token", firstExtra(intent, "external_token", EXTRA_TOKEN, "trigger_token", "token", "auth_token"))
            .put("trigger_package_name", senderPackage.ifBlank {
                firstExtra(intent, "trigger_package_name", EXTRA_PACKAGE_NAME, "package_name", "caller_package")
            })
            .put("referrer", firstExtra(intent, "referrer", EXTRA_REFERRER, "source", "caller"))
            .put("extras", extrasJson)
    }

    private fun firstExtra(intent: Intent, vararg keys: String): String {
        return keys.firstNotNullOfOrNull { key ->
            intent.getStringExtra(key)?.takeIf { it.isNotBlank() }
        }.orEmpty()
    }

    private fun putPrimitiveExtra(target: JSONObject, key: String, value: Any?) {
        when (value) {
            null -> return
            is String, is Number, is Boolean -> target.put(key, value)
            else -> target.put(key, value.toString())
        }
    }

    companion object {
        const val ACTION_EXTERNAL_TRIGGER = "com.nousresearch.hermesagent.EXTERNAL_TRIGGER"
        const val EXTRA_TRIGGER_ID = "com.nousresearch.hermesagent.extra.TRIGGER_ID"
        const val EXTRA_TOKEN = "com.nousresearch.hermesagent.extra.TOKEN"
        const val EXTRA_PACKAGE_NAME = "com.nousresearch.hermesagent.extra.PACKAGE_NAME"
        const val EXTRA_REFERRER = "com.nousresearch.hermesagent.extra.REFERRER"

        private val RESERVED_EXTRA_KEYS = setOf(
            "trigger_id",
            "external_trigger_id",
            "event_id",
            EXTRA_TRIGGER_ID,
            "external_token",
            "trigger_token",
            "token",
            "auth_token",
            EXTRA_TOKEN,
            "trigger_package_name",
            "package_name",
            "caller_package",
            EXTRA_PACKAGE_NAME,
            "referrer",
            "source",
            "caller",
            EXTRA_REFERRER,
        )
    }
}
