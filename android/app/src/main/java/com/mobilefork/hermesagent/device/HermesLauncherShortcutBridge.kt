package com.mobilefork.hermesagent.device

import android.content.Context
import android.content.Intent
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.drawable.Icon
import android.os.Build
import com.mobilefork.hermesagent.MainActivity
import com.mobilefork.hermesagent.R
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest

object HermesLauncherShortcutBridge {
    const val ACTION_RUN_AUTOMATION_SHORTCUT = "com.mobilefork.hermesagent.RUN_AUTOMATION_SHORTCUT"
    const val EXTRA_AUTOMATION_ID = "com.mobilefork.hermesagent.extra.AUTOMATION_ID"

    fun createShortcutJson(context: Context, arguments: JSONObject): String {
        val appContext = context.applicationContext
        val record = recordFromArguments(appContext, arguments)
            ?: return errorJson("create_launcher_shortcut requires an existing automation_id or id")
        val label = stringArgument(arguments, "label", "shortcut_label", "title", allowEmpty = true)
            ?.ifBlank { record.label }
            ?: record.label
        if (label.indexOf('\u0000') >= 0) {
            return errorJson("launcher shortcut label must not contain NUL bytes")
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N_MR1) {
            return JSONObject()
                .put("success", false)
                .put("requires_api", Build.VERSION_CODES.N_MR1)
                .put("current_api", Build.VERSION.SDK_INT)
                .put("message", "Launcher shortcuts require Android 7.1 or newer.")
                .toString()
        }
        val manager = shortcutManager(appContext)
            ?: return errorJson("Launcher shortcut manager is not available on this device")
        val shortcutId = shortcutIdForAutomation(record.id)
        val shortcut = shortcutInfo(appContext, shortcutId, record.id, label, record.label)
        val createDynamic = arguments.optBoolean("dynamic", arguments.optBoolean("dynamic_shortcut", true))
        val requestPin = arguments.optBoolean("pin", arguments.optBoolean("request_pin", true))
        var dynamicCreated = false
        var pinStarted = false
        var dynamicError = ""
        if (createDynamic) {
            val dynamicResult = runCatching {
                val existing = manager.dynamicShortcuts.any { it.id == shortcutId }
                if (existing) {
                    manager.updateShortcuts(listOf(shortcut))
                } else {
                    manager.addDynamicShortcuts(listOf(shortcut))
                }
            }
            dynamicCreated = dynamicResult.getOrDefault(false)
            dynamicError = dynamicResult.exceptionOrNull()?.message.orEmpty()
        }
        val pinSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && manager.isRequestPinShortcutSupported
        if (requestPin && pinSupported) {
            val pinResult = runCatching { manager.requestPinShortcut(shortcut, null) }
            pinStarted = pinResult.getOrDefault(false)
        }
        return JSONObject()
            .put("success", dynamicCreated || pinStarted)
            .put("action", "create_launcher_shortcut")
            .put("automation_id", record.id)
            .put("shortcut_id", shortcutId)
            .put("label", label.take(MAX_SHORTCUT_LABEL_CHARS))
            .put("dynamic_shortcut_requested", createDynamic)
            .put("dynamic_shortcut_created", dynamicCreated)
            .put("dynamic_shortcut_error", dynamicError)
            .put("pin_requested", requestPin)
            .put("pin_request_supported", pinSupported)
            .put("pin_request_started", pinStarted)
            .put("message", shortcutMessage(dynamicCreated, pinStarted, pinSupported, requestPin))
            .toString()
    }

    fun listShortcutsJson(context: Context): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N_MR1) {
            return JSONObject()
                .put("success", true)
                .put("supported", false)
                .put("current_api", Build.VERSION.SDK_INT)
                .put("dynamic_shortcuts", JSONArray())
                .put("pinned_shortcuts", JSONArray())
                .toString()
        }
        val manager = shortcutManager(context.applicationContext)
            ?: return errorJson("Launcher shortcut manager is not available on this device")
        return JSONObject()
            .put("success", true)
            .put("supported", true)
            .put("dynamic_shortcuts", shortcutsToJson(manager.dynamicShortcuts))
            .put("pinned_shortcuts", shortcutsToJson(manager.pinnedShortcuts))
            .put("max_shortcuts_per_activity", manager.maxShortcutCountPerActivity)
            .toString()
    }

    fun removeShortcutJson(context: Context, arguments: JSONObject): String {
        val automationId = stringArgument(arguments, "automation_id", "id", "task_id")?.trim().orEmpty()
        val shortcutId = stringArgument(arguments, "shortcut_id")?.trim()
            ?: automationId.takeIf { it.isNotBlank() }?.let(::shortcutIdForAutomation)
            ?: return errorJson("remove_launcher_shortcut requires shortcut_id or automation_id")
        if (shortcutId.indexOf('\u0000') >= 0) {
            return errorJson("launcher shortcut id must not contain NUL bytes")
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N_MR1) {
            return JSONObject()
                .put("success", true)
                .put("supported", false)
                .put("shortcut_id", shortcutId)
                .put("message", "Launcher shortcuts are not supported on this Android version.")
                .toString()
        }
        val manager = shortcutManager(context.applicationContext)
            ?: return errorJson("Launcher shortcut manager is not available on this device")
        manager.removeDynamicShortcuts(listOf(shortcutId))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N_MR1) {
            runCatching { manager.disableShortcuts(listOf(shortcutId), "Hermes automation shortcut removed") }
        }
        return JSONObject()
            .put("success", true)
            .put("action", "remove_launcher_shortcut")
            .put("shortcut_id", shortcutId)
            .put("automation_id", automationId)
            .put("message", "Removed dynamic launcher shortcut; pinned launcher copies may need manual removal.")
            .toString()
    }

    fun isShortcutIntent(intent: Intent?): Boolean {
        return intent?.action == ACTION_RUN_AUTOMATION_SHORTCUT
    }

    fun handleShortcutIntentJson(context: Context, intent: Intent?): String {
        if (!isShortcutIntent(intent)) {
            return errorJson("Unsupported launcher shortcut intent: ${intent?.action.orEmpty()}")
        }
        val automationId = intent?.getStringExtra(EXTRA_AUTOMATION_ID)
            ?: intent?.getStringExtra(HermesAutomationScheduler.EXTRA_AUTOMATION_ID)
            ?: ""
        if (automationId.isBlank()) {
            return errorJson("Launcher shortcut requires automation_id")
        }
        val result = JSONObject(HermesAutomationBridge.runAutomationJson(context.applicationContext, automationId, "launcher_shortcut"))
        return result
            .put("launcher_shortcut", true)
            .put("automation_id", automationId)
            .toString()
    }

    fun shortcutIntent(context: Context, automationId: String): Intent {
        return Intent(context.applicationContext, MainActivity::class.java)
            .setAction(ACTION_RUN_AUTOMATION_SHORTCUT)
            .putExtra(EXTRA_AUTOMATION_ID, automationId)
            .putExtra(HermesAutomationScheduler.EXTRA_AUTOMATION_ID, automationId)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
    }

    private fun recordFromArguments(context: Context, arguments: JSONObject): HermesAutomationRecord? {
        val automationId = stringArgument(arguments, "automation_id", "id", "task_id", "automation")?.trim().orEmpty()
        if (automationId.indexOf('\u0000') >= 0) {
            return null
        }
        return HermesAutomationStore(context).get(automationId)
    }

    private fun shortcutInfo(
        context: Context,
        shortcutId: String,
        automationId: String,
        label: String,
        longLabel: String,
    ): ShortcutInfo {
        return ShortcutInfo.Builder(context, shortcutId)
            .setShortLabel(label.take(MAX_SHORTCUT_LABEL_CHARS).ifBlank { "Hermes task" })
            .setLongLabel(longLabel.take(MAX_SHORTCUT_LONG_LABEL_CHARS).ifBlank { label.take(MAX_SHORTCUT_LONG_LABEL_CHARS) })
            .setIcon(Icon.createWithResource(context, R.drawable.ic_nav_hermes))
            .setIntent(shortcutIntent(context, automationId))
            .build()
    }

    private fun shortcutsToJson(shortcuts: List<ShortcutInfo>): JSONArray {
        return JSONArray().apply {
            shortcuts.forEach { shortcut ->
                put(
                    JSONObject()
                        .put("shortcut_id", shortcut.id)
                        .put("short_label", shortcut.shortLabel.toString())
                        .put("long_label", shortcut.longLabel?.toString().orEmpty())
                        .put("enabled", shortcut.isEnabled)
                        .put("automation_id", shortcut.intent?.getStringExtra(EXTRA_AUTOMATION_ID).orEmpty()),
                )
            }
        }
    }

    private fun shortcutManager(context: Context): ShortcutManager? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N_MR1) {
            context.getSystemService(ShortcutManager::class.java)
        } else {
            null
        }
    }

    private fun shortcutIdForAutomation(automationId: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(automationId.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
            .take(32)
        return "automation_$digest"
    }

    private fun shortcutMessage(
        dynamicCreated: Boolean,
        pinStarted: Boolean,
        pinSupported: Boolean,
        requestPin: Boolean,
    ): String {
        return when {
            dynamicCreated && pinStarted -> "Created dynamic shortcut and requested pinned launcher shortcut."
            dynamicCreated && requestPin && !pinSupported -> "Created dynamic shortcut; this launcher does not support pin requests."
            dynamicCreated -> "Created dynamic launcher shortcut."
            pinStarted -> "Requested pinned launcher shortcut."
            requestPin && !pinSupported -> "Launcher shortcut was not created; this launcher does not support pin requests."
            else -> "Launcher shortcut was not created."
        }
    }

    private fun stringArgument(arguments: JSONObject, vararg keys: String, allowEmpty: Boolean = false): String? {
        return keys.firstNotNullOfOrNull { key ->
            if (arguments.has(key) && !arguments.isNull(key)) {
                arguments.optString(key).takeIf { allowEmpty || it.isNotBlank() }
            } else {
                null
            }
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .toString()
    }

    private const val MAX_SHORTCUT_LABEL_CHARS = 40
    private const val MAX_SHORTCUT_LONG_LABEL_CHARS = 80
}
