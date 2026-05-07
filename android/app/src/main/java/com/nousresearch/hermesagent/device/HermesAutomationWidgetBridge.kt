package com.nousresearch.hermesagent.device

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Build
import android.widget.RemoteViews
import com.nousresearch.hermesagent.R
import org.json.JSONArray
import org.json.JSONObject

object HermesAutomationWidgetBridge {
    const val ACTION_RUN_AUTOMATION_WIDGET = "com.nousresearch.hermesagent.RUN_AUTOMATION_WIDGET"
    const val TRIGGER_HOME_SCREEN_WIDGET = "home_screen_widget"

    fun setWidgetAutomationJson(context: Context, arguments: JSONObject): String {
        val appContext = context.applicationContext
        val appWidgetId = widgetIdArgument(arguments)
        if (appWidgetId != null && appWidgetId <= 0) {
            return errorJson("home screen widget id must be a positive integer")
        }
        val record = recordFromArguments(appContext, arguments)
            ?: return errorJson("set_home_screen_widget_automation requires an existing automation_id or id")
        val label = stringArgument(arguments, "label", "widget_label", "title", allowEmpty = true)
            ?.ifBlank { record.label }
            ?: record.label
        if (label.indexOf('\u0000') >= 0) {
            return errorJson("home screen widget label must not contain NUL bytes")
        }

        preferences(appContext)
            .edit()
            .putString(automationKey(appWidgetId), record.id)
            .putString(labelKey(appWidgetId), label.take(MAX_WIDGET_LABEL_CHARS))
            .apply()

        updateWidgets(appContext)
        val requestPin = arguments.optBoolean("pin", arguments.optBoolean("request_pin", false))
        val pinResult = if (requestPin) requestPinWidget(appContext) else PinResult(pinSupported(appContext), false, "")

        return JSONObject()
            .put("success", true)
            .put("action", "set_home_screen_widget_automation")
            .put("configured", true)
            .put("automation_id", record.id)
            .put("app_widget_id", appWidgetId ?: JSONObject.NULL)
            .put("label", label.take(MAX_WIDGET_LABEL_CHARS))
            .put("installed_widget_ids", JSONArray(installedWidgetIds(appContext)))
            .put("pin_requested", requestPin)
            .put("pin_request_supported", pinResult.supported)
            .put("pin_request_started", pinResult.started)
            .put("pin_request_error", pinResult.error)
            .put("message", widgetMessage(appWidgetId, requestPin, pinResult))
            .toString()
    }

    fun getWidgetAutomationJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val appWidgetId = widgetIdArgument(arguments)
        if (appWidgetId != null && appWidgetId <= 0) {
            return errorJson("home screen widget id must be a positive integer")
        }
        return widgetStatusJson(appContext, appWidgetId)
            .put("action", "get_home_screen_widget_automation")
            .toString()
    }

    fun listWidgetsJson(context: Context): String {
        val appContext = context.applicationContext
        val installedIds = installedWidgetIds(appContext)
        val installed = JSONArray().apply {
            installedIds.forEach { put(widgetStatusJson(appContext, it)) }
        }
        val stored = JSONArray().apply {
            storedWidgetIds(appContext).forEach { put(widgetStatusJson(appContext, it)) }
        }
        return JSONObject()
            .put("success", true)
            .put("action", "list_home_screen_widgets")
            .put("supported", true)
            .put("installed_widget_ids", JSONArray(installedIds))
            .put("installed_widgets", installed)
            .put("stored_widget_configs", stored)
            .put("default_config", widgetStatusJson(appContext, null))
            .put("pin_request_supported", pinSupported(appContext))
            .toString()
    }

    fun clearWidgetAutomationJson(context: Context, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        val appWidgetId = widgetIdArgument(arguments)
        if (appWidgetId != null && appWidgetId <= 0) {
            return errorJson("home screen widget id must be a positive integer")
        }
        val clearAll = arguments.optBoolean("all", arguments.optBoolean("clear_all", false))
        val editor = preferences(appContext).edit()
        if (clearAll) {
            preferences(appContext).all.keys
                .filter { it == KEY_DEFAULT_AUTOMATION_ID || it == KEY_DEFAULT_LABEL || it.startsWith(KEY_WIDGET_AUTOMATION_ID_PREFIX) || it.startsWith(KEY_WIDGET_LABEL_PREFIX) }
                .forEach { editor.remove(it) }
        } else {
            editor.remove(automationKey(appWidgetId))
            editor.remove(labelKey(appWidgetId))
        }
        editor.apply()
        updateWidgets(appContext)
        return JSONObject()
            .put("success", true)
            .put("action", "clear_home_screen_widget_automation")
            .put("configured", false)
            .put("app_widget_id", appWidgetId ?: JSONObject.NULL)
            .put("cleared_all", clearAll)
            .put("message", if (clearAll) "Cleared all Hermes home-screen widget automations." else "Cleared the Hermes home-screen widget automation.")
            .toString()
    }

    fun runConfiguredAutomationJson(context: Context, appWidgetId: Int? = null): String {
        val appContext = context.applicationContext
        val config = configuredAutomation(appContext, appWidgetId)
            ?: return errorJson("No Hermes automation is configured for this home-screen widget")
        val result = JSONObject(
            HermesAutomationBridge.runAutomationJson(
                appContext,
                config.automationId,
                TRIGGER_HOME_SCREEN_WIDGET,
            ),
        )
        return result
            .put("home_screen_widget", true)
            .put("app_widget_id", appWidgetId ?: JSONObject.NULL)
            .put("automation_id", config.automationId)
            .put("label", config.label)
            .put("uses_default_config", config.usesDefault)
            .toString()
    }

    fun runConfiguredAutomationJson(context: Context, arguments: JSONObject): String {
        val appWidgetId = widgetIdArgument(arguments)
        if (appWidgetId != null && appWidgetId <= 0) {
            return errorJson("home screen widget id must be a positive integer")
        }
        return runConfiguredAutomationJson(context, appWidgetId)
    }

    fun updateWidgets(context: Context) {
        val appContext = context.applicationContext
        val manager = AppWidgetManager.getInstance(appContext)
        installedWidgetIds(appContext).forEach { updateWidget(appContext, manager, it) }
    }

    fun updateWidget(context: Context, appWidgetManager: AppWidgetManager, appWidgetId: Int) {
        val appContext = context.applicationContext
        val config = configuredAutomation(appContext, appWidgetId)
        val views = RemoteViews(appContext.packageName, R.layout.hermes_automation_widget)
        val label = config?.label?.ifBlank { DEFAULT_WIDGET_LABEL } ?: DEFAULT_WIDGET_LABEL
        val status = if (config == null) {
            appContext.getString(R.string.hermes_automation_widget_empty_status)
        } else {
            appContext.getString(R.string.hermes_automation_widget_ready_status)
        }
        views.setTextViewText(R.id.hermes_widget_title, label)
        views.setTextViewText(R.id.hermes_widget_status, status)
        views.setImageViewResource(R.id.hermes_widget_icon, R.drawable.ic_nav_hermes)
        val pendingIntent = runPendingIntent(appContext, appWidgetId)
        views.setOnClickPendingIntent(R.id.hermes_widget_root, pendingIntent)
        views.setOnClickPendingIntent(R.id.hermes_widget_button, pendingIntent)
        appWidgetManager.updateAppWidget(appWidgetId, views)
    }

    fun removeWidgetConfigs(context: Context, appWidgetIds: IntArray) {
        val editor = preferences(context.applicationContext).edit()
        appWidgetIds.forEach { appWidgetId ->
            editor.remove(automationKey(appWidgetId))
            editor.remove(labelKey(appWidgetId))
        }
        editor.apply()
    }

    private fun requestPinWidget(context: Context): PinResult {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return PinResult(false, false, "")
        }
        val manager = AppWidgetManager.getInstance(context.applicationContext)
        val supported = manager.isRequestPinAppWidgetSupported
        if (!supported) {
            return PinResult(false, false, "")
        }
        val result = runCatching {
            manager.requestPinAppWidget(providerComponent(context), null, null)
        }
        return PinResult(supported, result.getOrDefault(false), result.exceptionOrNull()?.message.orEmpty())
    }

    private fun pinSupported(context: Context): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            AppWidgetManager.getInstance(context.applicationContext).isRequestPinAppWidgetSupported
    }

    private fun widgetStatusJson(context: Context, appWidgetId: Int?): JSONObject {
        val config = configuredAutomation(context.applicationContext, appWidgetId)
        val record = config?.automationId?.let { HermesAutomationStore(context.applicationContext).get(it) }
        return JSONObject()
            .put("success", true)
            .put("configured", config != null)
            .put("app_widget_id", appWidgetId ?: JSONObject.NULL)
            .put("automation_id", config?.automationId.orEmpty())
            .put("label", config?.label.orEmpty())
            .put("uses_default_config", config?.usesDefault ?: false)
            .put("automation_exists", record != null)
            .put("automation", record?.toJson() ?: JSONObject.NULL)
    }

    private fun configuredAutomation(context: Context, appWidgetId: Int?): WidgetConfig? {
        if (appWidgetId != null) {
            configForKeys(context, automationKey(appWidgetId), labelKey(appWidgetId), usesDefault = false)?.let { return it }
        }
        return configForKeys(context, KEY_DEFAULT_AUTOMATION_ID, KEY_DEFAULT_LABEL, usesDefault = true)
    }

    private fun configForKeys(context: Context, automationKey: String, labelKey: String, usesDefault: Boolean): WidgetConfig? {
        val prefs = preferences(context.applicationContext)
        val automationId = prefs.getString(automationKey, "").orEmpty()
        if (automationId.isBlank() || automationId.indexOf('\u0000') >= 0) {
            return null
        }
        val label = prefs.getString(labelKey, DEFAULT_WIDGET_LABEL).orEmpty()
            .replace("\u0000", "")
            .ifBlank { DEFAULT_WIDGET_LABEL }
            .take(MAX_WIDGET_LABEL_CHARS)
        return WidgetConfig(automationId, label, usesDefault)
    }

    private fun recordFromArguments(context: Context, arguments: JSONObject): HermesAutomationRecord? {
        val automationId = stringArgument(arguments, "automation_id", "id", "task_id", "automation")?.trim().orEmpty()
        if (automationId.indexOf('\u0000') >= 0) {
            return null
        }
        return HermesAutomationStore(context).get(automationId)
    }

    private fun installedWidgetIds(context: Context): List<Int> {
        return AppWidgetManager.getInstance(context.applicationContext)
            .getAppWidgetIds(providerComponent(context))
            .toList()
    }

    private fun storedWidgetIds(context: Context): List<Int> {
        return preferences(context.applicationContext).all.keys
            .asSequence()
            .filter { it.startsWith(KEY_WIDGET_AUTOMATION_ID_PREFIX) }
            .mapNotNull { it.removePrefix(KEY_WIDGET_AUTOMATION_ID_PREFIX).toIntOrNull() }
            .sorted()
            .toList()
    }

    private fun providerComponent(context: Context) =
        ComponentName(context.applicationContext, HermesAutomationWidgetProvider::class.java)

    private fun runPendingIntent(context: Context, appWidgetId: Int): PendingIntent {
        val intent = Intent(context.applicationContext, HermesAutomationWidgetProvider::class.java)
            .setAction(ACTION_RUN_AUTOMATION_WIDGET)
            .putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, appWidgetId)
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
        return PendingIntent.getBroadcast(context.applicationContext, appWidgetId, intent, flags)
    }

    private fun preferences(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private fun automationKey(appWidgetId: Int?) =
        appWidgetId?.let { "$KEY_WIDGET_AUTOMATION_ID_PREFIX$it" } ?: KEY_DEFAULT_AUTOMATION_ID

    private fun labelKey(appWidgetId: Int?) =
        appWidgetId?.let { "$KEY_WIDGET_LABEL_PREFIX$it" } ?: KEY_DEFAULT_LABEL

    private fun widgetIdArgument(arguments: JSONObject): Int? {
        return WIDGET_ID_KEYS.firstNotNullOfOrNull { key ->
            if (arguments.has(key) && !arguments.isNull(key)) {
                arguments.optString(key).trim().toIntOrNull() ?: -1
            } else {
                null
            }
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

    private fun widgetMessage(appWidgetId: Int?, requestPin: Boolean, pinResult: PinResult): String {
        return when {
            appWidgetId != null -> "Configured the selected Hermes home-screen widget."
            requestPin && pinResult.started -> "Configured the default Hermes widget automation and asked Android to pin a widget."
            requestPin && !pinResult.supported -> "Configured the default Hermes widget automation; this launcher does not support widget pin requests."
            requestPin -> "Configured the default Hermes widget automation; Android did not start a widget pin request."
            else -> "Configured the default Hermes home-screen widget automation."
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .toString()
    }

    private data class WidgetConfig(
        val automationId: String,
        val label: String,
        val usesDefault: Boolean,
    )

    private data class PinResult(
        val supported: Boolean,
        val started: Boolean,
        val error: String,
    )

    private val WIDGET_ID_KEYS = listOf("app_widget_id", "widget_id", "appWidgetId")
    private const val PREFS_NAME = "hermes_automation_widget"
    private const val KEY_DEFAULT_AUTOMATION_ID = "default_automation_id"
    private const val KEY_DEFAULT_LABEL = "default_label"
    private const val KEY_WIDGET_AUTOMATION_ID_PREFIX = "widget_automation_id_"
    private const val KEY_WIDGET_LABEL_PREFIX = "widget_label_"
    private const val DEFAULT_WIDGET_LABEL = "Hermes task"
    private const val MAX_WIDGET_LABEL_CHARS = 40
}
