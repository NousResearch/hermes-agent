package com.nousresearch.hermesagent.device

import android.content.ComponentName
import android.content.Context
import android.os.Build
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import org.json.JSONObject

object HermesQuickSettingsTileBridge {
    fun setTileAutomationJson(context: Context, arguments: JSONObject): String {
        val appContext = context.applicationContext
        val record = recordFromArguments(appContext, arguments)
            ?: return errorJson("set_quick_settings_tile_automation requires an existing automation_id or id")
        val label = stringArgument(arguments, "label", "tile_label", "title", allowEmpty = true)
            ?.ifBlank { record.label }
            ?: record.label
        if (label.indexOf('\u0000') >= 0) {
            return errorJson("quick settings tile label must not contain NUL bytes")
        }
        preferences(appContext)
            .edit()
            .putString(KEY_AUTOMATION_ID, record.id)
            .putString(KEY_LABEL, label.take(MAX_TILE_LABEL_CHARS))
            .apply()
        requestTileListeningState(appContext)
        return JSONObject()
            .put("success", true)
            .put("action", "set_quick_settings_tile_automation")
            .put("configured", true)
            .put("automation_id", record.id)
            .put("label", label.take(MAX_TILE_LABEL_CHARS))
            .put("message", "Configured the Hermes Quick Settings tile for this saved automation.")
            .toString()
    }

    fun getTileAutomationJson(context: Context): String {
        val appContext = context.applicationContext
        val config = configuredAutomation(appContext)
        val record = config?.automationId?.let { HermesAutomationStore(appContext).get(it) }
        return JSONObject()
            .put("success", true)
            .put("action", "get_quick_settings_tile_automation")
            .put("configured", config != null)
            .put("automation_id", config?.automationId.orEmpty())
            .put("label", config?.label.orEmpty())
            .put("automation_exists", record != null)
            .put("automation", record?.toJson() ?: JSONObject.NULL)
            .toString()
    }

    fun clearTileAutomationJson(context: Context): String {
        val appContext = context.applicationContext
        preferences(appContext)
            .edit()
            .remove(KEY_AUTOMATION_ID)
            .remove(KEY_LABEL)
            .apply()
        requestTileListeningState(appContext)
        return JSONObject()
            .put("success", true)
            .put("action", "clear_quick_settings_tile_automation")
            .put("configured", false)
            .put("message", "Cleared the Hermes Quick Settings tile automation.")
            .toString()
    }

    fun runConfiguredAutomationJson(context: Context): String {
        val appContext = context.applicationContext
        val config = configuredAutomation(appContext)
            ?: return errorJson("No Hermes automation is configured for the Quick Settings tile")
        val result = JSONObject(
            HermesAutomationBridge.runAutomationJson(
                appContext,
                config.automationId,
                TRIGGER_QUICK_SETTINGS_TILE,
            ),
        )
        return result
            .put("quick_settings_tile", true)
            .put("automation_id", config.automationId)
            .put("label", config.label)
            .toString()
    }

    fun updateTile(context: Context, tile: Tile?) {
        tile ?: return
        val config = configuredAutomation(context.applicationContext)
        tile.label = config?.label?.ifBlank { DEFAULT_TILE_LABEL } ?: DEFAULT_TILE_LABEL
        tile.state = if (config == null) Tile.STATE_INACTIVE else Tile.STATE_ACTIVE
        tile.updateTile()
    }

    private fun recordFromArguments(context: Context, arguments: JSONObject): HermesAutomationRecord? {
        val automationId = stringArgument(arguments, "automation_id", "id", "task_id", "automation")?.trim().orEmpty()
        if (automationId.indexOf('\u0000') >= 0) {
            return null
        }
        return HermesAutomationStore(context).get(automationId)
    }

    private fun configuredAutomation(context: Context): TileConfig? {
        val prefs = preferences(context.applicationContext)
        val automationId = prefs.getString(KEY_AUTOMATION_ID, "").orEmpty()
        if (automationId.isBlank() || automationId.indexOf('\u0000') >= 0) {
            return null
        }
        val label = prefs.getString(KEY_LABEL, DEFAULT_TILE_LABEL).orEmpty()
            .replace("\u0000", "")
            .ifBlank { DEFAULT_TILE_LABEL }
            .take(MAX_TILE_LABEL_CHARS)
        return TileConfig(automationId, label)
    }

    private fun preferences(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private fun requestTileListeningState(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            runCatching {
                TileService.requestListeningState(
                    context.applicationContext,
                    ComponentName(context.applicationContext, HermesQuickSettingsTileService::class.java),
                )
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

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("error", message)
            .toString()
    }

    private data class TileConfig(
        val automationId: String,
        val label: String,
    )

    const val TRIGGER_QUICK_SETTINGS_TILE = "quick_settings_tile"
    private const val PREFS_NAME = "hermes_quick_settings_tile"
    private const val KEY_AUTOMATION_ID = "automation_id"
    private const val KEY_LABEL = "label"
    private const val DEFAULT_TILE_LABEL = "Hermes task"
    private const val MAX_TILE_LABEL_CHARS = 40
}
