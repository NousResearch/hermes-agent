package com.mobilefork.hermesagent.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.util.Locale

enum class McpConfigurationMode(val persistedValue: String) {
    SIMPLE("simple"),
    ADVANCED("advanced"),
    ;

    companion object {
        fun fromPersistedValue(value: String): McpConfigurationMode {
            return when (value.trim().lowercase(Locale.US)) {
                ADVANCED.persistedValue -> ADVANCED
                else -> SIMPLE
            }
        }
    }
}

data class McpSettings(
    val mode: McpConfigurationMode = McpConfigurationMode.SIMPLE,
    val configText: String = McpSettingsDefaults.simpleConfigText(),
    val providerPromptCacheResendEnabled: Boolean = false,
    val lastStatusMessage: String = McpSettingsMessages.SIMPLE_READY,
    val lastReloadEpochMs: Long = 0L,
)

data class McpConfigActionResult(
    val success: Boolean,
    val statusMessage: String,
    val configText: String,
    val serverCount: Int = 0,
    val lastReloadEpochMs: Long = 0L,
)

object McpSettingsMessages {
    const val SIMPLE_READY = "MCP simple onboarding is ready. Auto setup writes a local native-tools config."
    const val ADVANCED_READY = "Advanced MCP config editing is ready. Save validates JSON before reload."
    const val CACHE_RESEND_ENABLED =
        "Provider cache resend enabled. Hermes may resend stable prior context only for providers that allow it."
    const val CACHE_RESEND_DISABLED =
        "Provider cache resend disabled. Hermes must not resend cached prior/tool-output context."
}

object McpSettingsDefaults {
    const val MAX_CONFIG_TEXT_CHARS = 50_000

    fun simpleConfigText(): String {
        return JSONObject()
            .put(
                "mcpServers",
                JSONObject()
                    .put(
                        "hermes-native-tools",
                        JSONObject()
                            .put("transport", "native")
                            .put("enabled", true)
                            .put("autoStart", true)
                            .put("description", "Hermes Android local tools exposed to the agent runtime"),
                    ),
            )
            .put(
                "client",
                JSONObject()
                    .put("reloadPolicy", "manual")
                    .put("safeStatusMessages", true),
            )
            .toString(2)
    }

    fun normalizeConfigText(value: String): String {
        return value
            .replace("\r\n", "\n")
            .replace('\r', '\n')
            .filter { it == '\n' || it == '\t' || it >= ' ' }
            .trim()
            .take(MAX_CONFIG_TEXT_CHARS)
    }
}

object McpPromptCacheResendPolicy {
    private val providerIdsThatDisallowCachedContextResend = setOf(
        "none",
        "local",
        "offline",
        "llama_cpp",
        "llama-cpp",
        "litert_lm",
        "litert-lm",
        "custom-no-cache",
        "no-cache",
    )

    fun shouldResendCachedContext(providerId: String, settings: McpSettings): Boolean {
        return settings.providerPromptCacheResendEnabled && !providerDisallowsCachedContextResend(providerId)
    }

    fun providerDisallowsCachedContextResend(providerId: String): Boolean {
        val normalized = providerId.trim().lowercase(Locale.US)
        return normalized.isBlank() || normalized in providerIdsThatDisallowCachedContextResend
    }

    fun statusFor(providerId: String, settings: McpSettings): String {
        if (!settings.providerPromptCacheResendEnabled) {
            return McpSettingsMessages.CACHE_RESEND_DISABLED
        }
        return if (providerDisallowsCachedContextResend(providerId)) {
            "Provider cache resend is enabled globally, but ${providerId.ifBlank { "this provider" }} disallows cached context resend."
        } else {
            McpSettingsMessages.CACHE_RESEND_ENABLED
        }
    }
}

class McpSettingsStore(context: Context) {
    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val configFile = File(appContext.filesDir, CONFIG_FILE_RELATIVE_PATH)

    fun load(): McpSettings {
        val configText = readConfigText()
        return McpSettings(
            mode = McpConfigurationMode.fromPersistedValue(
                preferences.getString(KEY_MODE, McpConfigurationMode.SIMPLE.persistedValue).orEmpty(),
            ),
            configText = configText,
            providerPromptCacheResendEnabled = preferences.getBoolean(KEY_PROVIDER_PROMPT_CACHE_RESEND, false),
            lastStatusMessage = preferences.getString(KEY_LAST_STATUS, McpSettingsMessages.SIMPLE_READY).orEmpty()
                .ifBlank { McpSettingsMessages.SIMPLE_READY },
            lastReloadEpochMs = preferences.getLong(KEY_LAST_RELOAD_EPOCH_MS, 0L),
        )
    }

    fun configFilePath(): String = configFile.absolutePath

    fun saveMode(mode: McpConfigurationMode): McpSettings {
        val status = when (mode) {
            McpConfigurationMode.SIMPLE -> McpSettingsMessages.SIMPLE_READY
            McpConfigurationMode.ADVANCED -> McpSettingsMessages.ADVANCED_READY
        }
        preferences.edit()
            .putString(KEY_MODE, mode.persistedValue)
            .putString(KEY_LAST_STATUS, status)
            .apply()
        return load()
    }

    fun saveProviderPromptCacheResendEnabled(enabled: Boolean): McpSettings {
        val status = if (enabled) {
            McpSettingsMessages.CACHE_RESEND_ENABLED
        } else {
            McpSettingsMessages.CACHE_RESEND_DISABLED
        }
        preferences.edit()
            .putBoolean(KEY_PROVIDER_PROMPT_CACHE_RESEND, enabled)
            .putString(KEY_LAST_STATUS, status)
            .apply()
        return load()
    }

    fun detectExistingConfiguration(): McpConfigActionResult {
        if (!configFile.isFile) {
            val configText = McpSettingsDefaults.simpleConfigText()
            val status = "No MCP config file found. Auto fill can create ${CONFIG_FILE_RELATIVE_PATH}."
            saveStatus(status)
            return McpConfigActionResult(
                success = false,
                statusMessage = status,
                configText = configText,
            )
        }
        return validateConfigText(
            rawText = readConfigText(),
            successPrefix = "Detected MCP config",
            persistStatus = true,
        )
    }

    fun autoFillSimpleConfiguration(): McpConfigActionResult {
        val configText = McpSettingsDefaults.simpleConfigText()
        val result = validateConfigText(
            rawText = configText,
            successPrefix = "Auto-filled MCP config",
            persistStatus = false,
        )
        val status = "${result.statusMessage} Review it, then use Auto setup to save and reload."
        saveStatus(status)
        return result.copy(statusMessage = status)
    }

    fun autoSetupSimpleConfiguration(nowEpochMs: Long = System.currentTimeMillis()): McpConfigActionResult {
        val configText = McpSettingsDefaults.simpleConfigText()
        val validation = validateConfigText(
            rawText = configText,
            successPrefix = "Auto setup prepared MCP config",
            persistStatus = false,
        )
        if (!validation.success) {
            saveStatus(validation.statusMessage)
            return validation
        }
        writeConfigText(configText)
        preferences.edit()
            .putString(KEY_MODE, McpConfigurationMode.SIMPLE.persistedValue)
            .apply()
        return reloadServers(nowEpochMs)
    }

    fun saveAdvancedConfigTextAndReload(
        rawText: String,
        nowEpochMs: Long = System.currentTimeMillis(),
    ): McpConfigActionResult {
        val normalized = McpSettingsDefaults.normalizeConfigText(rawText)
        val validation = validateConfigText(
            rawText = normalized,
            successPrefix = "Advanced MCP config validated",
            persistStatus = false,
        )
        if (!validation.success) {
            saveStatus(validation.statusMessage)
            return validation.copy(configText = normalized)
        }
        writeConfigText(normalized)
        preferences.edit()
            .putString(KEY_MODE, McpConfigurationMode.ADVANCED.persistedValue)
            .apply()
        return reloadServers(nowEpochMs)
    }

    fun reloadServers(nowEpochMs: Long = System.currentTimeMillis()): McpConfigActionResult {
        val configText = readConfigText()
        val validation = validateConfigText(
            rawText = configText,
            successPrefix = "Reloaded MCP config",
            persistStatus = false,
        )
        if (!validation.success) {
            saveStatus(validation.statusMessage)
            return validation
        }
        val status = when (validation.serverCount) {
            0 -> "Reloaded MCP config. No enabled server definitions were found."
            1 -> "Reloaded 1 MCP server definition from local config."
            else -> "Reloaded ${validation.serverCount} MCP server definitions from local config."
        }
        preferences.edit()
            .putString(KEY_LAST_STATUS, status)
            .putLong(KEY_LAST_RELOAD_EPOCH_MS, nowEpochMs)
            .apply()
        return validation.copy(statusMessage = status, lastReloadEpochMs = nowEpochMs)
    }

    private fun readConfigText(): String {
        if (configFile.isFile) {
            return McpSettingsDefaults.normalizeConfigText(configFile.readText())
                .ifBlank { McpSettingsDefaults.simpleConfigText() }
        }
        return preferences.getString(KEY_CONFIG_TEXT, null)
            ?.let(McpSettingsDefaults::normalizeConfigText)
            ?.takeIf { it.isNotBlank() }
            ?: McpSettingsDefaults.simpleConfigText()
    }

    private fun writeConfigText(configText: String) {
        val normalized = McpSettingsDefaults.normalizeConfigText(configText)
        configFile.parentFile?.mkdirs()
        configFile.writeText(normalized)
        preferences.edit()
            .putString(KEY_CONFIG_TEXT, normalized)
            .apply()
    }

    private fun validateConfigText(
        rawText: String,
        successPrefix: String,
        persistStatus: Boolean,
    ): McpConfigActionResult {
        val normalized = McpSettingsDefaults.normalizeConfigText(rawText)
        if (normalized.isBlank()) {
            val status = "MCP config is empty. Add a JSON object before reloading."
            if (persistStatus) saveStatus(status)
            return McpConfigActionResult(false, status, normalized)
        }
        val json = try {
            JSONObject(normalized)
        } catch (error: JSONException) {
            val status = "MCP config JSON is invalid: ${error.message.orEmpty().ifBlank { error::class.java.simpleName }}"
            if (persistStatus) saveStatus(status)
            return McpConfigActionResult(false, status, normalized)
        }
        val serverCount = countEnabledServers(json)
        val status = when (serverCount) {
            0 -> "$successPrefix. No enabled server definitions found."
            1 -> "$successPrefix with 1 enabled server definition."
            else -> "$successPrefix with $serverCount enabled server definitions."
        }
        if (persistStatus) saveStatus(status)
        return McpConfigActionResult(
            success = true,
            statusMessage = status,
            configText = json.toString(2),
            serverCount = serverCount,
        )
    }

    private fun countEnabledServers(json: JSONObject): Int {
        val mcpServers = json.optJSONObject("mcpServers")
        if (mcpServers != null) {
            return mcpServers.keys().asSequence().count { serverName ->
                isServerEnabled(mcpServers.opt(serverName))
            }
        }
        val servers = json.optJSONArray("servers") ?: JSONArray()
        var count = 0
        for (index in 0 until servers.length()) {
            if (isServerEnabled(servers.opt(index))) {
                count += 1
            }
        }
        return count
    }

    private fun isServerEnabled(value: Any?): Boolean {
        val server = value as? JSONObject ?: return value != null
        return if (server.has("enabled")) {
            server.optBoolean("enabled", true)
        } else {
            true
        }
    }

    private fun saveStatus(status: String) {
        preferences.edit()
            .putString(KEY_LAST_STATUS, status)
            .apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_mcp_settings"
        private const val CONFIG_FILE_RELATIVE_PATH = "hermes-home/mcp/mcp_config.json"
        private const val KEY_MODE = "mode"
        private const val KEY_CONFIG_TEXT = "config_text"
        private const val KEY_PROVIDER_PROMPT_CACHE_RESEND = "provider_prompt_cache_resend_enabled"
        private const val KEY_LAST_STATUS = "last_status"
        private const val KEY_LAST_RELOAD_EPOCH_MS = "last_reload_epoch_ms"
    }
}
