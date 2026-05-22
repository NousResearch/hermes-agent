package com.mobilefork.hermesagent.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class AppSettings(
    val provider: String = "openrouter",
    val baseUrl: String = "",
    val model: String = "",
    val corr3xtBaseUrl: String = "",
    val dataSaverMode: Boolean = false,
    val offlineAirplaneMode: Boolean = false,
    val portalEnabled: Boolean = true,
    val onDeviceBackend: String = "none",
    val liteRtLmSpeculativeDecodingMode: String = "auto",
    val languageTag: String = "en",
    val chatDisplayMode: String = "compact",
    val keywordHighlightingEnabled: Boolean = true,
    val themePrimaryHex: String = "#8C7BFF",
    val themeSecondaryHex: String = "#C6A15B",
    val themeBackgroundHex: String = "#090B10",
    val themeSurfaceHex: String = "#11141C",
    val themeSurfaceVariantHex: String = "#1B202B",
    val themeCardShape: String = "rounded",
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("provider", provider)
            .put("base_url", baseUrl)
            .put("model", model)
            .put("corr3xt_base_url", corr3xtBaseUrl)
            .put("data_saver_mode", dataSaverMode)
            .put("offline_airplane_mode", offlineAirplaneMode)
            .put("portal_enabled", portalEnabled)
            .put("on_device_backend", onDeviceBackend)
            .put("litert_lm_speculative_decoding_mode", liteRtLmSpeculativeDecodingMode)
            .put("language_tag", languageTag)
            .put("chat_display_mode", chatDisplayMode)
            .put("keyword_highlighting_enabled", keywordHighlightingEnabled)
            .put("theme_primary_hex", themePrimaryHex)
            .put("theme_secondary_hex", themeSecondaryHex)
            .put("theme_background_hex", themeBackgroundHex)
            .put("theme_surface_hex", themeSurfaceHex)
            .put("theme_surface_variant_hex", themeSurfaceVariantHex)
            .put("theme_card_shape", themeCardShape)
    }

    companion object {
        const val EXPORT_KIND = "hermes_android_app_settings_bundle"
        const val EXPORT_SCHEMA_VERSION = 1

        val REDACTED_SECRET_FIELDS: JSONArray
            get() = JSONArray()
                .put("api_key")
                .put("access_token")
                .put("refresh_token")
                .put("provider_credentials")
                .put("cookie")
                .put("authorization")

        fun fromJson(json: JSONObject, fallback: AppSettings = AppSettings()): AppSettings {
            return fallback.copy(
                provider = json.optString("provider", fallback.provider).ifBlank { fallback.provider },
                baseUrl = json.optString("base_url", fallback.baseUrl),
                model = json.optString("model", fallback.model),
                corr3xtBaseUrl = json.optString("corr3xt_base_url", fallback.corr3xtBaseUrl),
                dataSaverMode = optBoolean(json, "data_saver_mode", fallback.dataSaverMode),
                offlineAirplaneMode = optBoolean(json, "offline_airplane_mode", fallback.offlineAirplaneMode),
                portalEnabled = optBoolean(json, "portal_enabled", fallback.portalEnabled),
                onDeviceBackend = json.optString("on_device_backend", fallback.onDeviceBackend).ifBlank { fallback.onDeviceBackend },
                liteRtLmSpeculativeDecodingMode = json.optString(
                    "litert_lm_speculative_decoding_mode",
                    fallback.liteRtLmSpeculativeDecodingMode,
                ).ifBlank { fallback.liteRtLmSpeculativeDecodingMode },
                languageTag = json.optString("language_tag", fallback.languageTag).ifBlank { fallback.languageTag },
                chatDisplayMode = json.optString("chat_display_mode", fallback.chatDisplayMode).ifBlank { fallback.chatDisplayMode },
                keywordHighlightingEnabled = optBoolean(
                    json,
                    "keyword_highlighting_enabled",
                    fallback.keywordHighlightingEnabled,
                ),
                themePrimaryHex = json.optString("theme_primary_hex", fallback.themePrimaryHex).ifBlank { fallback.themePrimaryHex },
                themeSecondaryHex = json.optString("theme_secondary_hex", fallback.themeSecondaryHex).ifBlank { fallback.themeSecondaryHex },
                themeBackgroundHex = json.optString("theme_background_hex", fallback.themeBackgroundHex).ifBlank { fallback.themeBackgroundHex },
                themeSurfaceHex = json.optString("theme_surface_hex", fallback.themeSurfaceHex).ifBlank { fallback.themeSurfaceHex },
                themeSurfaceVariantHex = json.optString(
                    "theme_surface_variant_hex",
                    fallback.themeSurfaceVariantHex,
                ).ifBlank { fallback.themeSurfaceVariantHex },
                themeCardShape = json.optString("theme_card_shape", fallback.themeCardShape).ifBlank { fallback.themeCardShape },
            )
        }

        fun exportBundle(settings: AppSettings, exportedAtEpochMs: Long = System.currentTimeMillis()): JSONObject {
            val settingsJson = settings.toJson()
            return JSONObject()
                .put("kind", EXPORT_KIND)
                .put("schema_version", EXPORT_SCHEMA_VERSION)
                .put("exported_at_epoch_ms", exportedAtEpochMs)
                .put("secrets_included", false)
                .put("portable_field_count", settingsJson.length())
                .put("redacted_secret_fields", REDACTED_SECRET_FIELDS)
                .put("settings", settingsJson)
        }

        private fun optBoolean(json: JSONObject, key: String, fallback: Boolean): Boolean {
            return if (json.has(key) && !json.isNull(key)) json.optBoolean(key, fallback) else fallback
        }
    }
}

class AppSettingsStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): AppSettings {
        return AppSettings(
            provider = preferences.getString(KEY_PROVIDER, "openrouter").orEmpty(),
            baseUrl = preferences.getString(KEY_BASE_URL, "").orEmpty(),
            model = preferences.getString(KEY_MODEL, "").orEmpty(),
            corr3xtBaseUrl = preferences.getString(KEY_CORR3XT_BASE_URL, "").orEmpty(),
            dataSaverMode = preferences.getBoolean(KEY_DATA_SAVER_MODE, false),
            offlineAirplaneMode = preferences.getBoolean(KEY_OFFLINE_AIRPLANE_MODE, false),
            portalEnabled = preferences.getBoolean(KEY_PORTAL_ENABLED, true),
            onDeviceBackend = preferences.getString(KEY_ON_DEVICE_BACKEND, "none").orEmpty(),
            liteRtLmSpeculativeDecodingMode = preferences.getString(
                KEY_LITERT_LM_SPECULATIVE_DECODING_MODE,
                "auto",
            ).orEmpty(),
            languageTag = preferences.getString(KEY_LANGUAGE_TAG, "en").orEmpty(),
            chatDisplayMode = preferences.getString(KEY_CHAT_DISPLAY_MODE, "compact").orEmpty(),
            keywordHighlightingEnabled = preferences.getBoolean(KEY_KEYWORD_HIGHLIGHTING_ENABLED, true),
            themePrimaryHex = preferences.getString(KEY_THEME_PRIMARY_HEX, "#8C7BFF").orEmpty(),
            themeSecondaryHex = preferences.getString(KEY_THEME_SECONDARY_HEX, "#C6A15B").orEmpty(),
            themeBackgroundHex = preferences.getString(KEY_THEME_BACKGROUND_HEX, "#090B10").orEmpty(),
            themeSurfaceHex = preferences.getString(KEY_THEME_SURFACE_HEX, "#11141C").orEmpty(),
            themeSurfaceVariantHex = preferences.getString(KEY_THEME_SURFACE_VARIANT_HEX, "#1B202B").orEmpty(),
            themeCardShape = preferences.getString(KEY_THEME_CARD_SHAPE, "rounded").orEmpty(),
        )
    }

    fun save(settings: AppSettings) {
        preferences.edit()
            .putString(KEY_PROVIDER, settings.provider)
            .putString(KEY_BASE_URL, settings.baseUrl)
            .putString(KEY_MODEL, settings.model)
            .putString(KEY_CORR3XT_BASE_URL, settings.corr3xtBaseUrl)
            .putBoolean(KEY_DATA_SAVER_MODE, settings.dataSaverMode)
            .putBoolean(KEY_OFFLINE_AIRPLANE_MODE, settings.offlineAirplaneMode)
            .putBoolean(KEY_PORTAL_ENABLED, settings.portalEnabled)
            .putString(KEY_ON_DEVICE_BACKEND, settings.onDeviceBackend)
            .putString(KEY_LITERT_LM_SPECULATIVE_DECODING_MODE, settings.liteRtLmSpeculativeDecodingMode)
            .putString(KEY_LANGUAGE_TAG, settings.languageTag)
            .putString(KEY_CHAT_DISPLAY_MODE, settings.chatDisplayMode)
            .putBoolean(KEY_KEYWORD_HIGHLIGHTING_ENABLED, settings.keywordHighlightingEnabled)
            .putString(KEY_THEME_PRIMARY_HEX, settings.themePrimaryHex)
            .putString(KEY_THEME_SECONDARY_HEX, settings.themeSecondaryHex)
            .putString(KEY_THEME_BACKGROUND_HEX, settings.themeBackgroundHex)
            .putString(KEY_THEME_SURFACE_HEX, settings.themeSurfaceHex)
            .putString(KEY_THEME_SURFACE_VARIANT_HEX, settings.themeSurfaceVariantHex)
            .putString(KEY_THEME_CARD_SHAPE, settings.themeCardShape)
            .apply()
    }

    fun exportBundleJson(): JSONObject = AppSettings.exportBundle(load())

    fun importBundleJson(bundle: JSONObject): AppSettings {
        val settingsJson = bundle.optJSONObject("settings") ?: bundle
        val imported = AppSettings.fromJson(settingsJson, load())
        save(imported)
        return imported
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_settings"
        private const val KEY_PROVIDER = "provider"
        private const val KEY_BASE_URL = "base_url"
        private const val KEY_MODEL = "model"
        private const val KEY_CORR3XT_BASE_URL = "corr3xt_base_url"
        private const val KEY_DATA_SAVER_MODE = "data_saver_mode"
        private const val KEY_OFFLINE_AIRPLANE_MODE = "offline_airplane_mode"
        private const val KEY_PORTAL_ENABLED = "portal_enabled"
        private const val KEY_ON_DEVICE_BACKEND = "on_device_backend"
        private const val KEY_LITERT_LM_SPECULATIVE_DECODING_MODE = "litert_lm_speculative_decoding_mode"
        private const val KEY_LANGUAGE_TAG = "language_tag"
        private const val KEY_CHAT_DISPLAY_MODE = "chat_display_mode"
        private const val KEY_KEYWORD_HIGHLIGHTING_ENABLED = "keyword_highlighting_enabled"
        private const val KEY_THEME_PRIMARY_HEX = "theme_primary_hex"
        private const val KEY_THEME_SECONDARY_HEX = "theme_secondary_hex"
        private const val KEY_THEME_BACKGROUND_HEX = "theme_background_hex"
        private const val KEY_THEME_SURFACE_HEX = "theme_surface_hex"
        private const val KEY_THEME_SURFACE_VARIANT_HEX = "theme_surface_variant_hex"
        private const val KEY_THEME_CARD_SHAPE = "theme_card_shape"
    }
}
