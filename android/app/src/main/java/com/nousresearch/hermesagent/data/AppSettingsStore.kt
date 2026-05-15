package com.nousresearch.hermesagent.data

import android.content.Context

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
)

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
            .apply()
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
    }
}
