package com.nousresearch.hermesagent.data

import android.content.Context

data class DeviceCapabilityState(
    val sharedFolderUri: String = "",
    val sharedFolderLabel: String = "",
    val backgroundPersistenceEnabled: Boolean = false,
)

class DeviceCapabilityStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): DeviceCapabilityState {
        return DeviceCapabilityState(
            sharedFolderUri = preferences.getString(KEY_SHARED_FOLDER_URI, "").orEmpty(),
            sharedFolderLabel = preferences.getString(KEY_SHARED_FOLDER_LABEL, "").orEmpty(),
            backgroundPersistenceEnabled = preferences.getBoolean(KEY_BACKGROUND_PERSISTENCE_ENABLED, false),
        )
    }

    fun saveSharedFolder(uri: String, label: String) {
        preferences.edit()
            .putString(KEY_SHARED_FOLDER_URI, uri)
            .putString(KEY_SHARED_FOLDER_LABEL, label)
            .apply()
    }

    fun clearSharedFolder() {
        preferences.edit()
            .remove(KEY_SHARED_FOLDER_URI)
            .remove(KEY_SHARED_FOLDER_LABEL)
            .apply()
    }

    fun saveBackgroundPersistenceEnabled(enabled: Boolean) {
        preferences.edit()
            .putBoolean(KEY_BACKGROUND_PERSISTENCE_ENABLED, enabled)
            .apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_device_capabilities"
        private const val KEY_SHARED_FOLDER_URI = "shared_folder_uri"
        private const val KEY_SHARED_FOLDER_LABEL = "shared_folder_label"
        private const val KEY_BACKGROUND_PERSISTENCE_ENABLED = "background_persistence_enabled"
    }
}
