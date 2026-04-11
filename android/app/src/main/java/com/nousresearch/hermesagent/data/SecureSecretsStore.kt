package com.nousresearch.hermesagent.data

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

class SecureSecretsStore(context: Context) {
    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val preferences = EncryptedSharedPreferences.create(
        context,
        PREFS_NAME,
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

    fun loadApiKey(provider: String): String {
        return preferences.getString(providerKey(provider), "").orEmpty()
    }

    fun saveApiKey(provider: String, apiKey: String) {
        preferences.edit().putString(providerKey(provider), apiKey).apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_secrets"

        private fun providerKey(provider: String): String {
            return provider.lowercase().replace('-', '_') + "_api_key"
        }
    }
}
