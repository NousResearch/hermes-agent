package com.nousresearch.hermesagent.data

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

interface AuthSessionSecretStore {
    fun loadAuthSessionSecrets(methodId: String): AuthSessionSecrets
    fun saveAuthSessionSecrets(methodId: String, secrets: AuthSessionSecrets)
    fun clearAuthSessionSecrets(methodId: String)
}

class SecureSecretsStore(context: Context) : AuthSessionSecretStore {
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

    override fun loadAuthSessionSecrets(methodId: String): AuthSessionSecrets {
        val prefix = authSessionPrefix(methodId)
        return AuthSessionSecrets(
            accessToken = preferences.getString("${prefix}_access_token", "").orEmpty(),
            refreshToken = preferences.getString("${prefix}_refresh_token", "").orEmpty(),
            sessionToken = preferences.getString("${prefix}_session_token", "").orEmpty(),
            apiKey = preferences.getString("${prefix}_api_key", "").orEmpty(),
        )
    }

    override fun saveAuthSessionSecrets(methodId: String, secrets: AuthSessionSecrets) {
        preferences.edit()
            .putString("${authSessionPrefix(methodId)}_access_token", secrets.accessToken)
            .putString("${authSessionPrefix(methodId)}_refresh_token", secrets.refreshToken)
            .putString("${authSessionPrefix(methodId)}_session_token", secrets.sessionToken)
            .putString("${authSessionPrefix(methodId)}_api_key", secrets.apiKey)
            .apply()
    }

    override fun clearAuthSessionSecrets(methodId: String) {
        preferences.edit()
            .remove("${authSessionPrefix(methodId)}_access_token")
            .remove("${authSessionPrefix(methodId)}_refresh_token")
            .remove("${authSessionPrefix(methodId)}_session_token")
            .remove("${authSessionPrefix(methodId)}_api_key")
            .apply()
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_secrets"

        private fun providerKey(provider: String): String {
            return provider.lowercase().replace('-', '_') + "_api_key"
        }

        private fun authSessionPrefix(methodId: String): String {
            return "auth_session_" + methodId.lowercase().replace('-', '_')
        }
    }
}

data class AuthSessionSecrets(
    val accessToken: String = "",
    val refreshToken: String = "",
    val sessionToken: String = "",
    val apiKey: String = "",
)
