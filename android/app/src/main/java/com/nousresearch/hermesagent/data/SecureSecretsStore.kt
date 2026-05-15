package com.nousresearch.hermesagent.data

import android.content.Context
import android.util.Base64
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import org.json.JSONObject

interface AuthSessionSecretStore {
    fun loadAuthSessionSecrets(methodId: String): AuthSessionSecrets
    fun saveAuthSessionSecrets(methodId: String, secrets: AuthSessionSecrets)
    fun clearAuthSessionSecrets(methodId: String)
}

class SecureSecretsStore(context: Context) : AuthSessionSecretStore {
    private val appContext = context.applicationContext

    private val masterKey by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        MasterKey.Builder(appContext)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
    }

    private val preferences by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        EncryptedSharedPreferences.create(
            appContext,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    fun loadApiKey(provider: String): String {
        val key = providerKey(provider)
        val associatedData = "api_key:$key"
        return when (val decoded = loadSealedSecret(key, associatedData)) {
            is SecretIntegrityEnvelope.Opened -> decoded.value
            is SecretIntegrityEnvelope.Legacy -> {
                saveSealedSecret(key, associatedData, decoded.value)
                decoded.value
            }
            SecretIntegrityEnvelope.Invalid,
            SecretIntegrityEnvelope.Missing,
            -> ""
        }
    }

    fun saveApiKey(provider: String, apiKey: String) {
        val key = providerKey(provider)
        saveSealedSecret(key, "api_key:$key", apiKey)
    }

    override fun loadAuthSessionSecrets(methodId: String): AuthSessionSecrets {
        val prefix = authSessionPrefix(methodId)
        val sealedKey = "${prefix}_sealed_v1"
        val sealedValue = readStringSafely(sealedKey)
        if (sealedValue.isNotBlank()) {
            return when (val decoded = SecretIntegrityEnvelope.open(
                storedValue = sealedValue,
                associatedData = "auth_session:$prefix",
                keyBytes = integrityKey,
            )) {
                is SecretIntegrityEnvelope.Opened -> authSessionSecretsFromJson(decoded.value)
                is SecretIntegrityEnvelope.Legacy -> {
                    val secrets = authSessionSecretsFromJson(decoded.value)
                    saveAuthSessionSecrets(methodId, secrets)
                    secrets
                }
                SecretIntegrityEnvelope.Invalid,
                SecretIntegrityEnvelope.Missing,
                -> AuthSessionSecrets()
            }
        }

        val legacySecrets = loadLegacyAuthSessionSecrets(prefix)
        if (legacySecrets.hasAnySecret()) {
            saveAuthSessionSecrets(methodId, legacySecrets)
        }
        return legacySecrets
    }

    override fun saveAuthSessionSecrets(methodId: String, secrets: AuthSessionSecrets) {
        val prefix = authSessionPrefix(methodId)
        preferences.edit()
            .putString(
                "${prefix}_sealed_v1",
                SecretIntegrityEnvelope.seal(
                    value = secrets.toJsonString(),
                    associatedData = "auth_session:$prefix",
                    keyBytes = integrityKey,
                ),
            )
            .remove("${prefix}_access_token")
            .remove("${prefix}_refresh_token")
            .remove("${prefix}_session_token")
            .remove("${prefix}_api_key")
            .apply()
    }

    override fun clearAuthSessionSecrets(methodId: String) {
        val prefix = authSessionPrefix(methodId)
        preferences.edit()
            .remove("${prefix}_sealed_v1")
            .remove("${prefix}_access_token")
            .remove("${prefix}_refresh_token")
            .remove("${prefix}_session_token")
            .remove("${prefix}_api_key")
            .apply()
    }

    private val integrityKey: ByteArray by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        loadOrCreateIntegrityKey()
    }

    private fun loadSealedSecret(storageKey: String, associatedData: String): SecretIntegrityEnvelope.OpenResult {
        return SecretIntegrityEnvelope.open(
            storedValue = readStringSafely(storageKey),
            associatedData = associatedData,
            keyBytes = integrityKey,
        )
    }

    private fun saveSealedSecret(storageKey: String, associatedData: String, value: String) {
        preferences.edit()
            .putString(
                storageKey,
                SecretIntegrityEnvelope.seal(
                    value = value,
                    associatedData = associatedData,
                    keyBytes = integrityKey,
                ),
            )
            .apply()
    }

    private fun loadLegacyAuthSessionSecrets(prefix: String): AuthSessionSecrets {
        return AuthSessionSecrets(
            accessToken = readStringSafely("${prefix}_access_token"),
            refreshToken = readStringSafely("${prefix}_refresh_token"),
            sessionToken = readStringSafely("${prefix}_session_token"),
            apiKey = readStringSafely("${prefix}_api_key"),
        )
    }

    private fun readStringSafely(key: String): String {
        return runCatching { preferences.getString(key, "") }.getOrDefault("").orEmpty()
    }

    private fun loadOrCreateIntegrityKey(): ByteArray {
        decodeStorageBase64(readStringSafely(INTEGRITY_KEY_PREF))
            ?.takeIf { it.size >= SecretIntegrityEnvelope.MIN_KEY_BYTES }
            ?.let { return it }

        val keyBytes = ByteArray(SecretIntegrityEnvelope.MIN_KEY_BYTES)
        SecureRandom().nextBytes(keyBytes)
        preferences.edit()
            .putString(INTEGRITY_KEY_PREF, encodeStorageBase64(keyBytes))
            .commit()
        return keyBytes
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_secrets"
        private const val INTEGRITY_KEY_PREF = "__hermes_secret_integrity_hmac_key_v1"

        private fun providerKey(provider: String): String {
            return provider.lowercase().replace('-', '_') + "_api_key"
        }

        private fun authSessionPrefix(methodId: String): String {
            return "auth_session_" + methodId.lowercase().replace('-', '_')
        }

        private fun encodeStorageBase64(bytes: ByteArray): String =
            Base64.encodeToString(bytes, Base64.NO_WRAP)

        private fun decodeStorageBase64(value: String): ByteArray? =
            runCatching { Base64.decode(value, Base64.NO_WRAP) }.getOrNull()
    }
}

data class AuthSessionSecrets(
    val accessToken: String = "",
    val refreshToken: String = "",
    val sessionToken: String = "",
    val apiKey: String = "",
)

private fun AuthSessionSecrets.hasAnySecret(): Boolean {
    return accessToken.isNotBlank() ||
        refreshToken.isNotBlank() ||
        sessionToken.isNotBlank() ||
        apiKey.isNotBlank()
}

private fun AuthSessionSecrets.toJsonString(): String {
    return JSONObject()
        .put("accessToken", accessToken)
        .put("refreshToken", refreshToken)
        .put("sessionToken", sessionToken)
        .put("apiKey", apiKey)
        .toString()
}

private fun authSessionSecretsFromJson(value: String): AuthSessionSecrets {
    return runCatching {
        val json = JSONObject(value)
        AuthSessionSecrets(
            accessToken = json.optString("accessToken"),
            refreshToken = json.optString("refreshToken"),
            sessionToken = json.optString("sessionToken"),
            apiKey = json.optString("apiKey"),
        )
    }.getOrDefault(AuthSessionSecrets())
}

internal object SecretIntegrityEnvelope {
    const val MIN_KEY_BYTES = 32
    private const val PREFIX = "hermes_secret_v1."
    private const val MAC_ALGORITHM = "HmacSHA256"
    private const val DOMAIN_SEPARATOR = "hermes-secret-envelope-v1"

    sealed interface OpenResult
    data object Missing : OpenResult
    data object Invalid : OpenResult
    data class Legacy(val value: String) : OpenResult
    data class Opened(val value: String) : OpenResult

    fun seal(value: String, associatedData: String, keyBytes: ByteArray): String {
        val payload = encode(value.toByteArray(StandardCharsets.UTF_8))
        val mac = mac(payload = payload, associatedData = associatedData, keyBytes = keyBytes)
        val envelope = JSONObject()
            .put("payload", payload)
            .put("mac", encode(mac))
            .toString()
        return PREFIX + encode(envelope.toByteArray(StandardCharsets.UTF_8))
    }

    fun open(storedValue: String?, associatedData: String, keyBytes: ByteArray): OpenResult {
        val stored = storedValue?.takeIf { it.isNotBlank() } ?: return Missing
        if (!stored.startsWith(PREFIX)) {
            return Legacy(stored)
        }
        return runCatching {
            val envelope = JSONObject(
                String(decode(stored.removePrefix(PREFIX)), StandardCharsets.UTF_8),
            )
            val payload = envelope.optString("payload")
            val storedMac = decode(envelope.optString("mac"))
            val expectedMac = mac(payload = payload, associatedData = associatedData, keyBytes = keyBytes)
            if (payload.isBlank() || !MessageDigest.isEqual(storedMac, expectedMac)) {
                Invalid
            } else {
                Opened(String(decode(payload), StandardCharsets.UTF_8))
            }
        }.getOrDefault(Invalid)
    }

    private fun mac(payload: String, associatedData: String, keyBytes: ByteArray): ByteArray {
        require(keyBytes.size >= MIN_KEY_BYTES) { "Secret integrity key must be at least $MIN_KEY_BYTES bytes" }
        val mac = Mac.getInstance(MAC_ALGORITHM)
        mac.init(SecretKeySpec(keyBytes, MAC_ALGORITHM))
        val material = DOMAIN_SEPARATOR + "\u0000" + associatedData + "\u0000" + payload
        return mac.doFinal(material.toByteArray(StandardCharsets.UTF_8))
    }

    private fun encode(bytes: ByteArray): String =
        Base64.encodeToString(bytes, Base64.NO_WRAP)

    private fun decode(value: String): ByteArray =
        Base64.decode(value, Base64.NO_WRAP)
}
