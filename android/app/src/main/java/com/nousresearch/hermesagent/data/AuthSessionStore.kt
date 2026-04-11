package com.nousresearch.hermesagent.data

import android.content.Context
import android.net.Uri
import org.json.JSONObject

class AuthSessionStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun loadSessions(): List<AuthSession> {
        return AuthCatalog.options.map { option -> loadSession(option.id) ?: defaultSession(option) }
    }

    fun loadSession(methodId: String): AuthSession? {
        val raw = preferences.getString(sessionKey(methodId), null) ?: return null
        return runCatching {
            val json = JSONObject(raw)
            AuthSession(
                methodId = json.optString("methodId", methodId),
                label = json.optString("label", AuthCatalog.find(methodId)?.label.orEmpty()),
                scope = json.optString("scope", AuthScope.AppAccount.name)
                    .let { runCatching { AuthScope.valueOf(it) }.getOrDefault(AuthScope.AppAccount) },
                runtimeProvider = json.optString("runtimeProvider"),
                signedIn = json.optBoolean("signedIn", false),
                status = json.optString("status", "Not signed in"),
                email = json.optString("email"),
                phone = json.optString("phone"),
                displayName = json.optString("displayName"),
                accessToken = json.optString("accessToken"),
                refreshToken = json.optString("refreshToken"),
                sessionToken = json.optString("sessionToken"),
                apiKey = json.optString("apiKey"),
                baseUrl = json.optString("baseUrl"),
                model = json.optString("model"),
                updatedAtEpochMs = json.optLong("updatedAtEpochMs", System.currentTimeMillis()),
            )
        }.getOrNull()
    }

    fun saveSession(session: AuthSession) {
        val json = JSONObject()
            .put("methodId", session.methodId)
            .put("label", session.label)
            .put("scope", session.scope.name)
            .put("runtimeProvider", session.runtimeProvider)
            .put("signedIn", session.signedIn)
            .put("status", session.status)
            .put("email", session.email)
            .put("phone", session.phone)
            .put("displayName", session.displayName)
            .put("accessToken", session.accessToken)
            .put("refreshToken", session.refreshToken)
            .put("sessionToken", session.sessionToken)
            .put("apiKey", session.apiKey)
            .put("baseUrl", session.baseUrl)
            .put("model", session.model)
            .put("updatedAtEpochMs", session.updatedAtEpochMs)
        preferences.edit().putString(sessionKey(session.methodId), json.toString()).apply()
    }

    fun clearSession(methodId: String) {
        preferences.edit().remove(sessionKey(methodId)).apply()
    }

    fun savePendingRequest(request: PendingAuthRequest) {
        val json = JSONObject()
            .put("state", request.state)
            .put("methodId", request.methodId)
            .put("startUrl", request.startUrl)
            .put("createdAtEpochMs", request.createdAtEpochMs)
        preferences.edit().putString(KEY_PENDING_REQUEST, json.toString()).apply()
    }

    fun loadPendingRequest(): PendingAuthRequest? {
        val raw = preferences.getString(KEY_PENDING_REQUEST, null) ?: return null
        return runCatching {
            val json = JSONObject(raw)
            PendingAuthRequest(
                state = json.optString("state"),
                methodId = json.optString("methodId"),
                startUrl = json.optString("startUrl"),
                createdAtEpochMs = json.optLong("createdAtEpochMs", System.currentTimeMillis()),
            )
        }.getOrNull()
    }

    fun clearPendingRequest() {
        preferences.edit().remove(KEY_PENDING_REQUEST).apply()
    }

    fun consumeAuthCallback(uri: Uri): AuthSession? {
        if (!isAuthCallback(uri)) {
            return null
        }

        val pending = loadPendingRequest()
        val methodId = uri.getQueryParameter("method")
            ?.takeIf { it.isNotBlank() }
            ?: pending?.methodId
            ?: return null
        val option = AuthCatalog.find(methodId) ?: return null
        val callbackState = uri.getQueryParameter("state").orEmpty()
        val stateMismatch = pending != null && pending.state.isNotBlank() && callbackState != pending.state
        val error = uri.getQueryParameter("error").orEmpty()
        val signedIn = !stateMismatch && error.isBlank() && (
            !uri.getQueryParameter("email").isNullOrBlank()
                || !uri.getQueryParameter("phone").isNullOrBlank()
                || !uri.getQueryParameter("api_key").isNullOrBlank()
                || !uri.getQueryParameter("access_token").isNullOrBlank()
                || !uri.getQueryParameter("session_token").isNullOrBlank()
        )

        val session = AuthSession(
            methodId = option.id,
            label = option.label,
            scope = option.scope,
            runtimeProvider = uri.getQueryParameter("provider")
                ?.takeIf { it.isNotBlank() }
                ?: option.runtimeProvider,
            signedIn = signedIn,
            status = when {
                stateMismatch -> "Auth callback rejected: state mismatch"
                error.isNotBlank() -> "Auth failed: $error"
                signedIn -> "Signed in with ${option.label}"
                else -> "Auth callback received but no credentials were returned"
            },
            email = uri.getQueryParameter("email").orEmpty(),
            phone = uri.getQueryParameter("phone").orEmpty(),
            displayName = uri.getQueryParameter("display_name").orEmpty(),
            accessToken = uri.getQueryParameter("access_token").orEmpty(),
            refreshToken = uri.getQueryParameter("refresh_token").orEmpty(),
            sessionToken = uri.getQueryParameter("session_token").orEmpty(),
            apiKey = uri.getQueryParameter("api_key").orEmpty(),
            baseUrl = uri.getQueryParameter("base_url").orEmpty(),
            model = uri.getQueryParameter("model").orEmpty(),
        )
        saveSession(session)
        clearPendingRequest()
        return session
    }

    fun hasSignedInAppAccount(): Boolean {
        return loadSessions().any { it.scope == AuthScope.AppAccount && it.signedIn }
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_auth"
        private const val KEY_PENDING_REQUEST = "pending_request"
        const val CALLBACK_SCHEME = "hermesagent"
        const val CALLBACK_HOST = "auth"
        const val CALLBACK_PATH = "/callback"
        const val CALLBACK_URI = "$CALLBACK_SCHEME://$CALLBACK_HOST$CALLBACK_PATH"

        fun isAuthCallback(uri: Uri?): Boolean {
            if (uri == null) return false
            return uri.scheme.equals(CALLBACK_SCHEME, ignoreCase = true)
                && uri.host.equals(CALLBACK_HOST, ignoreCase = true)
                && (uri.path ?: "").startsWith(CALLBACK_PATH)
        }

        private fun sessionKey(methodId: String): String = "session_${methodId.lowercase()}"

        private fun defaultSession(option: AuthOption): AuthSession {
            return AuthSession(
                methodId = option.id,
                label = option.label,
                scope = option.scope,
                runtimeProvider = option.runtimeProvider,
                baseUrl = option.defaultBaseUrl,
                model = option.defaultModel,
                status = "Not signed in",
                signedIn = false,
            )
        }
    }
}
