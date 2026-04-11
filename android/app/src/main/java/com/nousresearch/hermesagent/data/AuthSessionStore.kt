package com.nousresearch.hermesagent.data

import android.content.Context
import android.net.Uri
import org.json.JSONObject

class AuthSessionStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    data class AuthCallbackEvaluation(
        val session: AuthSession? = null,
        val consumed: Boolean = false,
        val clearPending: Boolean = false,
    )

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
                updatedAtEpochMs = json.optLong("updatedAtEpochMs", 0),
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
        val evaluation = evaluateAuthCallback(uri, loadPendingRequest())
        if (!evaluation.consumed) {
            return null
        }
        if (evaluation.clearPending) {
            clearPendingRequest()
        }
        evaluation.session?.let(::saveSession)
        return evaluation.session
    }

    fun hasSignedInAppAccount(): Boolean {
        return loadSessions().any { it.scope == AuthScope.AppAccount && it.signedIn }
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_auth"
        private const val KEY_PENDING_REQUEST = "pending_request"
        private const val PENDING_REQUEST_MAX_AGE_MS = 15 * 60 * 1000L
        private const val DEFAULT_STATUS = "Not signed in"
        private const val MAX_STATE_LENGTH = 160
        private const val MAX_STATUS_LENGTH = 240
        private const val MAX_EMAIL_LENGTH = 320
        private const val MAX_PHONE_LENGTH = 64
        private const val MAX_NAME_LENGTH = 120
        private const val MAX_MODEL_LENGTH = 160
        private const val MAX_URL_LENGTH = 2048
        private const val MAX_TOKEN_LENGTH = 8192

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

        fun isPendingRequestExpired(
            request: PendingAuthRequest,
            nowEpochMs: Long = System.currentTimeMillis(),
        ): Boolean {
            return nowEpochMs - request.createdAtEpochMs > PENDING_REQUEST_MAX_AGE_MS
        }

        internal fun evaluateAuthCallback(
            uri: Uri?,
            pending: PendingAuthRequest?,
            nowEpochMs: Long = System.currentTimeMillis(),
        ): AuthCallbackEvaluation {
            if (!isAuthCallback(uri)) {
                return AuthCallbackEvaluation()
            }
            val callbackUri = uri ?: return AuthCallbackEvaluation()
            val methodIdFromUri = normalizeMethodId(callbackUri.getQueryParameter("method"))
            val expectedOption = pending?.methodId?.let(AuthCatalog::find)
            val callbackOption = methodIdFromUri?.let(AuthCatalog::find)
            val option = expectedOption ?: callbackOption
                ?: return AuthCallbackEvaluation(consumed = true, clearPending = pending != null)

            if (pending == null) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback rejected: no pending sign-in request", nowEpochMs),
                    consumed = true,
                    clearPending = false,
                )
            }

            if (isPendingRequestExpired(pending, nowEpochMs)) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback expired. Start sign-in again.", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }

            if (methodIdFromUri != null && methodIdFromUri != pending.methodId) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback rejected: method mismatch", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }

            val callbackState = sanitizeValue(callbackUri.getQueryParameter("state"), MAX_STATE_LENGTH)
            if (callbackState.isBlank() || pending.state.isBlank() || callbackState != pending.state) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback rejected: state mismatch", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }

            val error = sanitizeStatus(
                callbackUri.getQueryParameter("error_description")
                    .orEmpty()
                    .ifBlank { callbackUri.getQueryParameter("error").orEmpty() }
            )
            if (error.isNotBlank()) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth failed: $error", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }

            val runtimeProvider = normalizeProvider(callbackUri.getQueryParameter("provider")).ifBlank {
                option.runtimeProvider
            }
            if (option.scope == AuthScope.RuntimeProvider && runtimeProvider != option.runtimeProvider) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback rejected: provider mismatch", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }

            val email = sanitizeValue(callbackUri.getQueryParameter("email"), MAX_EMAIL_LENGTH)
            val phone = sanitizeValue(callbackUri.getQueryParameter("phone"), MAX_PHONE_LENGTH)
            val displayName = sanitizeValue(callbackUri.getQueryParameter("display_name"), MAX_NAME_LENGTH)
            val accessToken = sanitizeToken(callbackUri.getQueryParameter("access_token"))
            val refreshToken = sanitizeToken(callbackUri.getQueryParameter("refresh_token"))
            val sessionToken = sanitizeToken(callbackUri.getQueryParameter("session_token"))
            val apiKey = sanitizeToken(callbackUri.getQueryParameter("api_key"))
            val baseUrl = sanitizeHttpUrl(callbackUri.getQueryParameter("base_url")).ifBlank {
                option.defaultBaseUrl
            }
            val model = sanitizeValue(callbackUri.getQueryParameter("model"), MAX_MODEL_LENGTH)
                .ifBlank { option.defaultModel }

            val hasIdentity = email.isNotBlank() || phone.isNotBlank() || displayName.isNotBlank()
            val hasProviderCredentials = apiKey.isNotBlank() || accessToken.isNotBlank() || sessionToken.isNotBlank()
            if (option.scope == AuthScope.AppAccount && !hasIdentity) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback rejected: no account identity returned", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }
            if (option.scope == AuthScope.RuntimeProvider && !hasProviderCredentials) {
                return AuthCallbackEvaluation(
                    session = failureSession(option, "Auth callback rejected: no provider credentials were returned", nowEpochMs),
                    consumed = true,
                    clearPending = true,
                )
            }

            val session = AuthSession(
                methodId = option.id,
                label = option.label,
                scope = option.scope,
                runtimeProvider = runtimeProvider,
                signedIn = true,
                status = successStatus(option),
                email = email,
                phone = phone,
                displayName = displayName,
                accessToken = accessToken,
                refreshToken = refreshToken,
                sessionToken = sessionToken,
                apiKey = apiKey,
                baseUrl = baseUrl,
                model = model,
                updatedAtEpochMs = nowEpochMs,
            )
            return AuthCallbackEvaluation(
                session = session,
                consumed = true,
                clearPending = true,
            )
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
                status = DEFAULT_STATUS,
                signedIn = false,
                updatedAtEpochMs = 0,
            )
        }

        private fun failureSession(option: AuthOption, status: String, nowEpochMs: Long): AuthSession {
            return defaultSession(option).copy(
                status = sanitizeStatus(status),
                updatedAtEpochMs = nowEpochMs,
            )
        }

        private fun successStatus(option: AuthOption): String {
            return sanitizeStatus("Signed in with ${option.label}")
        }

        private fun normalizeMethodId(value: String?): String? {
            return sanitizeValue(value, MAX_NAME_LENGTH).lowercase().ifBlank { null }
        }

        private fun normalizeProvider(value: String?): String {
            return sanitizeValue(value, MAX_NAME_LENGTH).lowercase()
        }

        private fun sanitizeStatus(value: String): String {
            return sanitizeValue(value, MAX_STATUS_LENGTH)
        }

        private fun sanitizeToken(value: String?): String {
            return sanitizeValue(value, MAX_TOKEN_LENGTH)
        }

        private fun sanitizeHttpUrl(value: String?): String {
            val candidate = sanitizeValue(value, MAX_URL_LENGTH)
            if (candidate.isBlank()) {
                return ""
            }
            val parsed = runCatching { Uri.parse(candidate) }.getOrNull() ?: return ""
            val scheme = parsed.scheme?.lowercase().orEmpty()
            val authority = parsed.encodedAuthority.orEmpty()
            if (scheme !in setOf("http", "https") || authority.isBlank()) {
                return ""
            }
            val normalizedPath = parsed.encodedPath
                ?.trim()
                ?.trimEnd('/')
                ?.takeIf { it.isNotBlank() && it != "/" }
            return Uri.Builder()
                .scheme(scheme)
                .encodedAuthority(authority)
                .apply {
                    if (!normalizedPath.isNullOrBlank()) {
                        encodedPath(normalizedPath)
                    }
                }
                .build()
                .toString()
                .trimEnd('/')
        }

        private fun sanitizeValue(value: String?, maxLength: Int): String {
            return value.orEmpty()
                .replace(Regex("[\\u0000-\\u001F]"), " ")
                .trim()
                .take(maxLength)
        }
    }
}
