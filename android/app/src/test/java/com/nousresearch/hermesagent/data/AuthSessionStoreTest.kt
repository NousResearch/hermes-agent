package com.nousresearch.hermesagent.data

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class AuthSessionStoreTest {
    @Test
    fun evaluateAuthCallback_rejectsMissingPendingRequest() {
        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=google&state=expected&email=user@example.com"),
            pending = null,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertFalse(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: no pending sign-in request", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_rejectsExpiredPendingRequest() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "google",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 1L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=google&state=expected&email=user@example.com"),
            pending = pending,
            nowEpochMs = (16 * 60 * 1000L),
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback expired. Start sign-in again.", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_rejectsMethodMismatch() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "google",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=email&state=expected&email=user@example.com"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertEquals("google", result.session?.methodId)
        assertEquals("Auth callback rejected: method mismatch", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_rejectsProviderMismatchForRuntimeProvider() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "chatgpt",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=chatgpt&state=expected&provider=anthropic&access_token=token"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: provider mismatch", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_requiresProviderCredentials() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "chatgpt",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=chatgpt&state=expected&provider=chatgpt-web"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: no provider credentials were returned", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_requiresAppIdentity() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "google",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=google&state=expected"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: no account identity returned", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_acceptsRuntimeProviderCredentials() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "chatgpt",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse(
                "${AuthSessionStore.CALLBACK_URI}?method=chatgpt&state=expected&provider=chatgpt-web&access_token=access&session_token=session&base_url=https%3A%2F%2Fchatgpt.com%2Fbackend-api%2Ff%3Fignored%3D1&model=gpt-5-thinking"
            ),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertTrue(result.session?.signedIn ?: false)
        assertEquals("chatgpt-web", result.session?.runtimeProvider)
        assertEquals("https://chatgpt.com/backend-api/f", result.session?.baseUrl)
        assertEquals("gpt-5-thinking", result.session?.model)
        assertEquals("Signed in with ChatGPT", result.session?.status)
    }

    @Test
    fun saveSession_keepsRuntimeProviderCredentialsOutOfPlainAuthPreferences() {
        val context = cleanContext()
        val store = AuthSessionStore(context, InMemoryAuthSessionSecretStore())
        val session = AuthSession(
            methodId = "chatgpt",
            label = "ChatGPT",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "chatgpt-web",
            signedIn = true,
            status = "Signed in with ChatGPT",
            accessToken = "access-secret",
            refreshToken = "refresh-secret",
            sessionToken = "session-secret",
            apiKey = "api-secret",
            baseUrl = "https://chatgpt.com/backend-api/f",
            model = "gpt-5-thinking",
        )

        store.saveSession(session)

        val raw = plainAuthPreferences(context).getString("session_chatgpt", "").orEmpty()
        assertTrue(raw.isNotBlank())
        assertFalse(raw.contains("access-secret"))
        assertFalse(raw.contains("refresh-secret"))
        assertFalse(raw.contains("session-secret"))
        assertFalse(raw.contains("api-secret"))
        assertFalse(raw.contains("accessToken"))
        assertFalse(raw.contains("refreshToken"))
        assertFalse(raw.contains("sessionToken"))
        assertFalse(raw.contains("apiKey"))

        val loaded = store.loadSession("chatgpt")
        assertNotNull(loaded)
        assertEquals("access-secret", loaded?.accessToken)
        assertEquals("refresh-secret", loaded?.refreshToken)
        assertEquals("session-secret", loaded?.sessionToken)
        assertEquals("api-secret", loaded?.apiKey)
    }

    @Test
    fun pendingAuthRequestPersistsOpenRouterPkceMetadata() {
        val context = cleanContext()
        val store = AuthSessionStore(context, InMemoryAuthSessionSecretStore())
        val pending = PendingAuthRequest(
            state = "state-123",
            methodId = "openrouter",
            startUrl = "https://openrouter.ai/auth",
            authProvider = "openrouter-oauth",
            codeVerifier = "verifier-123",
            codeChallengeMethod = "S256",
            createdAtEpochMs = 123L,
        )

        store.savePendingRequest(pending)

        val loaded = store.loadPendingRequest()
        assertEquals("state-123", loaded?.state)
        assertEquals("openrouter", loaded?.methodId)
        assertEquals("openrouter-oauth", loaded?.authProvider)
        assertEquals("verifier-123", loaded?.codeVerifier)
        assertEquals("S256", loaded?.codeChallengeMethod)
        assertEquals(123L, loaded?.createdAtEpochMs)
    }

    @Test
    fun loadSession_migratesLegacyPlaintextCredentialsIntoSecureStore() {
        val context = cleanContext()
        val secretStore = InMemoryAuthSessionSecretStore()
        plainAuthPreferences(context).edit().putString(
            "session_chatgpt",
            JSONObject()
                .put("methodId", "chatgpt")
                .put("label", "ChatGPT")
                .put("scope", AuthScope.RuntimeProvider.name)
                .put("runtimeProvider", "chatgpt-web")
                .put("signedIn", true)
                .put("status", "Signed in with ChatGPT")
                .put("accessToken", "legacy-access")
                .put("refreshToken", "legacy-refresh")
                .put("sessionToken", "legacy-session")
                .put("apiKey", "legacy-api")
                .put("baseUrl", "https://chatgpt.com/backend-api/f")
                .put("model", "gpt-5-thinking")
                .put("updatedAtEpochMs", 100L)
                .toString(),
        ).commit()

        val loaded = AuthSessionStore(context, secretStore).loadSession("chatgpt")

        assertEquals("legacy-access", loaded?.accessToken)
        assertEquals("legacy-refresh", loaded?.refreshToken)
        assertEquals("legacy-session", loaded?.sessionToken)
        assertEquals("legacy-api", loaded?.apiKey)
        val migratedRaw = plainAuthPreferences(context).getString("session_chatgpt", "").orEmpty()
        assertFalse(migratedRaw.contains("legacy-access"))
        assertFalse(migratedRaw.contains("accessToken"))
    }

    private fun cleanContext(): Context {
        val context = RuntimeEnvironment.getApplication().applicationContext
        context.deleteSharedPreferences("hermes_android_auth")
        context.deleteSharedPreferences("hermes_android_secrets")
        return context
    }

    private fun plainAuthPreferences(context: Context) =
        context.getSharedPreferences("hermes_android_auth", Context.MODE_PRIVATE)

    private class InMemoryAuthSessionSecretStore : AuthSessionSecretStore {
        private val secretsByMethod = mutableMapOf<String, AuthSessionSecrets>()

        override fun loadAuthSessionSecrets(methodId: String): AuthSessionSecrets =
            secretsByMethod[methodId].orEmpty()

        override fun saveAuthSessionSecrets(methodId: String, secrets: AuthSessionSecrets) {
            secretsByMethod[methodId] = secrets
        }

        override fun clearAuthSessionSecrets(methodId: String) {
            secretsByMethod.remove(methodId)
        }

        private fun AuthSessionSecrets?.orEmpty(): AuthSessionSecrets = this ?: AuthSessionSecrets()
    }
}
