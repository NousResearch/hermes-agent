package com.nousresearch.hermesagent

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.nousresearch.hermesagent.data.AuthScope
import com.nousresearch.hermesagent.data.AuthSession
import com.nousresearch.hermesagent.data.AuthSessionStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AuthSecureStorageInstrumentedTest {
    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun qwenOAuthCallbackCredentialsStayOutOfPlainSessionPreferences() {
        context.deleteSharedPreferences("hermes_android_auth")
        context.deleteSharedPreferences("hermes_android_secrets")

        val store = AuthSessionStore(context)
        val session = AuthSession(
            methodId = "qwen-oauth",
            label = "Qwen OAuth",
            scope = AuthScope.RuntimeProvider,
            runtimeProvider = "qwen-oauth",
            signedIn = true,
            status = "Signed in with Qwen OAuth",
            accessToken = "qwen-access-secret",
            refreshToken = "qwen-refresh-secret",
            sessionToken = "qwen-session-secret",
            apiKey = "qwen-api-secret",
            baseUrl = "https://portal.qwen.ai/v1",
            model = "qwen3-coder-plus",
        )

        try {
            store.saveSession(session)

            val raw = context
                .getSharedPreferences("hermes_android_auth", Context.MODE_PRIVATE)
                .getString("session_qwen-oauth", "")
                .orEmpty()
            assertTrue(raw.isNotBlank())
            assertFalse(raw.contains("qwen-access-secret"))
            assertFalse(raw.contains("qwen-refresh-secret"))
            assertFalse(raw.contains("qwen-session-secret"))
            assertFalse(raw.contains("qwen-api-secret"))
            assertFalse(raw.contains("accessToken"))
            assertFalse(raw.contains("refreshToken"))
            assertFalse(raw.contains("sessionToken"))
            assertFalse(raw.contains("apiKey"))

            val loaded = AuthSessionStore(context).loadSession("qwen-oauth")
            assertNotNull(loaded)
            assertEquals("qwen-access-secret", loaded?.accessToken)
            assertEquals("qwen-refresh-secret", loaded?.refreshToken)
            assertEquals("qwen-session-secret", loaded?.sessionToken)
            assertEquals("qwen-api-secret", loaded?.apiKey)
        } finally {
            store.clearSession("qwen-oauth")
        }
    }
}
