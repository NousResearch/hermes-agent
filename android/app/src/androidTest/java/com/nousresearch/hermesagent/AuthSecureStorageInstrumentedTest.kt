package com.nousresearch.hermesagent

import android.app.Application
import android.content.Context
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.chaquo.python.Python
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.AuthScope
import com.nousresearch.hermesagent.data.AuthSession
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.data.SecureSecretsStore
import com.nousresearch.hermesagent.ui.settings.SettingsViewModel
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

    private val app: Application
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

    @Test
    fun settingsImportSavedQwenOAuthBundleMirrorsPythonCredentialIntoEncryptedPrefs() {
        context.deleteSharedPreferences("hermes_android_settings")
        context.deleteSharedPreferences("hermes_android_secrets")
        HermesRuntimeManager.ensurePythonStarted(app)
        val python = Python.getInstance()
        python.getModule("hermes_android.auth_bridge").callAttr(
            "write_provider_auth_bundle",
            "qwen-oauth",
            "",
            "qwen-import-access",
            "",
            "qwen-import-refresh",
            "https://portal.qwen.ai/v1",
        )
        try {
            AppSettingsStore(app).save(
                AppSettings(
                    provider = "qwen-oauth",
                    baseUrl = "https://portal.qwen.ai/v1",
                    model = "qwen3-coder-plus",
                )
            )

            lateinit var viewModel: SettingsViewModel
            InstrumentationRegistry.getInstrumentation().runOnMainSync {
                viewModel = SettingsViewModel(app)
                viewModel.importSavedProviderCredential()
            }

            val deadline = SystemClock.elapsedRealtime() + 60_000L
            var status = ""
            while (SystemClock.elapsedRealtime() < deadline) {
                status = viewModel.uiState.value.status
                if (status.contains("Imported saved Hermes credential") || status.contains("failed")) {
                    break
                }
                Thread.sleep(250L)
            }
            assertTrue(status, status.contains("Imported saved Hermes credential"))
            assertEquals("qwen-import-access", SecureSecretsStore(app).loadApiKey("qwen-oauth"))
            assertEquals("qwen-import-access", viewModel.uiState.value.apiKey)
        } finally {
            python.getModule("hermes_android.auth_bridge").callAttr("clear_provider_auth_bundle", "qwen-oauth")
        }
    }
}
