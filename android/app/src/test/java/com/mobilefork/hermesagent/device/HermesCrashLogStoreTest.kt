package com.mobilefork.hermesagent.device

import android.content.Context
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesCrashLogStoreTest {
    private val context: Context = RuntimeEnvironment.getApplication()

    @Before
    fun resetCrashLogs() {
        HermesCrashLogStore.clearAllForTest(context)
    }

    @Test
    fun redactsSensitiveValuesBeforePersistingCrash() {
        val crash = HermesCrashLogStore.recordCrashForTest(
            context = context,
            throwable = IllegalStateException(
                "failed for ady@example.com Authorization: Bearer bearer-token-123456789 " +
                    "OPENAI_API_KEY=sk-test123456789 phone +1 415 555 1212 C:\\Users\\Ady\\secret.txt",
            ),
            threadName = "main-C:\\Users\\Ady",
            nowMs = 1_717_171_717_000L,
        )
        val persistedText = HermesCrashLogStore.exportLogsText(context)
        val status = HermesCrashLogStore.statusSnapshot(context)

        assertEquals("java.lang.IllegalStateException", crash.getString("exception_type"))
        assertTrue(status.hasLastCrash)
        assertTrue(status.previewLines.any { it.contains("IllegalStateException") || it.contains("[REDACTED") })
        assertFalse(persistedText.contains("ady@example.com"))
        assertFalse(persistedText.contains("bearer-token-123456789"))
        assertFalse(persistedText.contains("sk-test123456789"))
        assertFalse(persistedText.contains("+1 415 555 1212"))
        assertFalse(persistedText.contains("C:\\Users\\Ady"))
        assertTrue(persistedText.contains("[REDACTED_EMAIL]"))
        assertTrue(persistedText.contains("[REDACTED_BEARER]"))
        assertTrue(persistedText.contains("[REDACTED_SECRET]"))
        assertTrue(persistedText.contains("[REDACTED_PHONE]"))
        assertTrue(persistedText.contains("[REDACTED_USER]"))
    }

    @Test
    fun diagnosticsBridgeExposesCrashStatusAndExport() {
        HermesCrashLogStore.recordCrashForTest(
            context = context,
            throwable = RuntimeException("session_token=qwen-secret-token user jane@example.net"),
            nowMs = 1_717_171_718_000L,
        )

        val status = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "crash_log_status"))
        val export = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "diagnostics_log_export"))

        assertTrue(status.getBoolean("success"))
        assertTrue(status.getBoolean("last_crash_present"))
        assertEquals("diagnostics_log_export", status.getString("export_action"))
        assertTrue(export.getBoolean("success"))
        assertEquals("text/plain", export.getString("mime_type"))
        assertTrue(export.getString("content").contains("java.lang.RuntimeException"))
        assertFalse(export.getString("content").contains("qwen-secret-token"))
        assertFalse(export.getString("content").contains("jane@example.net"))
        assertTrue(export.getString("content").contains("[REDACTED_SECRET]"))
        assertTrue(export.getString("content").contains("[REDACTED_EMAIL]"))
    }

    @Test
    fun clearLastCrashKeepsDiagnosticsExportUsable() {
        HermesCrashLogStore.recordCrashForTest(
            context = context,
            throwable = IllegalArgumentException("api_key=sk-clear-test123456"),
            nowMs = 1_717_171_719_000L,
        )

        HermesCrashLogStore.clearLastCrash(context)
        val status = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "crash_log_status"))
        val export = HermesCrashLogStore.exportLogsText(context)

        assertFalse(status.getBoolean("last_crash_present"))
        assertTrue(export.contains("Recent diagnostic events"))
        assertFalse(export.contains("sk-clear-test123456"))
        assertTrue(export.contains("[REDACTED_SECRET]"))
    }
}
