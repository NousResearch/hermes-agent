package com.mobilefork.hermesagent

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.mobilefork.hermesagent.data.LocalModelDownloadRecord
import com.mobilefork.hermesagent.data.LocalModelDownloadStore
import com.mobilefork.hermesagent.device.HermesDeviceDiagnosticsBridge
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class HermesPhoneGoalPreflightInstrumentedTest {
    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun phoneGoalPreflightRunsInsideAndroidProcessAndReportsModelImportReadiness() {
        val modelFile = File(context.filesDir, "phone-goal-preflight.gguf").apply {
            writeText("HERMES_PHONE_GOAL_PREFLIGHT")
        }
        LocalModelDownloadStore(context).apply {
            saveDownloads(emptyList())
            upsertDownload(
                LocalModelDownloadRecord(
                    id = "phone-goal-preflight-model",
                    title = "Phone goal preflight model",
                    sourceUrl = "",
                    repoOrUrl = "",
                    filePath = modelFile.name,
                    revision = "local",
                    runtimeFlavor = "GGUF",
                    destinationFileName = modelFile.name,
                    destinationPath = modelFile.absolutePath,
                    downloadManagerId = -1L,
                    totalBytes = modelFile.length(),
                    downloadedBytes = modelFile.length(),
                    status = "completed",
                    statusMessage = "Imported for phone-goal preflight",
                ),
            )
            setPreferredDownloadId("phone-goal-preflight-model")
        }

        val result = JSONObject(
            HermesDeviceDiagnosticsBridge.performActionJson(context, "social_gmail_goal_preflight"),
        )

        assertTrue(result.toString(), result.getBoolean("success"))
        assertEquals("social_gmail_goal_preflight", result.getString("action"))
        assertTrue(result.getBoolean("local_model_import_button_supported"))
        assertTrue(result.getBoolean("local_model_import_uses_android_open_document"))
        assertTrue(result.getJSONObject("preferred_local_model").getBoolean("ready"))
        assertEquals(modelFile.length(), result.getJSONObject("preferred_local_model").getLong("file_bytes"))
        assertTrue(result.getBoolean("physical_phone_required"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.getJSONObject("tiktok").getJSONArray("candidate_package_names").toString().contains("com.zhiliaoapp.musically"))
        assertEquals("com.instagram.android", result.getJSONObject("instagram").getJSONArray("candidate_package_names").getString(0))
        assertEquals("com.google.android.gm", result.getJSONObject("gmail").getJSONArray("candidate_package_names").getString(0))
        if (result.getJSONObject("android_device_identity").getBoolean("likely_emulator")) {
            assertFalse(result.getBoolean("physical_phone_detected"))
            assertFalse("The emulator should not prove the full physical-phone social/Gmail goal", result.getBoolean("ready_for_full_goal"))
        }

        modelFile.delete()
    }
}
