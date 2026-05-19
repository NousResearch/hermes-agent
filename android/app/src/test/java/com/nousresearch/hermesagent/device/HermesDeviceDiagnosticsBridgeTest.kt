package com.nousresearch.hermesagent.device

import android.content.Context
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesDeviceDiagnosticsBridgeTest {
    private val context: Context = RuntimeEnvironment.getApplication()

    @Test
    fun mapsWifiFrequenciesToChannels() {
        assertEquals(1, HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(2412))
        assertEquals(6, HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(2437))
        assertEquals(11, HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(2462))
        assertEquals(14, HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(2484))
        assertEquals(36, HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(5180))
        assertEquals(1, HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(5955))
        assertNull(HermesDeviceDiagnosticsBridge.channelForFrequencyMhz(1000))
    }

    @Test
    fun emulatorDetectionSeparatesVirtualTargetsFromPhysicalPhoneEvidence() {
        assertTrue(
            HermesDeviceDiagnosticsBridge.isLikelyEmulatorDevice(
                fingerprint = "google/sdk_gphone64_x86_64/emu64xa:15/AP31/userdebug/test-keys",
                model = "sdk_gphone64_x86_64",
                manufacturer = "Google",
                brand = "google",
                device = "emu64xa",
                product = "sdk_gphone64_x86_64",
                hardware = "ranchu",
            ),
        )
        assertFalse(
            HermesDeviceDiagnosticsBridge.isLikelyEmulatorDevice(
                fingerprint = "google/oriole/oriole:15/AP31/user/release-keys",
                model = "Pixel 6",
                manufacturer = "Google",
                brand = "google",
                device = "oriole",
                product = "oriole",
                hardware = "oriole",
            ),
        )
    }

    @Test
    fun signalCapabilityStatusIsHonestAboutRfHardwareLimits() {
        val result = HermesDeviceDiagnosticsBridge.signalCapabilityStatusJson(context)

        assertTrue(result.getBoolean("success"))
        assertFalse(result.getBoolean("am_fm_public_android_scan_supported"))
        assertFalse(result.getBoolean("general_radio_spectrum_supported"))
        assertFalse(result.getBoolean("microwave_spectrum_supported"))
        assertTrue(result.getBoolean("requires_external_sdr_for_broad_rf"))
        assertTrue(result.has("bluetooth_scan_permission_status"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.getJSONArray("limits").length() >= 2)
    }

    @Test
    fun radioSignalStatusCreatesAmFmCapabilityCardsWithoutPretendingPublicScannerAccess() {
        val result = HermesDeviceDiagnosticsBridge.radioSignalStatusJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("radio_signal_status", result.getString("action"))
        assertFalse(result.getBoolean("am_fm_public_android_scan_supported"))
        assertTrue(result.getBoolean("requires_external_sdr_for_broad_rf"))
        assertTrue(result.getJSONArray("radio_bands").toString().contains("FM broadcast"))
        assertEquals("signal_graph_card", result.getJSONArray("cards").getJSONObject(0).getString("type"))
        assertFalse(result.getJSONArray("radio_bands").getJSONObject(2).getBoolean("supported"))
        assertTrue(result.getJSONArray("radio_bands").getJSONObject(2).getBoolean("requires_external_hardware"))
    }

    @Test
    fun sensorSnapshotCreatesGraphRowsForMotionCards() {
        val result = HermesDeviceDiagnosticsBridge.sensorSnapshotJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("sensor_snapshot", result.getString("action"))
        val card = result.getJSONArray("cards").getJSONObject(0)
        assertEquals("signal_graph_card", card.getString("type"))
        assertEquals("sensor_vector", card.getString("graph_type"))
        assertEquals(result.getJSONArray("sensor_samples").length(), card.getInt("row_count"))
    }

    @Test
    fun socDetectionCoversMediatekAndSnapdragonWithoutAdrenoAssumptions() {
        assertTrue(HermesDeviceDiagnosticsBridge.isLikelyMediatekSoc(listOf("MediaTek", "mt6893", "Dimensity 1200")))
        assertFalse(HermesDeviceDiagnosticsBridge.isLikelyMediatekSoc(listOf("Qualcomm", "sm8550", "Snapdragon 8 Gen 2")))
        assertTrue(HermesDeviceDiagnosticsBridge.isLikelySnapdragonSoc(listOf("Qualcomm", "sm8550", "Snapdragon 8 Gen 2")))
        assertFalse(HermesDeviceDiagnosticsBridge.isLikelySnapdragonSoc(listOf("MediaTek", "mt6768", "Helio")))
    }

    @Test
    fun toolCatalogExposesDiagnosticsToolAndHindsightTranslation() {
        val result = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "tool_catalog"))
        val tools = result.getJSONArray("native_tools")
        val names = buildSet {
            for (index in 0 until tools.length()) {
                add(tools.getJSONObject(index).getString("name"))
            }
        }

        assertTrue(names.contains("android_device_diagnostics_tool"))
        assertTrue(names.contains("hindsight_memory_tool"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("top_apps"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_scan"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_status"))
        assertTrue(result.getJSONObject("hindsight_memory_translation").has("retain"))
    }

    @Test
    fun socialGmailPreflightReportsPackagesAccessibilityAndPreferredModel() {
        val modelFile = java.io.File(context.filesDir, "preflight-model.gguf").apply {
            writeText("HERMES_PREFLIGHT_MODEL")
            deleteOnExit()
        }
        LocalModelDownloadStore(context).apply {
            saveDownloads(emptyList())
            upsertDownload(
                LocalModelDownloadRecord(
                    id = "preflight-model",
                    title = "Preflight Qwen GGUF",
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
                    statusMessage = "Imported for preflight",
                ),
            )
            setPreferredDownloadId("preflight-model")
        }

        val result = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "social_gmail_goal_preflight"))

        assertTrue(result.getBoolean("success"))
        assertEquals("social_gmail_goal_preflight", result.getString("action"))
        assertTrue(result.getBoolean("physical_phone_required"))
        assertFalse(result.getBoolean("physical_phone_detected"))
        assertTrue(result.getJSONObject("android_device_identity").getBoolean("likely_emulator"))
        assertTrue(result.getJSONObject("tiktok").getJSONArray("candidate_package_names").toString().contains("com.zhiliaoapp.musically"))
        assertEquals("com.instagram.android", result.getJSONObject("instagram").getJSONArray("candidate_package_names").getString(0))
        assertEquals("com.google.android.gm", result.getJSONObject("gmail").getJSONArray("candidate_package_names").getString(0))
        assertTrue(result.getBoolean("local_model_import_button_supported"))
        assertTrue(result.getBoolean("local_model_import_uses_android_open_document"))
        assertFalse(result.getJSONObject("gmail_latest_email_summary_strategy").getBoolean("direct_mailbox_read_supported"))
        assertTrue(result.getJSONObject("preferred_local_model").getBoolean("ready"))
        assertEquals(modelFile.length(), result.getJSONObject("preferred_local_model").getLong("file_bytes"))
        assertFalse(result.getBoolean("ready_for_full_goal"))
        assertTrue(result.getJSONArray("external_send_safety_checks").toString().contains("adybag14@gmail.com"))
    }
}
