package com.nousresearch.hermesagent.device

import android.content.Context
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import org.json.JSONArray
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
    fun ratesWifiChannelsByCrowdingAndRecommendsQuietNonOverlappingChannel() {
        val rows = HermesDeviceDiagnosticsBridge.wifiChannelRatingRowsForMeasurements(
            listOf(
                HermesDeviceDiagnosticsBridge.WifiChannelMeasurement(channel = 1, band = "2.4GHz", rssiDbm = -35),
                HermesDeviceDiagnosticsBridge.WifiChannelMeasurement(channel = 1, band = "2.4GHz", rssiDbm = -62),
                HermesDeviceDiagnosticsBridge.WifiChannelMeasurement(channel = 6, band = "2.4GHz", rssiDbm = -48),
            ),
        )

        val best = rows.getJSONObject(0)
        val channelOne = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getInt("channel") == 1 }

        assertEquals("2.4GHz", best.getString("band"))
        assertEquals(11, best.getInt("channel"))
        assertEquals(0, best.getInt("overlap_count"))
        assertTrue(best.getInt("score") > channelOne.getInt("score"))
        assertTrue(best.getString("recommendation").contains("Best current option"))
    }

    @Test
    fun enrichesWifiRowsWithVendorSecurityAndFilterFacets() {
        assertEquals("AC:BC:32", HermesDeviceDiagnosticsBridge.wifiBssidOui("ac:bc:32:12:34:56"))
        assertEquals("Apple", HermesDeviceDiagnosticsBridge.wifiOuiVendorLabel("AC:BC:32"))
        assertEquals("Locally administered / randomized", HermesDeviceDiagnosticsBridge.wifiOuiVendorLabel("DA:A1:19"))
        assertEquals("WPA3", HermesDeviceDiagnosticsBridge.wifiSecurityLabel("[WPA3-SAE-CCMP][ESS]"))
        assertEquals("fair", HermesDeviceDiagnosticsBridge.wifiSignalQualityLabel(-67))

        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesNet")
                    .put("bssid", "AC:BC:32:12:34:56")
                    .put("rssi_dbm", -42)
                    .put("frequency_mhz", 5180)
                    .put("band", "5GHz")
                    .put("capabilities", "[WPA2-PSK-CCMP][ESS]"),
            )
            .put(
                JSONObject()
                    .put("ssid", "HermesGuest")
                    .put("bssid", "AC:BC:32:65:43:21")
                    .put("bssid_oui", "AC:BC:32")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -68)
                    .put("frequency_mhz", 2412)
                    .put("band", "2.4GHz")
                    .put("security_mode", "Open"),
            )

        val vendors = HermesDeviceDiagnosticsBridge.wifiVendorSummaryJson(networks)
        val filters = HermesDeviceDiagnosticsBridge.wifiAnalyzerFilterSummaryJson(networks)
        val securityFilter = (0 until filters.length())
            .map { filters.getJSONObject(it) }
            .first { it.getString("key") == "security" }

        assertEquals("Apple", vendors.getJSONObject(0).getString("vendor"))
        assertEquals(2, vendors.getJSONObject(0).getInt("network_count"))
        assertTrue(vendors.getJSONObject(0).getJSONArray("bssid_ouis").toString().contains("AC:BC:32"))
        assertEquals(4, filters.length())
        assertTrue(securityFilter.getJSONArray("options").toString().contains("WPA2"))
        assertTrue(securityFilter.getJSONArray("options").toString().contains("Open"))
    }

    @Test
    fun summarizesBluetoothMetadataForAgentAndCards() {
        assertEquals("near", HermesDeviceDiagnosticsBridge.bluetoothProximityLabel(-48))
        assertEquals("room", HermesDeviceDiagnosticsBridge.bluetoothProximityLabel(-67))
        assertEquals(2.0, HermesDeviceDiagnosticsBridge.estimateBluetoothDistanceMeters(-65, -59)!!, 0.01)

        val devices = JSONArray()
            .put(
                JSONObject()
                    .put("device_name", "Heart Strap")
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("paired", true)
                    .put("connectable", true)
                    .put("rssi_dbm", -49)
                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                    .put("manufacturer_ids", JSONArray().put("0x004C")),
            )
            .put(
                JSONObject()
                    .put("device_name", "Beacon")
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("connectable", false)
                    .put("rssi_dbm", -73)
                    .put("manufacturer_ids", JSONArray().put("0x004C")),
            )

        val rows = HermesDeviceDiagnosticsBridge.bluetoothMetadataSummaryRows(devices)
        val category = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("summary_type") == "device_category" }
        val manufacturer = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("summary_type") == "manufacturer_id" }
        val service = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("summary_type") == "service_uuid" }

        assertEquals("wearable_health", category.getString("label"))
        assertEquals(2, category.getInt("count"))
        assertEquals(1, category.getInt("paired_count"))
        assertEquals("0x004C", manufacturer.getString("label"))
        assertEquals(2, manufacturer.getInt("count"))
        assertTrue(service.getString("label").contains("180d"))
        assertTrue(service.getString("recommendation").contains("BLE service UUID"))
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
        assertTrue(result.getJSONObject("soc_profile").has("supported_abis"))
        assertTrue(result.getJSONObject("soc_profile").has("native_abi_strategy"))
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
        val capabilityCard = result.getJSONArray("cards").getJSONObject(1)
        assertEquals("Sensor Hardware", capabilityCard.getString("title"))
        assertEquals("sensor_capability", capabilityCard.getString("graph_type"))
        assertEquals(result.getJSONArray("sensor_capabilities").length(), capabilityCard.getInt("row_count"))
        assertTrue(result.has("sensor_capability_count"))
        assertTrue(result.has("motion_sensor_count"))
        assertTrue(result.has("wake_up_sensor_count"))
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
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_channel_rating"))
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
