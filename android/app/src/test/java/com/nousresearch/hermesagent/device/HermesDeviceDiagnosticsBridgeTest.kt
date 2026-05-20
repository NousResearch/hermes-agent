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
    fun buildsWifiAccessPointExportRowsAndAnalyzerSummaries() {
        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesNet")
                    .put("display_ssid", "HermesNet")
                    .put("bssid", "AC:BC:32:12:34:56")
                    .put("bssid_oui", "AC:BC:32")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -41)
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("band", "5GHz")
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("wifi_standard", "802.11ac")
                    .put("security_mode", "WPA2")
                    .put("capabilities", "[WPA2-PSK-CCMP][ESS]")
                    .put("estimated_distance_meters", 1.25),
            )
            .put(
                JSONObject()
                    .put("ssid", "Lab, Guest")
                    .put("display_ssid", "Lab, Guest")
                    .put("bssid", "DA:A1:19:65:43:21")
                    .put("rssi_dbm", -72)
                    .put("frequency_mhz", 2412)
                    .put("channel", 1)
                    .put("band", "2.4GHz")
                    .put("channel_width", "20MHz")
                    .put("wifi_standard", "802.11n")
                    .put("security_mode", "Open")
                    .put("capabilities", "[ESS]"),
            )

        val details = HermesDeviceDiagnosticsBridge.wifiAccessPointDetailRows(networks, limit = 10)
        val export = HermesDeviceDiagnosticsBridge.wifiAccessPointExportJson(details, "both", generatedAtMs = 1234L)
        val csv = HermesDeviceDiagnosticsBridge.wifiAccessPointCsv(details)
        val security = HermesDeviceDiagnosticsBridge.wifiSecuritySummaryJson(networks)
        val widths = HermesDeviceDiagnosticsBridge.wifiChannelWidthSummaryJson(networks)
        val standards = HermesDeviceDiagnosticsBridge.wifiStandardSummaryJson(networks)

        assertEquals(2, details.length())
        assertEquals(1, details.getJSONObject(0).getInt("rank"))
        assertEquals("HermesNet", details.getJSONObject(0).getString("display_ssid"))
        assertEquals(80, details.getJSONObject(0).getInt("channel_width_mhz"))
        assertEquals("802.11ac", details.getJSONObject(0).getString("wifi_standard"))
        assertEquals("both", export.getString("format"))
        assertEquals(2, export.getInt("row_count"))
        assertEquals("wifi_access_point_export_csv", export.getString("csv_key"))
        assertTrue(csv.startsWith("rank,display_ssid,hidden_ssid"))
        assertTrue(csv.contains("\"Lab, Guest\""))
        assertEquals("WPA2", security.getJSONObject(0).getString("security_mode"))
        assertTrue(security.getJSONObject(1).getString("recommendation").contains("Open AP group"))
        assertEquals("80MHz", widths.getJSONObject(0).getString("channel_width"))
        assertEquals("802.11ac", standards.getJSONObject(0).getString("wifi_standard"))
    }

    @Test
    fun buildsWifiSignalHistoryRowsForTrendCards() {
        val firstScan = JSONArray().put(
            JSONObject()
                .put("ssid", "HermesNet")
                .put("bssid", "AC:BC:32:12:34:56")
                .put("bssid_vendor", "Apple")
                .put("rssi_dbm", -66)
                .put("frequency_mhz", 5180)
                .put("channel", 36)
                .put("band", "5GHz")
                .put("security_mode", "WPA2"),
        )
        val secondScan = JSONArray().put(
            JSONObject()
                .put("ssid", "HermesNet")
                .put("bssid", "AC:BC:32:12:34:56")
                .put("bssid_vendor", "Apple")
                .put("rssi_dbm", -55)
                .put("frequency_mhz", 5180)
                .put("channel", 36)
                .put("band", "5GHz")
                .put("security_mode", "WPA2"),
        )

        val store = HermesDeviceDiagnosticsBridge.mergeWifiSignalHistory(
            HermesDeviceDiagnosticsBridge.mergeWifiSignalHistory(JSONObject(), firstScan, 1_000L),
            secondScan,
            2_000L,
        )
        val rows = HermesDeviceDiagnosticsBridge.wifiSignalHistoryRowsFromStore(store, 2_500L)
        val row = rows.getJSONObject(0)

        assertEquals("HermesNet", row.getString("ssid"))
        assertEquals("Apple", row.getString("bssid_vendor"))
        assertEquals(2, row.getInt("sample_count"))
        assertEquals(-55, row.getInt("current_rssi_dbm"))
        assertEquals(-60, row.getInt("average_rssi_dbm"))
        assertEquals(11, row.getInt("trend_db"))
        assertEquals("improving", row.getString("trend_label"))
        assertEquals(2, row.getJSONArray("rssi_series").length())
        assertEquals(500L, row.getLong("last_seen_ms"))
    }

    @Test
    fun deviceStateWriterToleratesDiagnosticsSnapshotServices() {
        val stateFile = java.io.File(context.filesDir, "hermes-home/android-device-state.json")
        stateFile.delete()

        DeviceStateWriter.write(context)

        assertTrue(stateFile.isFile)
        val payload = JSONObject(stateFile.readText())
        assertTrue(payload.getBoolean("device_diagnostics_tool_available"))
        assertTrue(payload.has("usage_access_granted"))
        assertTrue(payload.has("wifi_scan_permission_status"))
        assertTrue(payload.has("hindsight_promoted_memory_count"))
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
        assertTrue(result.getJSONObject("soc_profile").has("soc_family"))
        assertTrue(result.getJSONObject("soc_profile").has("gpu_family_hint"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_backend_strategy"))
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
    fun sensorAnalyzerReportExposesReadinessRoutesAndSamplingPolicyWithoutForcingSnapshot() {
        val result = HermesDeviceDiagnosticsBridge.sensorAnalyzerReportJson(context)
        val features = result.getJSONArray("sensor_analyzer_feature_matrix")
        val routes = result.getJSONArray("sensor_analyzer_workflow_routes")
        val policies = result.getJSONArray("sensor_sampling_policy_matrix")
        val featureLabels = buildSet {
            for (index in 0 until features.length()) add(features.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }
        val policyLabels = buildSet {
            for (index in 0 until policies.length()) add(policies.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("sensor_analyzer_report", result.getString("action"))
        assertTrue(result.has("sensor_sampling_status"))
        assertFalse(result.getJSONObject("sensor_sampling_status").getBoolean("active_sample_requested"))
        assertTrue(featureLabels.contains("Motion and orientation sensors"))
        assertTrue(featureLabels.contains("Accelerometer access"))
        assertTrue(featureLabels.contains("Gyroscope access"))
        assertTrue(featureLabels.contains("Sensor hardware metadata"))
        assertTrue(featureLabels.contains("Sensor privacy and power boundary"))
        assertTrue(routeLabels.contains("Route one-shot motion sample"))
        assertTrue(routeLabels.contains("Route sensor hardware metadata"))
        assertTrue(routeLabels.contains("Route sampling policy explanation"))
        assertTrue(policyLabels.contains("Passive report default"))
        assertTrue(policyLabels.contains("Bounded one-shot timeout"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.getJSONArray("cards").toString().contains("Sensor Analyzer Readiness"))
        assertTrue(result.getInt("sensor_analyzer_feature_count") >= 9)
        assertTrue(result.getInt("sensor_analyzer_workflow_route_count") >= 6)
        assertTrue(result.getInt("sensor_sampling_policy_count") >= 5)
    }

    @Test
    fun socDetectionCoversMediatekAndSnapdragonWithoutAdrenoAssumptions() {
        assertTrue(HermesDeviceDiagnosticsBridge.isLikelyMediatekSoc(listOf("MediaTek", "mt6893", "Dimensity 1200")))
        assertFalse(HermesDeviceDiagnosticsBridge.isLikelyMediatekSoc(listOf("Qualcomm", "sm8550", "Snapdragon 8 Gen 2")))
        assertTrue(HermesDeviceDiagnosticsBridge.isLikelySnapdragonSoc(listOf("Qualcomm", "sm8550", "Snapdragon 8 Gen 2")))
        assertFalse(HermesDeviceDiagnosticsBridge.isLikelySnapdragonSoc(listOf("MediaTek", "mt6768", "Helio")))

        val dimensityProfile = HermesAndroidHardwareProfile.classify(listOf("MediaTek Dimensity 9300 mt6989 Immortalis-G720"))
        val powerVrProfile = HermesAndroidHardwareProfile.classify(listOf("MediaTek Helio P35 mt6765 PowerVR Rogue GE8320"))
        val tensorProfile = HermesAndroidHardwareProfile.classify(listOf("Google Tensor gs201 Mali-G710"))
        val exynosProfile = HermesAndroidHardwareProfile.classify(listOf("Samsung Exynos 2400 Xclipse 940"))
        val unisocProfile = HermesAndroidHardwareProfile.classify(listOf("Unisoc T820 ums9230 Mali-G57"))

        assertEquals("mediatek", dimensityProfile.socFamily)
        assertEquals("mali_immortalis", dimensityProfile.gpuFamily)
        assertEquals("ARM MediaTek/Mali Immortalis", HermesAndroidHardwareProfile.accelerationLabel(dimensityProfile))
        assertEquals("mediatek", powerVrProfile.socFamily)
        assertEquals("powervr_img", powerVrProfile.gpuFamily)
        assertEquals("google_tensor", tensorProfile.socFamily)
        assertEquals("mali", tensorProfile.gpuFamily)
        assertEquals("samsung_exynos", exynosProfile.socFamily)
        assertEquals("xclipse", exynosProfile.gpuFamily)
        assertEquals("unisoc", unisocProfile.socFamily)
        assertEquals("mali", unisocProfile.gpuFamily)
        assertTrue(
            HermesAndroidHardwareProfile.nativeAbiStrategy(listOf("arm64-v8a", "armeabi-v7a"))
                .contains("Adreno, Mali, Immortalis, Xclipse, and PowerVR/IMG"),
        )
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
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_ap_details"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_export"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("signal_awareness_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("soc_compatibility_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("agent_environment_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_scan"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("sensor_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_status"))
        assertTrue(result.getJSONObject("hindsight_memory_translation").has("retain"))
    }

    @Test
    fun socCompatibilityReportExposesBackendPolicyRoutesAndMediatekCoverageCards() {
        val result = HermesDeviceDiagnosticsBridge.socCompatibilityReportJson(context)
        val backend = result.getJSONArray("soc_backend_matrix")
        val routes = result.getJSONArray("soc_backend_policy_routes")
        val constraints = result.getJSONArray("soc_backend_constraint_matrix")
        val backendLabels = buildSet {
            for (index in 0 until backend.length()) add(backend.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }
        val constraintLabels = buildSet {
            for (index in 0 until constraints.length()) add(constraints.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("soc_compatibility_report", result.getString("action"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_backend_strategy"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("likely_mediatek"))
        assertTrue(result.has("likely_snapdragon"))
        assertTrue(result.has("likely_mali_gpu"))
        assertTrue(result.has("supports_arm64"))
        assertTrue(backendLabels.contains("Detected SOC family"))
        assertTrue(backendLabels.contains("Native ABI selection"))
        assertTrue(backendLabels.contains("LiteRT-LM accelerator policy"))
        assertTrue(backendLabels.contains("MediaTek/Mali/PowerVR coverage"))
        assertTrue(routeLabels.contains("Route SOC compatibility report"))
        assertTrue(routeLabels.contains("Route full agent environment"))
        assertTrue(constraintLabels.contains("Avoid Adreno-only assumptions"))
        assertTrue(constraintLabels.contains("GPU probe then CPU fallback"))
        assertTrue(constraintLabels.contains("x86 emulator is not phone GPU proof"))
        assertTrue(result.getJSONArray("cards").toString().contains("SOC Compatibility"))
        assertTrue(result.getInt("soc_backend_feature_count") >= 7)
        assertTrue(result.getInt("ready_soc_backend_feature_count") >= 4)
        assertTrue(result.getInt("soc_backend_route_count") >= 5)
        assertTrue(result.getInt("soc_backend_constraint_count") >= 5)
    }

    @Test
    fun wifiAnalyzerReportExposesReadinessRoutesAndScanPolicyWithoutForcingRefresh() {
        val result = HermesDeviceDiagnosticsBridge.wifiAnalyzerReportJson(context)
        val features = result.getJSONArray("wifi_analyzer_feature_matrix")
        val routes = result.getJSONArray("wifi_analyzer_workflow_routes")
        val policies = result.getJSONArray("wifi_scan_policy_matrix")
        val featureLabels = buildSet {
            for (index in 0 until features.length()) add(features.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }
        val policyLabels = buildSet {
            for (index in 0 until policies.length()) add(policies.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("wifi_analyzer_report", result.getString("action"))
        assertTrue(result.has("wifi_scan_permission_status"))
        assertTrue(result.has("wifi_scan_status"))
        assertTrue(featureLabels.contains("Identify nearby access points"))
        assertTrue(featureLabels.contains("Channel signal graph"))
        assertTrue(featureLabels.contains("Band, security, signal, and SSID filters"))
        assertTrue(featureLabels.contains("Vendor/OUI lookup"))
        assertTrue(featureLabels.contains("HT/VHT/HE/EHT width and standard metadata"))
        assertTrue(featureLabels.contains("Wi-Fi safety boundary"))
        assertTrue(routeLabels.contains("Route best-channel analysis"))
        assertTrue(routeLabels.contains("Route full AP metadata"))
        assertTrue(routeLabels.contains("Route AP export"))
        assertTrue(policyLabels.contains("Android scan throttling"))
        assertTrue(policyLabels.contains("Passive report default"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.getJSONArray("cards").toString().contains("Wi-Fi Analyzer Readiness"))
        assertTrue(result.getInt("wifi_analyzer_feature_count") >= 8)
        assertTrue(result.getInt("wifi_analyzer_workflow_route_count") >= 6)
        assertTrue(result.getInt("wifi_scan_policy_count") >= 5)
    }

    @Test
    fun bluetoothAnalyzerReportExposesReadinessRoutesAndScanPolicyWithoutForcingRefresh() {
        val result = HermesDeviceDiagnosticsBridge.bluetoothAnalyzerReportJson(context)
        val features = result.getJSONArray("bluetooth_analyzer_feature_matrix")
        val routes = result.getJSONArray("bluetooth_analyzer_workflow_routes")
        val policies = result.getJSONArray("bluetooth_scan_policy_matrix")
        val featureLabels = buildSet {
            for (index in 0 until features.length()) add(features.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }
        val policyLabels = buildSet {
            for (index in 0 until policies.length()) add(policies.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("bluetooth_analyzer_report", result.getString("action"))
        assertTrue(result.has("bluetooth_scan_permission_status"))
        assertTrue(result.has("bluetooth_scan_status"))
        assertFalse(result.getJSONObject("bluetooth_scan_status").getBoolean("refresh_requested"))
        assertTrue(featureLabels.contains("Identify paired devices"))
        assertTrue(featureLabels.contains("Scan nearby BLE devices"))
        assertTrue(featureLabels.contains("RSSI proximity graph"))
        assertTrue(featureLabels.contains("Service UUID metadata"))
        assertTrue(featureLabels.contains("Manufacturer ID metadata"))
        assertTrue(featureLabels.contains("Bluetooth safety boundary"))
        assertTrue(routeLabels.contains("Route nearby Bluetooth scan"))
        assertTrue(routeLabels.contains("Route service/manufacturer metadata"))
        assertTrue(routeLabels.contains("Route scan policy explanation"))
        assertTrue(policyLabels.contains("Connect and scan permissions"))
        assertTrue(policyLabels.contains("Active scan cadence"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.getJSONArray("cards").toString().contains("Bluetooth Analyzer Readiness"))
        assertTrue(result.getInt("bluetooth_analyzer_feature_count") >= 8)
        assertTrue(result.getInt("bluetooth_analyzer_workflow_route_count") >= 5)
        assertTrue(result.getInt("bluetooth_scan_policy_count") >= 6)
    }

    @Test
    fun signalAwarenessReportFusesWirelessRadioSensorsAndSocContext() {
        val result = HermesDeviceDiagnosticsBridge.signalAwarenessReportJson(context)
        val awareness = result.getJSONArray("signal_awareness_matrix")
        val routes = result.getJSONArray("signal_workflow_routes")
        val constraints = result.getJSONArray("signal_constraint_matrix")
        val awarenessLabels = buildSet {
            for (index in 0 until awareness.length()) add(awareness.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }
        val constraintLabels = buildSet {
            for (index in 0 until constraints.length()) add(constraints.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("signal_awareness_report", result.getString("action"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_backend_strategy"))
        assertTrue(result.getJSONObject("signal_capability_status").has("requires_external_sdr_for_broad_rf"))
        assertTrue(result.has("cached_wifi_signal_history"))
        assertTrue(awarenessLabels.contains("Wi-Fi scan surface"))
        assertTrue(awarenessLabels.contains("Bluetooth proximity metadata"))
        assertTrue(awarenessLabels.contains("Radio/RF limits"))
        assertTrue(awarenessLabels.contains("SOC backend compatibility"))
        assertTrue(routeLabels.contains("Route Wi-Fi analyzer work"))
        assertTrue(routeLabels.contains("Route Bluetooth proximity work"))
        assertTrue(constraintLabels.contains("AM/FM tuner public API"))
        assertTrue(constraintLabels.contains("Broad RF and microwave hardware"))
        assertTrue(result.getJSONArray("cards").toString().contains("Signal Awareness"))
        assertTrue(result.getInt("signal_awareness_count") >= 8)
        assertTrue(result.getInt("signal_workflow_route_count") >= 6)
        assertTrue(result.getInt("signal_constraint_count") >= 5)
    }

    @Test
    fun agentEnvironmentReportSummarizesKaiParityAndSystemInputs() {
        val result = HermesDeviceDiagnosticsBridge.agentEnvironmentReportJson(context)
        val capabilities = result.getJSONArray("agent_capability_matrix")
        val kaiParity = result.getJSONArray("kai_parity_matrix")
        val readiness = result.getJSONArray("workflow_readiness_matrix")
        val capabilityText = capabilities.toString()
        val kaiText = kaiParity.toString()
        val readinessText = readiness.toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_environment_report", result.getString("action"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_backend_strategy"))
        assertTrue(result.getJSONObject("signal_capability_status").has("requires_external_sdr_for_broad_rf"))
        assertTrue(capabilityText.contains("Wi-Fi analyzer"))
        assertTrue(capabilityText.contains("Bluetooth scanner"))
        assertTrue(capabilityText.contains("SOC and LiteRT backend policy"))
        assertTrue(capabilityText.contains("Persistent hindsight memory"))
        assertTrue(kaiText.contains("Persistent memory"))
        assertTrue(kaiText.contains("Autonomous heartbeat"))
        assertTrue(kaiText.contains("Image attachments and screen vision"))
        assertTrue(readinessText.contains("Analyze nearby Wi-Fi"))
        assertTrue(readinessText.contains("Run local multimodal agent"))
        assertTrue(result.getJSONArray("cards").toString().contains("Kai Parity"))
        assertTrue(result.getInt("agent_capability_count") >= 8)
        assertTrue(result.getInt("kai_parity_count") >= 6)
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
