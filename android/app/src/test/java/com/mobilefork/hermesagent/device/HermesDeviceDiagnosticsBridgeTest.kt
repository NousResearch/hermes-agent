package com.mobilefork.hermesagent.device

import android.content.Context
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.LocalModelDownloadRecord
import com.mobilefork.hermesagent.data.LocalModelDownloadStore
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
    fun ratesWifi6GhzWithPreferredCandidateChannelsBeyondObservedAccessPoints() {
        val rows = HermesDeviceDiagnosticsBridge.wifiChannelRatingRowsForMeasurements(
            listOf(
                HermesDeviceDiagnosticsBridge.WifiChannelMeasurement(
                    channel = 5,
                    band = "6GHz",
                    rssiDbm = -38,
                    widthLabel = "320MHz",
                ),
            ),
        )
        val channels = buildSet {
            for (index in 0 until rows.length()) {
                val row = rows.getJSONObject(index)
                if (row.getString("band") == "6GHz") add(row.getInt("channel"))
            }
        }
        val observed = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("band") == "6GHz" && it.getInt("channel") == 5 }
        val quiet = rows.getJSONObject(0)

        assertTrue(channels.contains(5))
        assertTrue(channels.contains(21))
        assertTrue(channels.contains(37))
        assertEquals(5975, observed.getInt("frequency_hint_mhz"))
        assertEquals("6GHz", quiet.getString("band"))
        assertTrue(quiet.getInt("score") > observed.getInt("score"))
        assertTrue(quiet.getString("recommendation").contains("Best current option"))
    }

    @Test
    fun buildsWifiChannelUtilizationRowsFromVisibleApPressure() {
        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesNet")
                    .put("display_ssid", "HermesNet")
                    .put("rssi_dbm", -36)
                    .put("frequency_mhz", 2412)
                    .put("channel", 1)
                    .put("band", "2.4GHz")
                    .put("channel_width", "40MHz")
                    .put("security_mode", "WPA3"),
            )
            .put(
                JSONObject()
                    .put("ssid", "HermesGuest")
                    .put("rssi_dbm", -55)
                    .put("frequency_mhz", 2422)
                    .put("channel", 3)
                    .put("band", "2.4GHz")
                    .put("channel_width", "20MHz")
                    .put("security_mode", "WPA2"),
            )
            .put(
                JSONObject()
                    .put("ssid", "Lab5G")
                    .put("rssi_dbm", -70)
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("band", "5GHz")
                    .put("channel_width", "80MHz")
                    .put("security_mode", "WPA2"),
            )

        val rows = HermesDeviceDiagnosticsBridge.wifiChannelUtilizationRowsForNetworks(networks)
        val first = rows.getJSONObject(0)

        assertTrue(rows.length() >= 2)
        assertEquals("2.4GHz", first.getString("band"))
        assertEquals(1, first.getInt("channel"))
        assertTrue(first.getInt("channel_pressure_score") >= 50)
        assertEquals("crowded", first.getString("utilization_label"))
        assertTrue(first.getJSONArray("security_modes").toString().contains("WPA3"))
        assertTrue(first.getJSONArray("sample_ssids").toString().contains("HermesNet"))
        assertTrue(first.getString("recommendation").contains("Crowded") || first.getString("recommendation").contains("Heavily"))
    }

    @Test
    fun buildsWifiChannelGraphRowsWithWidthSpansAndOverlapPressure() {
        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesWide")
                    .put("display_ssid", "HermesWide")
                    .put("bssid", "AC:BC:32:12:34:56")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -36)
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("band", "5GHz")
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("security_mode", "WPA3")
                    .put("estimated_distance_meters", 1.1),
            )
            .put(
                JSONObject()
                    .put("ssid", "HermesNarrow")
                    .put("display_ssid", "HermesNarrow")
                    .put("bssid", "AC:BC:32:65:43:21")
                    .put("rssi_dbm", -50)
                    .put("frequency_mhz", 5200)
                    .put("channel", 40)
                    .put("band", "5GHz")
                    .put("channel_width", "20MHz")
                    .put("channel_width_mhz", 20)
                    .put("security_mode", "WPA2"),
            )
            .put(
                JSONObject()
                    .put("ssid", "FarLab")
                    .put("display_ssid", "FarLab")
                    .put("bssid", "DA:A1:19:11:22:33")
                    .put("rssi_dbm", -74)
                    .put("frequency_mhz", 5745)
                    .put("channel", 149)
                    .put("band", "5GHz")
                    .put("channel_width", "40MHz")
                    .put("channel_width_mhz", 40)
                    .put("security_mode", "WPA2"),
            )

        val rows = HermesDeviceDiagnosticsBridge.wifiChannelGraphRows(networks)
        val primary = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("display_ssid") == "HermesWide" }

        assertEquals(3, rows.length())
        assertEquals("5GHz", primary.getString("band"))
        assertEquals(36, primary.getInt("channel"))
        assertEquals("channel_width_envelope", primary.getString("graph_shape"))
        assertTrue(primary.getInt("graph_width_channels") > 1)
        assertTrue(primary.getInt("channel_span_start") < 36)
        assertTrue(primary.getInt("channel_span_end") > 36)
        assertTrue(primary.getInt("frequency_span_start_mhz") < 5180)
        assertTrue(primary.getInt("frequency_span_end_mhz") > 5180)
        assertTrue(primary.getInt("overlap_network_count") >= 1)
        assertTrue(primary.getInt("overlap_pressure_score") > 0)
        assertTrue(primary.getJSONArray("overlap_sample_ssids").toString().contains("HermesNarrow"))
        assertTrue(primary.getString("recommendation").contains("overlap") || primary.getString("recommendation").contains("Clear"))
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
    fun filtersWifiAnalyzerRowsByBandSecuritySignalSsidRssiAndHiddenState() {
        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesFast")
                    .put("display_ssid", "HermesFast")
                    .put("bssid", "AC:BC:32:12:34:56")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -49)
                    .put("frequency_mhz", 5180)
                    .put("band", "5GHz")
                    .put("security_mode", "WPA3")
                    .put("hidden_ssid", false),
            )
            .put(
                JSONObject()
                    .put("ssid", "HermesGuest")
                    .put("display_ssid", "HermesGuest")
                    .put("rssi_dbm", -52)
                    .put("frequency_mhz", 5180)
                    .put("band", "5GHz")
                    .put("security_mode", "Open")
                    .put("hidden_ssid", false),
            )
            .put(
                JSONObject()
                    .put("ssid", "Lab2G")
                    .put("display_ssid", "Lab2G")
                    .put("rssi_dbm", -45)
                    .put("frequency_mhz", 2412)
                    .put("band", "2.4GHz")
                    .put("security_mode", "WPA3")
                    .put("hidden_ssid", false),
            )
            .put(
                JSONObject()
                    .put("ssid", "")
                    .put("display_ssid", "")
                    .put("rssi_dbm", -48)
                    .put("frequency_mhz", 5180)
                    .put("band", "5GHz")
                    .put("security_mode", "WPA3")
                    .put("hidden_ssid", true),
            )

        val filtered = HermesDeviceDiagnosticsBridge.wifiFilteredNetworkRows(
            networks,
            JSONObject()
                .put("filter_band", "5GHz")
                .put("filter_security", "WPA3")
                .put("filter_signal", "excellent,good")
                .put("filter_ssid", "Hermes")
                .put("min_rssi_dbm", -60)
                .put("include_hidden", false),
        )
        val hiddenOnly = HermesDeviceDiagnosticsBridge.wifiFilteredNetworkRows(
            networks,
            JSONObject()
                .put("filter_band", "5GHz")
                .put("hidden_only", true),
        )

        assertEquals(1, filtered.length())
        assertEquals("HermesFast", filtered.getJSONObject(0).getString("ssid"))
        assertEquals(1, hiddenOnly.length())
        assertTrue(hiddenOnly.getJSONObject(0).getBoolean("hidden_ssid"))
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
    fun buildsWifiAccessPointSemanticRowsAndBandCoverageForAgentContext() {
        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesNet")
                    .put("display_ssid", "HermesNet")
                    .put("bssid", "AC:BC:32:12:34:56")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -42)
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("band", "5GHz")
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("wifi_standard", "802.11ax")
                    .put("security_mode", "WPA3")
                    .put("capabilities", "[WPA3-SAE-CCMP][ESS]"),
            )
            .put(
                JSONObject()
                    .put("ssid", "Cafe Guest")
                    .put("display_ssid", "Cafe Guest")
                    .put("bssid", "DA:A1:19:65:43:21")
                    .put("rssi_dbm", -59)
                    .put("frequency_mhz", 2412)
                    .put("channel", 1)
                    .put("band", "2.4GHz")
                    .put("channel_width", "20MHz")
                    .put("wifi_standard", "802.11n")
                    .put("security_mode", "Open")
                    .put("capabilities", "[WPS][ESS]")
                    .put("hidden_ssid", false),
            )

        val details = HermesDeviceDiagnosticsBridge.wifiAccessPointDetailRows(networks, limit = 10)
        val semanticRows = HermesDeviceDiagnosticsBridge.wifiAccessPointSemanticRows(details, limit = 10)
        val ratings = HermesDeviceDiagnosticsBridge.wifiChannelRatingRowsForNetworks(networks)
        val coverageRows = HermesDeviceDiagnosticsBridge.wifiBandCoverageRows(networks, ratings = ratings)
        val guest = (0 until semanticRows.length())
            .map { semanticRows.getJSONObject(it) }
            .first { it.getString("display_ssid") == "Cafe Guest" }
        val home = (0 until semanticRows.length())
            .map { semanticRows.getJSONObject(it) }
            .first { it.getString("display_ssid") == "HermesNet" }
        val band24 = (0 until coverageRows.length())
            .map { coverageRows.getJSONObject(it) }
            .first { it.getString("band") == "2.4GHz" }
        val band6 = (0 until coverageRows.length())
            .map { coverageRows.getJSONObject(it) }
            .first { it.getString("band") == "6GHz" }

        assertEquals("guest/public hotspot", guest.getString("semantic_label"))
        assertEquals("wps_attention", guest.getString("security_risk_label"))
        assertTrue(guest.getJSONArray("semantic_tags").toString().contains("guest_public_hotspot"))
        assertTrue(guest.getString("recommendation").contains("WPS"))
        assertEquals("private router/AP", home.getString("semantic_label"))
        assertEquals("strong_security", home.getString("security_risk_label"))
        assertEquals(1, band24.getInt("network_count"))
        assertEquals(1, band24.getInt("security_attention_count"))
        assertTrue(band24.getJSONArray("visible_channels").toString().contains("1"))
        assertEquals("not observed in latest scan", band6.getString("coverage_label"))
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
    fun statusJsonAvoidsUsageStatsAppOpsInRobolectric() {
        val status = HermesDeviceDiagnosticsBridge.statusJson(context)

        assertTrue(HermesDeviceDiagnosticsBridge.isRobolectricRuntime())
        assertFalse(status.getBoolean("usage_access_granted"))
        assertTrue(status.has("available_actions"))
    }

    @Test
    fun summarizesBluetoothMetadataForAgentAndCards() {
        assertEquals("near", HermesDeviceDiagnosticsBridge.bluetoothProximityLabel(-48))
        assertEquals("room", HermesDeviceDiagnosticsBridge.bluetoothProximityLabel(-67))
        assertEquals(2.0, HermesDeviceDiagnosticsBridge.estimateBluetoothDistanceMeters(-65, -59)!!, 0.01)
        assertEquals("Heart Rate", HermesDeviceDiagnosticsBridge.bluetoothServiceUuidLabel("0000180d-0000-1000-8000-00805f9b34fb"))
        assertEquals("Apple", HermesDeviceDiagnosticsBridge.bluetoothManufacturerIdLabel("0x004C"))

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
                    .put("service_labels", JSONArray().put("Heart Rate"))
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("manufacturer_names", JSONArray().put("Apple"))
                    .put("semantic_context", "services=Heart Rate | manufacturers=Apple"),
            )
            .put(
                JSONObject()
                    .put("device_name", "Beacon")
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("connectable", false)
                    .put("rssi_dbm", -73)
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("manufacturer_names", JSONArray().put("Apple")),
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
        assertEquals("Apple", manufacturer.getString("semantic_label"))
        assertEquals(2, manufacturer.getInt("count"))
        assertTrue(service.getString("label").contains("180d"))
        assertEquals("Heart Rate", service.getString("semantic_label"))
        assertTrue(service.getString("recommendation").contains("BLE service UUID"))
    }

    @Test
    fun buildsBluetoothSignalHistoryRowsFromNearbyRssiSamples() {
        val firstScan = JSONArray().put(
            JSONObject()
                .put("device_name", "Heart Strap")
                .put("advertised_name", "Heart Strap")
                .put("address", "AA:BB:CC:00:11:22")
                .put("device_type", "le")
                .put("device_category", "wearable_health")
                .put("rssi_dbm", -72)
                .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                .put("service_labels", JSONArray().put("Heart Rate"))
                .put("manufacturer_ids", JSONArray().put("0x004C"))
                .put("manufacturer_names", JSONArray().put("Apple"))
                .put("semantic_context", "services=Heart Rate | manufacturers=Apple"),
        )
        val secondScan = JSONArray().put(
            JSONObject()
                .put("device_name", "Heart Strap")
                .put("advertised_name", "Heart Strap")
                .put("address", "AA:BB:CC:00:11:22")
                .put("device_type", "le")
                .put("device_category", "wearable_health")
                .put("rssi_dbm", -58)
                .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                .put("service_labels", JSONArray().put("Heart Rate"))
                .put("manufacturer_ids", JSONArray().put("0x004C"))
                .put("manufacturer_names", JSONArray().put("Apple"))
                .put("semantic_context", "services=Heart Rate | manufacturers=Apple"),
        )

        val store = HermesDeviceDiagnosticsBridge.mergeBluetoothSignalHistory(
            HermesDeviceDiagnosticsBridge.mergeBluetoothSignalHistory(JSONObject(), firstScan, 1_000L),
            secondScan,
            2_000L,
        )
        val rows = HermesDeviceDiagnosticsBridge.bluetoothSignalHistoryRowsFromStore(store, 2_500L)
        val row = rows.getJSONObject(0)

        assertEquals("Heart Strap", row.getString("device_name"))
        assertEquals("wearable_health", row.getString("device_category"))
        assertEquals(2, row.getInt("sample_count"))
        assertEquals(-58, row.getInt("current_rssi_dbm"))
        assertEquals(-65, row.getInt("average_rssi_dbm"))
        assertEquals(14, row.getInt("trend_db"))
        assertEquals("approaching", row.getString("trend_label"))
        assertEquals("room", row.getString("proximity_label"))
        assertTrue(row.getJSONArray("service_uuids").toString().contains("180d"))
        assertTrue(row.getJSONArray("service_labels").toString().contains("Heart Rate"))
        assertTrue(row.getJSONArray("manufacturer_names").toString().contains("Apple"))
        assertTrue(row.getString("semantic_context").contains("Heart Rate"))
        assertEquals(2, row.getJSONArray("rssi_series").length())
        assertEquals(500L, row.getLong("last_seen_ms"))
    }

    @Test
    fun buildsMotionSensorHistoryRowsFromBoundedSamples() {
        val firstSample = JSONArray().put(
            JSONObject()
                .put("sensor_type", "accelerometer")
                .put("sensor_label", "Accelerometer")
                .put("sensor_name", "BMI160 Accelerometer")
                .put("vendor", "Bosch")
                .put("unit", "m/s^2")
                .put("sampled", true)
                .put("available", true)
                .put("values", JSONArray().put(0.0).put(0.0).put(9.81))
                .put("accuracy_label", "high")
                .put("maximum_range", 19.6),
        )
        val secondSample = JSONArray().put(
            JSONObject()
                .put("sensor_type", "accelerometer")
                .put("sensor_label", "Accelerometer")
                .put("sensor_name", "BMI160 Accelerometer")
                .put("vendor", "Bosch")
                .put("unit", "m/s^2")
                .put("sampled", true)
                .put("available", true)
                .put("values", JSONArray().put(0.0).put(2.0).put(11.0))
                .put("accuracy_label", "high")
                .put("maximum_range", 19.6),
        )

        val store = HermesDeviceDiagnosticsBridge.mergeMotionSensorHistory(
            HermesDeviceDiagnosticsBridge.mergeMotionSensorHistory(JSONObject(), firstSample, 1_000L),
            secondSample,
            2_000L,
        )
        val rows = HermesDeviceDiagnosticsBridge.motionSensorHistoryRowsFromStore(store, 2_500L)
        val row = rows.getJSONObject(0)

        assertEquals("accelerometer", row.getString("sensor_type"))
        assertEquals("BMI160 Accelerometer", row.getString("sensor_name"))
        assertEquals(2, row.getInt("sample_count"))
        assertEquals(11.18, row.getDouble("current_magnitude"), 0.02)
        assertEquals(10.50, row.getDouble("average_magnitude"), 0.02)
        assertEquals("increasing", row.getString("trend_label"))
        assertEquals("drifting", row.getString("stability_label"))
        assertEquals(2, row.getJSONArray("magnitude_series").length())
        assertEquals(3, row.getJSONArray("current_values").length())
        assertEquals(500L, row.getLong("last_seen_ms"))
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
        assertTrue(result.getJSONArray("radio_bands").length() >= 6)
        assertEquals(result.getJSONArray("radio_bands").length(), result.getInt("radio_band_plan_count"))
        assertTrue(result.getJSONArray("radio_receiver_profiles").length() >= 5)
        assertTrue(result.getInt("radio_receiver_profile_count") >= 5)
        assertTrue(result.getInt("ready_radio_receiver_profile_count") >= 5)
        assertTrue(result.getJSONArray("radio_signal_feature_matrix").length() >= 6)
        assertTrue(result.getJSONArray("radio_signal_workflow_routes").length() >= 4)
        assertTrue(result.getJSONArray("radio_signal_constraint_matrix").length() >= 4)
        assertTrue(result.getJSONArray("radio_signal_graph_rows").length() >= 2)
        assertEquals(result.getJSONArray("radio_signal_graph_rows").length(), result.getInt("radio_signal_graph_row_count"))
        assertEquals(0, result.getInt("radio_signal_graph_sample_count"))
        assertTrue(result.getInt("radio_signal_feature_count") >= 6)
        assertTrue(result.getInt("ready_radio_signal_feature_count") >= 1)
        assertTrue(result.getInt("radio_signal_workflow_route_count") >= 4)
        assertTrue(result.getInt("radio_signal_constraint_count") >= 4)
    }

    @Test
    fun radioSignalStatusCreatesAmFmCapabilityCardsWithoutPretendingPublicScannerAccess() {
        val result = HermesDeviceDiagnosticsBridge.radioSignalStatusJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("radio_signal_status", result.getString("action"))
        assertFalse(result.getBoolean("am_fm_public_android_scan_supported"))
        assertTrue(result.getBoolean("requires_external_sdr_for_broad_rf"))
        val bands = result.getJSONArray("radio_bands")
        val bandRows = (0 until bands.length()).map { bands.getJSONObject(it) }
        val am = bandRows.first { it.getString("band") == "AM broadcast" }
        val fm = bandRows.first { it.getString("band") == "FM broadcast" }
        val wifi = bandRows.first { it.getString("band") == "Wi-Fi 2.4 GHz" }
        val bluetooth = bandRows.first { it.getString("band") == "Bluetooth 2.4 GHz" }
        val external = bandRows.first { it.getString("band") == "External SDR / broad RF" }
        val receiverProfiles = result.getJSONArray("radio_receiver_profiles")
        val receiverRows = (0 until receiverProfiles.length()).map { receiverProfiles.getJSONObject(it) }
        val fmReceiver = receiverRows.first { it.getString("receiver_id") == "fm_vendor_or_sdr" }
        val wifiReceiver = receiverRows.first { it.getString("receiver_id") == "wifi_public_metadata" }
        val sdrReceiver = receiverRows.first { it.getString("receiver_id") == "external_sdr_bridge" }
        val graphRows = result.getJSONArray("radio_signal_graph_rows")
        val cards = result.getJSONArray("cards")

        assertTrue(bands.length() >= 6)
        assertEquals(bands.length(), result.getInt("radio_band_plan_count"))
        assertFalse(am.getBoolean("public_android_scan_supported"))
        assertFalse(fm.getBoolean("public_android_scan_supported"))
        assertTrue(am.getString("access_path").contains("Broadcast Radio HAL"))
        assertTrue(wifi.getString("access_path").contains("wifi_channel_utilization"))
        assertTrue(bluetooth.getString("access_path").contains("bluetooth_signal_history"))
        assertFalse(external.getBoolean("supported"))
        assertTrue(external.getBoolean("requires_external_hardware"))
        assertTrue(receiverProfiles.length() >= 5)
        assertTrue(fmReceiver.getBoolean("requires_vendor_bridge"))
        assertEquals("radio_signal_graph", fmReceiver.getString("route_action"))
        assertTrue(fmReceiver.getJSONArray("station_metadata_fields").toString().contains("rds_program_service"))
        assertEquals("wifi_analyzer_report", wifiReceiver.getString("route_action"))
        assertTrue(wifiReceiver.getJSONArray("graph_row_schema").toString().contains("rssi_dbm"))
        assertTrue(sdrReceiver.getBoolean("requires_external_hardware"))
        assertEquals("radio_signal_graph", sdrReceiver.getString("route_action"))
        assertTrue(sdrReceiver.getJSONArray("sample_fields").toString().contains("center_frequency_hz"))
        assertTrue(graphRows.toString().contains("AM broadcast band"))
        assertTrue(graphRows.toString().contains("FM broadcast band"))
        assertEquals(graphRows.length(), result.getInt("radio_signal_graph_row_count"))
        assertTrue(result.getJSONArray("radio_signal_feature_matrix").length() >= 6)
        assertTrue(result.getJSONArray("radio_signal_workflow_routes").length() >= 4)
        assertTrue(result.getJSONArray("radio_signal_constraint_matrix").length() >= 4)
        assertEquals("signal_graph_card", cards.getJSONObject(0).getString("type"))
        assertEquals("Radio Band Plan", cards.getJSONObject(0).getString("title"))
        assertEquals("radio_frequency_capability", cards.getJSONObject(0).getString("graph_type"))
        assertEquals("AM/FM Signal Graph", cards.getJSONObject(1).getString("title"))
        assertEquals("radio_signal_graph", cards.getJSONObject(1).getString("graph_type"))
        assertEquals("Receiver Profiles", cards.getJSONObject(2).getString("title"))
        assertEquals("radio_receiver_profile", cards.getJSONObject(2).getString("graph_type"))
        assertEquals("Radio Signal Routes", cards.getJSONObject(3).getString("title"))
        assertEquals("radio_signal_workflow_routes", cards.getJSONObject(3).getString("graph_type"))
        assertEquals("Radio Scan Boundaries", cards.getJSONObject(4).getString("title"))
        assertEquals("radio_signal_constraint_matrix", cards.getJSONObject(4).getString("graph_type"))
    }

    @Test
    fun radioSignalGraphNormalizesBridgeSamplesForAmFmCards() {
        val result = HermesDeviceDiagnosticsBridge.radioSignalGraphJson(
            context,
            JSONObject()
                .put("sample_source", "unit_test_vendor_bridge")
                .put(
                    "radio_samples",
                    JSONArray()
                        .put(
                            JSONObject()
                                .put("station_label", "Hermes FM")
                                .put("frequency_mhz", 99.5)
                                .put("rssi_dbm", -58)
                                .put("snr_db", 31)
                                .put("modulation", "fm")
                                .put("receiver_id", "fm_vendor_or_sdr"),
                        )
                        .put(
                            JSONObject()
                                .put("station_label", "Hermes AM")
                                .put("frequency_khz", 1010)
                                .put("power_db", -72)
                                .put("modulation", "am")
                                .put("receiver_id", "am_vendor_or_sdr"),
                        ),
                ),
        )

        assertTrue(result.getBoolean("success"))
        assertEquals("radio_signal_graph", result.getString("action"))
        assertFalse(result.getBoolean("native_android_public_tuner_scan_supported"))
        assertTrue(result.getBoolean("requires_vendor_or_external_receiver_for_am_fm_samples"))
        assertTrue(result.getBoolean("radio_signal_graph_bridge_ready"))
        assertEquals(2, result.getInt("radio_signal_graph_sample_count"))
        assertTrue(result.getInt("radio_signal_graph_row_count") >= 4)
        val rows = result.getJSONArray("radio_signal_graph_rows").toString()
        assertTrue(rows.contains("Hermes FM"))
        assertTrue(rows.contains("99.5 MHz"))
        assertTrue(rows.contains("Hermes AM"))
        assertTrue(rows.contains("1010 kHz"))
        val cards = result.getJSONArray("cards")
        assertEquals("AM/FM Signal Graph", cards.getJSONObject(0).getString("title"))
        assertEquals("radio_signal_graph", cards.getJSONObject(0).getString("graph_type"))
        assertEquals("Radio Receiver Source Readiness", cards.getJSONObject(1).getString("title"))
        assertTrue(cards.toString().contains("radio_receiver_profile"))
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
        assertEquals("Motion Pose Estimate", capabilityCard.getString("title"))
        assertEquals("motion_pose_estimate", capabilityCard.getString("graph_type"))
        assertEquals(result.getJSONArray("motion_pose_estimates").length(), capabilityCard.getInt("row_count"))
        val historyCard = result.getJSONArray("cards").getJSONObject(2)
        assertEquals("Motion Sensor History", historyCard.getString("title"))
        assertEquals("motion_sensor_history", historyCard.getString("graph_type"))
        assertEquals(result.getJSONArray("motion_sensor_history").length(), historyCard.getInt("row_count"))
        val hardwareCard = result.getJSONArray("cards").getJSONObject(3)
        assertEquals("Sensor Hardware", hardwareCard.getString("title"))
        assertEquals("sensor_capability", hardwareCard.getString("graph_type"))
        assertEquals(result.getJSONArray("sensor_capabilities").length(), hardwareCard.getInt("row_count"))
        assertTrue(result.has("sampled_sensor_count"))
        assertTrue(result.has("motion_sensor_history_count"))
        assertTrue(result.has("motion_pose_estimate_count"))
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
        assertTrue(result.has("cached_motion_sensor_history_count"))
        assertTrue(result.has("motion_sensor_history_count"))
        assertTrue(result.has("motion_pose_estimate_count"))
        assertFalse(result.getJSONObject("sensor_sampling_status").getBoolean("active_sample_requested"))
        assertTrue(featureLabels.contains("Motion and orientation sensors"))
        assertTrue(featureLabels.contains("Motion trend history graph"))
        assertTrue(featureLabels.contains("Motion pose fusion rows"))
        assertTrue(featureLabels.contains("Accelerometer access"))
        assertTrue(featureLabels.contains("Gyroscope access"))
        assertTrue(featureLabels.contains("Sensor hardware metadata"))
        assertTrue(featureLabels.contains("Sensor privacy and power boundary"))
        assertTrue(routeLabels.contains("Route one-shot motion sample"))
        assertTrue(routeLabels.contains("Route motion trend history"))
        assertTrue(routeLabels.contains("Route motion pose fusion"))
        assertTrue(routeLabels.contains("Route sensor hardware metadata"))
        assertTrue(routeLabels.contains("Route sampling policy explanation"))
        assertTrue(policyLabels.contains("Passive report default"))
        assertTrue(policyLabels.contains("Bounded one-shot timeout"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.getJSONArray("cards").toString().contains("Sensor Analyzer Readiness"))
        assertTrue(result.getInt("sensor_analyzer_feature_count") >= 10)
        assertTrue(result.getInt("sensor_analyzer_workflow_route_count") >= 7)
        assertTrue(result.getInt("sensor_sampling_policy_count") >= 5)
    }

    @Test
    fun motionPoseEstimateRowsFuseImuVectorsForAgentCards() {
        val samples = JSONArray()
            .put(
                JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_label", "Accelerometer")
                    .put("available", true)
                    .put("sampled", true)
                    .put("values", JSONArray().put(0.0).put(0.0).put(9.80665)),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "magnetic_field")
                    .put("sensor_label", "Magnetic field")
                    .put("available", true)
                    .put("sampled", true)
                    .put("values", JSONArray().put(0.0).put(50.0).put(0.0)),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "gyroscope")
                    .put("sensor_label", "Gyroscope")
                    .put("available", true)
                    .put("sampled", true)
                    .put("values", JSONArray().put(0.01).put(0.02).put(0.03)),
            )

        val rows = HermesDeviceDiagnosticsBridge.motionPoseEstimateRows(samples)
        val pose = rows.getJSONObject(0)
        val angular = rows.getJSONObject(1)
        val acceleration = rows.getJSONObject(2)

        assertEquals(3, rows.length())
        assertEquals("Device pose estimate", pose.getString("label"))
        assertEquals("accelerometer+magnetic_field", pose.getString("pose_source"))
        assertEquals("high", pose.getString("confidence_label"))
        assertEquals("face_up", pose.getString("face_orientation_label"))
        assertTrue(pose.has("roll_degrees"))
        assertTrue(pose.has("pitch_degrees"))
        assertTrue(pose.has("azimuth_degrees"))
        assertEquals("Angular motion state", angular.getString("label"))
        assertEquals("gyroscope", angular.getString("pose_source"))
        assertTrue(angular.getString("motion_state_label").contains("steady"))
        assertEquals("Acceleration state", acceleration.getString("label"))
        assertEquals("accelerometer", acceleration.getString("pose_source"))
        assertEquals("steady_with_gravity", acceleration.getString("motion_state_label"))
    }

    @Test
    fun motionPoseEstimateRowsPreferRotationVectorWhenAvailable() {
        val samples = JSONArray()
            .put(
                JSONObject()
                    .put("sensor_type", "rotation_vector")
                    .put("available", true)
                    .put("sampled", true)
                    .put("values", JSONArray().put(0.0).put(0.0).put(0.0).put(1.0)),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("available", true)
                    .put("sampled", true)
                    .put("values", JSONArray().put(3.0).put(0.0).put(9.0)),
            )

        val pose = HermesDeviceDiagnosticsBridge.motionPoseEstimateRows(samples).getJSONObject(0)

        assertEquals("rotation_vector", pose.getString("pose_source"))
        assertEquals("high", pose.getString("confidence_label"))
        assertEquals(0.0, pose.getDouble("roll_degrees"), 0.1)
        assertEquals(0.0, pose.getDouble("pitch_degrees"), 0.1)
        assertEquals(0.0, pose.getDouble("azimuth_degrees"), 0.1)
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
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_channel_utilization"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_ap_details"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_export"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("signal_awareness_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("soc_compatibility_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("agent_environment_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("agent_card_manifest_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_scan"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("sensor_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("local_backend_runtime_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("device_performance_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("gpu_backend_risk_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_status"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_graph"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("motion_pose"))
        assertTrue(result.getJSONObject("hindsight_memory_translation").has("retain"))
    }

    @Test
    fun localBackendRuntimeReportExposesPassiveRuntimeHealthRows() {
        val result = HermesDeviceDiagnosticsBridge.localBackendRuntimeReportJson(context)
        val rows = result.getJSONArray("runtime_backend_matrix")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("local_backend_runtime_report", result.getString("action"))
        assertTrue(result.has("selected_on_device_backend"))
        assertTrue(result.has("offline_airplane_mode"))
        assertTrue(result.getJSONObject("current_local_backend").has("backend_kind"))
        assertTrue(result.getJSONObject("litert_runtime_health").has("gpu_policy"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(labels.contains("Selected on-device backend"))
        assertTrue(labels.contains("Current local backend state"))
        assertTrue(labels.contains("LiteRT-LM /health accelerator"))
        assertTrue(labels.contains("Model artifact compatibility"))
        assertTrue(labels.contains("Multimodal runtime policy"))
        assertTrue(labels.contains("MediaTek/non-Snapdragon fallback policy"))
        assertTrue(result.getJSONArray("cards").toString().contains("Runtime Backend Health"))
        assertTrue(result.getJSONArray("cards").toString().contains("Thermal & Memory Guardrails"))
        assertTrue(result.getInt("runtime_backend_feature_count") >= 6)
        assertTrue(result.getInt("runtime_stability_feature_count") >= 6)
    }

    @Test
    fun socCompatibilityReportExposesBackendPolicyRoutesAndMediatekCoverageCards() {
        val result = HermesDeviceDiagnosticsBridge.socCompatibilityReportJson(context)
        val backend = result.getJSONArray("soc_backend_matrix")
        val routes = result.getJSONArray("soc_backend_policy_routes")
        val constraints = result.getJSONArray("soc_backend_constraint_matrix")
        val runtimeRows = result.getJSONArray("runtime_backend_matrix")
        val stabilityRows = result.getJSONArray("runtime_stability_matrix")
        val backendLabels = buildSet {
            for (index in 0 until backend.length()) add(backend.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }
        val constraintLabels = buildSet {
            for (index in 0 until constraints.length()) add(constraints.getJSONObject(index).getString("label"))
        }
        val runtimeLabels = buildSet {
            for (index in 0 until runtimeRows.length()) add(runtimeRows.getJSONObject(index).getString("label"))
        }
        val stabilityLabels = buildSet {
            for (index in 0 until stabilityRows.length()) add(stabilityRows.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("soc_compatibility_report", result.getString("action"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_backend_strategy"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_backend_order"))
        assertTrue(result.getJSONObject("soc_profile").has("litert_lm_artifact_selection_policy"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("current_local_backend"))
        assertTrue(result.has("litert_runtime_health"))
        assertTrue(result.has("likely_mediatek"))
        assertTrue(result.has("likely_snapdragon"))
        assertTrue(result.has("likely_mali_gpu"))
        assertTrue(result.has("supports_arm64"))
        assertTrue(backendLabels.contains("Detected SOC family"))
        assertTrue(backendLabels.contains("Native ABI selection"))
        assertTrue(backendLabels.contains("LiteRT-LM accelerator policy"))
        assertTrue(backendLabels.contains("SOC-specific LiteRT artifact selection"))
        assertTrue(backendLabels.contains("MediaTek/Mali/PowerVR coverage"))
        assertTrue(
            result.getJSONObject("soc_profile")
                .getJSONObject("litert_lm_artifact_selection_policy")
                .getBoolean("generic_litertlm_preferred"),
        )
        assertTrue(
            result.getJSONObject("soc_profile")
                .getJSONObject("litert_lm_artifact_selection_policy")
                .getString("recommendation")
                .contains("MediaTek"),
        )
        assertTrue(routeLabels.contains("Route SOC compatibility report"))
        assertTrue(routeLabels.contains("Route full agent environment"))
        assertTrue(constraintLabels.contains("Avoid Adreno-only assumptions"))
        assertTrue(constraintLabels.contains("GPU probe then CPU fallback"))
        assertTrue(constraintLabels.contains("x86 emulator is not phone GPU proof"))
        assertTrue(result.getJSONArray("cards").toString().contains("SOC Compatibility"))
        assertTrue(result.getJSONArray("cards").toString().contains("Runtime Backend Health"))
        assertTrue(result.getJSONArray("cards").toString().contains("Thermal & Memory Guardrails"))
        assertTrue(runtimeLabels.contains("LiteRT-LM /health accelerator"))
        assertTrue(runtimeLabels.contains("MediaTek/non-Snapdragon fallback policy"))
        assertTrue(stabilityLabels.contains("Thermal throttling status"))
        assertTrue(stabilityLabels.contains("MediaTek/non-Adreno stability guardrail"))
        assertTrue(result.getInt("soc_backend_feature_count") >= 8)
        assertTrue(result.getInt("ready_soc_backend_feature_count") >= 4)
        assertTrue(result.getInt("soc_backend_route_count") >= 5)
        assertTrue(result.getInt("soc_backend_constraint_count") >= 5)
        assertTrue(result.getInt("runtime_backend_feature_count") >= 6)
        assertTrue(result.getInt("runtime_stability_feature_count") >= 6)
    }

    @Test
    fun gpuBackendRiskReportExposesOperationalRiskMatrixAndRoutes() {
        val result = HermesDeviceDiagnosticsBridge.gpuBackendRiskReportJson(context)
        val risks = result.getJSONArray("gpu_backend_risk_matrix")
        val routes = result.getJSONArray("gpu_backend_risk_routes")
        val riskLabels = buildSet {
            for (index in 0 until risks.length()) add(risks.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("gpu_backend_risk_report", result.getString("action"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("current_local_backend"))
        assertTrue(result.has("litert_runtime_health"))
        assertTrue(result.has("gpu_backend_risk_level"))
        assertTrue(result.has("gpu_backend_risk_score"))
        assertTrue(riskLabels.contains("Live accelerator acceptance"))
        assertTrue(riskLabels.contains("SOC/GPU policy coverage"))
        assertTrue(riskLabels.contains("Thermal throttle risk"))
        assertTrue(riskLabels.contains("Memory pressure risk"))
        assertTrue(riskLabels.contains("Power saver and battery heat"))
        assertTrue(riskLabels.contains("Model artifact fit"))
        assertTrue(riskLabels.contains("Phone validation scope"))
        assertTrue(risks.getJSONObject(0).has("risk_level"))
        assertTrue(risks.getJSONObject(0).has("risk_score"))
        assertTrue(risks.getJSONObject(0).has("mitigation"))
        assertTrue(routes.toString().contains("gpu_backend_risk_report"))
        assertTrue(routeLabels.contains("Route GPU backend risk triage"))
        assertTrue(routeLabels.contains("Route live runtime health"))
        assertTrue(routeLabels.contains("Route SOC and artifact policy"))
        assertTrue(routeLabels.contains("Route stability guardrails"))
        assertTrue(routeLabels.contains("Route phone workflow preflight"))
        assertTrue(result.getJSONArray("cards").toString().contains("GPU Backend Risk"))
        assertTrue(result.getJSONArray("cards").toString().contains("Backend Risk Routes"))
        assertTrue(result.getInt("gpu_backend_risk_count") >= 8)
        assertTrue(result.getInt("gpu_backend_risk_route_count") >= 5)
    }

    @Test
    fun devicePerformanceReportExposesThermalMemoryPowerGuardrailsForRuntimeStability() {
        val result = HermesDeviceDiagnosticsBridge.devicePerformanceReportJson(context)
        val rows = result.getJSONArray("runtime_stability_matrix")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("device_performance_report", result.getString("action"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("thermal_status_label"))
        assertTrue(result.has("power_save_mode"))
        assertTrue(result.has("low_ram_device"))
        assertTrue(result.has("memory_class_mb"))
        assertTrue(result.has("large_memory_class_mb"))
        assertTrue(result.has("media_performance_class"))
        assertTrue(result.has("battery_status_label"))
        assertTrue(labels.contains("Thermal throttling status"))
        assertTrue(labels.contains("Low-RAM and memory class"))
        assertTrue(labels.contains("Battery and power saver state"))
        assertTrue(labels.contains("Android media performance class"))
        assertTrue(labels.contains("MediaTek/non-Adreno stability guardrail"))
        assertTrue(labels.contains("Local inference cadence policy"))
        assertTrue(result.getJSONArray("cards").toString().contains("Thermal & Memory Guardrails"))
        assertTrue(result.getInt("runtime_stability_feature_count") >= 6)
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
        assertTrue(result.has("wifi_access_point_semantics"))
        assertTrue(result.has("wifi_band_coverage"))
        assertTrue(featureLabels.contains("Identify nearby access points"))
        assertTrue(featureLabels.contains("Channel signal graph"))
        assertTrue(featureLabels.contains("Channel utilization occupancy"))
        assertTrue(featureLabels.contains("Pause/resume scan control"))
        assertTrue(featureLabels.contains("Band coverage and 2.4/5/6GHz visibility"))
        assertTrue(featureLabels.contains("Band, security, signal, and SSID filters"))
        assertTrue(featureLabels.contains("Agent AP semantic and risk labels"))
        assertTrue(featureLabels.contains("Vendor/OUI lookup"))
        assertTrue(featureLabels.contains("HT/VHT/HE/EHT width and standard metadata"))
        assertTrue(featureLabels.contains("Wi-Fi safety boundary"))
        assertTrue(routeLabels.contains("Route best-channel analysis"))
        assertTrue(routeLabels.contains("Route full AP metadata"))
        assertTrue(routeLabels.contains("Route AP export"))
        assertTrue(routeLabels.contains("Route pause or resume scan mode"))
        assertTrue(policyLabels.contains("Android scan throttling"))
        assertTrue(policyLabels.contains("Pause/resume scan mode"))
        assertTrue(policyLabels.contains("Passive report default"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.has("wifi_scan_control"))
        assertTrue(result.getJSONObject("wifi_scan_control").getBoolean("pause_resume_supported"))
        assertTrue(result.getJSONArray("cards").toString().contains("Wi-Fi Analyzer Readiness"))
        assertTrue(result.getInt("wifi_analyzer_feature_count") >= 8)
        assertTrue(result.getInt("wifi_analyzer_workflow_route_count") >= 6)
        assertTrue(result.getInt("wifi_scan_policy_count") >= 5)
    }

    @Test
    fun wifiScanModePausedSuppressesActiveRefreshAndResumedRequestsIt() {
        val paused = HermesDeviceDiagnosticsBridge.wifiScanJson(
            context,
            JSONObject()
                .put("refresh", true)
                .put("scan_mode", "paused"),
        )
        val resumed = HermesDeviceDiagnosticsBridge.wifiScanJson(
            context,
            JSONObject().put("scan_mode", "resumed"),
        )

        val pausedControl = paused.getJSONObject("wifi_scan_control")
        val resumedControl = resumed.getJSONObject("wifi_scan_control")

        assertEquals("paused", pausedControl.getString("scan_mode"))
        assertTrue(pausedControl.getBoolean("pause_resume_supported"))
        assertTrue(pausedControl.getBoolean("user_refresh_requested"))
        assertFalse(pausedControl.getBoolean("effective_refresh_requested"))
        assertTrue(pausedControl.getBoolean("refresh_suppressed_by_pause"))
        assertEquals("resumed", resumedControl.getString("scan_mode"))
        assertTrue(resumedControl.getBoolean("effective_refresh_requested"))
        assertTrue(resumedControl.getBoolean("resumed_requests_active_scan"))
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
        assertTrue(result.has("bluetooth_service_label_count"))
        assertTrue(result.has("bluetooth_manufacturer_name_count"))
        assertTrue(featureLabels.contains("Identify paired devices"))
        assertTrue(featureLabels.contains("Scan nearby BLE devices"))
        assertTrue(featureLabels.contains("Pause/resume BLE scan control"))
        assertTrue(featureLabels.contains("RSSI proximity graph"))
        assertTrue(featureLabels.contains("RSSI trend history graph"))
        assertTrue(featureLabels.contains("Service UUID labels"))
        assertTrue(featureLabels.contains("Manufacturer names"))
        assertTrue(featureLabels.contains("Bluetooth safety boundary"))
        assertTrue(routeLabels.contains("Route nearby Bluetooth scan"))
        assertTrue(routeLabels.contains("Route Bluetooth signal history"))
        assertTrue(routeLabels.contains("Route pause or resume BLE scan mode"))
        assertTrue(routeLabels.contains("Route service/manufacturer semantics"))
        assertTrue(routeLabels.contains("Route scan policy explanation"))
        assertTrue(policyLabels.contains("Connect and scan permissions"))
        assertTrue(policyLabels.contains("Active scan cadence"))
        assertTrue(policyLabels.contains("Pause/resume BLE scan mode"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.has("bluetooth_scan_control"))
        assertTrue(result.getJSONObject("bluetooth_scan_control").getBoolean("pause_resume_supported"))
        assertTrue(result.getJSONArray("cards").toString().contains("Bluetooth Analyzer Readiness"))
        assertTrue(result.getInt("bluetooth_analyzer_feature_count") >= 9)
        assertTrue(result.getInt("bluetooth_analyzer_workflow_route_count") >= 6)
        assertTrue(result.getInt("bluetooth_scan_policy_count") >= 6)
    }

    @Test
    fun bluetoothScanModePausedSuppressesActiveRefreshAndResumedRequestsIt() {
        val paused = HermesDeviceDiagnosticsBridge.bluetoothScanJson(
            context,
            JSONObject()
                .put("refresh", true)
                .put("scan_mode", "paused"),
        )
        val resumed = HermesDeviceDiagnosticsBridge.bluetoothScanJson(
            context,
            JSONObject().put("scan_mode", "resumed"),
        )

        val pausedControl = paused.getJSONObject("bluetooth_scan_control")
        val resumedControl = resumed.getJSONObject("bluetooth_scan_control")

        assertEquals("paused", pausedControl.getString("scan_mode"))
        assertTrue(pausedControl.getBoolean("pause_resume_supported"))
        assertTrue(pausedControl.getBoolean("user_refresh_requested"))
        assertFalse(pausedControl.getBoolean("effective_refresh_requested"))
        assertTrue(pausedControl.getBoolean("refresh_suppressed_by_pause"))
        assertEquals("resumed", resumedControl.getString("scan_mode"))
        assertTrue(resumedControl.getBoolean("effective_refresh_requested"))
        assertTrue(resumedControl.getBoolean("resumed_requests_active_scan"))
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
        assertTrue(result.has("cached_bluetooth_signal_history"))
        assertTrue(result.getJSONArray("radio_bands").length() >= 6)
        assertTrue(result.getJSONArray("radio_receiver_profiles").length() >= 5)
        assertTrue(result.getJSONArray("radio_signal_feature_matrix").length() >= 6)
        assertTrue(result.getJSONArray("radio_signal_workflow_routes").length() >= 4)
        assertTrue(result.getJSONArray("radio_signal_constraint_matrix").length() >= 4)
        assertTrue(result.getJSONArray("gpu_backend_risk_matrix").length() >= 8)
        assertTrue(result.getJSONArray("gpu_backend_risk_routes").length() >= 5)
        assertTrue(result.has("gpu_backend_risk_level"))
        assertTrue(result.has("gpu_backend_risk_score"))
        assertTrue(result.getInt("radio_band_plan_count") >= 6)
        assertTrue(result.getInt("radio_receiver_profile_count") >= 5)
        assertTrue(result.getInt("radio_signal_feature_count") >= 6)
        assertTrue(awarenessLabels.contains("Wi-Fi scan surface"))
        assertTrue(awarenessLabels.contains("Bluetooth proximity metadata"))
        assertTrue(awarenessLabels.contains("Cached Bluetooth trend memory"))
        assertTrue(awarenessLabels.contains("Radio/RF limits"))
        assertTrue(awarenessLabels.contains("SOC backend compatibility"))
        assertTrue(awarenessLabels.contains("GPU backend risk triage"))
        assertTrue(routeLabels.contains("Route Wi-Fi analyzer work"))
        assertTrue(routeLabels.contains("Route Bluetooth proximity work"))
        assertTrue(routeLabels.contains("Route Bluetooth trend work"))
        assertTrue(routeLabels.contains("Route backend risk triage"))
        assertTrue(constraintLabels.contains("AM/FM tuner public API"))
        assertTrue(constraintLabels.contains("Broad RF and microwave hardware"))
        assertTrue(result.getJSONArray("cards").toString().contains("Signal Awareness"))
        assertTrue(result.getJSONArray("cards").toString().contains("GPU Backend Risk"))
        assertTrue(result.getJSONArray("cards").toString().contains("Radio Band Plan"))
        assertTrue(result.getJSONArray("cards").toString().contains("Receiver Profiles"))
        assertTrue(result.getInt("signal_awareness_count") >= 10)
        assertTrue(result.getInt("signal_workflow_route_count") >= 8)
        assertTrue(result.getInt("signal_constraint_count") >= 5)
    }

    @Test
    fun agentEnvironmentReportSummarizesKaiParityAndSystemInputs() {
        AppSettingsStore(context).save(
            AppSettings(customSystemPrompt = "Prefer local signal cards before broad answers."),
        )
        val result = HermesDeviceDiagnosticsBridge.agentEnvironmentReportJson(context)
        val capabilities = result.getJSONArray("agent_capability_matrix")
        val kaiParity = result.getJSONArray("kai_parity_matrix")
        val kaiOperations = result.getJSONArray("kai_operations_matrix")
        val readiness = result.getJSONArray("workflow_readiness_matrix")
        val capabilityText = capabilities.toString()
        val kaiText = kaiParity.toString()
        val kaiOperationsText = kaiOperations.toString()
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
        assertTrue(kaiText.contains("Customizable soul"))
        assertTrue(kaiText.contains("system prompt"))
        assertTrue(kaiText.contains("custom_system_prompt"))
        assertTrue(kaiText.contains("Multi-provider priority and fallback"))
        assertTrue(kaiText.contains("MCP and external tool equivalents"))
        assertTrue(kaiText.contains("Encrypted credentials and local storage"))
        assertTrue(kaiText.contains("Text to speech"))
        assertTrue(kaiText.contains("Autonomous heartbeat"))
        assertTrue(kaiText.contains("Image attachments and screen vision"))
        assertTrue(kaiText.contains("App settings and automation backup"))
        assertTrue(kaiText.contains("export_app_settings"))
        assertTrue(kaiOperationsText.contains("Provider priority and fallback route"))
        assertTrue(kaiOperationsText.contains("Persona and system prompt route"))
        assertTrue(kaiOperationsText.contains("Tool and MCP bridge route"))
        assertTrue(kaiOperationsText.contains("Encrypted credentials and backup route"))
        assertTrue(kaiOperationsText.contains("android_automation_tool:export_app_settings"))
        assertTrue(kaiOperationsText.contains("Scheduled task compatibility route"))
        assertTrue(kaiOperationsText.contains("schedule_task"))
        assertTrue(kaiOperationsText.contains("list_tasks"))
        assertTrue(kaiOperationsText.contains("cancel_task"))
        assertTrue(kaiOperationsText.contains("kai_task_compat"))
        assertTrue(kaiOperationsText.contains("background_ai_prompt_execution"))
        assertTrue(kaiOperationsText.contains("TTS and image conversation route"))
        assertTrue(kaiOperationsText.contains("Android shell boundary route"))
        assertTrue(result.getJSONObject("agent_persona_status").getBoolean("custom_system_prompt_enabled"))
        assertTrue(readinessText.contains("Analyze nearby Wi-Fi"))
        assertTrue(readinessText.contains("Run local multimodal agent"))
        assertTrue(readinessText.contains("Route Kai-style tool orchestration"))
        assertTrue(result.getJSONArray("cards").toString().contains("Kai Parity"))
        assertTrue(result.getJSONArray("cards").toString().contains("Kai Operations"))
        assertTrue(result.getInt("agent_capability_count") >= 8)
        assertTrue(result.getInt("kai_parity_count") >= 11)
        assertTrue(result.getInt("kai_operations_count") >= 8)
        assertTrue(result.getInt("ready_kai_operations_count") >= 5)
    }

    @Test
    fun agentObservationReportComposesGemmaVisibleSignalDashboard() {
        val result = HermesDeviceDiagnosticsBridge.agentObservationReportJson(context)
        val observations = result.getJSONArray("agent_observation_matrix")
        val signalContext = result.getJSONArray("agent_signal_context_matrix")
        val cardManifest = result.getJSONArray("agent_card_manifest")
        val routes = result.getJSONArray("agent_observation_routes")
        val observationText = observations.toString()
        val signalContextText = signalContext.toString()
        val cardManifestText = cardManifest.toString()
        val routeText = routes.toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_observation_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_analyzer_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("gpu_backend_risk_report"))
        assertTrue(result.getJSONObject("wifi_observation_summary").getJSONArray("card_titles").toString().contains("Wi-Fi Analyzer"))
        assertTrue(result.getJSONObject("backend_risk_observation_summary").getJSONArray("card_titles").toString().contains("GPU Backend Risk"))
        assertTrue(result.getJSONObject("agent_environment_observation_summary").getJSONArray("card_titles").toString().contains("Kai Operations"))
        assertTrue(observationText.contains("Gemma signal dashboard"))
        assertTrue(observationText.contains("Signal context fusion matrix"))
        assertTrue(observationText.contains("Agent-readable card manifest"))
        assertTrue(observationText.contains("Wi-Fi AP metadata and channel graphs"))
        assertTrue(observationText.contains("Bluetooth nearby metadata"))
        assertTrue(observationText.contains("Motion and sensor context"))
        assertTrue(observationText.contains("Radio and RF boundaries"))
        assertTrue(observationText.contains("GPU backend risk triage"))
        assertTrue(observationText.contains("SOC and local model readiness"))
        assertTrue(observationText.contains("Kai operations and interactive routes"))
        assertTrue(signalContextText.contains("Gemma signal context contract"))
        assertTrue(signalContextText.contains("Wi-Fi channel and band context"))
        assertTrue(signalContextText.contains("Bluetooth RSSI and identity context"))
        assertTrue(signalContextText.contains("Motion pose and sensor context"))
        assertTrue(signalContextText.contains("Radio hardware boundary context"))
        assertTrue(signalContextText.contains("Backend risk and fallback context"))
        assertTrue(signalContextText.contains("source_actions"))
        assertTrue(signalContextText.contains("card_graph_types"))
        assertTrue(cardManifestText.contains("wifi_analyzer_report"))
        assertTrue(cardManifestText.contains("wifi_channel_graph"))
        assertTrue(cardManifestText.contains("bluetooth_analyzer_report"))
        assertTrue(cardManifestText.contains("radio_signal_status"))
        assertTrue(cardManifestText.contains("gpu_backend_risk_report"))
        assertTrue(cardManifestText.contains("agent_environment_report"))
        assertTrue(cardManifestText.contains("refresh_policy"))
        assertTrue(cardManifestText.contains("permission_gate"))
        assertTrue(routeText.contains("Open Wi-Fi analyzer cards"))
        assertTrue(routeText.contains("Open signal context fusion card"))
        assertTrue(routeText.contains("Open GPU backend risk cards"))
        assertTrue(routeText.contains("Open SOC and Kai environment cards"))
        assertTrue(result.getJSONArray("cards").toString().contains("Agent Observation"))
        assertTrue(result.getJSONArray("cards").toString().contains("Signal Context Fusion"))
        assertTrue(result.getJSONArray("cards").toString().contains("Agent Card Manifest"))
        assertTrue(result.getJSONArray("cards").toString().contains("Observation Routes"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("Read agent_observation_matrix first"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("agent_signal_context_matrix"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("agent_card_manifest"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("gpu_backend_risk_report"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("wifi_channel_graph"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("bluetooth_signal_history"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("gpu_backend_risk_matrix"))
        assertTrue(result.getInt("agent_card_manifest_count") >= 18)
        assertTrue(result.getInt("ready_agent_card_manifest_count") >= 8)
        assertTrue(result.getInt("agent_signal_context_count") >= 7)
        assertTrue(result.getInt("ready_agent_signal_context_count") >= 4)
        assertTrue(result.getInt("agent_observation_count") >= 11)
        assertTrue(result.getInt("agent_observation_route_count") >= 8)
    }

    @Test
    fun agentCardManifestReportExposesDirectGraphCardManifest() {
        val result = HermesDeviceDiagnosticsBridge.agentCardManifestReportJson(context)
        val cardManifest = result.getJSONArray("agent_card_manifest")
        val cardManifestText = cardManifest.toString()
        val graphTypes = result.getJSONArray("agent_card_graph_types").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_card_manifest_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_analyzer_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("gpu_backend_risk_report"))
        assertTrue(cardManifestText.contains("wifi_channel_graph"))
        assertTrue(cardManifestText.contains("wifi_channel_rating"))
        assertTrue(cardManifestText.contains("bluetooth_signal_history"))
        assertTrue(cardManifestText.contains("radio_signal_graph"))
        assertTrue(cardManifestText.contains("gpu_backend_risk_report"))
        assertTrue(cardManifestText.contains("refresh_policy"))
        assertTrue(cardManifestText.contains("permission_gate"))
        assertTrue(graphTypes.contains("wifi_channel_graph"))
        assertTrue(graphTypes.contains("radio_signal_graph"))
        assertTrue(graphTypes.contains("gpu_backend_risk_matrix"))
        assertTrue(result.getJSONArray("cards").toString().contains("Agent Card Manifest"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("graph_type"))
        assertTrue(result.getInt("agent_card_manifest_count") >= 18)
        assertTrue(result.getInt("ready_agent_card_manifest_count") >= 8)
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
