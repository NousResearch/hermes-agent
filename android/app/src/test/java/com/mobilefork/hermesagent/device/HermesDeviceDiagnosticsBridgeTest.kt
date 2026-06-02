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
        assertTrue(result.getJSONArray("radio_receiver_bridge_schema").length() >= 3)
        assertTrue(result.getInt("radio_receiver_bridge_schema_count") >= 3)
        assertTrue(result.getJSONArray("radio_signal_feature_matrix").length() >= 6)
        assertTrue(result.getJSONArray("radio_signal_workflow_routes").length() >= 4)
        assertTrue(result.getJSONArray("radio_signal_constraint_matrix").length() >= 4)
        assertTrue(result.getJSONArray("radio_signal_graph_rows").length() >= 2)
        assertEquals(result.getJSONArray("radio_signal_graph_rows").length(), result.getInt("radio_signal_graph_row_count"))
        assertEquals(0, result.getInt("radio_signal_graph_sample_count"))
        assertFalse(result.getJSONObject("radio_signal_graph_sample_summary").getBoolean("bridge_ready"))
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
        assertTrue(result.getJSONArray("radio_receiver_bridge_schema").toString().contains("radio_samples_json"))
        assertTrue(result.getJSONArray("radio_receiver_bridge_schema").toString().contains("rds_radio_text"))
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
        assertTrue(cards.toString().contains("Radio Bridge Sample Schema"))
        assertTrue(cards.toString().contains("radio_receiver_bridge_schema"))
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
        assertTrue(result.getJSONObject("radio_signal_graph_sample_summary").getBoolean("bridge_ready"))
        assertEquals(1, result.getJSONObject("radio_signal_graph_sample_summary").getInt("fm_sample_count"))
        assertEquals(1, result.getJSONObject("radio_signal_graph_sample_summary").getInt("am_sample_count"))
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
        assertTrue(cards.toString().contains("Radio Bridge Sample Schema"))
        assertTrue(result.getJSONArray("radio_receiver_bridge_schema").toString().contains("station_label"))
    }

    @Test
    fun radioSignalGraphAcceptsDirectAndJsonBridgeSamples() {
        val result = HermesDeviceDiagnosticsBridge.radioSignalGraphJson(
            context,
            JSONObject()
                .put("sample_source", "unit_test_direct_bridge")
                .put("station_label", "Direct FM")
                .put("frequency_mhz", "101.7")
                .put("rssi_dbm", "-52")
                .put("snr_db", "27")
                .put("modulation", "fm")
                .put("receiver_id", "fm_vendor_or_sdr")
                .put("rds_program_service", "HERMES")
                .put("rds_radio_text", "Bridge supplied RDS text")
                .put(
                    "radio_samples_json",
                    """[{"station_label":"JSON AM","frequency_khz":880,"power_db":-68,"modulation":"am","receiver_id":"am_vendor_or_sdr"}]""",
                ),
        )

        assertTrue(result.getBoolean("success"))
        assertTrue(result.getBoolean("radio_signal_graph_bridge_ready"))
        assertEquals(2, result.getInt("radio_signal_graph_sample_count"))
        val summary = result.getJSONObject("radio_signal_graph_sample_summary")
        assertEquals(1, summary.getInt("fm_sample_count"))
        assertEquals(1, summary.getInt("am_sample_count"))
        assertEquals(1, summary.getInt("rds_sample_count"))
        val rows = result.getJSONArray("radio_signal_graph_rows").toString()
        assertTrue(rows.contains("Direct FM"))
        assertTrue(rows.contains("101.7 MHz"))
        assertTrue(rows.contains("HERMES"))
        assertTrue(rows.contains("Bridge supplied RDS text"))
        assertTrue(rows.contains("JSON AM"))
        assertTrue(rows.contains("880 kHz"))
        assertTrue(result.getJSONArray("radio_receiver_bridge_schema").toString().contains("direct_argument_fields"))
    }

    @Test
    fun radioSignalGraphAcceptsSdrBridgeAliasesAndReportsSampleMetadataReadiness() {
        val result = JSONObject(
            HermesDeviceDiagnosticsBridge.performActionJson(
                context,
                "sdr_bridge_samples",
                JSONObject()
                    .put("sample_source", "unit_test_sdr_bridge")
                    .put(
                        "sdr_samples_json",
                        """[{"label":"NOAA sample","center_frequency_hz":162550000,"power_db":-43.5,"snr_db":18,"receiver_id":"external_sdr_bridge","span_hz":200000,"sample_rate_hz":240000,"bin_width_hz":1250}]""",
                    ),
            ),
        )

        assertTrue(result.getBoolean("success"))
        assertEquals("radio_signal_graph", result.getString("action"))
        assertTrue(result.getBoolean("radio_signal_graph_bridge_ready"))
        assertEquals(1, result.getInt("radio_signal_graph_sample_count"))
        assertEquals(1, result.getJSONObject("radio_signal_graph_sample_summary").getInt("external_sdr_sample_count"))
        assertTrue(result.getJSONArray("accepted_radio_bridge_sample_json_keys").toString().contains("sdr_samples_json"))
        assertTrue(result.getJSONArray("accepted_radio_bridge_sample_array_keys").toString().contains("radio_bridge_samples"))
        val metadataRows = result.getJSONArray("radio_bridge_sample_metadata")
        assertEquals(metadataRows.length(), result.getInt("radio_bridge_sample_metadata_count"))
        assertEquals(1, result.getInt("ready_radio_bridge_sample_metadata_count"))
        val metadata = metadataRows.getJSONObject(0)
        assertEquals("NOAA sample", metadata.getString("label"))
        assertEquals("sample ready", metadata.getString("value_label"))
        assertTrue(metadata.getBoolean("external_sdr_sample"))
        assertTrue(metadata.getBoolean("span_metadata_present"))
        assertTrue(metadata.getBoolean("sample_rate_metadata_present"))
        assertTrue(metadata.getBoolean("required_metadata_ready"))
        assertEquals(100, metadata.getInt("metadata_completeness_score"))
        assertTrue(result.getJSONArray("cards").toString().contains("Radio Bridge Sample Metadata"))
        assertTrue(result.getJSONArray("cards").toString().contains("radio_bridge_sample_metadata"))
    }

    @Test
    fun radioSignalAdvisorRanksReceiversAndBuildsDecisionRows() {
        val status = HermesDeviceDiagnosticsBridge.radioSignalStatusJson(context)
        val graph = HermesDeviceDiagnosticsBridge.radioSignalGraphJson(
            context,
            JSONObject()
                .put("sample_source", "unit_test_radio_advisor")
                .put(
                    "radio_samples",
                    JSONArray()
                        .put(
                            JSONObject()
                                .put("station_label", "Hermes FM")
                                .put("frequency_mhz", 99.5)
                                .put("rssi_dbm", -48)
                                .put("snr_db", 31)
                                .put("receiver_id", "fm_vendor_or_sdr")
                                .put("modulation", "fm"),
                        )
                        .put(
                            JSONObject()
                                .put("label", "NOAA sample")
                                .put("center_frequency_hz", 162550000)
                                .put("power_db", -43.5)
                                .put("receiver_id", "external_sdr_bridge")
                                .put("span_hz", 200000)
                                .put("sample_rate_hz", 240000),
                        ),
                ),
        )

        val candidates = HermesDeviceDiagnosticsBridge.radioReceiverCandidateRows(status, graph)
        val advisorRows = HermesDeviceDiagnosticsBridge.radioSignalAdvisorRows(status, graph)
        val advisorLabels = buildSet {
            for (index in 0 until advisorRows.length()) add(advisorRows.getJSONObject(index).getString("label"))
        }

        assertEquals("fm_vendor_or_sdr", candidates.getJSONObject(0).getString("receiver_id"))
        assertTrue(candidates.getJSONObject(0).getInt("candidate_score") >= candidates.getJSONObject(1).getInt("candidate_score"))
        assertEquals(1, candidates.getJSONObject(0).getInt("sample_count"))
        assertTrue(advisorLabels.contains("Receiver source decision"))
        assertTrue(advisorLabels.contains("AM/FM scan decision"))
        assertTrue(advisorLabels.contains("External SDR sample decision"))
        assertTrue(advisorLabels.contains("Bridge metadata completeness decision"))
        assertTrue(advisorLabels.contains("Wi-Fi/Bluetooth radio metadata decision"))
        assertTrue(advisorLabels.contains("Radio safety boundary decision"))
    }

    @Test
    fun radioSignalAdvisorReportRoutesCardsAndPreservesBridgeSamples() {
        val result = HermesDeviceDiagnosticsBridge.radioSignalAdvisorReportJson(
            context,
            JSONObject()
                .put("sample_source", "unit_test_radio_advisor_report")
                .put(
                    "radio_samples_json",
                    """[{"station_label":"Advisor FM","frequency_mhz":101.7,"rssi_dbm":-52,"receiver_id":"fm_vendor_or_sdr","modulation":"fm"}]""",
                ),
        )

        assertTrue(result.getBoolean("success"))
        assertEquals("radio_signal_advisor_report", result.getString("action"))
        assertTrue(result.getBoolean("radio_signal_graph_bridge_ready"))
        assertTrue(result.has("radio_signal_advisor_matrix"))
        assertTrue(result.has("radio_receiver_candidates"))
        assertTrue(result.has("gemma_radio_advisor_directives"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("radio_signal_graph"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("sdr_bridge_samples"))
        assertTrue(result.getJSONArray("cards").toString().contains("Radio Signal Advisor"))
        assertTrue(result.getJSONArray("cards").toString().contains("Radio Receiver Candidates"))
        assertTrue(result.getJSONArray("radio_signal_graph_sample_rows").toString().contains("Advisor FM"))
        assertEquals(
            result.getJSONArray("radio_signal_advisor_matrix").length(),
            result.getInt("radio_signal_advisor_count"),
        )
    }

    @Test
    fun radioSignalDecisionPacketReportExposesClaimSafeDecisionCards() {
        val result = HermesDeviceDiagnosticsBridge.radioSignalDecisionPacketReportJson(
            context,
            JSONObject()
                .put("sample_source", "unit_test_radio_decision_packet")
                .put(
                    "radio_samples_json",
                    """[{"station_label":"Decision FM","frequency_mhz":99.9,"rssi_dbm":-50,"receiver_id":"fm_vendor_or_sdr","modulation":"fm"},{"station_label":"Decision AM","frequency_khz":880,"rssi_dbm":-61,"receiver_id":"am_vendor_or_sdr","modulation":"am"}]""",
                ),
        )

        assertTrue(result.getBoolean("success"))
        assertEquals("radio_signal_decision_packet_report", result.getString("action"))
        assertTrue(result.getBoolean("radio_signal_graph_bridge_ready"))
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        assertTrue(sourceActions.contains("radio_signal_advisor_report"))
        assertTrue(sourceActions.contains("radio_signal_graph"))
        assertTrue(sourceActions.contains("rf_coexistence_report"))
        assertTrue(sourceActions.contains("mediatek_signal_stack_report"))

        val packetText = result.getJSONArray("radio_signal_decision_packet").toString()
        assertTrue(packetText.contains("Receiver source packet"))
        assertTrue(packetText.contains("sample packet"))
        assertTrue(packetText.contains("decision_status"))
        assertTrue(packetText.contains("active_refresh_action"))
        assertTrue(packetText.contains("passive_fallback_action"))
        assertTrue(packetText.contains("claim_scope"))
        assertTrue(packetText.contains("mediatek_sensitive"))
        assertTrue(packetText.contains("rf_coexistence_sensitive"))
        assertTrue(result.getJSONArray("radio_signal_decision_routes").toString().contains("radio_signal_graph"))
        assertTrue(result.getJSONArray("radio_signal_decision_routes").toString().contains("mediatek_signal_stack_report"))
        assertTrue(result.getJSONArray("radio_signal_claim_boundaries").toString().contains("Public Android radio boundary"))
        assertTrue(result.getJSONArray("radio_signal_decision_graph_types").toString().contains("radio_signal_decision_packet"))
        assertTrue(result.getJSONArray("cards").toString().contains("Radio Signal Decision"))
        assertTrue(result.getJSONArray("gemma_radio_signal_decision_directives").toString().contains("claim_boundaries"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "mediatek_radio_decision_packet"))
        assertEquals("radio_signal_decision_packet_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("radio_signal_decision_packet_report"))
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
        val qualityCard = result.getJSONArray("cards").getJSONObject(4)
        assertEquals("Motion Sensor Quality", qualityCard.getString("title"))
        assertEquals("motion_sensor_quality", qualityCard.getString("graph_type"))
        assertEquals(result.getJSONArray("motion_sensor_quality").length(), qualityCard.getInt("row_count"))
        assertTrue(result.has("sampled_sensor_count"))
        assertTrue(result.has("motion_sensor_history_count"))
        assertTrue(result.has("motion_pose_estimate_count"))
        assertTrue(result.has("motion_sensor_quality_count"))
        assertTrue(result.has("motion_sensor_quality_level"))
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
        assertTrue(featureLabels.contains("Motion sensor quality gates"))
        assertTrue(featureLabels.contains("Accelerometer access"))
        assertTrue(featureLabels.contains("Gyroscope access"))
        assertTrue(featureLabels.contains("Sensor hardware metadata"))
        assertTrue(featureLabels.contains("Sensor privacy and power boundary"))
        assertTrue(routeLabels.contains("Route one-shot motion sample"))
        assertTrue(routeLabels.contains("Route motion trend history"))
        assertTrue(routeLabels.contains("Route motion pose fusion"))
        assertTrue(routeLabels.contains("Route motion quality gates"))
        assertTrue(routeLabels.contains("Route sensor hardware metadata"))
        assertTrue(routeLabels.contains("Route sampling policy explanation"))
        assertTrue(policyLabels.contains("Passive report default"))
        assertTrue(policyLabels.contains("Bounded one-shot timeout"))
        assertTrue(policyLabels.contains("Analysis and privacy boundary"))
        assertTrue(result.getJSONArray("cards").toString().contains("Sensor Analyzer Readiness"))
        assertTrue(result.getJSONArray("cards").toString().contains("Motion Sensor Quality"))
        assertTrue(result.has("motion_sensor_quality_count"))
        assertTrue(result.has("ready_motion_sensor_quality_count"))
        assertTrue(result.has("motion_sensor_quality_score"))
        assertTrue(result.has("motion_sensor_quality_level"))
        assertTrue(result.getInt("sensor_analyzer_feature_count") >= 10)
        assertTrue(result.getInt("sensor_analyzer_workflow_route_count") >= 7)
        assertTrue(result.getInt("sensor_sampling_policy_count") >= 5)
    }

    @Test
    fun sensorWorkflowAdvisorRanksMotionCandidatesAndBuildsDecisionRows() {
        val analyzerReport = JSONObject()
            .put("sensor_service_available", true)
            .put("motion_sensor_count", 4)
            .put("motion_sensor_quality_score", 91)
            .put("motion_sensor_quality_level", "ready")
            .put("available_sensor_types", JSONArray().put("accelerometer").put("gyroscope").put("rotation_vector").put("light"))
            .put(
                "sensor_sampling_status",
                JSONObject()
                    .put("sensor_service_available", true)
                    .put("active_sample_requested", false)
                    .put("requested_sensor_count", 4)
                    .put("requested_available_count", 4)
                    .put("timeout_ms", 800),
            )
            .put(
                "sensor_capabilities",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("sensor_type", "accelerometer")
                            .put("sensor_label", "Accelerometer")
                            .put("available", true)
                            .put("sensor_name", "Hermes Accel")
                            .put("vendor", "Android")
                            .put("power_ma", 0.5)
                            .put("min_delay_us", 5000)
                            .put("resolution", 0.001)
                            .put("fifo_max_event_count", 64)
                            .put("wake_up", true),
                    )
                    .put(
                        JSONObject()
                            .put("sensor_type", "gyroscope")
                            .put("sensor_label", "Gyroscope")
                            .put("available", true)
                            .put("sensor_name", "Hermes Gyro")
                            .put("vendor", "Android")
                            .put("power_ma", 0.7)
                            .put("min_delay_us", 5000)
                            .put("resolution", 0.001),
                    )
                    .put(
                        JSONObject()
                            .put("sensor_type", "light")
                            .put("sensor_label", "Light")
                            .put("available", true)
                            .put("power_ma", 0.2),
                    ),
            )
            .put("motion_sensor_quality", JSONArray().put(JSONObject().put("label", "Motion-aware workflow readiness").put("ready", true)))
            .put("motion_sensor_history", JSONArray().put(JSONObject().put("sensor_type", "accelerometer").put("sample_count", 3)))
            .put("motion_pose_estimates", JSONArray().put(JSONObject().put("pose_type", "device_pose").put("confidence_label", "high")))
        val performanceReport = JSONObject()
            .put("thermal_status_label", "nominal")
            .put("power_save_mode", false)
            .put("low_ram_device", false)
            .put("runtime_stability_matrix", JSONArray().put(JSONObject().put("label", "Memory class").put("ready", true)))

        val candidates = HermesDeviceDiagnosticsBridge.sensorWorkflowCandidateRows(analyzerReport)
        val advisorRows = HermesDeviceDiagnosticsBridge.sensorWorkflowAdvisorRows(analyzerReport, performanceReport)
        val advisorLabels = buildSet {
            for (index in 0 until advisorRows.length()) add(advisorRows.getJSONObject(index).getString("label"))
        }

        assertEquals("Accelerometer", candidates.getJSONObject(0).getString("label"))
        assertTrue(candidates.getJSONObject(0).getInt("candidate_score") >= candidates.getJSONObject(1).getInt("candidate_score"))
        assertEquals("accelerometer", candidates.getJSONObject(0).getString("sensor_type"))
        assertTrue(advisorLabels.contains("Motion readiness decision"))
        assertTrue(advisorLabels.contains("Accelerometer workflow readiness"))
        assertTrue(advisorLabels.contains("Gyroscope workflow readiness"))
        assertTrue(advisorLabels.contains("Pose and motion history decision"))
        assertTrue(advisorLabels.contains("Sampling and watcher decision"))
        assertTrue(advisorLabels.contains("Thermal and system runway decision"))
        assertTrue(advisorLabels.contains("Gemma workflow recommendation"))
    }

    @Test
    fun sensorWorkflowAdvisorReportRoutesPassiveCardsAndSourceActions() {
        val result = HermesDeviceDiagnosticsBridge.sensorWorkflowAdvisorReportJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("sensor_workflow_advisor_report", result.getString("action"))
        assertTrue(result.has("sensor_workflow_advisor_matrix"))
        assertTrue(result.has("sensor_workflow_candidates"))
        assertTrue(result.has("gemma_sensor_workflow_directives"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("motion_sensor_quality"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("device_performance_report"))
        assertTrue(result.getJSONArray("cards").toString().contains("Sensor Workflow Advisor"))
        assertTrue(result.getJSONArray("cards").toString().contains("Sensor Workflow Candidates"))
        assertEquals(
            result.getJSONArray("sensor_workflow_advisor_matrix").length(),
            result.getInt("sensor_workflow_advisor_count"),
        )
    }

    @Test
    fun motionSensorDecisionPacketReportExposesClaimSafeDecisionCards() {
        val result = HermesDeviceDiagnosticsBridge.motionSensorDecisionPacketReportJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("motion_sensor_decision_packet_report", result.getString("action"))
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        assertTrue(sourceActions.contains("sensor_analyzer_report"))
        assertTrue(sourceActions.contains("sensor_workflow_advisor_report"))
        assertTrue(sourceActions.contains("motion_sensor_quality"))
        assertTrue(sourceActions.contains("motion_sensor_history"))
        assertTrue(sourceActions.contains("motion_pose"))
        assertTrue(sourceActions.contains("mediatek_signal_stack_report"))

        val packetText = result.getJSONArray("motion_sensor_decision_packet").toString()
        assertTrue(packetText.contains("Motion sensor decision packet"))
        assertTrue(packetText.contains("Accelerometer workflow packet"))
        assertTrue(packetText.contains("Gyroscope and pose packet"))
        assertTrue(packetText.contains("decision_status"))
        assertTrue(packetText.contains("active_refresh_action"))
        assertTrue(packetText.contains("passive_fallback_action"))
        assertTrue(packetText.contains("claim_scope"))
        assertTrue(packetText.contains("mediatek_sensitive"))
        assertTrue(packetText.contains("sensor_privacy_sensitive"))
        assertTrue(result.getJSONArray("motion_sensor_decision_routes").toString().contains("motion_pose"))
        assertTrue(result.getJSONArray("motion_sensor_decision_routes").toString().contains("mediatek_signal_stack_report"))
        assertTrue(result.getJSONArray("motion_sensor_claim_boundaries").toString().contains("Android SensorManager boundary"))
        assertTrue(result.getJSONArray("motion_sensor_decision_graph_types").toString().contains("motion_sensor_decision_packet"))
        assertTrue(result.getJSONArray("cards").toString().contains("Motion Sensor Decision"))
        assertTrue(result.getJSONArray("gemma_motion_sensor_decision_directives").toString().contains("claim_boundaries"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "mediatek_motion_decision_packet"))
        assertEquals("motion_sensor_decision_packet_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("motion_sensor_decision_packet_report"))
        assertTrue(availableActions.contains("motion_sensor_quality"))
    }

    @Test
    fun motionSensorQualityRowsGateFusionFreshnessCalibrationAndStability() {
        val capabilities = JSONArray()
            .put(
                JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_label", "Accelerometer")
                    .put("available", true)
                    .put("min_delay_us", 5000)
                    .put("power_ma", 0.6),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "gyroscope")
                    .put("sensor_label", "Gyroscope")
                    .put("available", true)
                    .put("min_delay_us", 5000)
                    .put("power_ma", 0.8),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "rotation_vector")
                    .put("sensor_label", "Rotation vector")
                    .put("available", true)
                    .put("min_delay_us", 10000)
                    .put("power_ma", 0.5),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "magnetic_field")
                    .put("sensor_label", "Magnetic field")
                    .put("available", true)
                    .put("min_delay_us", 20000)
                    .put("power_ma", 0.3),
            )
        val history = JSONArray()
            .put(
                JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_label", "Accelerometer")
                    .put("current_magnitude", 9.81)
                    .put("last_seen_ms", 500)
                    .put("sample_count", 3)
                    .put("stability_label", "stable")
                    .put("accuracy_label", "high"),
            )
            .put(
                JSONObject()
                    .put("sensor_type", "gyroscope")
                    .put("sensor_label", "Gyroscope")
                    .put("current_magnitude", 0.03)
                    .put("last_seen_ms", 600)
                    .put("sample_count", 3)
                    .put("stability_label", "stable")
                    .put("accuracy_label", "high"),
            )
        val poses = JSONArray()
            .put(
                JSONObject()
                    .put("pose_type", "device_pose")
                    .put("pose_source", "rotation_vector")
                    .put("confidence_label", "high")
                    .put("heading_label", "N")
                    .put("tilt_degrees", 2.0)
                    .put("fraction", 0.95),
            )

        val rows = HermesDeviceDiagnosticsBridge.motionSensorQualityRows(
            capabilities = capabilities,
            motionHistoryRows = history,
            motionPoseEstimates = poses,
            sensorServiceAvailable = true,
            activeSampleRequested = false,
            timeoutMs = 800,
        )
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val text = rows.toString()

        assertEquals(8, rows.length())
        assertTrue(labels.contains("IMU fusion source coverage"))
        assertTrue(labels.contains("Pose confidence gate"))
        assertTrue(labels.contains("Gyroscope stability gate"))
        assertTrue(labels.contains("Acceleration stability gate"))
        assertTrue(labels.contains("Cached sample freshness"))
        assertTrue(labels.contains("Accuracy and calibration gate"))
        assertTrue(labels.contains("Sampling cadence and power"))
        assertTrue(labels.contains("Motion-aware workflow readiness"))
        assertTrue(text.contains("rotation_vector"))
        assertTrue(text.contains("high"))
        assertTrue(rows.getJSONObject(0).getBoolean("ready"))
        assertTrue(rows.getJSONObject(7).getBoolean("ready"))
    }

    @Test
    fun motionSensorQualityReportExposesDirectCardsAndCounts() {
        val result = HermesDeviceDiagnosticsBridge.motionSensorQualityJson(context)

        assertEquals("motion_sensor_quality", result.getString("action"))
        assertTrue(result.has("motion_sensor_quality"))
        assertTrue(result.has("motion_sensor_quality_count"))
        assertTrue(result.has("ready_motion_sensor_quality_count"))
        assertTrue(result.has("motion_sensor_quality_score"))
        assertTrue(result.has("motion_sensor_quality_level"))
        val card = result.getJSONArray("cards").getJSONObject(0)
        assertEquals("Motion Sensor Quality", card.getString("title"))
        assertEquals("motion_sensor_quality", card.getString("graph_type"))
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
        assertTrue(names.contains("hy_memory_tool"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("top_apps"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_channel_rating"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_channel_utilization"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_ap_details"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_export"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_signal_advisor_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("wifi_channel_decision_packet_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("signal_awareness_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("agent_signal_evidence_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("soc_compatibility_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("agent_environment_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("agent_card_manifest_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_scan"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("bluetooth_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("sensor_analyzer_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("local_backend_runtime_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("non_adreno_backend_advisor_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("device_performance_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("gpu_backend_risk_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("local_inference_compatibility_report"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_status"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_graph"))
        assertTrue(result.getJSONArray("diagnostics_actions").toString().contains("radio_signal_advisor_report"))
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
        assertTrue(result.has("accelerator_preflight_matrix"))
        assertTrue(labels.contains("Selected on-device backend"))
        assertTrue(labels.contains("Current local backend state"))
        assertTrue(labels.contains("LiteRT-LM /health accelerator"))
        assertTrue(labels.contains("Model artifact compatibility"))
        assertTrue(labels.contains("Multimodal runtime policy"))
        assertTrue(labels.contains("MediaTek/non-Snapdragon fallback policy"))
        assertTrue(result.getJSONArray("cards").toString().contains("Runtime Backend Health"))
        assertTrue(result.getJSONArray("cards").toString().contains("Accelerator Preflight"))
        assertTrue(result.getJSONArray("cards").toString().contains("Thermal & Memory Guardrails"))
        assertTrue(result.getInt("runtime_backend_feature_count") >= 6)
        assertTrue(result.getInt("accelerator_preflight_count") >= 7)
        assertTrue(result.getInt("runtime_stability_feature_count") >= 6)
    }

    @Test
    fun acceleratorPreflightReportExposesAbiOpenClDelegateAndRuntimeProofRows() {
        val result = HermesDeviceDiagnosticsBridge.acceleratorPreflightReportJson(context)
        val rows = result.getJSONArray("accelerator_preflight_matrix")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val text = rows.toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("accelerator_preflight_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("local_backend_runtime_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("current_local_backend"))
        assertTrue(result.has("litert_runtime_health"))
        assertTrue(labels.contains("ABI and package lane"))
        assertTrue(labels.contains("Non-Adreno GPU policy"))
        assertTrue(labels.contains("OpenCL library visibility"))
        assertTrue(labels.contains("LiteRT backend attempt order"))
        assertTrue(labels.contains("Live /health accelerator proof"))
        assertTrue(labels.contains("Model artifact fit"))
        assertTrue(labels.contains("Thermal and memory startup runway"))
        assertTrue(text.contains("opencl_library_visible"))
        assertTrue(text.contains("opencl_probe_loads_library"))
        assertTrue(text.contains("non_adreno_policy_active"))
        assertTrue(text.contains("translated_arm64_on_x86"))
        assertTrue(result.getJSONArray("cards").toString().contains("Accelerator Preflight"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("accelerator_preflight_matrix"))
        assertTrue(result.getInt("accelerator_preflight_count") >= 7)
    }

    @Test
    fun socCompatibilityReportExposesBackendPolicyRoutesAndMediatekCoverageCards() {
        val result = HermesDeviceDiagnosticsBridge.socCompatibilityReportJson(context)
        val backend = result.getJSONArray("soc_backend_matrix")
        val routes = result.getJSONArray("soc_backend_policy_routes")
        val constraints = result.getJSONArray("soc_backend_constraint_matrix")
        val runtimeRows = result.getJSONArray("runtime_backend_matrix")
        val acceleratorRows = result.getJSONArray("accelerator_preflight_matrix")
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
        assertTrue(acceleratorRows.toString().contains("OpenCL library visibility"))
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
        assertTrue(result.getJSONArray("cards").toString().contains("Accelerator Preflight"))
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
        assertTrue(result.getInt("accelerator_preflight_count") >= 7)
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
    fun localInferenceCompatibilityReportFusesSocRuntimeRiskAndValidation() {
        val result = HermesDeviceDiagnosticsBridge.localInferenceCompatibilityReportJson(context)
        val rows = result.getJSONArray("local_inference_compatibility_matrix")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val text = rows.toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("local_inference_compatibility_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("soc_compatibility_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("gpu_backend_risk_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("current_local_backend"))
        assertTrue(result.has("litert_runtime_health"))
        assertTrue(result.has("gpu_backend_risk_matrix"))
        assertTrue(result.has("local_inference_compatibility_score"))
        assertTrue(result.has("local_inference_compatibility_level"))
        assertTrue(labels.contains("SOC and GPU family coverage"))
        assertTrue(labels.contains("Live accelerator acceptance"))
        assertTrue(labels.contains("Model artifact fit"))
        assertTrue(labels.contains("Thermal memory and power runway"))
        assertTrue(labels.contains("MediaTek and non-Adreno fallback policy"))
        assertTrue(labels.contains("Phone validation scope"))
        assertTrue(labels.contains("Agent drill-down route"))
        assertTrue(text.contains("local_backend_runtime_report"))
        assertTrue(text.contains("device_performance_report"))
        assertTrue(text.contains("gpu_backend_risk_report"))
        assertTrue(result.getJSONArray("cards").toString().contains("Local Inference Compatibility"))
        assertTrue(result.getJSONArray("cards").toString().contains("GPU Backend Risk"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("local_inference_compatibility_matrix"))
        assertTrue(result.getInt("local_inference_compatibility_count") >= 7)
        assertTrue(result.getInt("ready_local_inference_compatibility_count") >= 3)
    }

    @Test
    fun mediatekReadinessReportExposesDedicatedNonAdrenoReadinessProfile() {
        val result = HermesDeviceDiagnosticsBridge.mediatekReadinessReportJson(context)
        val rows = result.getJSONArray("mediatek_readiness_matrix")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("mediatek_readiness_report", result.getString("action"))
        assertTrue(sourceActions.contains("soc_compatibility_report"))
        assertTrue(sourceActions.contains("gpu_backend_risk_report"))
        assertTrue(sourceActions.contains("local_backend_runtime_report"))
        assertTrue(sourceActions.contains("device_performance_report"))
        assertTrue(sourceActions.contains("local_inference_compatibility_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("current_local_backend"))
        assertTrue(result.has("litert_runtime_health"))
        assertTrue(result.has("gpu_backend_risk_matrix"))
        assertTrue(result.has("runtime_backend_matrix"))
        assertTrue(result.has("runtime_stability_matrix"))
        assertTrue(result.has("local_inference_compatibility_matrix"))
        assertTrue(result.has("mediatek_readiness_score"))
        assertTrue(result.has("mediatek_readiness_level"))
        assertTrue(result.has("ready_mediatek_readiness_count"))
        assertTrue(labels.contains("MediaTek family detection"))
        assertTrue(labels.contains("Mali and Immortalis GPU path"))
        assertTrue(labels.contains("PowerVR/IMG fallback path"))
        assertTrue(labels.contains("SOC-aware artifact selection"))
        assertTrue(labels.contains("Runtime accelerator proof"))
        assertTrue(labels.contains("Thermal memory runway"))
        assertTrue(labels.contains("Physical ARM validation"))
        assertTrue(labels.contains("Safe fallback policy"))
        assertTrue(cards.contains("MediaTek Readiness"))
        assertTrue(cards.contains("mediatek_readiness_matrix"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("Dimensity"))
        assertTrue(result.getInt("mediatek_readiness_count") >= 8)
        assertTrue(result.getInt("ready_mediatek_readiness_count") >= 3)
    }

    @Test
    fun mediatekSignalStackReportFusesSocWirelessRadioSensorsAndClaimBoundaries() {
        val result = HermesDeviceDiagnosticsBridge.mediatekSignalStackReportJson(context)
        val stackRows = result.getJSONArray("mediatek_signal_stack_matrix")
        val routeRows = result.getJSONArray("mediatek_signal_refresh_routes")
        val boundaryRows = result.getJSONArray("mediatek_signal_claim_boundaries")
        val stackLabels = buildSet {
            for (index in 0 until stackRows.length()) add(stackRows.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routeRows.length()) add(routeRows.getJSONObject(index).getString("label"))
        }
        val boundaryLabels = buildSet {
            for (index in 0 until boundaryRows.length()) add(boundaryRows.getJSONObject(index).getString("label"))
        }
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val cards = result.getJSONArray("cards").toString()
        val graphTypes = result.getJSONArray("mediatek_signal_graph_types").toString()
        val directives = result.getJSONArray("gemma_mediatek_signal_directives").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("mediatek_signal_stack_report", result.getString("action"))
        assertTrue(sourceActions.contains("mediatek_readiness_report"))
        assertTrue(sourceActions.contains("soc_compatibility_report"))
        assertTrue(sourceActions.contains("signal_awareness_report"))
        assertTrue(sourceActions.contains("rf_coexistence_report"))
        assertTrue(sourceActions.contains("wifi_analyzer_report"))
        assertTrue(sourceActions.contains("bluetooth_analyzer_report"))
        assertTrue(sourceActions.contains("sensor_analyzer_report"))
        assertTrue(sourceActions.contains("radio_signal_status"))
        assertTrue(sourceActions.contains("local_inference_compatibility_report"))
        assertTrue(sourceActions.contains("gpu_backend_risk_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("mediatek_readiness_summary"))
        assertTrue(result.has("soc_compatibility_summary"))
        assertTrue(result.has("signal_awareness_summary"))
        assertTrue(result.has("rf_coexistence_summary"))
        assertTrue(result.has("wifi_analyzer_summary"))
        assertTrue(result.has("bluetooth_analyzer_summary"))
        assertTrue(result.has("sensor_analyzer_summary"))
        assertTrue(result.has("radio_signal_summary"))
        assertTrue(result.has("local_inference_summary"))
        assertTrue(result.has("backend_risk_summary"))
        assertTrue(result.has("gpu_backend_risk_level"))
        assertTrue(result.has("gpu_backend_risk_score"))
        assertTrue(stackLabels.contains("SOC and backend anchor"))
        assertTrue(stackLabels.contains("Wi-Fi analyzer evidence"))
        assertTrue(stackLabels.contains("Bluetooth nearby evidence"))
        assertTrue(stackLabels.contains("Radio and SDR boundary"))
        assertTrue(stackLabels.contains("Motion and IMU evidence"))
        assertTrue(stackLabels.contains("RF coexistence fusion"))
        assertTrue(stackLabels.contains("Local Gemma runtime overlay"))
        assertTrue(routeLabels.contains("Open fused MediaTek signal stack"))
        assertTrue(routeLabels.contains("Refresh Wi-Fi and Bluetooth evidence"))
        assertTrue(routeLabels.contains("Sample motion sensors"))
        assertTrue(routeLabels.contains("Attach radio bridge or SDR samples"))
        assertTrue(routeLabels.contains("Check runtime acceleration boundary"))
        assertTrue(boundaryLabels.contains("Backend policy is not a live signal"))
        assertTrue(boundaryLabels.contains("Cached signal rows need refresh for live claims"))
        assertTrue(boundaryLabels.contains("AM/FM and broad RF require bridge proof"))
        assertTrue(boundaryLabels.contains("GPU acceleration needs runtime proof"))
        assertTrue(boundaryLabels.contains("Physical MediaTek device proof remains separate"))
        assertTrue(cards.contains("MediaTek Signal Stack"))
        assertTrue(cards.contains("mediatek_signal_stack_matrix"))
        assertTrue(cards.contains("MediaTek Signal Routes"))
        assertTrue(cards.contains("mediatek_signal_refresh_routes"))
        assertTrue(cards.contains("MediaTek Claim Boundaries"))
        assertTrue(cards.contains("mediatek_signal_claim_boundaries"))
        assertTrue(graphTypes.contains("mediatek_signal_stack_matrix"))
        assertTrue(graphTypes.contains("mediatek_signal_refresh_routes"))
        assertTrue(graphTypes.contains("mediatek_signal_claim_boundaries"))
        assertTrue(graphTypes.contains("rf_coexistence_matrix"))
        assertTrue(graphTypes.contains("gpu_backend_risk_matrix"))
        assertTrue(graphTypes.contains("local_inference_compatibility_matrix"))
        assertTrue(directives.contains("mediatek_signal_claim_boundaries"))
        assertTrue(result.getInt("mediatek_signal_stack_count") >= 7)
        assertEquals(stackRows.length(), result.getInt("mediatek_signal_stack_count"))
        assertEquals(routeRows.length(), result.getInt("mediatek_signal_refresh_route_count"))
        assertEquals(boundaryRows.length(), result.getInt("mediatek_signal_claim_boundary_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "mediatek_signal_compatibility_report"))
        assertEquals("mediatek_signal_stack_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("mediatek_signal_stack_report"))
    }

    @Test
    fun mediatekDeviceValidationReportCombinesPhoneProofSignalsAndReleaseGates() {
        val result = HermesDeviceDiagnosticsBridge.mediatekDeviceValidationReportJson(context)
        val validationRows = result.getJSONArray("mediatek_device_validation_matrix")
        val routeRows = result.getJSONArray("live_signal_validation_routes")
        val gateRows = result.getJSONArray("release_device_proof_gates")
        val validationText = validationRows.toString()
        val routeText = routeRows.toString()
        val gateText = gateRows.toString()
        val validationLabels = buildSet {
            for (index in 0 until validationRows.length()) add(validationRows.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routeRows.length()) add(routeRows.getJSONObject(index).getString("label"))
        }
        val gateLabels = buildSet {
            for (index in 0 until gateRows.length()) add(gateRows.getJSONObject(index).getString("label"))
        }
        val cards = result.getJSONArray("cards").toString()
        val graphTypes = result.getJSONArray("mediatek_device_validation_graph_types").toString()
        val directives = result.getJSONArray("gemma_device_validation_directives").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("mediatek_device_validation_report", result.getString("action"))
        assertTrue(sourceActions.contains("mediatek_signal_stack_report"))
        assertTrue(sourceActions.contains("mediatek_backend_launch_checklist_report"))
        assertTrue(sourceActions.contains("agent_signal_proof_audit_report"))
        assertTrue(sourceActions.contains("agent_signal_permission_runbook_report"))
        assertTrue(sourceActions.contains("agent_release_validation_report"))
        assertTrue(sourceActions.contains("gpu_backend_risk_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("mediatek_signal_stack_summary"))
        assertTrue(result.has("signal_proof_audit_summary"))
        assertTrue(result.has("release_validation_summary"))
        assertTrue(validationLabels.contains("Physical MediaTek/non-Adreno identity"))
        assertTrue(validationLabels.contains("LiteRT/Gemma backend launch on phone"))
        assertTrue(validationLabels.contains("Live Wi-Fi analyzer proof"))
        assertTrue(validationLabels.contains("Live Bluetooth nearby proof"))
        assertTrue(validationLabels.contains("Motion sensor runtime proof"))
        assertTrue(validationLabels.contains("AM/FM or SDR receiver proof"))
        assertTrue(validationLabels.contains("Top-card observation proof"))
        assertTrue(validationLabels.contains("GitHub release APK proof"))
        assertTrue(validationLabels.contains("Claim boundary"))
        assertTrue(validationText.contains("physical_device_validation_required"))
        assertTrue(validationText.contains("claim_scope"))
        assertTrue(routeLabels.contains("Open physical device validation"))
        assertTrue(routeLabels.contains("Run live Wi-Fi proof"))
        assertTrue(routeLabels.contains("Run live Bluetooth proof"))
        assertTrue(routeLabels.contains("Sample live motion proof"))
        assertTrue(routeLabels.contains("Attach radio or SDR proof"))
        assertTrue(routeLabels.contains("Check backend and release proof"))
        assertTrue(gateLabels.contains("GitHub release workflow gate"))
        assertTrue(gateLabels.contains("Signed APK/AAB checksum gate"))
        assertTrue(gateLabels.contains("Physical phone backend gate"))
        assertTrue(gateLabels.contains("F-Droid metadata and Fastlane gate"))
        assertTrue(cards.contains("MediaTek Device Validation"))
        assertTrue(cards.contains("mediatek_device_validation_matrix"))
        assertTrue(cards.contains("Live Signal Validation Routes"))
        assertTrue(cards.contains("live_signal_validation_routes"))
        assertTrue(cards.contains("Release Device Proof Gates"))
        assertTrue(cards.contains("release_device_proof_gates"))
        assertTrue(graphTypes.contains("mediatek_device_validation_matrix"))
        assertTrue(graphTypes.contains("agent_signal_proof_audit_matrix"))
        assertTrue(graphTypes.contains("agent_release_artifact_gates"))
        assertTrue(directives.contains("physical_device_validation_required"))
        assertTrue(result.getInt("mediatek_device_validation_count") >= 9)
        assertTrue(result.getInt("live_signal_validation_route_count") >= 6)
        assertTrue(result.getInt("release_device_proof_gate_count") >= 4)
        assertEquals(validationRows.length(), result.getInt("mediatek_device_validation_count"))
        assertEquals(routeRows.length(), result.getInt("live_signal_validation_route_count"))
        assertEquals(gateRows.length(), result.getInt("release_device_proof_gate_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "physical_mediatek_validation_report"))
        assertEquals("mediatek_device_validation_report", dispatched.getString("action"))
        val nonAdrenoAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "non_adreno_device_validation_report"))
        assertEquals("mediatek_device_validation_report", nonAdrenoAlias.getString("action"))
        val phoneAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "phone_signal_validation_report"))
        assertEquals("mediatek_device_validation_report", phoneAlias.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("mediatek_device_validation_report"))
    }

    @Test
    fun deviceValidationEvidenceExportBundlesPhoneArtifactsReleaseRoutesAndClaimBoundaries() {
        val result = HermesDeviceDiagnosticsBridge.deviceValidationEvidenceExportReportJson(context)
        val manifest = result.getJSONArray("device_validation_evidence_manifest")
        val artifacts = result.getJSONArray("device_validation_required_artifacts")
        val phoneRoutes = result.getJSONArray("phone_validation_command_routes")
        val githubRoutes = result.getJSONArray("github_release_evidence_routes")
        val fdroidRoutes = result.getJSONArray("fdroid_evidence_routes")
        val artifactLabels = buildSet {
            for (index in 0 until artifacts.length()) add(artifacts.getJSONObject(index).getString("label"))
        }
        val manifestText = manifest.toString()
        val artifactText = artifacts.toString()
        val phoneRouteText = phoneRoutes.toString()
        val githubRouteText = githubRoutes.toString()
        val fdroidRouteText = fdroidRoutes.toString()
        val graphTypes = result.getJSONArray("device_validation_evidence_graph_types").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val directives = result.getJSONArray("gemma_device_validation_export_directives").toString()
        val bundle = result.getJSONObject("device_validation_evidence_export_bundle")

        assertTrue(result.getBoolean("success"))
        assertEquals("device_validation_evidence_export_report", result.getString("action"))
        assertTrue(sourceActions.contains("mediatek_device_validation_report"))
        assertTrue(sourceActions.contains("agent_signal_proof_audit_report"))
        assertTrue(sourceActions.contains("agent_release_validation_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("mediatek_device_validation_summary"))
        assertTrue(result.has("signal_replay_export_summary"))
        assertTrue(manifestText.contains("Portable device evidence schema"))
        assertTrue(manifestText.contains("Device identity boundary"))
        assertTrue(artifactLabels.contains("Physical phone identity capture"))
        assertTrue(artifactLabels.contains("APK/package/signing proof"))
        assertTrue(artifactLabels.contains("Wi-Fi live scan proof"))
        assertTrue(artifactLabels.contains("Bluetooth live scan proof"))
        assertTrue(artifactLabels.contains("Motion sensor sample proof"))
        assertTrue(artifactLabels.contains("AM/FM or SDR receiver sample proof"))
        assertTrue(artifactLabels.contains("LiteRT /health backend proof"))
        assertTrue(artifactLabels.contains("Top-card observation proof"))
        assertTrue(artifactLabels.contains("GitHub release asset checksum proof"))
        assertTrue(artifactLabels.contains("F-Droid metadata/Fastlane proof"))
        assertTrue(artifactText.contains("capture_command"))
        assertTrue(artifactText.contains("physical_device_validation_required"))
        assertTrue(artifactText.contains("release_validation_required"))
        assertTrue(phoneRouteText.contains("adb shell getprop"))
        assertTrue(phoneRouteText.contains("operator_capture_route"))
        assertTrue(githubRouteText.contains("agent_release_validation_report"))
        assertTrue(githubRouteText.contains("gh release view"))
        assertTrue(fdroidRouteText.contains("fdroid_release_metadata_matrix"))
        assertTrue(fdroidRouteText.contains("fastlane"))
        assertTrue(graphTypes.contains("device_validation_evidence_manifest"))
        assertTrue(graphTypes.contains("device_validation_required_artifacts"))
        assertTrue(graphTypes.contains("github_release_evidence_routes"))
        assertTrue(graphTypes.contains("fdroid_evidence_routes"))
        assertEquals("device_validation_evidence_export", bundle.getString("bundle_kind"))
        assertTrue(bundle.getJSONArray("sections").toString().contains("agent_signal_replay_export_report"))
        assertTrue(directives.contains("claim_scope"))
        assertEquals(manifest.length(), result.getInt("device_validation_evidence_manifest_count"))
        assertEquals(artifacts.length(), result.getInt("device_validation_required_artifact_count"))
        assertEquals(phoneRoutes.length(), result.getInt("phone_validation_command_route_count"))
        assertEquals(githubRoutes.length(), result.getInt("github_release_evidence_route_count"))
        assertEquals(fdroidRoutes.length(), result.getInt("fdroid_evidence_route_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "phone_validation_evidence_export"))
        assertEquals("device_validation_evidence_export_report", dispatched.getString("action"))
        val releaseAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "github_release_device_evidence_export"))
        assertEquals("device_validation_evidence_export_report", releaseAlias.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("device_validation_evidence_export_report"))
    }

    @Test
    fun agentSignalObservationPacketBundlesVisibleSignalRoutesFreshnessAndBoundaries() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalObservationPacketReportJson(context)
        val packetRows = result.getJSONArray("agent_signal_observation_packet")
        val visualRows = result.getJSONArray("agent_signal_observation_visual_slots")
        val routeRows = result.getJSONArray("agent_signal_observation_graph_routes")
        val boundaryRows = result.getJSONArray("agent_signal_observation_claim_boundaries")
        val packetText = packetRows.toString()
        val visualText = visualRows.toString()
        val routeText = routeRows.toString()
        val boundaryText = boundaryRows.toString()
        val graphTypes = result.getJSONArray("agent_signal_observation_packet_graph_types").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val directives = result.getJSONArray("gemma_signal_observation_packet_directives").toString()
        val bundle = result.getJSONObject("agent_signal_observation_packet_bundle")

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_observation_packet_report", result.getString("action"))
        assertTrue(sourceActions.contains("agent_signal_evidence_report"))
        assertTrue(sourceActions.contains("agent_signal_card_deck_report"))
        assertTrue(sourceActions.contains("agent_signal_replay_freshness_audit_report"))
        assertTrue(sourceActions.contains("device_validation_evidence_export_report"))
        assertTrue(result.has("signal_evidence_summary"))
        assertTrue(result.has("signal_replay_freshness_summary"))
        assertTrue(packetText.contains("Wi-Fi Analyzer observation packet"))
        assertTrue(packetText.contains("Bluetooth proximity observation packet"))
        assertTrue(packetText.contains("Motion and IMU observation packet"))
        assertTrue(packetText.contains("SDR radio observation packet"))
        assertTrue(packetText.contains("backend observation packet"))
        assertTrue(packetText.contains("observation_status"))
        assertTrue(packetText.contains("active_refresh_action"))
        assertTrue(packetText.contains("passive_fallback_action"))
        assertTrue(packetText.contains("visible_card_required"))
        assertTrue(visualText.contains("Wi-Fi channel and AP graph"))
        assertTrue(visualText.contains("Replay freshness and proof guard card"))
        assertTrue(routeText.contains("wifi_channel_graph"))
        assertTrue(routeText.contains("bluetooth_signal_history"))
        assertTrue(routeText.contains("motion_pose_estimate"))
        assertTrue(routeText.contains("radio_signal_graph"))
        assertTrue(routeText.contains("device_validation_evidence_manifest"))
        assertTrue(boundaryText.contains("Passive observation packet boundary"))
        assertTrue(boundaryText.contains("Active scan boundary"))
        assertTrue(boundaryText.contains("Physical phone and release proof boundary"))
        assertTrue(graphTypes.contains("agent_signal_observation_packet"))
        assertTrue(graphTypes.contains("agent_signal_observation_visual_slots"))
        assertTrue(graphTypes.contains("wifi_channel_graph"))
        assertTrue(graphTypes.contains("device_validation_evidence_manifest"))
        assertEquals("agent_signal_observation_packet", bundle.getString("bundle_kind"))
        assertTrue(bundle.getJSONArray("sections").toString().contains("agent_signal_evidence_report"))
        assertTrue(bundle.getJSONArray("wifi_analyzer_parity_keys").toString().contains("channel_rating"))
        assertTrue(bundle.getJSONArray("kai_parity_keys").toString().contains("interactive_top_cards"))
        assertTrue(directives.contains("claim_scope"))
        assertEquals(packetRows.length(), result.getInt("agent_signal_observation_packet_count"))
        assertEquals(visualRows.length(), result.getInt("agent_signal_observation_visual_slot_count"))
        assertEquals(routeRows.length(), result.getInt("agent_signal_observation_graph_route_count"))
        assertEquals(boundaryRows.length(), result.getInt("agent_signal_observation_claim_boundary_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "gemma_signal_observation_packet"))
        assertEquals("agent_signal_observation_packet_report", dispatched.getString("action"))
        val contextAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "signal_context_packet_report"))
        assertEquals("agent_signal_observation_packet_report", contextAlias.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_observation_packet_report"))
    }

    @Test
    fun nonAdrenoBackendAdvisorReportFusesLaunchSequenceAndRuntimeProofCards() {
        val result = HermesDeviceDiagnosticsBridge.nonAdrenoBackendAdvisorReportJson(context)
        val rows = result.getJSONArray("non_adreno_backend_advisor_matrix")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val launchSequence = result.getJSONArray("non_adreno_backend_launch_sequence").toString()
        val cards = result.getJSONArray("cards").toString()
        val directives = result.getJSONArray("gemma_non_adreno_backend_directives").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("non_adreno_backend_advisor_report", result.getString("action"))
        assertTrue(sourceActions.contains("soc_compatibility_report"))
        assertTrue(sourceActions.contains("accelerator_preflight_report"))
        assertTrue(sourceActions.contains("gpu_backend_risk_report"))
        assertTrue(launchSequence.contains("local_backend_runtime_report"))
        assertTrue(launchSequence.contains("local_inference_compatibility_report"))
        assertTrue(result.has("android_device_identity"))
        assertTrue(result.has("soc_profile"))
        assertTrue(result.has("device_performance_profile"))
        assertTrue(result.has("preferred_local_model"))
        assertTrue(result.has("current_local_backend"))
        assertTrue(result.has("litert_runtime_health"))
        assertTrue(result.has("accelerator_preflight_matrix"))
        assertTrue(result.has("runtime_backend_matrix"))
        assertTrue(result.has("runtime_stability_matrix"))
        assertTrue(result.has("gpu_backend_risk_matrix"))
        assertTrue(result.has("non_adreno_backend_advisor_score"))
        assertTrue(result.has("non_adreno_backend_advisor_level"))
        assertTrue(result.has("gpu_backend_risk_level"))
        assertTrue(result.has("gpu_backend_risk_score"))
        assertTrue(labels.contains("Classify device family before launch"))
        assertTrue(labels.contains("Choose artifact lane without Qualcomm bias"))
        assertTrue(labels.contains("Preflight delegate order"))
        assertTrue(labels.contains("Prove live accelerator state"))
        assertTrue(labels.contains("Keep CPU fallback explicit"))
        assertTrue(labels.contains("Gate launch by phone pressure"))
        assertTrue(labels.contains("Separate emulator proof from phone proof"))
        assertTrue(rows.toString().contains("accelerator_preflight_report"))
        assertTrue(rows.toString().contains("gpu_backend_risk_matrix"))
        assertTrue(cards.contains("Non-Adreno Backend Advisor"))
        assertTrue(cards.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(cards.contains("Accelerator Preflight"))
        assertTrue(cards.contains("Runtime Backend Health"))
        assertTrue(directives.contains("before starting local inference"))
        assertTrue(result.getInt("non_adreno_backend_advisor_count") >= 7)
        assertTrue(result.getInt("ready_non_adreno_backend_advisor_count") >= 3)

        val alias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "mediatek_backend_advisor_report"))
        assertEquals("non_adreno_backend_advisor_report", alias.getString("action"))
    }

    @Test
    fun mediatekBackendLaunchChecklistReportOrdersInferenceLaunchGates() {
        val result = HermesDeviceDiagnosticsBridge.mediatekBackendLaunchChecklistReportJson(context)
        val rows = result.getJSONArray("mediatek_backend_launch_checklist_matrix")
        val labels = buildList {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val rowText = rows.toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("mediatek_backend_launch_checklist_report", result.getString("action"))
        assertTrue(sourceActions.contains("mediatek_readiness_report"))
        assertTrue(sourceActions.contains("non_adreno_backend_advisor_report"))
        assertTrue(sourceActions.contains("accelerator_preflight_report"))
        assertTrue(sourceActions.contains("local_backend_runtime_report"))
        assertTrue(sourceActions.contains("device_performance_report"))
        assertEquals("Classify MediaTek or non-Adreno family", labels[0])
        assertTrue(labels.contains("Choose compatible LiteRT-LM artifact"))
        assertTrue(labels.contains("Run accelerator preflight"))
        assertTrue(labels.contains("Start local runtime before judging accelerator"))
        assertTrue(labels.contains("Verify GPU proof or name CPU fallback"))
        assertTrue(labels.contains("Apply thermal memory guardrails"))
        assertTrue(labels.contains("Separate emulator proof from phone proof"))
        assertTrue(labels.contains("Keep fallback and next action visible"))
        assertTrue(rowText.contains("launch_step"))
        assertTrue(rowText.contains("launch_gate_status"))
        assertTrue(rowText.contains("live_runtime_proof"))
        assertTrue(rowText.contains("cpu_fallback_explicit"))
        assertTrue(cards.contains("MediaTek Launch Checklist"))
        assertTrue(cards.contains("mediatek_backend_launch_checklist_matrix"))
        assertTrue(cards.contains("Non-Adreno Backend Advisor"))
        assertTrue(result.getJSONArray("gemma_mediatek_launch_directives").toString().contains("launch_step"))
        assertEquals(rows.length(), result.getInt("mediatek_backend_launch_checklist_count"))
        assertTrue(result.getInt("ready_mediatek_backend_launch_checklist_count") > 0)

        val alias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "mediatek_launch_checklist_report"))
        assertEquals("mediatek_backend_launch_checklist_report", alias.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("mediatek_backend_launch_checklist_report"))
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
        assertTrue(result.has("wifi_connection_status"))
        assertTrue(result.has("wifi_connection_link"))
        assertTrue(result.has("wifi_access_point_semantics"))
        assertTrue(result.has("wifi_band_coverage"))
        assertTrue(featureLabels.contains("Identify nearby access points"))
        assertTrue(featureLabels.contains("Current connection link telemetry"))
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
        assertTrue(routeLabels.contains("Route current connection link"))
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
        assertTrue(result.getJSONArray("cards").toString().contains("Wi-Fi Link"))
        assertTrue(result.getInt("wifi_analyzer_feature_count") >= 8)
        assertTrue(result.getInt("wifi_analyzer_workflow_route_count") >= 6)
        assertTrue(result.getInt("wifi_scan_policy_count") >= 5)
    }

    @Test
    fun wifiSignalAdvisorRowsBuildGemmaVisibleDecisionRows() {
        val networks = JSONArray()
            .put(
                JSONObject()
                    .put("ssid", "HermesMesh")
                    .put("display_ssid", "HermesMesh")
                    .put("bssid", "AA:BB:CC:DD:EE:01")
                    .put("rssi_dbm", -72)
                    .put("frequency_mhz", 2437)
                    .put("channel", 6)
                    .put("band", "2.4GHz")
                    .put("channel_width", "20MHz")
                    .put("channel_width_mhz", 20)
                    .put("security_mode", "WPA2")
                    .put("wifi_standard", "802.11n"),
            )
            .put(
                JSONObject()
                    .put("ssid", "HermesMesh")
                    .put("display_ssid", "HermesMesh")
                    .put("bssid", "AA:BB:CC:DD:EE:02")
                    .put("rssi_dbm", -50)
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("band", "5GHz")
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("security_mode", "WPA3")
                    .put("wifi_standard", "802.11ax"),
            )
            .put(
                JSONObject()
                    .put("ssid", "Cafe")
                    .put("display_ssid", "Cafe")
                    .put("bssid", "DA:A1:19:00:00:01")
                    .put("rssi_dbm", -46)
                    .put("frequency_mhz", 2412)
                    .put("channel", 1)
                    .put("band", "2.4GHz")
                    .put("channel_width", "20MHz")
                    .put("channel_width_mhz", 20)
                    .put("security_mode", "Open")
                    .put("wifi_standard", "802.11n"),
            )
        val analyzer = JSONObject()
            .put(
                "wifi_scan_permission_status",
                JSONObject()
                    .put("can_read_scan_results", true)
                    .put("fine_location_granted", true)
                    .put("nearby_wifi_devices_granted", true)
                    .put("location_enabled", true),
            )
            .put(
                "wifi_connection_status",
                JSONObject()
                    .put("connected", true)
                    .put("wifi_enabled", true)
                    .put("ssid", "HermesMesh")
                    .put("bssid", "AA:BB:CC:DD:EE:01")
                    .put("rssi_dbm", -72)
                    .put("band", "2.4GHz")
                    .put("channel", 6),
            )
            .put(
                "wifi_connection_link",
                JSONArray().put(
                    JSONObject()
                        .put("ready", true)
                        .put("label", "Current Wi-Fi association"),
                ),
            )
            .put(
                "recommended_wifi_channels",
                JSONArray().put(
                    JSONObject()
                        .put("band", "5GHz")
                        .put("channel", 36)
                        .put("score", 92)
                        .put("rating_label", "excellent")
                        .put("recommendation", "Best current option"),
                ),
            )
            .put(
                "wifi_channel_utilization",
                JSONArray().put(
                    JSONObject()
                        .put("band", "2.4GHz")
                        .put("channel", 6)
                        .put("channel_pressure_score", 78)
                        .put("utilization_label", "heavily_used")
                        .put("overlap_count", 4)
                        .put("strongest_rssi_dbm", -45),
                ),
            )
            .put("wifi_networks", networks)
        val coexistence = JSONObject()
            .put(
                "rf_coexistence_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("ready", true)
                        .put("label", "2.4GHz crowding"),
                ),
            )
            .put("rf_coexistence_risk_level", "moderate")
            .put("rf_coexistence_risk_score", 40)

        val rows = HermesDeviceDiagnosticsBridge.wifiSignalAdvisorRows(analyzer, coexistence)
        val roaming = HermesDeviceDiagnosticsBridge.wifiRoamingCandidateRows(analyzer)
        val text = rows.toString()

        assertEquals(6, rows.length())
        assertTrue(text.contains("Current link decision"))
        assertTrue(text.contains("Best channel recommendation"))
        assertTrue(text.contains("Congestion and overlap decision"))
        assertTrue(text.contains("Roaming candidate decision"))
        assertTrue(text.contains("Permission and refresh decision"))
        assertTrue(text.contains("2.4 GHz coexistence decision"))
        assertTrue(text.contains("wifi_signal_advisor"))
        assertTrue(text.contains("wifi_roaming_candidates"))
        assertTrue(roaming.length() >= 2)
        assertEquals("HermesMesh", roaming.getJSONObject(0).getString("ssid"))
        assertTrue(roaming.getJSONObject(0).getInt("roaming_score") >= roaming.getJSONObject(1).getInt("roaming_score"))
    }

    @Test
    fun wifiSignalAdvisorReportExposesDecisionCardsAndRoutes() {
        val result = HermesDeviceDiagnosticsBridge.wifiSignalAdvisorReportJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("wifi_signal_advisor_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_channel_rating"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("rf_coexistence_report"))
        assertTrue(result.getJSONArray("wifi_signal_advisor_matrix").length() >= 6)
        assertTrue(result.getJSONArray("wifi_signal_advisor_matrix").toString().contains("Permission and refresh decision"))
        assertTrue(result.getJSONArray("cards").toString().contains("Wi-Fi Advisor"))
        assertTrue(result.getJSONArray("cards").toString().contains("Wi-Fi Roaming Candidates"))
        assertTrue(result.getJSONArray("gemma_wifi_advisor_directives").toString().contains("wifi_signal_advisor_matrix"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "wifi_signal_advisor_report"))
        assertEquals("wifi_signal_advisor_report", dispatched.getString("action"))
    }

    @Test
    fun wifiChannelDecisionPacketReportExposesClaimSafeDecisionCards() {
        val result = HermesDeviceDiagnosticsBridge.wifiChannelDecisionPacketReportJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("wifi_channel_decision_packet_report", result.getString("action"))
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        assertTrue(sourceActions.contains("wifi_analyzer_report"))
        assertTrue(sourceActions.contains("wifi_signal_advisor_report"))
        assertTrue(sourceActions.contains("rf_coexistence_report"))
        assertTrue(sourceActions.contains("mediatek_signal_stack_report"))

        val packetText = result.getJSONArray("wifi_channel_decision_packet").toString()
        assertTrue(packetText.contains("Channel recommendation packet"))
        assertTrue(packetText.contains("decision_status"))
        assertTrue(packetText.contains("active_refresh_action"))
        assertTrue(packetText.contains("passive_fallback_action"))
        assertTrue(packetText.contains("claim_scope"))
        assertTrue(packetText.contains("mediatek_sensitive"))
        assertTrue(packetText.contains("rf_coexistence_sensitive"))
        assertTrue(result.getJSONArray("wifi_channel_decision_routes").toString().contains("wifi_channel_utilization"))
        assertTrue(result.getJSONArray("wifi_channel_decision_claim_boundaries").toString().contains("backend boundary"))
        assertTrue(result.getJSONArray("wifi_channel_decision_graph_types").toString().contains("wifi_channel_decision_packet"))
        assertTrue(result.getJSONArray("cards").toString().contains("Wi-Fi Channel Decision"))
        assertTrue(result.getJSONArray("gemma_wifi_channel_decision_directives").toString().contains("claim_boundaries"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "gemma_wifi_channel_decision"))
        assertEquals("wifi_channel_decision_packet_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("wifi_channel_decision_packet_report"))
    }

    @Test
    fun wifiConnectionLinkReportExposesCurrentAssociationRows() {
        val result = HermesDeviceDiagnosticsBridge.wifiConnectionLinkReportJson(context)
        val rows = result.getJSONArray("wifi_connection_link")
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("wifi_connection_link", result.getString("action"))
        assertTrue(result.has("wifi_scan_permission_status"))
        assertTrue(result.has("wifi_connection_status"))
        assertTrue(result.has("wifi_connected"))
        assertTrue(result.has("wifi_current_band"))
        assertTrue(result.has("matched_scan_row_available"))
        assertTrue(labels.contains("Current Wi-Fi association"))
        assertTrue(labels.contains("Link RSSI quality"))
        assertTrue(labels.contains("Link speed telemetry"))
        assertTrue(labels.contains("Connected channel and band"))
        assertTrue(labels.contains("Standard and security context"))
        assertTrue(labels.contains("Current AP scan match"))
        assertTrue(result.getJSONArray("cards").toString().contains("wifi_connection_link"))
        assertTrue(result.getInt("wifi_connection_link_count") >= 6)
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
    fun bluetoothAdvisorRanksCandidatesAndBuildsDecisionRowsForGemmaCards() {
        val analyzerReport = JSONObject()
            .put(
                "bluetooth_scan_permission_status",
                JSONObject()
                    .put("bluetooth_connect_granted", true)
                    .put("bluetooth_scan_granted", true)
                    .put("can_read_paired_devices", true)
                    .put("can_scan_nearby_devices", true),
            )
            .put("bluetooth_scan_status", JSONObject().put("returned_device_count", 2))
            .put("bluetooth_scan_control", JSONObject().put("scan_mode", "paused"))
            .put("bluetooth_service_label_count", 1)
            .put("bluetooth_manufacturer_name_count", 1)
            .put(
                "bluetooth_devices",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("device_name", "Heart Strap")
                            .put("advertised_name", "Hermes Heart")
                            .put("address", "AA:BB:CC:00:11:22")
                            .put("device_type", "le")
                            .put("bond_state", "bonded")
                            .put("paired", true)
                            .put("connectable", true)
                            .put("device_category", "wearable_health")
                            .put("rssi_dbm", -47)
                            .put("proximity_label", "near")
                            .put("service_labels", JSONArray().put("Heart Rate"))
                            .put("manufacturer_names", JSONArray().put("Apple"))
                            .put("semantic_context", "services=Heart Rate | manufacturers=Apple"),
                    )
                    .put(
                        JSONObject()
                            .put("device_name", "Unknown Beacon")
                            .put("device_type", "le")
                            .put("rssi_dbm", -88),
                    ),
            )
            .put(
                "bluetooth_metadata_summary",
                JSONArray().put(
                    JSONObject()
                        .put("summary_type", "service_uuid")
                        .put("label", "0000180d-0000-1000-8000-00805f9b34fb")
                        .put("semantic_label", "Heart Rate")
                        .put("count", 1),
                ),
            )
            .put(
                "bluetooth_signal_history",
                JSONArray().put(
                    JSONObject()
                        .put("device_name", "Hermes Heart")
                        .put("current_rssi_dbm", -49)
                        .put("sample_count", 3)
                        .put("trend_label", "approaching")
                        .put("last_seen_ms", 1200),
                ),
            )
        val coexistenceReport = JSONObject()
            .put("rf_coexistence_risk_level", "moderate")
            .put("rf_coexistence_risk_score", 42)
            .put(
                "rf_coexistence_matrix",
                JSONArray().put(JSONObject().put("label", "Bluetooth 2.4 GHz").put("ready", true)),
            )

        val candidates = HermesDeviceDiagnosticsBridge.bluetoothDeviceCandidateRows(analyzerReport)
        val advisorRows = HermesDeviceDiagnosticsBridge.bluetoothSignalAdvisorRows(analyzerReport, coexistenceReport)
        val advisorLabels = buildSet {
            for (index in 0 until advisorRows.length()) add(advisorRows.getJSONObject(index).getString("label"))
        }

        assertEquals("Hermes Heart", candidates.getJSONObject(0).getString("label"))
        assertTrue(candidates.getJSONObject(0).getInt("candidate_score") > candidates.getJSONObject(1).getInt("candidate_score"))
        assertEquals("Heart Rate", candidates.getJSONObject(0).getJSONArray("service_labels").getString(0))
        assertTrue(advisorLabels.contains("Nearby device decision"))
        assertTrue(advisorLabels.contains("Proximity and trend decision"))
        assertTrue(advisorLabels.contains("Service and manufacturer decision"))
        assertTrue(advisorLabels.contains("Permission and refresh decision"))
        assertTrue(advisorLabels.contains("Device detail/export decision"))
        assertTrue(advisorLabels.contains("2.4 GHz coexistence decision"))
    }

    @Test
    fun bluetoothSignalAdvisorReportRoutesPassiveCardsAndSourceActions() {
        val result = HermesDeviceDiagnosticsBridge.bluetoothSignalAdvisorReportJson(
            context,
            JSONObject().put("refresh", true),
        )

        assertTrue(result.getBoolean("success"))
        assertEquals("bluetooth_signal_advisor_report", result.getString("action"))
        assertTrue(result.has("bluetooth_signal_advisor_matrix"))
        assertTrue(result.has("bluetooth_device_candidates"))
        assertTrue(result.has("gemma_bluetooth_advisor_directives"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_signal_history"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_device_details"))
        assertTrue(result.getJSONArray("cards").toString().contains("Bluetooth Advisor"))
        assertTrue(result.getJSONArray("cards").toString().contains("Bluetooth Device Candidates"))
        assertEquals(
            result.getJSONArray("bluetooth_signal_advisor_matrix").length(),
            result.getInt("bluetooth_signal_advisor_count"),
        )
    }

    @Test
    fun bluetoothNearbyDecisionPacketReportExposesClaimSafeDecisionCards() {
        val result = HermesDeviceDiagnosticsBridge.bluetoothNearbyDecisionPacketReportJson(context)

        assertTrue(result.getBoolean("success"))
        assertEquals("bluetooth_nearby_decision_packet_report", result.getString("action"))
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        assertTrue(sourceActions.contains("bluetooth_analyzer_report"))
        assertTrue(sourceActions.contains("bluetooth_signal_advisor_report"))
        assertTrue(sourceActions.contains("rf_coexistence_report"))
        assertTrue(sourceActions.contains("mediatek_signal_stack_report"))

        val packetText = result.getJSONArray("bluetooth_nearby_decision_packet").toString()
        assertTrue(packetText.contains("Nearby Bluetooth candidate packet"))
        assertTrue(packetText.contains("decision_status"))
        assertTrue(packetText.contains("active_refresh_action"))
        assertTrue(packetText.contains("passive_fallback_action"))
        assertTrue(packetText.contains("claim_scope"))
        assertTrue(packetText.contains("mediatek_sensitive"))
        assertTrue(packetText.contains("rf_coexistence_sensitive"))
        assertTrue(result.getJSONArray("bluetooth_nearby_decision_routes").toString().contains("bluetooth_scan"))
        assertTrue(result.getJSONArray("bluetooth_nearby_decision_routes").toString().contains("mediatek_signal_stack_report"))
        assertTrue(result.getJSONArray("bluetooth_nearby_claim_boundaries").toString().contains("backend boundary"))
        assertTrue(result.getJSONArray("bluetooth_nearby_decision_graph_types").toString().contains("bluetooth_nearby_decision_packet"))
        assertTrue(result.getJSONArray("cards").toString().contains("Bluetooth Nearby Decision"))
        assertTrue(result.getJSONArray("gemma_bluetooth_nearby_directives").toString().contains("claim_boundaries"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "mediatek_bluetooth_decision_packet"))
        assertEquals("bluetooth_nearby_decision_packet_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("bluetooth_nearby_decision_packet_report"))
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
    fun filtersBluetoothRowsForAgentCardsByServiceManufacturerProximityAndRssi() {
        val devices = JSONArray()
            .put(
                JSONObject()
                    .put("device_name", "Heart Strap")
                    .put("advertised_name", "Hermes Heart")
                    .put("address", "AA:BB:CC:00:11:22")
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("rssi_dbm", -48)
                    .put("proximity_label", "near")
                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                    .put("service_labels", JSONArray().put("Heart Rate"))
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("manufacturer_names", JSONArray().put("Apple"))
                    .put("semantic_context", "services=Heart Rate | manufacturers=Apple"),
            )
            .put(
                JSONObject()
                    .put("device_name", "Speaker")
                    .put("address", "AA:BB:CC:00:11:33")
                    .put("device_type", "classic")
                    .put("device_category", "audio")
                    .put("rssi_dbm", -77)
                    .put("proximity_label", "far")
                    .put("service_labels", JSONArray().put("Audio Sink"))
                    .put("manufacturer_names", JSONArray().put("Sony")),
            )

        val filtered = HermesDeviceDiagnosticsBridge.bluetoothFilteredDeviceRows(
            devices,
            JSONObject()
                .put("filter_bluetooth_service", "Heart Rate")
                .put("filter_bluetooth_manufacturer", "Apple")
                .put("filter_bluetooth_proximity", "near")
                .put("filter_bluetooth_category", "wearable")
                .put("min_rssi_dbm", -60),
        )

        assertEquals(1, filtered.length())
        assertEquals("Heart Strap", filtered.getJSONObject(0).getString("device_name"))
        assertEquals("near", filtered.getJSONObject(0).getString("proximity_label"))
    }

    @Test
    fun bluetoothDeviceDetailsExpandMetadataForInspectionAndExport() {
        val devices = JSONArray()
            .put(
                JSONObject()
                    .put("device_name", "Heart Strap")
                    .put("advertised_name", "Hermes Heart")
                    .put("address", "AA:BB:CC:00:11:22")
                    .put("device_type", "le")
                    .put("bond_state", "bonded")
                    .put("device_category", "wearable_health")
                    .put("major_device_class", "wearable")
                    .put("rssi_dbm", -48)
                    .put("proximity_label", "near")
                    .put("estimated_distance_meters", 1.4)
                    .put("tx_power_dbm", -8)
                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                    .put("service_labels", JSONArray().put("Heart Rate"))
                    .put("service_data_uuids", JSONArray().put("0000180f-0000-1000-8000-00805f9b34fb"))
                    .put("service_data_labels", JSONArray().put("Battery Service"))
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("manufacturer_names", JSONArray().put("Apple"))
                    .put("manufacturer_data_count", 1)
                    .put("manufacturer_data_bytes", 12)
                    .put("scan_record_bytes", 48),
            )

        val detailRows = HermesDeviceDiagnosticsBridge.bluetoothDeviceDetailRows(devices)
        val detail = detailRows.getJSONObject(0)
        val result = HermesDeviceDiagnosticsBridge.bluetoothDeviceDetailsJson(
            context,
            JSONObject()
                .put("action", "bluetooth_export")
                .put("export_format", "both")
                .put("bluetooth_devices", devices),
        )

        assertEquals(1, detailRows.length())
        assertEquals("Hermes Heart", detail.getString("display_label"))
        assertEquals("health or fitness device", detail.getString("semantic_label"))
        assertEquals("Heart Rate", detail.getJSONArray("service_labels").getString(0))
        assertEquals("Apple", detail.getJSONArray("manufacturer_names").getString(0))
        assertTrue(detail.getString("evidence_summary").contains("Heart Rate"))
        assertTrue(detail.getInt("metadata_completeness_score") >= 80)
        assertEquals("bluetooth_export", result.getString("action"))
        assertEquals(1, result.getInt("bluetooth_device_detail_count"))
        assertEquals(1, result.getInt("bluetooth_filtered_device_count"))
        assertEquals("both", result.getJSONObject("bluetooth_device_export").getString("format"))
        assertTrue(result.getJSONObject("bluetooth_device_export").getJSONArray("included_fields").toString().contains("metadata_completeness_score"))
        assertTrue(result.getString("bluetooth_device_export_csv").contains("Hermes Heart"))
        assertTrue(result.getJSONArray("cards").toString().contains("bluetooth_device_detail"))
        assertTrue(result.getJSONObject("bluetooth_scan_status").has("returned_device_count"))
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
    fun agentSignalEvidenceReportBundlesCurrentEvidenceForGemma() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalEvidenceReportJson(context)
        val evidence = result.getJSONArray("signal_evidence_matrix")
        val routes = result.getJSONArray("signal_evidence_routes")
        val graphTypes = result.getJSONArray("signal_evidence_graph_types").toString()
        val evidenceText = evidence.toString()
        val routeText = routes.toString()
        val labels = buildSet {
            for (index in 0 until evidence.length()) add(evidence.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_evidence_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_channel_graph"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_signal_history"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_device_details"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_export"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("local_inference_compatibility_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONObject("wifi_evidence_summary").getJSONArray("card_titles").toString().contains("Wi-Fi"))
        assertTrue(result.getJSONObject("bluetooth_evidence_summary").getJSONArray("card_titles").toString().contains("Bluetooth"))
        assertTrue(result.getJSONObject("local_inference_evidence_summary").getJSONArray("card_titles").toString().contains("Local Inference Compatibility"))
        assertTrue(result.getJSONObject("accelerator_preflight_evidence_summary").getJSONArray("card_titles").toString().contains("Accelerator Preflight"))
        assertTrue(labels.contains("Current signal evidence bundle"))
        assertTrue(labels.contains("Wi-Fi AP and channel evidence"))
        assertTrue(labels.contains("Bluetooth proximity evidence"))
        assertTrue(labels.contains("Motion and sensor evidence"))
        assertTrue(labels.contains("AM/FM and RF boundary evidence"))
        assertTrue(labels.contains("Local inference readiness evidence"))
        assertTrue(labels.contains("Permission and refresh evidence"))
        assertTrue(labels.contains("Evidence drill-down route"))
        assertTrue(evidenceText.contains("source_actions"))
        assertTrue(evidenceText.contains("card_graph_types"))
        assertTrue(evidenceText.contains("wifi_channel_graph"))
        assertTrue(evidenceText.contains("bluetooth_device_detail"))
        assertTrue(evidenceText.contains("bluetooth_device_details"))
        assertTrue(evidenceText.contains("bluetooth_export"))
        assertTrue(evidenceText.contains("bluetooth_signal_history"))
        assertTrue(evidenceText.contains("motion_pose_estimate"))
        assertTrue(evidenceText.contains("radio_signal_graph"))
        assertTrue(evidenceText.contains("radio_receiver_bridge_schema"))
        assertTrue(evidenceText.contains("local_inference_compatibility_matrix"))
        assertTrue(evidenceText.contains("accelerator_preflight_matrix"))
        assertTrue(evidenceText.contains("accelerator_preflight_report"))
        assertTrue(routeText.contains("Open Wi-Fi graph evidence"))
        assertTrue(routeText.contains("Open Bluetooth proximity evidence"))
        assertTrue(routeText.contains("Open Bluetooth device detail evidence"))
        assertTrue(routeText.contains("Open motion pose evidence"))
        assertTrue(routeText.contains("Open radio boundary evidence"))
        assertTrue(routeText.contains("Open local inference evidence"))
        assertTrue(routeText.contains("Open card manifest evidence"))
        assertTrue(graphTypes.contains("signal_evidence_matrix"))
        assertTrue(graphTypes.contains("wifi_channel_graph"))
        assertTrue(graphTypes.contains("bluetooth_device_detail"))
        assertTrue(graphTypes.contains("radio_receiver_bridge_schema"))
        assertTrue(graphTypes.contains("local_inference_compatibility_matrix"))
        assertTrue(graphTypes.contains("accelerator_preflight_matrix"))
        assertTrue(result.getJSONArray("cards").toString().contains("Signal Evidence Bundle"))
        assertTrue(result.getJSONArray("cards").toString().contains("Signal Evidence Routes"))
        assertTrue(result.getJSONArray("cards").toString().contains("Accelerator Preflight"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("signal_evidence_matrix"))
        assertTrue(result.getInt("signal_evidence_count") >= 9)
        assertTrue(result.getInt("signal_evidence_route_count") >= 8)
        assertTrue(result.getInt("signal_evidence_graph_type_count") >= 12)
    }

    @Test
    fun agentSignalBriefingReportBuildsTopCardSlotsAndMetadataKeys() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalBriefingReportJson(context)
        val briefing = result.getJSONArray("agent_signal_briefing_matrix")
        val slots = result.getJSONArray("agent_top_card_slots")
        val metadata = result.getJSONArray("agent_signal_metadata_keys")
        val briefingText = briefing.toString()
        val slotText = slots.toString()
        val metadataText = metadata.toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_briefing_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_channel_graph"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_device_details"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("motion_sensor_quality"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("radio_signal_graph"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("mediatek_readiness_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(briefingText.contains("Wi-Fi graph evidence"))
        assertTrue(briefingText.contains("Bluetooth metadata evidence"))
        assertTrue(briefingText.contains("Motion and sensor evidence"))
        assertTrue(briefingText.contains("Radio boundary and bridge evidence"))
        assertTrue(briefingText.contains("MediaTek and backend evidence"))
        assertTrue(briefingText.contains("accelerator_preflight_matrix"))
        assertTrue(slotText.contains("open_next_action"))
        assertTrue(slotText.contains("refresh_policy"))
        assertTrue(slotText.contains("permission_gate"))
        assertTrue(metadataText.contains("wifi_channel_graph"))
        assertTrue(metadataText.contains("bluetooth_device_details"))
        assertTrue(metadataText.contains("motion_sensor_quality"))
        assertTrue(metadataText.contains("radio_signal_graph"))
        assertTrue(metadataText.contains("accelerator_preflight_report"))
        assertTrue(metadataText.contains("gpu_backend_risk_matrix"))
        assertTrue(metadataText.contains("accelerator_preflight_matrix"))
        assertTrue(result.getJSONArray("gemma_signal_briefing_directives").toString().contains("agent_top_card_slots"))
        assertTrue(cards.contains("Agent Signal Briefing"))
        assertTrue(cards.contains("agent_signal_briefing_matrix"))
        assertTrue(cards.contains("Top Card Slots"))
        assertTrue(cards.contains("agent_top_card_slots"))
        assertTrue(cards.contains("Gemma Metadata Keys"))
        assertTrue(cards.contains("agent_signal_metadata_keys"))
        assertTrue(result.getInt("agent_signal_briefing_count") >= 7)
        assertTrue(result.getInt("agent_top_card_slot_count") >= 6)
        assertTrue(result.getInt("agent_signal_metadata_key_count") >= 6)
    }

    @Test
    fun agentSignalCardDeckReportPreloadsExpandedTopCardsForGemma() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalCardDeckReportJson(context)
        val manifest = result.getJSONArray("agent_signal_card_deck_manifest")
        val manifestText = manifest.toString()
        val cards = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_card_deck_report", result.getString("action"))
        assertTrue(sourceActions.contains("wifi_analyzer_report"))
        assertTrue(sourceActions.contains("bluetooth_signal_advisor_report"))
        assertTrue(sourceActions.contains("radio_signal_advisor_report"))
        assertTrue(sourceActions.contains("sensor_workflow_advisor_report"))
        assertTrue(sourceActions.contains("non_adreno_backend_advisor_report"))
        assertTrue(sourceActions.contains("agent_release_validation_report"))
        assertTrue(manifestText.contains("Wi-Fi Channel Graph"))
        assertTrue(manifestText.contains("Wi-Fi Channel Ratings"))
        assertTrue(manifestText.contains("Bluetooth Nearby Advisor"))
        assertTrue(manifestText.contains("Bluetooth Device Candidates"))
        assertTrue(manifestText.contains("Radio Signal Advisor"))
        assertTrue(manifestText.contains("Motion Sensor Workflow"))
        assertTrue(manifestText.contains("Non-Adreno Backend Advisor"))
        assertTrue(manifestText.contains("Release Validation"))
        assertTrue(manifestText.contains("top_card_slot"))
        assertTrue(manifestText.contains("open_next_action"))
        assertTrue(manifestText.contains("refresh_policy"))
        assertTrue(manifestText.contains("permission_gate"))
        assertTrue(manifestText.contains("wifi_channel_graph"))
        assertTrue(manifestText.contains("wifi_channel_rating"))
        assertTrue(manifestText.contains("bluetooth_signal_advisor_matrix"))
        assertTrue(manifestText.contains("bluetooth_device_candidates"))
        assertTrue(manifestText.contains("radio_signal_advisor_matrix"))
        assertTrue(manifestText.contains("sensor_workflow_advisor_matrix"))
        assertTrue(manifestText.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(manifestText.contains("agent_release_validation_matrix"))
        assertTrue(cards.contains("Expanded Signal Card Deck"))
        assertTrue(cards.contains("agent_signal_card_deck_manifest"))
        assertTrue(cards.contains("Wi-Fi Channel Graph"))
        assertTrue(cards.contains("Wi-Fi Channel Ratings"))
        assertTrue(cards.contains("Bluetooth Nearby Advisor"))
        assertTrue(cards.contains("Bluetooth Device Candidates"))
        assertTrue(cards.contains("Radio Signal Advisor"))
        assertTrue(cards.contains("Motion Sensor Workflow"))
        assertTrue(cards.contains("Non-Adreno Backend Advisor"))
        assertTrue(cards.contains("Release Validation"))
        assertTrue(result.getJSONArray("gemma_signal_card_deck_directives").toString().contains("agent_signal_card_deck_manifest"))
        assertTrue(result.getInt("agent_signal_card_deck_count") >= 8)

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "agent_signal_card_deck_report"))
        assertEquals("agent_signal_card_deck_report", dispatched.getString("action"))
        val alias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "expanded_signal_cards"))
        assertEquals("agent_signal_card_deck_report", alias.getString("action"))
    }

    @Test
    fun agentSignalCardRefreshPlanMapsExpandedCardsToLiveAndPassiveRoutes() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalCardRefreshPlanReportJson(context)
        val rows = result.getJSONArray("agent_signal_card_refresh_plan_matrix")
        val rowText = rows.toString()
        val cards = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_card_refresh_plan_report", result.getString("action"))
        assertTrue(sourceActions.contains("agent_signal_card_deck_report"))
        assertTrue(sourceActions.contains("agent_signal_permission_runbook_report"))
        assertTrue(sourceActions.contains("agent_signal_workflow_handoff_report"))
        assertTrue(sourceActions.contains("wifi_scan"))
        assertTrue(sourceActions.contains("bluetooth_scan"))
        assertTrue(sourceActions.contains("radio_signal_graph"))
        assertTrue(sourceActions.contains("accelerator_preflight_report"))
        assertTrue(rowText.contains("Wi-Fi Channel Graph"))
        assertTrue(rowText.contains("Wi-Fi Channel Ratings"))
        assertTrue(rowText.contains("Bluetooth Nearby Advisor"))
        assertTrue(rowText.contains("Radio Signal Advisor"))
        assertTrue(rowText.contains("Motion Sensor Workflow"))
        assertTrue(rowText.contains("Non-Adreno Backend Advisor"))
        assertTrue(rowText.contains("Release Validation"))
        assertTrue(rowText.contains("active_refresh_arguments"))
        assertTrue(rowText.contains("passive_fallback_action"))
        assertTrue(rowText.contains("settings_actions"))
        assertTrue(rowText.contains("card_refresh_status"))
        assertTrue(rowText.contains("radio_bridge_samples_json"))
        assertTrue(rowText.contains("physical_device_validation_required"))
        assertTrue(rowText.contains("wifi_scan"))
        assertTrue(rowText.contains("bluetooth_scan"))
        assertTrue(rowText.contains("motion_sensor_history"))
        assertTrue(cards.contains("Signal Card Refresh Plan"))
        assertTrue(cards.contains("agent_signal_card_refresh_plan_matrix"))
        assertTrue(result.getJSONArray("gemma_signal_card_refresh_plan_directives").toString().contains("active_refresh_arguments"))
        assertEquals(rows.length(), result.getInt("agent_signal_card_refresh_plan_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "top_card_refresh_plan"))
        assertEquals("agent_signal_card_refresh_plan_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_card_refresh_plan_report"))
    }

    @Test
    fun agentSignalCardRefreshStatusSummarizesReadyBlockedAndBridgeHints() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalCardRefreshStatusReportJson(context)
        val rows = result.getJSONArray("agent_signal_card_refresh_status_matrix")
        val rowText = rows.toString()
        val cards = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val radioRow = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("label") == "Radio Signal Advisor" }
        val releaseRow = (0 until rows.length())
            .map { rows.getJSONObject(it) }
            .first { it.getString("label") == "Release Validation" }

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_card_refresh_status_report", result.getString("action"))
        assertTrue(sourceActions.contains("agent_signal_card_refresh_plan_report"))
        assertTrue(sourceActions.contains("open_wifi_settings"))
        assertTrue(sourceActions.contains("radio_signal_graph"))
        assertTrue(rowText.contains("Wi-Fi Channel Graph"))
        assertTrue(rowText.contains("Bluetooth Nearby Advisor"))
        assertTrue(rowText.contains("status_label"))
        assertTrue(rowText.contains("status_hint"))
        assertTrue(rowText.contains("next_best_action"))
        assertTrue(rowText.contains("ready_for_active_refresh"))
        assertTrue(rowText.contains("passive_refresh_available"))
        assertTrue(rowText.contains("open_settings_action"))
        assertEquals("bridge_required", radioRow.getString("status_label"))
        assertEquals("radio_signal_advisor_report", radioRow.getString("passive_source_action"))
        assertTrue(radioRow.getString("status_hint").contains("bridge samples"))
        assertEquals("ready", releaseRow.getString("status_label"))
        assertTrue(releaseRow.getBoolean("ready_for_active_refresh"))
        assertTrue(cards.contains("Signal Card Refresh Status"))
        assertTrue(cards.contains("agent_signal_card_refresh_status_matrix"))
        assertTrue(result.getJSONArray("gemma_signal_card_refresh_status_directives").toString().contains("status_label"))
        assertEquals(rows.length(), result.getInt("agent_signal_card_refresh_status_count"))
        assertTrue(result.getInt("ready_agent_signal_card_refresh_status_count") > 0)

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "top_card_refresh_status"))
        assertEquals("agent_signal_card_refresh_status_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_card_refresh_status_report"))
    }

    @Test
    fun agentSignalSessionSnapshotReportFusesRfCardsRefreshAndBackendGates() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalSessionSnapshotReportJson(context)
        val snapshot = result.getJSONArray("agent_signal_session_snapshot_matrix")
        val domains = result.getJSONArray("agent_signal_session_domain_matrix")
        val routes = result.getJSONArray("agent_signal_session_action_routes")
        val cards = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val snapshotText = snapshot.toString()
        val domainText = domains.toString()
        val routeText = routes.toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_session_snapshot_report", result.getString("action"))
        assertTrue(sourceActions.contains("rf_coexistence_report"))
        assertTrue(sourceActions.contains("agent_signal_card_deck_report"))
        assertTrue(sourceActions.contains("agent_signal_card_refresh_status_report"))
        assertTrue(sourceActions.contains("mediatek_backend_launch_checklist_report"))
        assertTrue(sourceActions.contains("accelerator_preflight_report"))
        assertTrue(snapshot.length() >= 7)
        assertTrue(domains.length() >= 7)
        assertTrue(routes.length() >= 6)
        assertTrue(snapshotText.contains("Session signal posture"))
        assertTrue(snapshotText.contains("Top-card session deck"))
        assertTrue(snapshotText.contains("RF coexistence session risk"))
        assertTrue(snapshotText.contains("MediaTek launch session gate"))
        assertTrue(snapshotText.contains("Live refresh session readiness"))
        assertTrue(domainText.contains("Wi-Fi graph domain"))
        assertTrue(domainText.contains("MediaTek backend launch domain"))
        assertTrue(routeText.contains("agent_signal_card_deck_report"))
        assertTrue(routeText.contains("rf_coexistence_report"))
        assertTrue(routeText.contains("mediatek_backend_launch_checklist_report"))
        assertTrue(result.has("gemma_signal_session_snapshot_directives"))
        assertTrue(result.has("agent_signal_session_status"))
        assertTrue(cards.contains("Agent Signal Session Snapshot"))
        assertTrue(cards.contains("agent_signal_session_snapshot_matrix"))
        assertTrue(cards.contains("Session Domain Coverage"))
        assertTrue(cards.contains("agent_signal_session_domain_matrix"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "signal_session_snapshot"))
        assertEquals("agent_signal_session_snapshot_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_session_snapshot_report"))
    }

    @Test
    fun agentSignalProofAuditSeparatesActivePassiveBridgePhoneAndReleaseProof() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalProofAuditReportJson(context)
        val rows = result.getJSONArray("agent_signal_proof_audit_matrix")
        val boundaries = result.getJSONArray("agent_signal_claim_boundary_matrix")
        val cards = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val rowText = rows.toString()
        val boundaryText = boundaries.toString()
        val rowLabels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }
        val boundaryLabels = buildSet {
            for (index in 0 until boundaries.length()) add(boundaries.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_proof_audit_report", result.getString("action"))
        assertTrue(sourceActions.contains("agent_signal_session_snapshot_report"))
        assertTrue(sourceActions.contains("agent_signal_card_refresh_status_report"))
        assertTrue(sourceActions.contains("agent_signal_permission_runbook_report"))
        assertTrue(sourceActions.contains("wifi_analyzer_report"))
        assertTrue(sourceActions.contains("bluetooth_analyzer_report"))
        assertTrue(sourceActions.contains("radio_signal_status"))
        assertTrue(sourceActions.contains("accelerator_preflight_report"))
        assertTrue(rows.length() >= 8)
        assertTrue(boundaries.length() >= 5)
        assertTrue(rowLabels.contains("Session snapshot proof"))
        assertTrue(rowLabels.contains("Wi-Fi active proof"))
        assertTrue(rowLabels.contains("Bluetooth active proof"))
        assertTrue(rowLabels.contains("Motion sensor proof"))
        assertTrue(rowLabels.contains("AM/FM or SDR bridge proof"))
        assertTrue(rowLabels.contains("MediaTek backend proof"))
        assertTrue(rowLabels.contains("Top-card refresh proof"))
        assertTrue(rowLabels.contains("Release proof boundary"))
        assertTrue(rowText.contains("proof_status"))
        assertTrue(rowText.contains("claim_scope"))
        assertTrue(rowText.contains("passive_fallback_action"))
        assertTrue(rowText.contains("active_evidence_present"))
        assertTrue(rowText.contains("passive_evidence_present"))
        assertTrue(rowText.contains("bridge_required"))
        assertTrue(rowText.contains("physical_device_validation_required"))
        assertTrue(rowText.contains("release_validation_required"))
        assertTrue(boundaryLabels.contains("Active evidence boundary"))
        assertTrue(boundaryText.contains("claim_boundary"))
        assertTrue(cards.contains("Signal Proof Audit"))
        assertTrue(cards.contains("agent_signal_proof_audit_matrix"))
        assertTrue(cards.contains("Signal Claim Boundaries"))
        assertTrue(cards.contains("agent_signal_claim_boundary_matrix"))
        assertTrue(result.getJSONArray("gemma_signal_proof_audit_directives").toString().contains("active_evidence_present"))
        assertEquals(rows.length(), result.getInt("agent_signal_proof_audit_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "signal_proof_audit"))
        assertEquals("agent_signal_proof_audit_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_proof_audit_report"))
    }

    @Test
    fun agentSignalReplayExportReportBuildsPortableSignalReplayBundle() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalReplayExportReportJson(context)
        val manifest = result.getJSONArray("agent_signal_replay_export_manifest")
        val frames = result.getJSONArray("agent_signal_replay_frame_index")
        val metadata = result.getJSONArray("agent_signal_replay_metadata_keys")
        val graphTypes = result.getJSONArray("agent_signal_replay_export_graph_types").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val cards = result.getJSONArray("cards").toString()
        val bundle = result.getJSONObject("agent_signal_replay_export_bundle")
        val directives = result.getJSONArray("gemma_signal_replay_export_directives").toString()
        val manifestLabels = buildSet {
            for (index in 0 until manifest.length()) add(manifest.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_replay_export_report", result.getString("action"))
        assertEquals("signal_replay_export", bundle.getString("bundle_kind"))
        assertEquals(1, bundle.getInt("schema_version"))
        assertTrue(sourceActions.contains("agent_signal_evidence_report"))
        assertTrue(sourceActions.contains("agent_signal_timeline_report"))
        assertTrue(sourceActions.contains("agent_signal_proof_audit_report"))
        assertTrue(sourceActions.contains("agent_signal_session_snapshot_report"))
        assertTrue(sourceActions.contains("wifi_channel_graph"))
        assertTrue(sourceActions.contains("bluetooth_signal_history"))
        assertTrue(sourceActions.contains("radio_signal_graph"))
        assertTrue(sourceActions.contains("mediatek_backend_launch_checklist_report"))
        assertTrue(sourceActions.contains("agent_release_validation_report"))
        assertTrue(graphTypes.contains("agent_signal_replay_export_manifest"))
        assertTrue(graphTypes.contains("agent_signal_replay_frame_index"))
        assertTrue(graphTypes.contains("agent_signal_replay_metadata_keys"))
        assertTrue(graphTypes.contains("signal_evidence_matrix"))
        assertTrue(graphTypes.contains("agent_signal_timeline"))
        assertTrue(graphTypes.contains("agent_signal_proof_audit_matrix"))
        assertTrue(graphTypes.contains("agent_signal_session_snapshot_matrix"))
        assertTrue(graphTypes.contains("wifi_channel_graph"))
        assertTrue(graphTypes.contains("bluetooth_signal_history"))
        assertTrue(graphTypes.contains("motion_sensor_history"))
        assertTrue(graphTypes.contains("radio_signal_graph"))
        assertTrue(graphTypes.contains("mediatek_backend_launch_checklist_matrix"))
        assertTrue(manifest.length() >= 9)
        assertTrue(frames.length() >= 9)
        assertTrue(metadata.length() >= 5)
        assertTrue(manifestLabels.contains("Portable replay schema"))
        assertTrue(manifestLabels.contains("Evidence bundle frame"))
        assertTrue(manifestLabels.contains("Proof boundary frame"))
        assertTrue(frames.toString().contains("frame_key"))
        assertTrue(frames.toString().contains("proof_status"))
        assertTrue(metadata.toString().contains("metadata_keys"))
        assertTrue(bundle.getJSONArray("sections").length() >= 8)
        assertEquals(frames.length(), bundle.getInt("frame_count"))
        assertEquals(frames.length(), result.getInt("agent_signal_replay_frame_count"))
        assertTrue(cards.contains("Signal Replay Export"))
        assertTrue(cards.contains("Replay Frame Index"))
        assertTrue(cards.contains("Replay Claim Boundaries"))
        assertTrue(cards.contains("Replay Metadata Keys"))
        assertTrue(directives.contains("passive"))
        assertTrue(directives.contains("proof_status"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "signal_replay_export"))
        assertEquals("agent_signal_replay_export_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_replay_export_report"))
    }

    @Test
    fun agentSignalReplayFreshnessAuditConnectsReplayFramesToRefreshAndProofGates() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalReplayFreshnessAuditReportJson(context)
        val rows = result.getJSONArray("agent_signal_replay_freshness_matrix")
        val routes = result.getJSONArray("agent_signal_replay_refresh_routes")
        val summary = result.getJSONArray("agent_signal_replay_staleness_summary")
        val graphTypes = result.getJSONArray("agent_signal_replay_freshness_graph_types").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()
        val directives = result.getJSONArray("gemma_signal_replay_freshness_directives").toString()
        val cards = result.getJSONArray("cards").toString()
        val rowText = rows.toString()
        val routeText = routes.toString()
        val summaryText = summary.toString()
        val labels = buildSet {
            for (index in 0 until rows.length()) add(rows.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_replay_freshness_audit_report", result.getString("action"))
        assertTrue(sourceActions.contains("agent_signal_replay_export_report"))
        assertTrue(sourceActions.contains("agent_signal_card_refresh_status_report"))
        assertTrue(sourceActions.contains("agent_signal_proof_audit_report"))
        assertTrue(sourceActions.contains("agent_signal_timeline_report"))
        assertTrue(graphTypes.contains("agent_signal_replay_freshness_matrix"))
        assertTrue(graphTypes.contains("agent_signal_replay_refresh_routes"))
        assertTrue(graphTypes.contains("agent_signal_replay_staleness_summary"))
        assertTrue(graphTypes.contains("agent_signal_replay_frame_index"))
        assertTrue(rows.length() >= 9)
        assertTrue(routes.length() >= rows.length())
        assertTrue(summary.length() >= 3)
        assertTrue(labels.contains("Evidence bundle replay frame"))
        assertTrue(labels.contains("MediaTek/backend replay frame"))
        assertTrue(rowText.contains("freshness_status"))
        assertTrue(rowText.contains("staleness_risk"))
        assertTrue(rowText.contains("active_refresh_action"))
        assertTrue(rowText.contains("passive_fallback_action"))
        assertTrue(rowText.contains("permission_gate"))
        assertTrue(rowText.contains("hardware_gate"))
        assertTrue(rowText.contains("proof_status"))
        assertTrue(rowText.contains("phone_validation_required"))
        assertTrue(routeText.contains("route_type"))
        assertTrue(routeText.contains("agent_signal_card_refresh_status_report"))
        assertTrue(summaryText.contains("active_ready_replay_frames"))
        assertTrue(summaryText.contains("blocked_live_claim_replay_frames"))
        assertTrue(cards.contains("Replay Freshness Audit"))
        assertTrue(cards.contains("Replay Refresh Routes"))
        assertTrue(cards.contains("Replay Staleness Summary"))
        assertTrue(directives.contains("freshness_status"))
        assertEquals(rows.length(), result.getInt("agent_signal_replay_freshness_count"))
        assertEquals(routes.length(), result.getInt("agent_signal_replay_refresh_route_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "signal_replay_freshness"))
        assertEquals("agent_signal_replay_freshness_audit_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_replay_freshness_audit_report"))
    }

    @Test
    fun agentSignalTimelineReportFusesRecentSignalViewsAndRefreshRoutes() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalTimelineReportJson(context)
        val timeline = result.getJSONArray("agent_signal_timeline")
        val routes = result.getJSONArray("agent_signal_refresh_routes")
        val timelineText = timeline.toString()
        val routeText = routes.toString()
        val cards = result.getJSONArray("cards").toString()
        val timelineLabels = buildSet {
            for (index in 0 until timeline.length()) add(timeline.getJSONObject(index).getString("label"))
        }
        val routeLabels = buildSet {
            for (index in 0 until routes.length()) add(routes.getJSONObject(index).getString("label"))
        }

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_timeline_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_signal_briefing_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_channel_graph"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_signal_history"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("motion_pose"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("radio_signal_graph"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("non_adreno_backend_advisor_report"))
        assertTrue(timelineLabels.contains("Wi-Fi channel and link view"))
        assertTrue(timelineLabels.contains("Bluetooth proximity and identity view"))
        assertTrue(timelineLabels.contains("Motion sensor context view"))
        assertTrue(timelineLabels.contains("AM/FM and external radio bridge view"))
        assertTrue(timelineLabels.contains("MediaTek and local backend guardrail view"))
        assertTrue(timelineLabels.contains("Top-card planner view"))
        assertTrue(timelineText.contains("timeline_index"))
        assertTrue(timelineText.contains("freshness"))
        assertTrue(timelineText.contains("metadata_keys"))
        assertTrue(timelineText.contains("accelerator_preflight_matrix"))
        assertTrue(timelineText.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(timelineText.contains("non_adreno_backend_advisor_report"))
        assertTrue(routeLabels.contains("Refresh Wi-Fi channel view"))
        assertTrue(routeLabels.contains("Refresh Bluetooth proximity view"))
        assertTrue(routeLabels.contains("Sample motion sensor view"))
        assertTrue(routeLabels.contains("Supply radio bridge samples"))
        assertTrue(routeLabels.contains("Refresh backend guardrails"))
        assertTrue(routeLabels.contains("Refresh accelerator preflight"))
        assertTrue(routeText.contains("non_adreno_backend_advisor_report"))
        assertTrue(routeText.contains("vendor_radio_bridge_or_external_sdr"))
        assertTrue(result.getJSONArray("gemma_signal_timeline_directives").toString().contains("agent_signal_timeline"))
        assertTrue(cards.contains("Agent Signal Timeline"))
        assertTrue(cards.contains("agent_signal_timeline"))
        assertTrue(cards.contains("Signal Refresh Routes"))
        assertTrue(cards.contains("agent_signal_refresh_routes"))
        assertTrue(result.getInt("agent_signal_timeline_count") >= 6)
        assertTrue(result.getInt("agent_signal_refresh_route_count") >= 6)
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
        val toolSandbox = result.getJSONArray("agent_tool_sandbox_matrix")
        val mcpRegistry = result.getJSONArray("mcp_tool_server_registry")
        val mcpRoutes = result.getJSONArray("mcp_tool_server_routes")
        val readiness = result.getJSONArray("workflow_readiness_matrix")
        val capabilityText = capabilities.toString()
        val kaiText = kaiParity.toString()
        val kaiOperationsText = kaiOperations.toString()
        val toolSandboxText = toolSandbox.toString()
        val mcpRegistryText = mcpRegistry.toString()
        val mcpRoutesText = mcpRoutes.toString()
        val toolSandboxLabels = (0 until toolSandbox.length()).map { index ->
            toolSandbox.getJSONObject(index).getString("label")
        }
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
        assertTrue(toolSandboxLabels.contains("Native diagnostics tool surface"))
        assertTrue(toolSandboxLabels.contains("Android automation and heartbeat surface"))
        assertTrue(toolSandboxLabels.contains("Terminal/Linux workspace surface"))
        assertTrue(toolSandboxLabels.contains("Privileged Android action surface"))
        assertTrue(toolSandboxLabels.contains("UI/accessibility surface"))
        assertTrue(toolSandboxLabels.contains("External MCP/server parity surface"))
        assertTrue(toolSandboxLabels.contains("Memory/persona secure local surface"))
        assertTrue(toolSandboxLabels.contains("External send safety surface"))
        assertTrue(toolSandboxText.contains("android_device_diagnostics_tool:tool_catalog"))
        assertTrue(toolSandboxText.contains("terminal_tool"))
        assertTrue(toolSandboxText.contains("sandbox_scope"))
        assertTrue(toolSandboxText.contains("permission_gate"))
        assertTrue(toolSandboxText.contains("host_access"))
        assertTrue(toolSandboxText.contains("remote_dispatch_capable"))
        assertTrue(toolSandboxText.contains("mcp_parity_status"))
        assertTrue(toolSandboxText.contains("mcp_tool_server_registry_report"))
        assertTrue(mcpRegistryText.contains("Streamable HTTP MCP endpoint"))
        assertTrue(mcpRegistryText.contains("Context7 documentation server"))
        assertTrue(mcpRegistryText.contains("DeepWiki repository docs server"))
        assertTrue(mcpRegistryText.contains("Fetch web content server"))
        assertTrue(mcpRegistryText.contains("native_equivalent_action"))
        assertTrue(mcpRegistryText.contains("remote_endpoint_required"))
        assertTrue(mcpRoutesText.contains("Prefer native Hermes tools first"))
        assertTrue(mcpRoutesText.contains("Use HTTP automation for simple APIs"))
        assertTrue(mcpRoutesText.contains("Startup reconnect policy"))
        assertTrue(mcpRoutesText.contains("mcp_streamable_http_supported"))
        assertTrue(result.getJSONObject("agent_persona_status").getBoolean("custom_system_prompt_enabled"))
        assertTrue(readinessText.contains("Analyze nearby Wi-Fi"))
        assertTrue(readinessText.contains("Run local multimodal agent"))
        assertTrue(readinessText.contains("Route Kai-style tool orchestration"))
        assertTrue(result.getJSONArray("cards").toString().contains("Kai Parity"))
        assertTrue(result.getJSONArray("cards").toString().contains("Kai Operations"))
        assertTrue(result.getJSONArray("cards").toString().contains("Tool Sandbox Status"))
        assertTrue(result.getJSONArray("cards").toString().contains("MCP Tool Servers"))
        assertTrue(result.getJSONArray("cards").toString().contains("MCP Routing Policy"))
        assertTrue(result.getJSONArray("cards").toString().contains("mcp_tool_server_registry"))
        assertTrue(result.getInt("agent_capability_count") >= 8)
        assertTrue(result.getInt("kai_parity_count") >= 11)
        assertTrue(result.getInt("kai_operations_count") >= 8)
        assertTrue(result.getInt("ready_kai_operations_count") >= 5)
        assertTrue(result.getInt("agent_tool_sandbox_count") >= 8)
        assertTrue(result.getInt("ready_agent_tool_sandbox_count") >= 4)
        assertTrue(result.getInt("mcp_tool_server_count") >= 10)
        assertTrue(result.getInt("ready_mcp_tool_server_count") >= 5)
        assertTrue(result.getInt("mcp_tool_server_route_count") >= 5)
        assertTrue(result.getInt("ready_mcp_tool_server_route_count") >= 4)
    }

    @Test
    fun mcpToolServerRegistryReportExposesKaiCuratedServerParity() {
        val result = HermesDeviceDiagnosticsBridge.mcpToolServerRegistryReportJson(context)
        val registry = result.getJSONArray("mcp_tool_server_registry")
        val routes = result.getJSONArray("mcp_tool_server_routes")
        val registryText = registry.toString()
        val routeText = routes.toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("mcp_tool_server_registry_report", result.getString("action"))
        assertFalse(result.getBoolean("mcp_streamable_http_supported"))
        assertFalse(result.getBoolean("mcp_auto_reconnect_supported"))
        assertTrue(result.getBoolean("mcp_native_tool_bridge_available"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("tool_catalog"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("android_automation_tool:perform_http_request"))
        assertTrue(registryText.contains("Streamable HTTP MCP endpoint"))
        assertTrue(registryText.contains("Fetch web content server"))
        assertTrue(registryText.contains("DeepWiki repository docs server"))
        assertTrue(registryText.contains("Sequential Thinking planning server"))
        assertTrue(registryText.contains("Context7 documentation server"))
        assertTrue(registryText.contains("Globalping network probe server"))
        assertTrue(registryText.contains("CoinGecko market data server"))
        assertTrue(registryText.contains("Manifold Markets server"))
        assertTrue(registryText.contains("Find-A-Domain server"))
        assertTrue(registryText.contains("android_device_diagnostics_tool:tool_catalog"))
        assertTrue(registryText.contains("android_automation_tool:http_get"))
        assertTrue(registryText.contains("remote_endpoint_required"))
        assertTrue(registryText.contains("streamable_http_supported"))
        assertTrue(routeText.contains("Prefer native Hermes tools first"))
        assertTrue(routeText.contains("Use HTTP automation for simple APIs"))
        assertTrue(routeText.contains("Disclose remote MCP gaps"))
        assertTrue(routeText.contains("Startup reconnect policy"))
        assertTrue(routeText.contains("Keep credentials out of exports"))
        assertTrue(routeText.contains("mcp_streamable_http_supported"))
        assertTrue(cards.contains("MCP Tool Server Registry"))
        assertTrue(cards.contains("mcp_tool_server_registry"))
        assertTrue(cards.contains("MCP Routing Policy"))
        assertTrue(cards.contains("mcp_tool_server_routes"))
        assertTrue(result.getJSONArray("gemma_mcp_registry_directives").toString().contains("Streamable HTTP MCP"))
        assertTrue(result.getInt("mcp_tool_server_count") >= 10)
        assertTrue(result.getInt("ready_mcp_tool_server_count") >= 5)
        assertTrue(result.getInt("mcp_tool_server_route_count") >= 5)
        assertTrue(result.getInt("ready_mcp_tool_server_route_count") >= 4)
    }

    @Test
    fun agentCapabilityUpgradeReportAuditsFullObjectiveDomainsAndRoutes() {
        val result = HermesDeviceDiagnosticsBridge.agentCapabilityUpgradeReportJson(context)
        val objectives = result.getJSONArray("agent_upgrade_objective_matrix")
        val routes = result.getJSONArray("agent_upgrade_route_matrix")
        val objectiveText = objectives.toString()
        val routeText = routes.toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_capability_upgrade_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("wifi_analyzer_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("bluetooth_signal_advisor_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("sensor_workflow_advisor_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("radio_signal_advisor_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("mcp_tool_server_registry_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("mediatek_readiness_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("non_adreno_backend_advisor_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_card_priority_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_signal_workflow_handoff_report"))
        assertTrue(objectiveText.contains("WiFiAnalyzer-style Wi-Fi intelligence"))
        assertTrue(objectiveText.contains("Gemma-readable Wi-Fi top cards"))
        assertTrue(objectiveText.contains("Bluetooth nearby scanner and device understanding"))
        assertTrue(objectiveText.contains("SDR radio signal boundary"))
        assertTrue(objectiveText.contains("Accelerometer, gyroscope, and motion workflows"))
        assertTrue(objectiveText.contains("MediaTek and non-Adreno backend compatibility"))
        assertTrue(objectiveText.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(objectiveText.contains("Local Gemma multimodal readiness"))
        assertTrue(objectiveText.contains("Kai parity and MCP tool-server awareness"))
        assertTrue(objectiveText.contains("Cross-signal evidence for Gemma"))
        assertTrue(objectiveText.contains("Self-check and completion-boundary evidence"))
        assertTrue(objectiveText.contains("evidence_status"))
        assertTrue(objectiveText.contains("bridge_required"))
        assertTrue(objectiveText.contains("physical_device_validation_required"))
        assertTrue(objectiveText.contains("agent_upgrade_objective"))
        assertTrue(routeText.contains("Start with full upgrade audit"))
        assertTrue(routeText.contains("Open ranked top-card planner"))
        assertTrue(routeText.contains("Open signal workflow handoff"))
        assertTrue(routeText.contains("Verify non-Adreno backend launch advisor"))
        assertTrue(routeText.contains("Verify radio receiver bridge boundaries"))
        assertTrue(routeText.contains("Verify Kai MCP parity boundaries"))
        assertTrue(cards.contains("Upgrade Objective Matrix"))
        assertTrue(cards.contains("agent_upgrade_objective_matrix"))
        assertTrue(cards.contains("Upgrade Verification Routes"))
        assertTrue(cards.contains("agent_upgrade_route_matrix"))
        assertTrue(result.getJSONArray("gemma_upgrade_audit_directives").toString().contains("agent_upgrade_objective_matrix"))
        assertTrue(result.getInt("agent_upgrade_objective_count") >= 10)
        assertTrue(result.getInt("ready_agent_upgrade_objective_count") >= 7)
        assertTrue(result.getInt("agent_upgrade_route_count") >= 6)
        assertTrue(result.getInt("ready_agent_upgrade_route_count") >= 6)

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "agent_capability_upgrade_report"))
        assertEquals("agent_capability_upgrade_report", dispatched.getString("action"))
    }

    @Test
    fun agentObjectiveCoverageReportMapsRequirementsResearchAndGaps() {
        val result = HermesDeviceDiagnosticsBridge.agentObjectiveCoverageReportJson(context)
        val coverageRows = result.getJSONArray("agent_objective_coverage_matrix")
        val gapRows = result.getJSONArray("agent_objective_gap_matrix")
        val researchRows = result.getJSONArray("agent_research_parity_matrix")
        val coverageText = coverageRows.toString()
        val gapText = gapRows.toString()
        val researchText = researchRows.toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_objective_coverage_report", result.getString("action"))
        assertTrue(result.getJSONArray("research_source_urls").toString().contains("SimonSchubert"))
        assertTrue(result.getJSONArray("research_source_urls").toString().contains("WiFiAnalyzer"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_capability_upgrade_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("non_adreno_backend_advisor_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_release_validation_report"))
        assertTrue(coverageText.contains("WiFiAnalyzer-style Wi-Fi intelligence"))
        assertTrue(coverageText.contains("Bluetooth nearby scanner and device understanding"))
        assertTrue(coverageText.contains("MediaTek and non-Adreno backend compatibility"))
        assertTrue(coverageText.contains("coverage_status"))
        assertTrue(coverageText.contains("source_actions"))
        assertTrue(gapText.contains("Live signal permission proof"))
        assertTrue(gapText.contains("Release and CI proof"))
        assertTrue(gapText.contains("agent_release_validation_report"))
        assertTrue(gapText.contains("release_validation_required"))
        assertTrue(gapText.contains("physical_device_validation_required"))
        assertTrue(researchText.contains("Kai agent experience parity"))
        assertTrue(researchText.contains("WiFiAnalyzer graph and metadata parity"))
        assertTrue(researchText.contains("Hermes-only nearby signal expansion"))
        assertTrue(researchText.contains("research_source_url"))
        assertTrue(cards.contains("Objective Coverage"))
        assertTrue(cards.contains("agent_objective_coverage_matrix"))
        assertTrue(cards.contains("Objective Gaps"))
        assertTrue(cards.contains("agent_objective_gap_matrix"))
        assertTrue(cards.contains("Research Parity Map"))
        assertTrue(cards.contains("agent_research_parity_matrix"))
        assertTrue(result.getJSONArray("gemma_objective_coverage_directives").toString().contains("agent_objective_gap_matrix"))
        assertTrue(result.getInt("agent_objective_coverage_count") >= 10)
        assertTrue(result.getInt("agent_objective_gap_count") >= 3)
        assertTrue(result.getInt("agent_research_parity_count") >= 4)

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "agent_objective_coverage_report"))
        assertEquals("agent_objective_coverage_report", dispatched.getString("action"))
        val alias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "full_upgrade_coverage_report"))
        assertEquals("agent_objective_coverage_report", alias.getString("action"))
        val upgradeAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "agent_upgrade_coverage_report"))
        assertEquals("agent_objective_coverage_report", upgradeAlias.getString("action"))
        val hermesAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "hermes_upgrade_coverage_report"))
        assertEquals("agent_objective_coverage_report", hermesAlias.getString("action"))
    }

    @Test
    fun agentReleaseValidationReportMapsGithubFdroidAndArtifactGates() {
        val result = HermesDeviceDiagnosticsBridge.agentReleaseValidationReportJson(context)
        val validationRows = result.getJSONArray("agent_release_validation_matrix")
        val artifactRows = result.getJSONArray("agent_release_artifact_gates")
        val fdroidRows = result.getJSONArray("fdroid_release_metadata_matrix")
        val validationText = validationRows.toString()
        val artifactText = artifactRows.toString()
        val fdroidText = fdroidRows.toString()
        val cards = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_release_validation_report", result.getString("action"))
        assertTrue(result.getJSONObject("app_release_identity").toString().contains("android-release.yml"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_objective_coverage_report"))
        assertTrue(validationText.contains("Android CI workflow gate"))
        assertTrue(validationText.contains("Signed release workflow gate"))
        assertTrue(validationText.contains("Physical-device native validation"))
        assertTrue(validationText.contains("testDebugUnitTest"))
        assertTrue(validationText.contains("android-release.yml"))
        assertTrue(artifactText.contains("Signed APK and AAB assets"))
        assertTrue(artifactText.contains("SHA-256 checksum assets"))
        assertTrue(artifactText.contains("expected_tag_pattern"))
        assertTrue(fdroidText.contains("F-Droid metadata package alignment"))
        assertTrue(fdroidText.contains("Fastlane graphics in tagged tree"))
        assertTrue(fdroidText.contains("Future update discovery"))
        assertTrue(cards.contains("Release Validation"))
        assertTrue(cards.contains("agent_release_validation_matrix"))
        assertTrue(cards.contains("Release Artifact Gates"))
        assertTrue(cards.contains("agent_release_artifact_gates"))
        assertTrue(cards.contains("F-Droid Metadata Gates"))
        assertTrue(cards.contains("fdroid_release_metadata_matrix"))
        assertTrue(result.getJSONArray("gemma_release_validation_directives").toString().contains("release-ready"))
        assertTrue(result.getInt("agent_release_validation_count") >= 5)
        assertTrue(result.getInt("agent_release_artifact_gate_count") >= 3)
        assertTrue(result.getInt("fdroid_release_metadata_count") >= 3)

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "agent_release_validation_report"))
        assertEquals("agent_release_validation_report", dispatched.getString("action"))
        val alias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "release_workflow_audit"))
        assertEquals("agent_release_validation_report", alias.getString("action"))
        val githubReadinessAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "github_release_readiness_report"))
        assertEquals("agent_release_validation_report", githubReadinessAlias.getString("action"))
        val validationReadinessAlias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "release_validation_readiness_report"))
        assertEquals("agent_release_validation_report", validationReadinessAlias.getString("action"))
    }

    @Test
    fun agentSignalWorkflowHandoffReportRanksNextActionsAndBoundaries() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalWorkflowHandoffReportJson(context)
        val rows = result.getJSONArray("agent_signal_workflow_handoff_matrix")
        val routes = result.getJSONArray("agent_signal_next_action_routes")
        val rowText = rows.toString()
        val routeText = routes.toString()
        val cardText = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_workflow_handoff_report", result.getString("action"))
        assertTrue(sourceActions.contains("agent_signal_evidence_report"))
        assertTrue(sourceActions.contains("agent_card_priority_report"))
        assertTrue(sourceActions.contains("wifi_analyzer_report"))
        assertTrue(sourceActions.contains("bluetooth_analyzer_report"))
        assertTrue(sourceActions.contains("sensor_analyzer_report"))
        assertTrue(sourceActions.contains("radio_signal_advisor_report"))
        assertTrue(sourceActions.contains("mediatek_readiness_report"))
        assertTrue(sourceActions.contains("non_adreno_backend_advisor_report"))
        assertTrue(sourceActions.contains("mcp_tool_server_registry_report"))
        assertTrue(rowText.contains("Start with current evidence bundle"))
        assertTrue(rowText.contains("Open Wi-Fi Analyzer graph"))
        assertTrue(rowText.contains("Bluetooth details"))
        assertTrue(rowText.contains("motion sensor quality"))
        assertTrue(rowText.contains("radio receiver advisor"))
        assertTrue(rowText.contains("non-Adreno backend advisor"))
        assertTrue(rowText.contains("MediaTek"))
        assertTrue(rowText.contains("Kai"))
        assertTrue(rowText.contains("open_next_action"))
        assertTrue(rowText.contains("refresh_policy"))
        assertTrue(rowText.contains("permission_gate"))
        assertTrue(rowText.contains("bridge_required"))
        assertTrue(rowText.contains("physical_device_validation_required"))
        assertTrue(routeText.contains("Open signal workflow handoff"))
        assertTrue(routeText.contains("Open non-Adreno backend launch advisor"))
        assertTrue(routeText.contains("agent_signal_workflow_handoff_report"))
        assertTrue(routeText.contains("agent_question_patterns"))
        assertTrue(cardText.contains("Signal Workflow Handoff"))
        assertTrue(cardText.contains("agent_signal_workflow_handoff_matrix"))
        assertTrue(cardText.contains("Next Signal Actions"))
        assertTrue(cardText.contains("agent_signal_next_action_routes"))
        assertTrue(result.getJSONArray("gemma_signal_workflow_handoff_directives").toString().contains("open_next_action"))
        assertTrue(result.getInt("agent_signal_workflow_handoff_count") >= 9)
        assertTrue(result.getInt("ready_agent_signal_workflow_handoff_count") >= 5)
        assertTrue(result.getInt("agent_signal_next_action_route_count") >= 8)

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "agent_signal_workflow_handoff_report"))
        assertEquals("agent_signal_workflow_handoff_report", dispatched.getString("action"))
        val alias = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "next_signal_action_report"))
        assertEquals("agent_signal_workflow_handoff_report", alias.getString("action"))
    }

    @Test
    fun agentSignalPermissionRunbookReportMapsActiveRefreshGatesAndSettingsRoutes() {
        val result = HermesDeviceDiagnosticsBridge.agentSignalPermissionRunbookReportJson(context)
        val rows = result.getJSONArray("agent_signal_permission_runbook_matrix")
        val routes = result.getJSONArray("agent_signal_active_refresh_routes")
        val rowText = rows.toString()
        val routeText = routes.toString()
        val cardText = result.getJSONArray("cards").toString()
        val sourceActions = result.getJSONArray("source_report_actions").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_signal_permission_runbook_report", result.getString("action"))
        assertTrue(result.has("wifi_scan_permission_status"))
        assertTrue(result.has("bluetooth_scan_permission_status"))
        assertTrue(result.has("sensor_handoff_summary"))
        assertTrue(result.has("radio_status_handoff_summary"))
        assertTrue(result.has("accelerator_handoff_summary"))
        assertTrue(sourceActions.contains("wifi_scan"))
        assertTrue(sourceActions.contains("bluetooth_scan"))
        assertTrue(sourceActions.contains("radio_signal_graph"))
        assertTrue(sourceActions.contains("accelerator_preflight_report"))
        assertTrue(sourceActions.contains("open_app_settings"))
        assertTrue(rowText.contains("Prepare active Wi-Fi scan"))
        assertTrue(rowText.contains("Prepare active Bluetooth scan"))
        assertTrue(rowText.contains("Prepare motion sensor sample"))
        assertTrue(rowText.contains("SDR bridge samples"))
        assertTrue(rowText.contains("Prepare accelerator proof refresh"))
        assertTrue(rowText.contains("active_refresh_arguments"))
        assertTrue(rowText.contains("settings_actions"))
        assertTrue(rowText.contains("open_app_settings"))
        assertTrue(rowText.contains("open_location_settings"))
        assertTrue(rowText.contains("open_bluetooth_settings"))
        assertTrue(rowText.contains("passive_fallback_action"))
        assertTrue(rowText.contains("user_consent_required"))
        assertTrue(rowText.contains("bridge_required"))
        assertTrue(rowText.contains("physical_device_validation_required"))
        assertTrue(routeText.contains("Open Hermes app permissions"))
        assertTrue(routeText.contains("Run active Wi-Fi scan"))
        assertTrue(routeText.contains("Run active Bluetooth scan"))
        assertTrue(cardText.contains("Signal Permission Runbook"))
        assertTrue(cardText.contains("agent_signal_permission_runbook_matrix"))
        assertTrue(cardText.contains("Active Signal Refresh Routes"))
        assertTrue(cardText.contains("agent_signal_active_refresh_routes"))
        assertTrue(result.getJSONArray("gemma_signal_permission_runbook_directives").toString().contains("active_refresh_arguments"))
        assertEquals(rows.length(), result.getInt("agent_signal_permission_runbook_count"))
        assertEquals(routes.length(), result.getInt("agent_signal_active_refresh_route_count"))

        val dispatched = JSONObject(HermesDeviceDiagnosticsBridge.performActionJson(context, "signal_refresh_runbook_report"))
        assertEquals("agent_signal_permission_runbook_report", dispatched.getString("action"))
        val availableActions = HermesDeviceDiagnosticsBridge.statusJson(context).getJSONArray("available_actions").toString()
        assertTrue(availableActions.contains("agent_signal_permission_runbook_report"))
        assertTrue(availableActions.contains("open_location_settings"))
        assertTrue(availableActions.contains("open_bluetooth_settings"))
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
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONObject("wifi_observation_summary").getJSONArray("card_titles").toString().contains("Wi-Fi Analyzer"))
        assertTrue(result.getJSONObject("backend_risk_observation_summary").getJSONArray("card_titles").toString().contains("GPU Backend Risk"))
        assertTrue(result.getJSONObject("accelerator_preflight_observation_summary").getJSONArray("card_titles").toString().contains("Accelerator Preflight"))
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
        assertTrue(signalContextText.contains("radio_signal_advisor_matrix"))
        assertTrue(signalContextText.contains("radio_receiver_candidates"))
        assertTrue(signalContextText.contains("Backend risk and fallback context"))
        assertTrue(signalContextText.contains("accelerator_preflight_matrix"))
        assertTrue(signalContextText.contains("accelerator_preflight_report"))
        assertTrue(signalContextText.contains("source_actions"))
        assertTrue(signalContextText.contains("card_graph_types"))
        assertTrue(cardManifestText.contains("wifi_analyzer_report"))
        assertTrue(cardManifestText.contains("wifi_channel_graph"))
        assertTrue(cardManifestText.contains("bluetooth_analyzer_report"))
        assertTrue(cardManifestText.contains("bluetooth_device_details"))
        assertTrue(cardManifestText.contains("bluetooth_device_detail"))
        assertTrue(cardManifestText.contains("radio_signal_status"))
        assertTrue(cardManifestText.contains("radio_signal_advisor_report"))
        assertTrue(cardManifestText.contains("radio_signal_advisor_matrix"))
        assertTrue(cardManifestText.contains("radio_receiver_candidates"))
        assertTrue(cardManifestText.contains("radio_receiver_bridge_schema"))
        assertTrue(cardManifestText.contains("gpu_backend_risk_report"))
        assertTrue(cardManifestText.contains("accelerator_preflight_report"))
        assertTrue(cardManifestText.contains("accelerator_preflight_matrix"))
        assertTrue(cardManifestText.contains("non_adreno_backend_advisor_report"))
        assertTrue(cardManifestText.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(cardManifestText.contains("Non-Adreno backend advisor route"))
        assertTrue(cardManifestText.contains("agent_environment_report"))
        assertTrue(cardManifestText.contains("mcp_tool_server_registry_report"))
        assertTrue(cardManifestText.contains("mcp_tool_server_registry"))
        assertTrue(cardManifestText.contains("mcp_tool_server_routes"))
        assertTrue(cardManifestText.contains("agent_signal_workflow_handoff_report"))
        assertTrue(cardManifestText.contains("agent_signal_workflow_handoff_matrix"))
        assertTrue(cardManifestText.contains("agent_signal_next_action_routes"))
        assertTrue(cardManifestText.contains("agent_capability_upgrade_report"))
        assertTrue(cardManifestText.contains("agent_upgrade_objective_matrix"))
        assertTrue(cardManifestText.contains("agent_upgrade_route_matrix"))
        assertTrue(cardManifestText.contains("refresh_policy"))
        assertTrue(cardManifestText.contains("permission_gate"))
        assertTrue(routeText.contains("Open Wi-Fi analyzer cards"))
        assertTrue(routeText.contains("Open Bluetooth device detail cards"))
        assertTrue(routeText.contains("Open signal context fusion card"))
        assertTrue(routeText.contains("Open GPU backend risk cards"))
        assertTrue(routeText.contains("Open non-Adreno backend advisor cards"))
        assertTrue(routeText.contains("Open SOC and Kai environment cards"))
        assertTrue(result.getJSONArray("cards").toString().contains("Agent Observation"))
        assertTrue(result.getJSONArray("cards").toString().contains("Signal Context Fusion"))
        assertTrue(result.getJSONArray("cards").toString().contains("Agent Card Manifest"))
        assertTrue(result.getJSONArray("cards").toString().contains("Observation Routes"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("Read agent_observation_matrix first"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("agent_signal_context_matrix"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("agent_card_manifest"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("non_adreno_backend_advisor_report"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("wifi_channel_graph"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("bluetooth_device_detail"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("bluetooth_signal_history"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("radio_signal_advisor_matrix"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("radio_receiver_candidates"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("gpu_backend_risk_matrix"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("accelerator_preflight_matrix"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("mcp_tool_server_registry"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("mcp_tool_server_routes"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("agent_signal_workflow_handoff_matrix"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("agent_signal_next_action_routes"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("agent_upgrade_objective_matrix"))
        assertTrue(result.getJSONArray("agent_card_graph_types").toString().contains("agent_upgrade_route_matrix"))
        assertTrue(result.getInt("agent_card_manifest_count") >= 25)
        assertTrue(result.getInt("ready_agent_card_manifest_count") >= 9)
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
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("mcp_tool_server_registry_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_signal_workflow_handoff_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_capability_upgrade_report"))
        assertTrue(cardManifestText.contains("wifi_channel_graph"))
        assertTrue(cardManifestText.contains("wifi_channel_rating"))
        assertTrue(cardManifestText.contains("bluetooth_device_details"))
        assertTrue(cardManifestText.contains("bluetooth_device_detail"))
        assertTrue(cardManifestText.contains("bluetooth_signal_history"))
        assertTrue(cardManifestText.contains("radio_signal_graph"))
        assertTrue(cardManifestText.contains("radio_signal_advisor_report"))
        assertTrue(cardManifestText.contains("radio_signal_advisor_matrix"))
        assertTrue(cardManifestText.contains("radio_receiver_candidates"))
        assertTrue(cardManifestText.contains("gpu_backend_risk_report"))
        assertTrue(cardManifestText.contains("accelerator_preflight_report"))
        assertTrue(cardManifestText.contains("accelerator_preflight_matrix"))
        assertTrue(cardManifestText.contains("mcp_tool_server_registry_report"))
        assertTrue(cardManifestText.contains("mcp_tool_server_registry"))
        assertTrue(cardManifestText.contains("mcp_tool_server_routes"))
        assertTrue(cardManifestText.contains("agent_signal_workflow_handoff_report"))
        assertTrue(cardManifestText.contains("agent_signal_workflow_handoff_matrix"))
        assertTrue(cardManifestText.contains("agent_signal_next_action_routes"))
        assertTrue(cardManifestText.contains("agent_capability_upgrade_report"))
        assertTrue(cardManifestText.contains("agent_upgrade_objective_matrix"))
        assertTrue(cardManifestText.contains("agent_upgrade_route_matrix"))
        assertTrue(cardManifestText.contains("refresh_policy"))
        assertTrue(cardManifestText.contains("permission_gate"))
        assertTrue(graphTypes.contains("wifi_channel_graph"))
        assertTrue(graphTypes.contains("bluetooth_device_detail"))
        assertTrue(graphTypes.contains("radio_signal_graph"))
        assertTrue(graphTypes.contains("radio_signal_advisor_matrix"))
        assertTrue(graphTypes.contains("radio_receiver_candidates"))
        assertTrue(graphTypes.contains("radio_receiver_bridge_schema"))
        assertTrue(graphTypes.contains("gpu_backend_risk_matrix"))
        assertTrue(graphTypes.contains("accelerator_preflight_matrix"))
        assertTrue(graphTypes.contains("mcp_tool_server_registry"))
        assertTrue(graphTypes.contains("mcp_tool_server_routes"))
        assertTrue(graphTypes.contains("agent_signal_workflow_handoff_matrix"))
        assertTrue(graphTypes.contains("agent_signal_next_action_routes"))
        assertTrue(graphTypes.contains("agent_upgrade_objective_matrix"))
        assertTrue(graphTypes.contains("agent_upgrade_route_matrix"))
        assertTrue(result.getJSONArray("cards").toString().contains("Agent Card Manifest"))
        assertTrue(result.getJSONArray("gemma_observation_directives").toString().contains("graph_type"))
        assertTrue(result.getInt("agent_card_manifest_count") >= 25)
        assertTrue(result.getInt("ready_agent_card_manifest_count") >= 9)
    }

    @Test
    fun agentCardPriorityReportRanksTopCardsOpenSequenceAndKaiParity() {
        val result = HermesDeviceDiagnosticsBridge.agentCardPriorityReportJson(context)
        val priorities = result.getJSONArray("top_signal_card_priorities")
        val openSequence = result.getJSONArray("agent_card_open_sequence")
        val kaiParity = result.getJSONArray("kai_interactive_screen_parity")
        val priorityText = priorities.toString()
        val sequenceText = openSequence.toString()
        val kaiParityText = kaiParity.toString()
        val cardText = result.getJSONArray("cards").toString()

        assertTrue(result.getBoolean("success"))
        assertEquals("agent_card_priority_report", result.getString("action"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_self_check_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_observation_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_card_manifest_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("accelerator_preflight_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("non_adreno_backend_advisor_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("mcp_tool_server_registry_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_capability_upgrade_report"))
        assertTrue(result.getJSONArray("source_report_actions").toString().contains("agent_signal_workflow_handoff_report"))
        assertTrue(priorityText.contains("Self-check heartbeat and permission gates"))
        assertTrue(priorityText.contains("Top signal context fusion"))
        assertTrue(priorityText.contains("Wi-Fi channel and AP graph cards"))
        assertTrue(priorityText.contains("Bluetooth detail and RSSI trend cards"))
        assertTrue(priorityText.contains("RF coexistence and signal evidence"))
        assertTrue(priorityText.contains("Motion pose and sensor quality"))
        assertTrue(priorityText.contains("Radio advisor and bridge cards"))
        assertTrue(priorityText.contains("radio_signal_advisor_report"))
        assertTrue(priorityText.contains("radio_signal_advisor_matrix"))
        assertTrue(priorityText.contains("Non-Adreno backend launch advisor"))
        assertTrue(priorityText.contains("non_adreno_backend_advisor_report"))
        assertTrue(priorityText.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(priorityText.contains("MediaTek and backend guardrails"))
        assertTrue(priorityText.contains("Kai operations and interactive parity"))
        assertTrue(priorityText.contains("Kai MCP tool-server registry"))
        assertTrue(priorityText.contains("mcp_tool_server_registry_report"))
        assertTrue(priorityText.contains("mcp_tool_server_registry"))
        assertTrue(priorityText.contains("Full upgrade objective audit"))
        assertTrue(priorityText.contains("agent_capability_upgrade_report"))
        assertTrue(priorityText.contains("agent_upgrade_objective_matrix"))
        assertTrue(priorityText.contains("open_next_action"))
        assertTrue(priorityText.contains("refresh_policy"))
        assertTrue(priorityText.contains("permission_gate"))
        assertTrue(priorityText.contains("priority_rank"))
        assertTrue(priorityText.contains("agent_card_priority"))
        assertTrue(priorityText.contains("passive_backend_launch_advisor"))
        assertTrue(priorityText.contains("non_adreno_backend_advisor_matrix"))
        assertTrue(sequenceText.contains("Run planner self-check first"))
        assertTrue(sequenceText.contains("Open ranked top-card planner"))
        assertTrue(sequenceText.contains("Escalate to live scans only on request"))
        assertTrue(kaiParityText.contains("Persistent memory and persona context"))
        assertTrue(kaiParityText.contains("Generated screen and expandable card parity"))
        assertTrue(kaiParityText.contains("Provider fallback and model routing"))
        assertTrue(kaiParityText.contains("Tool and MCP bridge surface"))
        assertTrue(cardText.contains("Top Signal Cards"))
        assertTrue(cardText.contains("Card Open Sequence"))
        assertTrue(cardText.contains("Kai Interactive Parity"))
        assertTrue(cardText.contains("agent_card_priority_matrix"))
        assertTrue(cardText.contains("agent_card_open_sequence"))
        assertTrue(cardText.contains("kai_interactive_screen_parity"))
        assertTrue(result.getJSONArray("gemma_card_planner_directives").toString().contains("top_signal_card_priorities"))
        assertTrue(result.getJSONArray("gemma_card_planner_directives").toString().contains("open_next_action"))
        assertTrue(result.getJSONArray("gemma_card_planner_directives").toString().contains("kai_interactive_screen_parity"))
        assertTrue(result.getInt("top_signal_card_priority_count") >= 13)
        assertTrue(result.getInt("ready_top_signal_card_priority_count") >= 6)
        assertTrue(result.getInt("agent_card_open_sequence_count") >= 5)
        assertTrue(result.getInt("kai_interactive_screen_parity_count") >= 7)
        assertTrue(result.getInt("agent_card_manifest_count") >= 25)
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
