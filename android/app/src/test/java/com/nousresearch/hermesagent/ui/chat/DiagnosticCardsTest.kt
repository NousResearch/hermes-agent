package com.nousresearch.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DiagnosticCardsTest {
    @Test
    fun parsesWifiGraphRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("type", "signal_graph_card")
                        .put("title", "Wi-Fi Analyzer")
                        .put("body", "Nearby Wi-Fi signals.")
                        .put("graph_type", "wifi_channel_strength")
                        .put("row_count", 1)
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("ssid", "HermesNet")
                                    .put("rssi_dbm", -42)
                                    .put("frequency_mhz", 5180)
                                    .put("channel", 36)
                                    .put("band", "5 GHz")
                                    .put("channel_width", "80MHz")
                                    .put("security_mode", "WPA2")
                                    .put("bssid_vendor", "Apple")
                                    .put("estimated_distance_meters", 1.6),
                            ),
                        ),
                ),
            )
            .toString()

        val cards = extractDiagnosticCards(content)

        assertEquals(1, cards.size)
        assertEquals("Wi-Fi Analyzer", cards.single().title)
        assertEquals("wifi_channel_strength", cards.single().graphType)
        assertEquals("HermesNet", cards.single().rows.single().label)
        assertEquals("-42 dBm", cards.single().rows.single().valueLabel)
        assertTrue(cards.single().rows.single().detail.contains("ch 36"))
        assertTrue(cards.single().rows.single().detail.contains("WPA2"))
        assertTrue(cards.single().rows.single().detail.contains("Apple"))
        assertTrue(cards.single().rows.single().fraction > 0.8f)
    }

    @Test
    fun parsesBluetoothRowsEvenWhenOnlyPairedMetadataIsAvailable() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Bluetooth Nearby")
                        .put("body", "Paired or scanned devices.")
                        .put("graph_type", "bluetooth_rssi")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("device_name", "Headphones")
                                    .put("device_type", "dual")
                                    .put("device_category", "audio")
                                    .put("bond_state", "bonded")
                                    .put("paired", true)
                                    .put("proximity_label", "near")
                                    .put("service_uuid_count", 2)
                                    .put("manufacturer_data_count", 1),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Headphones", row.label)
        assertEquals("paired", row.valueLabel)
        assertTrue(row.detail.contains("bonded"))
        assertTrue(row.detail.contains("audio"))
        assertTrue(row.detail.contains("2 services"))
        assertTrue(row.fraction >= 0.4f)
    }

    @Test
    fun parsesBluetoothMetadataSummaryRows() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Bluetooth Metadata")
                        .put("body", "Summary rows.")
                        .put("graph_type", "bluetooth_metadata_summary")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("summary_type", "manufacturer_id")
                                    .put("label", "0x004C")
                                    .put("count", 2)
                                    .put("connectable_count", 1)
                                    .put("strongest_rssi_dbm", -50)
                                    .put("recommendation", "Manufacturer data advertised nearby."),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("0x004C", row.label)
        assertEquals("2 devices", row.valueLabel)
        assertTrue(row.detail.contains("manufacturer id"))
        assertTrue(row.detail.contains("1 connectable"))
        assertTrue(row.fraction > 0.7f)
    }

    @Test
    fun parsesWifiChannelRatingRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Wi-Fi Channel Rating")
                        .put("body", "Channel scores.")
                        .put("graph_type", "wifi_channel_rating")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("band", "2.4GHz")
                                    .put("channel", 11)
                                    .put("score", 96)
                                    .put("rating_label", "excellent")
                                    .put("network_count", 0)
                                    .put("overlap_count", 0)
                                    .put("recommendation", "Best current option: no overlapping APs."),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("2.4GHz ch 11", row.label)
        assertEquals("96/100 excellent", row.valueLabel)
        assertTrue(row.detail.contains("0 overlapping"))
        assertTrue(row.detail.contains("Best current option"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesWifiVendorSummaryRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Wi-Fi Vendors")
                        .put("body", "Vendor rows.")
                        .put("graph_type", "wifi_vendor_summary")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("vendor", "Apple")
                                    .put("network_count", 2)
                                    .put("strongest_rssi_dbm", -44)
                                    .put("bssid_ouis", JSONArray().put("AC:BC:32"))
                                    .put("recommendation", "Strong nearby vendor group."),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Apple", row.label)
        assertEquals("2 APs", row.valueLabel)
        assertTrue(row.detail.contains("AC:BC:32"))
        assertTrue(row.detail.contains("Strong nearby vendor group"))
        assertTrue(row.fraction > 0.75f)
    }

    @Test
    fun parsesWifiSignalHistoryRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Wi-Fi History")
                        .put("body", "Signal history.")
                        .put("graph_type", "wifi_signal_history")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("ssid", "HermesNet")
                                    .put("bssid_vendor", "Apple")
                                    .put("current_rssi_dbm", -55)
                                    .put("average_rssi_dbm", -60)
                                    .put("min_rssi_dbm", -66)
                                    .put("max_rssi_dbm", -55)
                                    .put("trend_db", 11)
                                    .put("trend_label", "improving")
                                    .put("sample_count", 2)
                                    .put("band", "5GHz")
                                    .put("channel", 36),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("HermesNet", row.label)
        assertEquals("-55 dBm improving", row.valueLabel)
        assertTrue(row.detail.contains("2 samples"))
        assertTrue(row.detail.contains("avg -60 dBm"))
        assertTrue(row.detail.contains("improving +11 dB"))
        assertTrue(row.fraction > 0.6f)
    }

    @Test
    fun parsesSensorVectorRowsFromMotionSamples() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Motion Sensors")
                        .put("body", "Sensor rows.")
                        .put("graph_type", "sensor_vector")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("sensor_type", "accelerometer")
                                    .put("sensor_label", "Accelerometer")
                                    .put("sampled", true)
                                    .put("unit", "m/s^2")
                                    .put("values", JSONArray().put(0.0).put(0.0).put(9.81)),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Accelerometer", row.label)
        assertEquals("9.81 m/s^2", row.valueLabel)
        assertTrue(row.fraction > 0.45f)
    }

    @Test
    fun parsesSensorCapabilityRowsWithHardwareMetadata() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Sensor Hardware")
                        .put("body", "Sensor metadata rows.")
                        .put("graph_type", "sensor_capability")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("sensor_type", "gyroscope")
                                    .put("sensor_label", "Gyroscope")
                                    .put("sensor_name", "BMI160 Gyroscope")
                                    .put("vendor", "Bosch")
                                    .put("available", true)
                                    .put("unit", "rad/s")
                                    .put("maximum_range", 34.91)
                                    .put("resolution", 0.001)
                                    .put("power_ma", 0.9)
                                    .put("min_delay_us", 5000)
                                    .put("reporting_mode", "continuous")
                                    .put("wake_up", true),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Gyroscope", row.label)
        assertEquals("range 34.91 rad/s", row.valueLabel)
        assertTrue(row.detail.contains("Bosch"))
        assertTrue(row.detail.contains("200.0 Hz"))
        assertTrue(row.detail.contains("wake-up"))
        assertTrue(row.fraction > 0.8f)
    }

    @Test
    fun parsesRadioCapabilityRowsAsLimitsWhenHardwareIsExternal() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "AM/FM Radio")
                        .put("body", "Capability rows.")
                        .put("graph_type", "radio_frequency_capability")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("band", "External SDR")
                                    .put("supported", false)
                                    .put("sampled", false)
                                    .put("requires_external_hardware", true)
                                    .put("reason", "Attach an SDR."),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("External SDR", row.label)
        assertEquals("external", row.valueLabel)
        assertTrue(row.detail.contains("Attach an SDR"))
        assertTrue(row.fraction in 0.4f..0.5f)
    }
}
