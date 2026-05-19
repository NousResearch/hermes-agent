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
                                    .put("bond_state", "bonded")
                                    .put("paired", true),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Headphones", row.label)
        assertEquals("paired", row.valueLabel)
        assertTrue(row.detail.contains("bonded"))
        assertTrue(row.fraction >= 0.4f)
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
