package com.nousresearch.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.InterruptedIOException

class NativeToolCallingChatClientTest {
    @Test
    fun skipsLocalFollowUpAfterExternalActivityHandoff() {
        val result = JSONObject()
            .put("success", true)
            .put("action", "open_uri")
            .put("external_activity_handoff", true)
            .put("message", "Started Android intent")

        assertTrue(NativeToolCallingChatClient.shouldSkipNativeFollowUpAfterToolResult(result.toString()))
    }

    @Test
    fun continuesLocalFollowUpAfterOrdinaryToolResult() {
        val result = JSONObject()
            .put("success", true)
            .put("path", "hermes-output.txt")

        assertFalse(NativeToolCallingChatClient.shouldSkipNativeFollowUpAfterToolResult(result.toString()))
    }

    @Test
    fun recoversHtmlGenerationTimeoutsWithFallbackDocument() {
        assertTrue(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                IllegalStateException("LiteRT-LM generation timed out after 45 seconds before producing a response"),
            )
        )
        assertTrue(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                InterruptedIOException("timeout"),
            )
        )
    }

    @Test
    fun doesNotHideNonTimeoutHtmlGenerationErrors() {
        assertFalse(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                IllegalStateException("image input is unavailable"),
            )
        )
    }

    @Test
    fun compactsLargeJsonToolResultForLocalModelContext() {
        val largeOutput = "scroll-item\n".repeat(700)
        val result = JSONObject()
            .put("exit_code", 0)
            .put("output", largeOutput)
            .put("cwd", "/data/data/com.nousresearch.hermesagent/files")
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertTrue(parsed.getInt("_original_chars") > compacted.length)
        assertEquals(0, parsed.getInt("exit_code"))
        assertTrue(parsed.getString("output").contains("scroll-item"))
        assertTrue(parsed.getString("output").contains("hermes context compressed"))
        assertTrue(compacted.length < result.length)
    }

    @Test
    fun compactsDiagnosticArraysButKeepsTopRowsReadable() {
        val networks = JSONArray()
        repeat(60) { index ->
            networks.put(
                JSONObject()
                    .put("ssid", "Lab-$index")
                    .put("bssid_oui", "AC:BC:32")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -30 - index)
                    .put("signal_quality", "excellent")
                    .put("frequency_mhz", 2412 + index)
                    .put("channel", index + 1)
                    .put("channel_width", "20MHz")
                    .put("security_mode", "WPA2")
                    .put("capabilities", "[WPA2-PSK-CCMP][ESS]".repeat(8)),
            )
        }
        val channelRatings = JSONArray()
        repeat(20) { index ->
            channelRatings.put(
                JSONObject()
                    .put("band", "2.4GHz")
                    .put("channel", index + 1)
                    .put("score", 100 - index)
                    .put("rating_label", "good")
                    .put("network_count", index)
                    .put("overlap_count", index + 1)
                    .put("strongest_rssi_dbm", -40 - index)
                    .put("recommendation", "Use if this is the highest-scored row."),
            )
        }
        val accessPointDetails = JSONArray()
        repeat(20) { index ->
            accessPointDetails.put(
                JSONObject()
                    .put("rank", index + 1)
                    .put("display_ssid", "Lab-$index")
                    .put("bssid", "AC:BC:32:00:00:$index")
                    .put("bssid_oui", "AC:BC:32")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -30 - index)
                    .put("signal_quality", "excellent")
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("wifi_standard", "802.11ac")
                    .put("security_mode", "WPA2")
                    .put("estimated_distance_m", 1.2),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "wifi_scan")
            .put("wifi_vendor_count", 1)
            .put("wifi_filter_count", 4)
            .put("wifi_access_point_detail_count", 20)
            .put("wifi_security_summary_count", 1)
            .put("wifi_width_summary_count", 1)
            .put("wifi_standard_summary_count", 1)
            .put("wifi_networks", networks)
            .put("wifi_access_point_details", accessPointDetails)
            .put("wifi_access_point_export", JSONObject().put("format", "json").put("row_count", 20).put("json_array_key", "wifi_access_point_details"))
            .put("wifi_channel_ratings", channelRatings)
            .put(
                "wifi_vendor_summary",
                JSONArray().put(
                    JSONObject()
                        .put("vendor", "Apple")
                        .put("network_count", 60)
                        .put("strongest_rssi_dbm", -30)
                        .put("bssid_ouis", JSONArray().put("AC:BC:32"))
                        .put("recommendation", "Strong nearby vendor group."),
                ),
            )
            .put(
                "wifi_analyzer_filters",
                JSONArray().put(
                    JSONObject()
                        .put("key", "security")
                        .put("label", "Security")
                        .put("options", JSONArray().put(JSONObject().put("value", "WPA2").put("count", 60))),
                ),
            )
            .put(
                "wifi_security_summary",
                JSONArray().put(JSONObject().put("security_mode", "WPA2").put("network_count", 60).put("strongest_rssi_dbm", -30)),
            )
            .put(
                "wifi_channel_width_summary",
                JSONArray().put(JSONObject().put("channel_width", "80MHz").put("channel_width_mhz", 80).put("network_count", 60)),
            )
            .put(
                "wifi_standard_summary",
                JSONArray().put(JSONObject().put("wifi_standard", "802.11ac").put("network_count", 60)),
            )
            .put(
                "recommended_wifi_channels",
                JSONArray().put(JSONObject().put("band", "2.4GHz").put("channel", 11).put("score", 96)),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Wi-Fi Analyzer").put("body", "60 signals")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val wifiNetworks = parsed.getJSONObject("wifi_networks")
        val wifiDetails = parsed.getJSONObject("wifi_access_point_details")
        val wifiRatings = parsed.getJSONObject("wifi_channel_ratings")
        val vendorSummary = parsed.getJSONArray("wifi_vendor_summary")
        val analyzerFilters = parsed.getJSONArray("wifi_analyzer_filters")
        val securitySummary = parsed.getJSONArray("wifi_security_summary")
        val widthSummary = parsed.getJSONArray("wifi_channel_width_summary")
        val standardSummary = parsed.getJSONArray("wifi_standard_summary")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(1, parsed.getInt("wifi_vendor_count"))
        assertEquals(4, parsed.getInt("wifi_filter_count"))
        assertEquals(20, parsed.getInt("wifi_access_point_detail_count"))
        assertEquals("array", wifiNetworks.getString("type"))
        assertEquals(60, wifiNetworks.getInt("original_count"))
        assertEquals(8, wifiNetworks.getJSONArray("items").length())
        assertEquals("Lab-0", wifiNetworks.getJSONArray("items").getJSONObject(0).getString("ssid"))
        assertEquals("Apple", wifiNetworks.getJSONArray("items").getJSONObject(0).getString("bssid_vendor"))
        assertEquals("WPA2", wifiNetworks.getJSONArray("items").getJSONObject(0).getString("security_mode"))
        assertEquals("array", wifiDetails.getString("type"))
        assertEquals(20, wifiDetails.getInt("original_count"))
        assertEquals("802.11ac", wifiDetails.getJSONArray("items").getJSONObject(0).getString("wifi_standard"))
        assertEquals(80, wifiDetails.getJSONArray("items").getJSONObject(0).getInt("channel_width_mhz"))
        assertEquals("array", wifiRatings.getString("type"))
        assertEquals(20, wifiRatings.getInt("original_count"))
        assertEquals(1, wifiRatings.getJSONArray("items").getJSONObject(0).getInt("channel"))
        assertEquals("Apple", vendorSummary.getJSONObject(0).getString("vendor"))
        assertEquals("security", analyzerFilters.getJSONObject(0).getString("key"))
        assertEquals("WPA2", securitySummary.getJSONObject(0).getString("security_mode"))
        assertEquals("80MHz", widthSummary.getJSONObject(0).getString("channel_width"))
        assertEquals("802.11ac", standardSummary.getJSONObject(0).getString("wifi_standard"))
        assertEquals(11, parsed.getJSONArray("recommended_wifi_channels").getJSONObject(0).getInt("channel"))
        assertEquals("Wi-Fi Analyzer", parsed.getJSONArray("cards").getJSONObject(0).getString("title"))
    }

    @Test
    fun compactsWifiSignalHistoryWithoutDroppingTrendMetadata() {
        val history = JSONArray()
        repeat(16) { index ->
            history.put(
                JSONObject()
                    .put("ssid", "Lab-$index")
                    .put("bssid_vendor", "Apple")
                    .put("current_rssi_dbm", -35 - index)
                    .put("average_rssi_dbm", -40 - index)
                    .put("min_rssi_dbm", -50 - index)
                    .put("max_rssi_dbm", -35 - index)
                    .put("trend_db", index)
                    .put("trend_label", "improving")
                    .put("sample_count", 4)
                    .put("rssi_series", JSONArray().put(JSONObject().put("observed_at_ms", index.toLong()).put("rssi_dbm", -35 - index))),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "wifi_scan")
            .put("wifi_history_network_count", 16)
            .put("wifi_signal_history", history)
            .put("cards", JSONArray().put(JSONObject().put("title", "Wi-Fi History").put("body", "16 AP trends")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val signalHistory = parsed.getJSONObject("wifi_signal_history")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(16, parsed.getInt("wifi_history_network_count"))
        assertEquals("array", signalHistory.getString("type"))
        assertEquals(16, signalHistory.getInt("original_count"))
        assertEquals(4, signalHistory.getJSONArray("items").length())
        assertEquals(-35, signalHistory.getJSONArray("items").getJSONObject(0).getInt("current_rssi_dbm"))
        assertEquals("improving", signalHistory.getJSONArray("items").getJSONObject(0).getString("trend_label"))
    }

    @Test
    fun compactsBluetoothAndRadioDiagnosticRowsWithoutDroppingSignalMetadata() {
        val devices = JSONArray()
        repeat(30) { index ->
            devices.put(
                JSONObject()
                    .put("device_name", "Beacon-$index")
                    .put("address", "AA:BB:CC:00:00:$index")
                    .put("rssi_dbm", -40 - index)
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("proximity_label", "near")
                    .put("scan_record", "ff".repeat(200)),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_scan")
            .put("bluetooth_metadata_count", 3)
            .put("bluetooth_service_uuid_count", 1)
            .put("bluetooth_manufacturer_id_count", 1)
            .put("bluetooth_devices", devices)
            .put(
                "bluetooth_metadata_summary",
                JSONArray().put(
                    JSONObject()
                        .put("summary_type", "manufacturer_id")
                        .put("label", "0x004C")
                        .put("count", 30)
                        .put("strongest_rssi_dbm", -40)
                        .put("recommendation", "Manufacturer data advertised nearby."),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Nearby").put("body", "30 devices")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedDevices = parsed.getJSONObject("bluetooth_devices")
        val metadataSummary = parsed.getJSONArray("bluetooth_metadata_summary")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(3, parsed.getInt("bluetooth_metadata_count"))
        assertEquals(1, parsed.getInt("bluetooth_service_uuid_count"))
        assertEquals(1, parsed.getInt("bluetooth_manufacturer_id_count"))
        assertEquals(30, compactedDevices.getInt("original_count"))
        assertEquals("Beacon-0", compactedDevices.getJSONArray("items").getJSONObject(0).getString("device_name"))
        assertEquals(-40, compactedDevices.getJSONArray("items").getJSONObject(0).getInt("rssi_dbm"))
        assertEquals("wearable_health", compactedDevices.getJSONArray("items").getJSONObject(0).getString("device_category"))
        assertEquals("near", compactedDevices.getJSONArray("items").getJSONObject(0).getString("proximity_label"))
        assertEquals("0x004C", metadataSummary.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsSensorCapabilitiesWithoutDroppingHardwareMetadata() {
        val capabilities = JSONArray()
        repeat(20) { index ->
            capabilities.put(
                JSONObject()
                    .put("sensor_type", if (index % 2 == 0) "accelerometer" else "gyroscope")
                    .put("sensor_label", if (index % 2 == 0) "Accelerometer" else "Gyroscope")
                    .put("sensor_name", "Motion Sensor $index")
                    .put("vendor", "Vendor $index")
                    .put("available", true)
                    .put("unit", if (index % 2 == 0) "m/s^2" else "rad/s")
                    .put("maximum_range", 19.6 + index)
                    .put("resolution", 0.001)
                    .put("power_ma", 0.8)
                    .put("min_delay_us", 5000)
                    .put("max_delay_us", 200000)
                    .put("reporting_mode", "continuous")
                    .put("wake_up", index == 0)
                    .put("fifo_max_event_count", 512),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "sensor_snapshot")
            .put("sensor_capability_count", 20)
            .put("motion_sensor_count", 20)
            .put("wake_up_sensor_count", 1)
            .put("sensor_capabilities", capabilities)
            .put("cards", JSONArray().put(JSONObject().put("title", "Sensor Hardware").put("body", "20 sensors")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedCapabilities = parsed.getJSONObject("sensor_capabilities")
        val first = compactedCapabilities.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(20, parsed.getInt("sensor_capability_count"))
        assertEquals(20, parsed.getInt("motion_sensor_count"))
        assertEquals(1, parsed.getInt("wake_up_sensor_count"))
        assertEquals(20, compactedCapabilities.getInt("original_count"))
        assertEquals("accelerometer", first.getString("sensor_type"))
        assertEquals("Motion Sensor 0", first.getString("sensor_name"))
        assertEquals(19.6, first.getDouble("maximum_range"), 0.01)
        assertEquals("continuous", first.getString("reporting_mode"))
        assertTrue(first.getBoolean("wake_up"))
    }

    @Test
    fun compactsCompletedNativeToolRoundsButKeepsLatestAssistantBlock() {
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", "tools enabled"))
            .put(JSONObject().put("role", "user").put("content", "Scroll TikTok, reply to DMs, then draft email"))
        repeat(6) { index ->
            messages
                .put(
                    assistantToolCall(
                        id = "call_$index",
                        name = "android_ui_tool",
                    ),
                )
                .put(
                    JSONObject()
                        .put("role", "tool")
                        .put("tool_call_id", "call_$index")
                        .put("name", "android_ui_tool")
                        .put(
                            "content",
                            NativeToolContextCompressor.compactToolResult(
                                "Scrolled feed $index\n" + "visible row $index ".repeat(1_800),
                            ),
                        ),
                )
        }

        val compacted = NativeToolContextCompressor.compactMessages(messages)

        assertTrue(compacted.length() < messages.length())
        assertEquals("system", compacted.getJSONObject(0).getString("role"))
        assertEquals("user", compacted.getJSONObject(1).getString("role"))
        assertTrue(compacted.getJSONObject(2).getString("content").contains("compacted prior native tool context"))
        val latestAssistant = compacted.getJSONObject(compacted.length() - 2)
        assertEquals("assistant", latestAssistant.getString("role"))
        assertEquals("call_5", latestAssistant.getJSONArray("tool_calls").getJSONObject(0).getString("id"))
        assertEquals("tool", compacted.getJSONObject(compacted.length() - 1).getString("role"))
    }

    private fun assistantToolCall(id: String, name: String): JSONObject {
        return JSONObject()
            .put("role", "assistant")
            .put("content", JSONObject.NULL)
            .put(
                "tool_calls",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("id", id)
                            .put("type", "function")
                            .put(
                                "function",
                                JSONObject()
                                    .put("name", name)
                                    .put("arguments", JSONObject().put("action", "scroll_forward").toString()),
                            ),
                    ),
            )
    }
}
