package com.nousresearch.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
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
    fun extractsExplicitAndroidDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_analyzer_report refresh=false max_results=12",
        )

        requireNotNull(parsed)
        assertEquals("wifi_analyzer_report", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals(12, parsed.getInt("max_results"))
    }

    @Test
    fun extractsExplicitSocCompatibilityDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=soc_compatibility_report",
        )

        requireNotNull(parsed)
        assertEquals("soc_compatibility_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitWifiChannelUtilizationDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_channel_utilization refresh=false",
        )

        requireNotNull(parsed)
        assertEquals("wifi_channel_utilization", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
    }

    @Test
    fun extractsExplicitBluetoothSignalHistoryDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=bluetooth_signal_history refresh=false",
        )

        requireNotNull(parsed)
        assertEquals("bluetooth_signal_history", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
    }

    @Test
    fun extractsExplicitMotionSensorHistoryDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=motion_sensor_history sample=true sensor_types=accelerometer,gyroscope",
        )

        requireNotNull(parsed)
        assertEquals("motion_sensor_history", parsed.getString("action"))
        assertTrue(parsed.getBoolean("sample"))
        assertEquals("accelerometer,gyroscope", parsed.getString("sensor_types"))
    }

    @Test
    fun extractsExplicitRadioAnalyzerDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=radio_analyzer_report",
        )

        requireNotNull(parsed)
        assertEquals("radio_analyzer_report", parsed.getString("action"))
    }

    @Test
    fun ignoresUnknownExplicitAndroidDiagnosticActions() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=network_intrusion",
        )

        assertNull(parsed)
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
        val channelUtilization = JSONArray()
        repeat(20) { index ->
            channelUtilization.put(
                JSONObject()
                    .put("band", "2.4GHz")
                    .put("channel", index + 1)
                    .put("channel_pressure_score", 90 - index)
                    .put("utilization_label", "crowded")
                    .put("network_count", index + 1)
                    .put("overlap_count", index + 2)
                    .put("strongest_rssi_dbm", -35 - index)
                    .put("average_rssi_dbm", -45 - index)
                    .put("max_channel_width_mhz", 40)
                    .put("wide_channel_count", 0)
                    .put("security_modes", JSONArray().put("WPA2"))
                    .put("sample_ssids", JSONArray().put("Lab-$index"))
                    .put("recommendation", "Crowded channel."),
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
            .put("wifi_channel_utilization_count", 20)
            .put("wifi_security_summary_count", 1)
            .put("wifi_width_summary_count", 1)
            .put("wifi_standard_summary_count", 1)
            .put("wifi_networks", networks)
            .put("wifi_access_point_details", accessPointDetails)
            .put("wifi_access_point_export", JSONObject().put("format", "json").put("row_count", 20).put("json_array_key", "wifi_access_point_details"))
            .put("wifi_channel_ratings", channelRatings)
            .put("wifi_channel_utilization", channelUtilization)
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
        val wifiUtilization = parsed.getJSONObject("wifi_channel_utilization")
        val vendorSummary = parsed.getJSONArray("wifi_vendor_summary")
        val analyzerFilters = parsed.getJSONArray("wifi_analyzer_filters")
        val securitySummary = parsed.getJSONArray("wifi_security_summary")
        val widthSummary = parsed.getJSONArray("wifi_channel_width_summary")
        val standardSummary = parsed.getJSONArray("wifi_standard_summary")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(1, parsed.getInt("wifi_vendor_count"))
        assertEquals(4, parsed.getInt("wifi_filter_count"))
        assertEquals(20, parsed.getInt("wifi_access_point_detail_count"))
        assertEquals(20, parsed.getInt("wifi_channel_utilization_count"))
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
        assertEquals("array", wifiUtilization.getString("type"))
        assertEquals(20, wifiUtilization.getInt("original_count"))
        assertEquals(90, wifiUtilization.getJSONArray("items").getJSONObject(0).getInt("channel_pressure_score"))
        assertEquals("crowded", wifiUtilization.getJSONArray("items").getJSONObject(0).getString("utilization_label"))
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
    fun compactsAgentEnvironmentReportWithoutDroppingReadinessRows() {
        val capabilities = JSONArray()
        repeat(20) { index ->
            capabilities.put(
                JSONObject()
                    .put("category", "capability_$index")
                    .put("label", "Capability $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed capability row $index")
                    .put("recommendation", "Use the matching native tool.")
                    .put("fraction", 0.8),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_environment_report")
            .put("agent_capability_count", 20)
            .put("ready_capability_count", 10)
            .put("kai_parity_count", 1)
            .put("workflow_readiness_count", 1)
            .put("agent_capability_matrix", capabilities)
            .put(
                "kai_parity_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "kai_parity")
                        .put("label", "Autonomous heartbeat")
                        .put("ready", true)
                        .put("value_label", "30s interval")
                        .put("parity_source", "Kai autonomous heartbeat"),
                ),
            )
            .put(
                "workflow_readiness_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "wireless_workflow")
                        .put("label", "Analyze nearby Wi-Fi")
                        .put("ready", true)
                        .put("value_label", "call wifi_ap_details"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Agent Environment").put("body", "20 rows")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val capabilityMatrix = parsed.getJSONObject("agent_capability_matrix")
        val kaiParity = parsed.getJSONArray("kai_parity_matrix")
        val readiness = parsed.getJSONArray("workflow_readiness_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(20, parsed.getInt("agent_capability_count"))
        assertEquals("array", capabilityMatrix.getString("type"))
        assertEquals(20, capabilityMatrix.getInt("original_count"))
        assertEquals("Capability 0", capabilityMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals(true, capabilityMatrix.getJSONArray("items").getJSONObject(0).getBoolean("ready"))
        assertEquals("Autonomous heartbeat", kaiParity.getJSONObject(0).getString("label"))
        assertEquals("Analyze nearby Wi-Fi", readiness.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsSignalAwarenessReportWithoutDroppingRoutesOrConstraints() {
        val awareness = JSONArray()
        repeat(18) { index ->
            awareness.put(
                JSONObject()
                    .put("category", "signal_awareness")
                    .put("label", "Signal row $index")
                    .put("ready", index % 3 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed signal awareness row $index")
                    .put("recommendation", "Use the right scanner.")
                    .put("tool_action", "wifi_ap_details")
                    .put("fraction", 0.75),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "signal_awareness_report")
            .put("signal_awareness_count", 18)
            .put("ready_signal_awareness_count", 6)
            .put("signal_workflow_route_count", 1)
            .put("signal_constraint_count", 1)
            .put("cached_wifi_history_network_count", 3)
            .put("signal_awareness_matrix", awareness)
            .put(
                "signal_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "signal_route")
                        .put("label", "Route Wi-Fi analyzer work")
                        .put("ready", true)
                        .put("value_label", "wifi_ap_details")
                        .put("tool_action", "wifi_ap_details"),
                ),
            )
            .put(
                "signal_constraint_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "signal_constraint")
                        .put("label", "AM/FM tuner public API")
                        .put("ready", false)
                        .put("value_label", "not public")
                        .put("constraint_type", "hardware_api"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Signal Awareness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val awarenessMatrix = parsed.getJSONObject("signal_awareness_matrix")
        val routes = parsed.getJSONArray("signal_workflow_routes")
        val constraints = parsed.getJSONArray("signal_constraint_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("signal_awareness_count"))
        assertEquals(3, parsed.getInt("cached_wifi_history_network_count"))
        assertEquals("array", awarenessMatrix.getString("type"))
        assertEquals("Signal row 0", awarenessMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("wifi_ap_details", awarenessMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route Wi-Fi analyzer work", routes.getJSONObject(0).getString("label"))
        assertEquals("AM/FM tuner public API", constraints.getJSONObject(0).getString("label"))
        assertEquals("hardware_api", constraints.getJSONObject(0).getString("constraint_type"))
    }

    @Test
    fun compactsSocCompatibilityReportWithoutDroppingBackendPolicyRows() {
        val backend = JSONArray()
        repeat(18) { index ->
            backend.put(
                JSONObject()
                    .put("category", "soc_backend_parity")
                    .put("label", "SOC backend row $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "backend value $index")
                    .put("detail", "Detailed SOC backend compatibility row $index with MediaTek Mali PowerVR coverage")
                    .put("recommendation", "Use soc_compatibility_report before local inference decisions.")
                    .put("tool_action", "soc_compatibility_report")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "soc_compatibility_report")
            .put("soc_backend_feature_count", 18)
            .put("ready_soc_backend_feature_count", 9)
            .put("soc_backend_route_count", 1)
            .put("soc_backend_constraint_count", 1)
            .put("likely_mediatek", true)
            .put("likely_snapdragon", false)
            .put("soc_backend_matrix", backend)
            .put(
                "soc_backend_policy_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "soc_backend_route")
                        .put("label", "Route SOC compatibility report")
                        .put("ready", true)
                        .put("value_label", "soc_compatibility_report")
                        .put("tool_action", "soc_compatibility_report"),
                ),
            )
            .put(
                "soc_backend_constraint_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "soc_backend_constraint")
                        .put("label", "Avoid Adreno-only assumptions")
                        .put("ready", true)
                        .put("value_label", "SOC-neutral")
                        .put("constraint_type", "soc_policy"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "SOC Compatibility").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val backendMatrix = parsed.getJSONObject("soc_backend_matrix")
        val routes = parsed.getJSONArray("soc_backend_policy_routes")
        val constraints = parsed.getJSONArray("soc_backend_constraint_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("soc_backend_feature_count"))
        assertEquals(9, parsed.getInt("ready_soc_backend_feature_count"))
        assertEquals(true, parsed.getBoolean("likely_mediatek"))
        assertEquals("array", backendMatrix.getString("type"))
        assertEquals("SOC backend row 0", backendMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("soc_compatibility_report", backendMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route SOC compatibility report", routes.getJSONObject(0).getString("label"))
        assertEquals("Avoid Adreno-only assumptions", constraints.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsWifiAnalyzerReportWithoutDroppingPolicyRoutes() {
        val features = JSONArray()
        repeat(18) { index ->
            features.put(
                JSONObject()
                    .put("category", "wifi_analyzer_parity")
                    .put("label", "Wi-Fi analyzer feature $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed Wi-Fi analyzer feature row $index")
                    .put("recommendation", "Route through the matching Wi-Fi diagnostic action.")
                    .put("feature_source", "WiFiAnalyzer parity")
                    .put("tool_action", "wifi_channel_rating")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "wifi_analyzer_report")
            .put("wifi_analyzer_feature_count", 18)
            .put("ready_wifi_analyzer_feature_count", 9)
            .put("wifi_analyzer_workflow_route_count", 1)
            .put("wifi_scan_policy_count", 1)
            .put("wifi_analyzer_feature_matrix", features)
            .put(
                "wifi_analyzer_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "wifi_analyzer_route")
                        .put("label", "Route best-channel analysis")
                        .put("ready", true)
                        .put("value_label", "wifi_channel_rating")
                        .put("tool_action", "wifi_channel_rating"),
                ),
            )
            .put(
                "wifi_scan_policy_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "wifi_scan_policy")
                        .put("label", "Android scan throttling")
                        .put("ready", true)
                        .put("value_label", "passive by default")
                        .put("constraint_type", "platform_policy")
                        .put("permission_gate", "android_wifi_scan"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Wi-Fi Analyzer Readiness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val featureMatrix = parsed.getJSONObject("wifi_analyzer_feature_matrix")
        val routes = parsed.getJSONArray("wifi_analyzer_workflow_routes")
        val policies = parsed.getJSONArray("wifi_scan_policy_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("wifi_analyzer_feature_count"))
        assertEquals(9, parsed.getInt("ready_wifi_analyzer_feature_count"))
        assertEquals("array", featureMatrix.getString("type"))
        assertEquals("Wi-Fi analyzer feature 0", featureMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("WiFiAnalyzer parity", featureMatrix.getJSONArray("items").getJSONObject(0).getString("feature_source"))
        assertEquals("wifi_channel_rating", featureMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route best-channel analysis", routes.getJSONObject(0).getString("label"))
        assertEquals("Android scan throttling", policies.getJSONObject(0).getString("label"))
        assertEquals("platform_policy", policies.getJSONObject(0).getString("constraint_type"))
        assertEquals("android_wifi_scan", policies.getJSONObject(0).getString("permission_gate"))
    }

    @Test
    fun compactsBluetoothAnalyzerReportWithoutDroppingPolicyRoutes() {
        val features = JSONArray()
        repeat(18) { index ->
            features.put(
                JSONObject()
                    .put("category", "bluetooth_analyzer_parity")
                    .put("label", "Bluetooth analyzer feature $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed Bluetooth analyzer feature row $index with scan, service UUID, manufacturer, and proximity context.")
                    .put("recommendation", "Route through the matching Bluetooth diagnostic action.")
                    .put("feature_source", "Android BluetoothLeScanner")
                    .put("tool_action", "bluetooth_scan")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_analyzer_report")
            .put("bluetooth_analyzer_feature_count", 18)
            .put("ready_bluetooth_analyzer_feature_count", 9)
            .put("bluetooth_analyzer_workflow_route_count", 1)
            .put("bluetooth_scan_policy_count", 1)
            .put("bluetooth_signal_history_count", 1)
            .put("bluetooth_analyzer_feature_matrix", features)
            .put(
                "bluetooth_analyzer_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_analyzer_route")
                        .put("label", "Route nearby Bluetooth scan")
                        .put("ready", true)
                        .put("value_label", "bluetooth_scan")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                "bluetooth_scan_policy_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_scan_policy")
                        .put("label", "Connect and scan permissions")
                        .put("ready", true)
                        .put("value_label", "connect and scan granted")
                        .put("constraint_type", "permission")
                        .put("permission_gate", "Bluetooth connect and scan"),
                ),
            )
            .put(
                "bluetooth_signal_history",
                JSONArray().put(
                    JSONObject()
                        .put("device_name", "Heart Strap")
                        .put("current_rssi_dbm", -58)
                        .put("average_rssi_dbm", -65)
                        .put("trend_label", "approaching")
                        .put("trend_db", 14),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Analyzer Readiness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val featureMatrix = parsed.getJSONObject("bluetooth_analyzer_feature_matrix")
        val routes = parsed.getJSONArray("bluetooth_analyzer_workflow_routes")
        val policies = parsed.getJSONArray("bluetooth_scan_policy_matrix")
        val history = parsed.getJSONArray("bluetooth_signal_history")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("bluetooth_analyzer_feature_count"))
        assertEquals(9, parsed.getInt("ready_bluetooth_analyzer_feature_count"))
        assertEquals("array", featureMatrix.getString("type"))
        assertEquals("Bluetooth analyzer feature 0", featureMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("Android BluetoothLeScanner", featureMatrix.getJSONArray("items").getJSONObject(0).getString("feature_source"))
        assertEquals("bluetooth_scan", featureMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route nearby Bluetooth scan", routes.getJSONObject(0).getString("label"))
        assertEquals("Connect and scan permissions", policies.getJSONObject(0).getString("label"))
        assertEquals("permission", policies.getJSONObject(0).getString("constraint_type"))
        assertEquals("Bluetooth connect and scan", policies.getJSONObject(0).getString("permission_gate"))
        assertEquals("Heart Strap", history.getJSONObject(0).getString("device_name"))
        assertEquals(1, parsed.getInt("bluetooth_signal_history_count"))
    }

    @Test
    fun compactsSensorAnalyzerReportWithoutDroppingPolicyRoutes() {
        val features = JSONArray()
        repeat(18) { index ->
            features.put(
                JSONObject()
                    .put("category", "sensor_analyzer_parity")
                    .put("label", "Sensor analyzer feature $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed Sensor Analyzer row $index with accelerometer, gyroscope, metadata, and sampling-policy context.")
                    .put("recommendation", "Route through the matching sensor diagnostic action.")
                    .put("feature_source", "Android SensorManager")
                    .put("tool_action", "sensor_snapshot")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "sensor_analyzer_report")
            .put("sensor_analyzer_feature_count", 18)
            .put("ready_sensor_analyzer_feature_count", 9)
            .put("sensor_analyzer_workflow_route_count", 1)
            .put("sensor_sampling_policy_count", 1)
            .put(
                "sensor_sampling_status",
                JSONObject()
                    .put("active_sample_requested", false)
                    .put("passive_report_default", true)
                    .put("requested_available_sensor_count", 2),
            )
            .put("sensor_analyzer_feature_matrix", features)
            .put(
                "sensor_analyzer_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "sensor_analyzer_route")
                        .put("label", "Route one-shot motion sample")
                        .put("ready", true)
                        .put("value_label", "sensor_snapshot")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                "sensor_sampling_policy_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "sensor_sampling_policy")
                        .put("label", "Passive report default")
                        .put("ready", true)
                        .put("value_label", "no live sample")
                        .put("constraint_type", "sampling_cadence"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Sensor Analyzer Readiness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val featureMatrix = parsed.getJSONObject("sensor_analyzer_feature_matrix")
        val routes = parsed.getJSONArray("sensor_analyzer_workflow_routes")
        val policies = parsed.getJSONArray("sensor_sampling_policy_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("sensor_analyzer_feature_count"))
        assertEquals(9, parsed.getInt("ready_sensor_analyzer_feature_count"))
        assertFalse(parsed.getJSONObject("sensor_sampling_status").getBoolean("active_sample_requested"))
        assertEquals("array", featureMatrix.getString("type"))
        assertEquals("Sensor analyzer feature 0", featureMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("Android SensorManager", featureMatrix.getJSONArray("items").getJSONObject(0).getString("feature_source"))
        assertEquals("sensor_snapshot", featureMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route one-shot motion sample", routes.getJSONObject(0).getString("label"))
        assertEquals("Passive report default", policies.getJSONObject(0).getString("label"))
        assertEquals("sampling_cadence", policies.getJSONObject(0).getString("constraint_type"))
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
