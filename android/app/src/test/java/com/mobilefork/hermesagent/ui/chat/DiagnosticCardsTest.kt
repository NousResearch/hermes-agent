package com.mobilefork.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DiagnosticCardsTest {
    @Test
    fun activityPreviewKeepsCollapsedRowsCompactButShowsAllCardsWhenExpanded() {
        val cards = (1..6).map { index ->
            DiagnosticCardSummary(
                title = "Signal card $index",
                body = "Agent-visible signal dashboard card $index",
            )
        }

        val collapsed = diagnosticCardsForActivityPreview(cards, expanded = false)
        val expanded = diagnosticCardsForActivityPreview(cards, expanded = true)

        assertEquals(COLLAPSED_ACTIVITY_DIAGNOSTIC_CARD_LIMIT, collapsed.size)
        assertEquals(listOf("Signal card 1", "Signal card 2", "Signal card 3"), collapsed.map { it.title })
        assertEquals(cards, expanded)
        assertEquals(3, hiddenDiagnosticCardCountForActivityPreview(cards, expanded = false))
        assertEquals(0, hiddenDiagnosticCardCountForActivityPreview(cards, expanded = true))
    }

    @Test
    fun activityPreviewPromotesGraphableSignalCardsWhenCollapsed() {
        val cards = listOf(
            DiagnosticCardSummary(
                title = "Tool Catalog",
                body = "Generic tool inventory.",
                graphType = "agent_card_manifest",
            ),
            DiagnosticCardSummary(
                title = "Workflow Routes",
                body = "Route matrix.",
                graphType = "signal_workflow_routes",
            ),
            DiagnosticCardSummary(
                title = "SOC Compatibility",
                body = "SOC backend rows.",
                graphType = "soc_backend_matrix",
            ),
            DiagnosticCardSummary(
                title = "Wi-Fi Channel Graph",
                body = "Wi-Fi channel envelopes.",
                graphType = "wifi_channel_graph",
            ),
            DiagnosticCardSummary(
                title = "Bluetooth Nearby",
                body = "Nearby Bluetooth RSSI.",
                graphType = "bluetooth_rssi",
            ),
            DiagnosticCardSummary(
                title = "AM/FM Signal Graph",
                body = "Radio samples.",
                graphType = "radio_signal_graph",
            ),
        )

        val collapsed = diagnosticCardsForActivityPreview(cards, expanded = false)

        assertEquals(
            listOf("Wi-Fi Channel Graph", "Bluetooth Nearby", "AM/FM Signal Graph"),
            collapsed.map { it.title },
        )
        assertEquals(cards, diagnosticCardsForActivityPreview(cards, expanded = true))
        assertEquals(1, diagnosticCardPreviewPriority(cards[3]))
        assertEquals(2, diagnosticCardPreviewPriority(cards[4]))
        assertEquals(3, diagnosticCardPreviewPriority(cards[5]))
        assertTrue(diagnosticCardPreviewPriority(cards[0]) > diagnosticCardPreviewPriority(cards[3]))
    }

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
    fun parsesWifiChannelGraphRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Wi-Fi Channel Graph")
                        .put("body", "Channel envelopes.")
                        .put("graph_type", "wifi_channel_graph")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("display_ssid", "HermesWide")
                                    .put("ssid", "HermesWide")
                                    .put("rssi_dbm", -38)
                                    .put("frequency_mhz", 5180)
                                    .put("channel", 36)
                                    .put("band", "5GHz")
                                    .put("channel_width", "80MHz")
                                    .put("channel_span_start", 28)
                                    .put("channel_span_end", 44)
                                    .put("overlap_pressure_score", 61)
                                    .put("overlap_network_count", 2)
                                    .put("overlap_sample_ssids", JSONArray().put("HermesNarrow").put("LabAP"))
                                    .put("security_mode", "WPA3")
                                    .put("bssid_vendor", "Apple"),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("HermesWide", row.label)
        assertEquals("-38 dBm", row.valueLabel)
        assertTrue(row.detail.contains("5GHz ch 36"))
        assertTrue(row.detail.contains("span 28-44"))
        assertTrue(row.detail.contains("80MHz"))
        assertTrue(row.detail.contains("61% overlap pressure"))
        assertTrue(row.detail.contains("2 overlaps"))
        assertTrue(row.detail.contains("near HermesNarrow, LabAP"))
        assertTrue(row.fraction > 0.8f)
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
                                    .put("semantic_label", "Apple")
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

        assertEquals("Apple", row.label)
        assertEquals("2 devices", row.valueLabel)
        assertTrue(row.detail.contains("manufacturer id"))
        assertTrue(row.detail.contains("raw 0x004C"))
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
    fun parsesWifiChannelUtilizationRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Wi-Fi Channel Utilization")
                        .put("body", "Observed channel pressure.")
                        .put("graph_type", "wifi_channel_utilization")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("band", "2.4GHz")
                                    .put("channel", 1)
                                    .put("channel_pressure_score", 72)
                                    .put("utilization_label", "crowded")
                                    .put("network_count", 2)
                                    .put("overlap_count", 3)
                                    .put("strongest_rssi_dbm", -36)
                                    .put("average_rssi_dbm", -52)
                                    .put("max_channel_width_mhz", 40)
                                    .put("security_modes", JSONArray().put("WPA3").put("WPA2"))
                                    .put("sample_ssids", JSONArray().put("HermesNet"))
                                    .put("recommendation", "Crowded channel."),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("2.4GHz ch 1", row.label)
        assertEquals("72% busy crowded", row.valueLabel)
        assertTrue(row.detail.contains("3 visible overlap"))
        assertTrue(row.detail.contains("40MHz max width"))
        assertTrue(row.detail.contains("HermesNet"))
        assertTrue(row.fraction > 0.7f)
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
    fun parsesWifiAccessPointDetailAndAnalyzerSummaryRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("title", "Wi-Fi AP Details")
                            .put("body", "AP details.")
                            .put("graph_type", "wifi_access_point_detail")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("display_ssid", "HermesNet")
                                        .put("bssid", "AC:BC:32:12:34:56")
                                        .put("bssid_vendor", "Apple")
                                        .put("rssi_dbm", -41)
                                        .put("frequency_mhz", 5180)
                                        .put("channel", 36)
                                        .put("band", "5GHz")
                                        .put("channel_width", "80MHz")
                                        .put("wifi_standard", "802.11ac")
                                        .put("security_mode", "WPA2")
                                        .put("estimated_distance_m", 1.25),
                                ),
                            ),
                    )
                    .put(
                        JSONObject()
                            .put("title", "Wi-Fi Security")
                            .put("body", "Security groups.")
                            .put("graph_type", "wifi_security_summary")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("security_mode", "WPA2")
                                        .put("network_count", 2)
                                        .put("strongest_rssi_dbm", -41)
                                        .put("bands", JSONArray().put("5GHz"))
                                        .put("channels", JSONArray().put("36"))
                                        .put("recommendation", "WPA2 AP group."),
                                ),
                            ),
                    )
                    .put(
                        JSONObject()
                            .put("title", "Wi-Fi Widths")
                            .put("body", "Width groups.")
                            .put("graph_type", "wifi_channel_width_summary")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("channel_width", "80MHz")
                                        .put("channel_width_mhz", 80)
                                        .put("network_count", 1)
                                        .put("recommendation", "Wide channel group."),
                                ),
                            ),
                    ),
            )
            .toString()

        val cards = extractDiagnosticCards(content)
        val apRow = cards[0].rows.single()
        val securityRow = cards[1].rows.single()
        val widthRow = cards[2].rows.single()

        assertEquals("HermesNet", apRow.label)
        assertEquals("-41 dBm", apRow.valueLabel)
        assertTrue(apRow.detail.contains("802.11ac"))
        assertTrue(apRow.detail.contains("AC:BC:32:12:34:56"))
        assertEquals("WPA2", securityRow.label)
        assertEquals("2 APs", securityRow.valueLabel)
        assertTrue(securityRow.detail.contains("ch 36"))
        assertEquals("80MHz", widthRow.label)
        assertTrue(widthRow.detail.contains("80 MHz effective"))
    }

    @Test
    fun parsesWifiSemanticAndBandCoverageRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("title", "Wi-Fi AP Semantics")
                            .put("body", "AP semantics.")
                            .put("graph_type", "wifi_access_point_semantics")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("display_ssid", "Cafe Guest")
                                        .put("semantic_label", "guest/public hotspot")
                                        .put("security_risk_label", "open_network")
                                        .put("security_mode", "Open")
                                        .put("band", "2.4GHz")
                                        .put("channel", 1)
                                        .put("rssi_dbm", -59)
                                        .put("semantic_tags", JSONArray().put("guest_public_hotspot").put("open_network"))
                                        .put("recommendation", "Treat as public."),
                                ),
                            ),
                    )
                    .put(
                        JSONObject()
                            .put("title", "Wi-Fi Band Coverage")
                            .put("body", "Band rows.")
                            .put("graph_type", "wifi_band_coverage")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("band", "5GHz")
                                        .put("network_count", 2)
                                        .put("visible_channels", JSONArray().put("36").put("40"))
                                        .put("observed_widths", JSONArray().put("80MHz"))
                                        .put("observed_standards", JSONArray().put("802.11ax"))
                                        .put("strongest_rssi_dbm", -42)
                                        .put("recommended_channel", 36)
                                        .put("recommended_score", 88)
                                        .put("recommendation", "Compare wide-channel contention."),
                                ),
                            ),
                    ),
            )
            .toString()

        val cards = extractDiagnosticCards(content)
        val semanticRow = cards[0].rows.single()
        val bandRow = cards[1].rows.single()

        assertEquals("Cafe Guest", semanticRow.label)
        assertTrue(semanticRow.valueLabel.contains("guest/public hotspot"))
        assertTrue(semanticRow.valueLabel.contains("open network"))
        assertTrue(semanticRow.detail.contains("Treat as public"))
        assertEquals("5GHz", bandRow.label)
        assertEquals("2 APs observed", bandRow.valueLabel)
        assertTrue(bandRow.detail.contains("best ch 36 88/100"))
        assertTrue(bandRow.fraction > 0.8f)
    }

    @Test
    fun parsesAgentEnvironmentRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("title", "Agent Environment")
                            .put("body", "Capability matrix.")
                            .put("graph_type", "agent_capability_matrix")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("category", "soc_backend")
                                        .put("label", "SOC and LiteRT backend policy")
                                        .put("ready", true)
                                        .put("value_label", "ARM MediaTek/Mali")
                                        .put("detail", "MediaTek | Mali | arm64-v8a")
                                        .put("recommendation", "GPU-first with CPU fallback.")
                                        .put("fraction", 0.95),
                                ),
                            ),
                    )
                    .put(
                        JSONObject()
                            .put("title", "Kai Parity")
                            .put("body", "Parity rows.")
                            .put("graph_type", "kai_parity_matrix")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("category", "kai_parity")
                                        .put("label", "Autonomous heartbeat")
                                        .put("ready", true)
                                        .put("value_label", "30s interval")
                                        .put("detail", "Operator heartbeat/status rows.")
                                        .put("recommendation", "Use for self-checks.")
                                        .put("fraction", 0.9),
                                ),
                            ),
                    )
                    .put(
                        JSONObject()
                            .put("title", "Kai Operations")
                            .put("body", "Operations rows.")
                            .put("graph_type", "kai_operations_matrix")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("category", "kai_operations")
                                        .put("label", "Tool and MCP bridge route")
                                        .put("ready", true)
                                        .put("value_label", "tool_catalog")
                                        .put("detail", "Terminal, file, UI, diagnostics, and memory tools.")
                                        .put("recommendation", "Call tool_catalog first.")
                                        .put("fraction", 0.85),
                                ),
                            ),
                    )
                    .put(
                        JSONObject()
                            .put("title", "Agent Observation")
                            .put("body", "Observation dashboard.")
                            .put("graph_type", "agent_observation_matrix")
                            .put(
                                "rows",
                                JSONArray().put(
                                    JSONObject()
                                        .put("category", "agent_observation")
                                        .put("label", "Gemma signal dashboard")
                                        .put("ready", true)
                                        .put("value_label", "single observation report")
                                        .put("detail", "Wi-Fi, Bluetooth, sensors, radio, SOC, and Kai rows.")
                                        .put("recommendation", "Use this first.")
                                        .put("fraction", 0.95),
                                ),
                            ),
                    ),
            )
            .toString()

        val cards = extractDiagnosticCards(content)
        val socRow = cards[0].rows.single()
        val heartbeatRow = cards[1].rows.single()
        val operationsRow = cards[2].rows.single()
        val observationRow = cards[3].rows.single()

        assertEquals("SOC and LiteRT backend policy", socRow.label)
        assertEquals("ARM MediaTek/Mali", socRow.valueLabel)
        assertTrue(socRow.detail.contains("soc backend"))
        assertTrue(socRow.detail.contains("GPU-first"))
        assertTrue(socRow.fraction > 0.9f)
        assertEquals("Autonomous heartbeat", heartbeatRow.label)
        assertEquals("30s interval", heartbeatRow.valueLabel)
        assertTrue(heartbeatRow.detail.contains("kai parity"))
        assertEquals("Tool and MCP bridge route", operationsRow.label)
        assertEquals("tool_catalog", operationsRow.valueLabel)
        assertTrue(operationsRow.detail.contains("kai operations"))
        assertTrue(operationsRow.detail.contains("tool_catalog"))
        assertEquals("Gemma signal dashboard", observationRow.label)
        assertEquals("single observation report", observationRow.valueLabel)
        assertTrue(observationRow.detail.contains("agent observation"))
        assertTrue(observationRow.detail.contains("Use this first"))
    }

    @Test
    fun parsesAgentSignalContextFusionRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Signal Context Fusion")
                        .put("body", "Fused signal context.")
                        .put("graph_type", "agent_signal_context_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "agent_signal_context")
                                    .put("label", "Wi-Fi channel and band context")
                                    .put("ready", true)
                                    .put("value_label", "2 AP(s), 3 band row(s)")
                                    .put("detail", "Channel rating and band coverage rows are available.")
                                    .put("recommendation", "Open source card for evidence.")
                                    .put("fraction", 0.9),
                            ),
                        ),
                ),
            )
            .toString()

        val card = extractDiagnosticCards(content).single()
        val row = card.rows.single()

        assertEquals("Signal Context Fusion", card.title)
        assertEquals("agent_signal_context_matrix", card.graphType)
        assertEquals("Wi-Fi channel and band context", row.label)
        assertEquals("2 AP(s), 3 band row(s)", row.valueLabel)
        assertTrue(row.detail.contains("agent signal context"))
        assertTrue(row.detail.contains("Open source card"))
        assertTrue(row.fraction > 0.85f)
    }

    @Test
    fun parsesAgentCardManifestRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Agent Card Manifest")
                        .put("body", "Card routes.")
                        .put("graph_type", "agent_card_manifest")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "agent_card_manifest")
                                    .put("label", "Wi-Fi Channel Graph")
                                    .put("ready", true)
                                    .put("value_label", "wifi_channel_graph via wifi_analyzer_report")
                                    .put("detail", "Card exposes channel rows.")
                                    .put("recommendation", "Open this expandable card for evidence.")
                                    .put("fraction", 0.95),
                            ),
                        ),
                ),
            )
            .toString()

        val card = extractDiagnosticCards(content).single()
        val row = card.rows.single()

        assertEquals("Agent Card Manifest", card.title)
        assertEquals("agent_card_manifest", card.graphType)
        assertEquals("Wi-Fi Channel Graph", row.label)
        assertEquals("wifi_channel_graph via wifi_analyzer_report", row.valueLabel)
        assertTrue(row.detail.contains("agent card manifest"))
        assertTrue(row.detail.contains("Open this expandable card"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesSignalAwarenessRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Signal Awareness")
                        .put("body", "Fused rows.")
                        .put("graph_type", "signal_awareness_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "signal_awareness")
                                    .put("label", "Bluetooth proximity metadata")
                                    .put("ready", true)
                                    .put("value_label", "scan ready")
                                    .put("detail", "Service UUID and manufacturer rows available.")
                                    .put("recommendation", "Use bluetooth_scan.")
                                    .put("fraction", 0.9),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Bluetooth proximity metadata", row.label)
        assertEquals("scan ready", row.valueLabel)
        assertTrue(row.detail.contains("Service UUID"))
        assertTrue(row.fraction > 0.8f)
    }

    @Test
    fun parsesRadioBandRowsWithPublicApiAndExternalHardwareLabels() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Radio Band Plan")
                        .put("body", "Radio rows.")
                        .put("graph_type", "radio_frequency_capability")
                        .put(
                            "rows",
                            JSONArray()
                                .put(
                                    JSONObject()
                                        .put("band", "Wi-Fi 2.4 GHz")
                                        .put("frequency_min_mhz", 2401.0)
                                        .put("frequency_max_mhz", 2484.0)
                                        .put("public_android_scan_supported", true)
                                        .put("access_path", "wifi_channel_utilization")
                                        .put("scan_state", "public_android_metadata_route")
                                        .put("reason", "Android exposes Wi-Fi RSSI and channel metadata."),
                                )
                                .put(
                                    JSONObject()
                                        .put("band", "External SDR / broad RF")
                                        .put("requires_external_hardware", true)
                                        .put("access_path", "USB SDR")
                                        .put("scan_state", "external_receiver_required")
                                        .put("reason", "Receiver bridge required."),
                                ),
                        ),
                ),
            )
            .toString()

        val rows = extractDiagnosticCards(content).single().rows

        assertEquals("Wi-Fi 2.4 GHz", rows[0].label)
        assertEquals("Android API", rows[0].valueLabel)
        assertTrue(rows[0].detail.contains("2401.0-2484.0 MHz"))
        assertTrue(rows[0].detail.contains("wifi_channel_utilization"))
        assertTrue(rows[0].detail.contains("public_android_metadata_route"))
        assertTrue(rows[0].fraction > 0.8f)
        assertEquals("external", rows[1].valueLabel)
        assertTrue(rows[1].detail.contains("external_receiver_required"))
        assertTrue(rows[1].fraction >= 0.4f)
    }

    @Test
    fun parsesRadioReceiverProfileRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Receiver Profiles")
                        .put("body", "Receiver schemas.")
                        .put("graph_type", "radio_receiver_profile")
                        .put(
                            "rows",
                            JSONArray()
                                .put(
                                    JSONObject()
                                        .put("label", "FM station receiver profile")
                                        .put("receiver_id", "fm_vendor_or_sdr")
                                        .put("source_type", "fm_broadcast")
                                        .put("frequency_min_mhz", 87.5)
                                        .put("frequency_max_mhz", 108.0)
                                        .put("vendor_bridge_possible", true)
                                        .put("requires_vendor_bridge", true)
                                        .put("scan_state", "vendor_bridge_required")
                                        .put("route_action", "radio_signal_graph")
                                        .put("access_path", "OEM Broadcast Radio HAL bridge")
                                        .put("graph_row_schema", JSONArray().put("frequency_mhz").put("rds_program_service").put("signal_dbuv_or_rssi_dbm"))
                                        .put("station_metadata_fields", JSONArray().put("frequency_mhz").put("rds_program_service"))
                                        .put("sample_fields", JSONArray().put("frequency_mhz").put("power_db"))
                                        .put("recommendation", "Use this profile as the required FM scan schema.")
                                        .put("fraction", 0.65),
                                ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("FM station receiver profile", row.label)
        assertEquals("vendor bridge", row.valueLabel)
        assertTrue(row.detail.contains("87.5-108.0 MHz"))
        assertTrue(row.detail.contains("route radio_signal_graph"))
        assertTrue(row.detail.contains("vendor_bridge_required"))
        assertTrue(row.detail.contains("schema frequency_mhz"))
        assertTrue(row.detail.contains("Use this profile"))
        assertTrue(row.fraction > 0.6f)
    }

    @Test
    fun parsesRadioSignalGraphRowsForBridgeSamplesAndBandBoundaries() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "AM/FM Signal Graph")
                        .put("body", "Bridge samples.")
                        .put("graph_type", "radio_signal_graph")
                        .put(
                            "rows",
                            JSONArray()
                                .put(
                                    JSONObject()
                                        .put("label", "Hermes FM")
                                        .put("band", "FM broadcast")
                                        .put("frequency_mhz", 99.5)
                                        .put("receiver_id", "fm_vendor_or_sdr")
                                        .put("modulation", "fm")
                                        .put("rssi_dbm", -58)
                                        .put("snr_db", 31)
                                        .put("sampled", true)
                                        .put("scan_state", "bridge_sample_reported")
                                        .put("recommendation", "Use as a receiver-provided sample."),
                                )
                                .put(
                                    JSONObject()
                                        .put("label", "AM broadcast band")
                                        .put("band", "AM broadcast")
                                        .put("frequency_min_khz", 530)
                                        .put("frequency_max_khz", 1700)
                                        .put("receiver_id", "am_vendor_or_sdr")
                                        .put("sampled", false)
                                        .put("value_label", "external receiver required")
                                        .put("scan_state", "external_or_vendor_receiver_required"),
                                ),
                        ),
                ),
            )
            .toString()

        val rows = extractDiagnosticCards(content).single().rows

        assertEquals("Hermes FM", rows[0].label)
        assertEquals("-58 dBm", rows[0].valueLabel)
        assertTrue(rows[0].detail.contains("99.5 MHz"))
        assertTrue(rows[0].detail.contains("receiver fm_vendor_or_sdr"))
        assertTrue(rows[0].detail.contains("SNR 31 dB"))
        assertTrue(rows[0].fraction >= 0.6f)
        assertEquals("AM broadcast band", rows[1].label)
        assertEquals("external receiver required", rows[1].valueLabel)
        assertTrue(rows[1].detail.contains("530-1700 kHz"))
        assertTrue(rows[1].detail.contains("external_or_vendor_receiver_required"))
    }

    @Test
    fun parsesRadioWorkflowRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Radio Signal Routes")
                        .put("body", "Route rows.")
                        .put("graph_type", "radio_signal_workflow_routes")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "radio_signal_route")
                                    .put("label", "Route Wi-Fi spectrum work")
                                    .put("ready", true)
                                    .put("value_label", "wifi_channel_utilization")
                                    .put("detail", "Use Wi-Fi channel utilization for graphable RF data.")
                                    .put("recommendation", "Run wifi_channel_utilization first.")
                                    .put("fraction", 0.9),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Route Wi-Fi spectrum work", row.label)
        assertEquals("wifi_channel_utilization", row.valueLabel)
        assertTrue(row.detail.contains("radio signal route"))
        assertTrue(row.detail.contains("Run wifi_channel_utilization"))
        assertTrue(row.fraction > 0.85f)
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
    fun parsesBluetoothSignalHistoryRowsForExpandableSignalCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Bluetooth Signal History")
                        .put("body", "Signal history.")
                        .put("graph_type", "bluetooth_signal_history")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("device_name", "Heart Strap")
                                    .put("device_type", "le")
                                    .put("device_category", "wearable_health")
                                    .put("proximity_label", "room")
                                    .put("current_rssi_dbm", -58)
                                    .put("average_rssi_dbm", -65)
                                    .put("min_rssi_dbm", -72)
                                    .put("max_rssi_dbm", -58)
                                    .put("trend_db", 14)
                                    .put("trend_label", "approaching")
                                    .put("sample_count", 2)
                                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                                    .put("service_labels", JSONArray().put("Heart Rate"))
                                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                                    .put("manufacturer_names", JSONArray().put("Apple")),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Heart Strap", row.label)
        assertEquals("-58 dBm approaching", row.valueLabel)
        assertTrue(row.detail.contains("wearable_health"))
        assertTrue(row.detail.contains("2 samples"))
        assertTrue(row.detail.contains("avg -65 dBm"))
        assertTrue(row.detail.contains("approaching +14 dB"))
        assertTrue(row.detail.contains("services Heart Rate"))
        assertTrue(row.detail.contains("manufacturers Apple"))
        assertTrue(row.detail.contains("manufacturers 0x004C"))
        assertTrue(row.fraction > 0.5f)
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
    fun parsesMotionSensorHistoryRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Motion Sensor History")
                        .put("body", "Motion trend rows.")
                        .put("graph_type", "motion_sensor_history")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("sensor_type", "accelerometer")
                                    .put("sensor_label", "Accelerometer")
                                    .put("sensor_name", "BMI160 Accelerometer")
                                    .put("vendor", "Bosch")
                                    .put("magnitude_unit", "m/s^2")
                                    .put("sample_count", 3)
                                    .put("current_magnitude", 11.18)
                                    .put("average_magnitude", 10.5)
                                    .put("min_magnitude", 9.81)
                                    .put("max_magnitude", 11.18)
                                    .put("trend_magnitude", 1.37)
                                    .put("trend_label", "increasing")
                                    .put("stability_label", "drifting")
                                    .put("current_values", JSONArray().put(0.0).put(2.0).put(11.0)),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Accelerometer", row.label)
        assertEquals("11.18 m/s^2 increasing", row.valueLabel)
        assertTrue(row.detail.contains("3 samples"))
        assertTrue(row.detail.contains("stability drifting"))
        assertTrue(row.detail.contains("range 9.81..11.18 m/s^2"))
        assertTrue(row.detail.contains("vector 0, 2, 11"))
        assertTrue(row.fraction > 0.5f)
    }

    @Test
    fun parsesMotionPoseEstimateRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Motion Pose Estimate")
                        .put("body", "Pose rows.")
                        .put("graph_type", "motion_pose_estimate")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("pose_type", "device_pose")
                                    .put("label", "Device pose estimate")
                                    .put("value_label", "face up | heading E")
                                    .put("pose_source", "accelerometer+magnetic_field")
                                    .put("source_sensors", JSONArray().put("accelerometer").put("magnetic_field"))
                                    .put("roll_degrees", 0.0)
                                    .put("pitch_degrees", 0.0)
                                    .put("tilt_degrees", 0.0)
                                    .put("azimuth_degrees", 90.0)
                                    .put("heading_label", "E")
                                    .put("confidence_label", "high")
                                    .put("workflow_hint", "Use for heading-aware workflows.")
                                    .put("fraction", 0.9),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Device pose estimate", row.label)
        assertEquals("face up | heading E", row.valueLabel)
        assertTrue(row.detail.contains("source accelerometer+magnetic_field"))
        assertTrue(row.detail.contains("sensors accelerometer, magnetic_field"))
        assertTrue(row.detail.contains("confidence high"))
        assertTrue(row.detail.contains("azimuth 90.0 deg"))
        assertTrue(row.detail.contains("heading E"))
        assertTrue(row.fraction > 0.8f)
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

    @Test
    fun parsesWifiAnalyzerReadinessRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Wi-Fi Analyzer Readiness")
                        .put("body", "Readiness rows.")
                        .put("graph_type", "wifi_analyzer_feature_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "wifi_analyzer_parity")
                                    .put("label", "Channel signal graph")
                                    .put("ready", true)
                                    .put("value_label", "24 channel row(s)")
                                    .put("detail", "Scores nearby channels.")
                                    .put("recommendation", "Use wifi_channel_rating.")
                                    .put("fraction", 0.92),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Channel signal graph", row.label)
        assertEquals("24 channel row(s)", row.valueLabel)
        assertTrue(row.detail.contains("wifi analyzer parity"))
        assertTrue(row.detail.contains("Use wifi_channel_rating"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesBluetoothAnalyzerReadinessRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Bluetooth Analyzer Readiness")
                        .put("body", "Readiness rows.")
                        .put("graph_type", "bluetooth_analyzer_feature_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "bluetooth_analyzer_parity")
                                    .put("label", "RSSI proximity graph")
                                    .put("ready", true)
                                    .put("value_label", "12 device row(s)")
                                    .put("detail", "Bluetooth RSSI proximity rows are available.")
                                    .put("recommendation", "Use bluetooth_scan.")
                                    .put("fraction", 0.91),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("RSSI proximity graph", row.label)
        assertEquals("12 device row(s)", row.valueLabel)
        assertTrue(row.detail.contains("bluetooth analyzer parity"))
        assertTrue(row.detail.contains("Use bluetooth_scan"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesSensorAnalyzerReadinessRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Sensor Analyzer Readiness")
                        .put("body", "Readiness rows.")
                        .put("graph_type", "sensor_analyzer_feature_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "sensor_analyzer_parity")
                                    .put("label", "Gyroscope access")
                                    .put("ready", true)
                                    .put("value_label", "gyroscope ready")
                                    .put("detail", "Gyroscope rows are available.")
                                    .put("recommendation", "Use sensor_snapshot.")
                                    .put("fraction", 0.93),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Gyroscope access", row.label)
        assertEquals("gyroscope ready", row.valueLabel)
        assertTrue(row.detail.contains("sensor analyzer parity"))
        assertTrue(row.detail.contains("Use sensor_snapshot"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesMotionSensorQualityRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Motion Sensor Quality")
                        .put("body", "Quality rows.")
                        .put("graph_type", "motion_sensor_quality")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "motion_sensor_quality")
                                    .put("label", "IMU fusion source coverage")
                                    .put("ready", true)
                                    .put("value_label", "4/6 source(s)")
                                    .put("detail", "accelerometer=true | gyroscope=true | rotation_vector=true")
                                    .put("recommendation", "Use motion_sensor_quality before orientation workflows.")
                                    .put("quality_signal", "fusion_sources")
                                    .put("tool_action", "motion_sensor_quality")
                                    .put("fraction", 0.94),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("IMU fusion source coverage", row.label)
        assertEquals("4/6 source(s)", row.valueLabel)
        assertTrue(row.detail.contains("motion sensor quality"))
        assertTrue(row.detail.contains("Use motion_sensor_quality"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesSocBackendRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "SOC Compatibility")
                        .put("body", "Backend rows.")
                        .put("graph_type", "soc_backend_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "soc_backend_parity")
                                    .put("label", "MediaTek/Mali/PowerVR coverage")
                                    .put("ready", true)
                                    .put("value_label", "MediaTek covered")
                                    .put("detail", "GPU probe plus CPU fallback is available.")
                                    .put("recommendation", "Use soc_compatibility_report.")
                                    .put("fraction", 0.95),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("MediaTek/Mali/PowerVR coverage", row.label)
        assertEquals("MediaTek covered", row.valueLabel)
        assertTrue(row.detail.contains("soc backend parity"))
        assertTrue(row.detail.contains("Use soc_compatibility_report"))
        assertTrue(row.fraction > 0.9f)
    }

    @Test
    fun parsesGpuBackendRiskRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "GPU Backend Risk")
                        .put("body", "Risk rows.")
                        .put("graph_type", "gpu_backend_risk_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "gpu_backend_risk")
                                    .put("label", "Live accelerator acceptance")
                                    .put("ready", true)
                                    .put("value_label", "gpu")
                                    .put("detail", "GPU accepted on MediaTek/Mali.")
                                    .put("recommendation", "Use local_backend_runtime_report.")
                                    .put("risk_level", "low")
                                    .put("risk_score", 10)
                                    .put("fraction", 0.9),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("Live accelerator acceptance", row.label)
        assertEquals("gpu", row.valueLabel)
        assertTrue(row.detail.contains("gpu backend risk"))
        assertTrue(row.detail.contains("Use local_backend_runtime_report"))
        assertTrue(row.fraction > 0.85f)
    }

    @Test
    fun parsesRuntimeBackendRowsForExpandableCards() {
        val content = JSONObject()
            .put(
                "cards",
                JSONArray().put(
                    JSONObject()
                        .put("title", "Runtime Backend Health")
                        .put("body", "Runtime rows.")
                        .put("graph_type", "runtime_backend_matrix")
                        .put(
                            "rows",
                            JSONArray().put(
                                JSONObject()
                                    .put("category", "runtime_backend")
                                    .put("label", "LiteRT-LM /health accelerator")
                                    .put("ready", true)
                                    .put("value_label", "gpu")
                                    .put("detail", "GPU was accepted on ARM MediaTek/Mali.")
                                    .put("recommendation", "Use local_backend_runtime_report.")
                                    .put("source_surface", "/health")
                                    .put("health_url", "http://127.0.0.1:15436/health")
                                    .put("fraction", 0.95),
                            ),
                        ),
                ),
            )
            .toString()

        val row = extractDiagnosticCards(content).single().rows.single()

        assertEquals("LiteRT-LM /health accelerator", row.label)
        assertEquals("gpu", row.valueLabel)
        assertTrue(row.detail.contains("runtime backend"))
        assertTrue(row.detail.contains("Use local_backend_runtime_report"))
        assertTrue(row.fraction > 0.9f)
    }
}
