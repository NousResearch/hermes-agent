package com.mobilefork.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.sqrt

internal data class DiagnosticCardSummary(
    val title: String,
    val body: String,
    val graphType: String? = null,
    val rowCount: Int = 0,
    val rows: List<DiagnosticGraphRow> = emptyList(),
)

internal data class DiagnosticGraphRow(
    val label: String,
    val valueLabel: String,
    val detail: String,
    val fraction: Float,
)

internal const val COLLAPSED_ACTIVITY_DIAGNOSTIC_CARD_LIMIT = 3

internal fun diagnosticCardsForActivityPreview(
    cards: List<DiagnosticCardSummary>,
    expanded: Boolean,
): List<DiagnosticCardSummary> {
    if (expanded) return cards
    return cards
        .withIndex()
        .sortedWith(
            compareBy<IndexedValue<DiagnosticCardSummary>> { diagnosticCardPreviewPriority(it.value) }
                .thenBy { it.index },
        )
        .take(COLLAPSED_ACTIVITY_DIAGNOSTIC_CARD_LIMIT)
        .map { it.value }
}

internal fun hiddenDiagnosticCardCountForActivityPreview(
    cards: List<DiagnosticCardSummary>,
    expanded: Boolean,
): Int {
    return if (expanded) 0 else (cards.size - COLLAPSED_ACTIVITY_DIAGNOSTIC_CARD_LIMIT).coerceAtLeast(0)
}

internal fun extractDiagnosticCards(content: String): List<DiagnosticCardSummary> {
    val parsed = runCatching { JSONObject(content.trim()) }.getOrNull() ?: return emptyList()
    val cards = parsed.optJSONArray("cards") ?: return emptyList()
    return buildList {
        for (index in 0 until cards.length()) {
            val card = cards.optJSONObject(index) ?: continue
            val title = card.optString("title").takeIf { it.isNotBlank() } ?: continue
            val graphType = card.optString("graph_type").takeIf { it.isNotBlank() }
            val rawRows = card.optJSONArray("rows")
            val rows = rawRows?.let { graphRows(graphType, it) }.orEmpty()
            val rowCount = card.optInt("row_count", rows.size).coerceAtLeast(rows.size)
            val body = buildString {
                append(card.optString("body").ifBlank { "Diagnostic card available." })
                if (graphType != null || rowCount > 0) {
                    append(" [")
                    append(graphType ?: "graph")
                    append(", ")
                    append(rowCount)
                    append(" rows]")
                }
            }
            add(
                DiagnosticCardSummary(
                    title = title,
                    body = body,
                    graphType = graphType,
                    rowCount = rowCount,
                    rows = rows,
                ),
            )
        }
    }
}

internal fun diagnosticCardPreviewPriority(card: DiagnosticCardSummary): Int {
    return when (card.graphType) {
        "signal_evidence_matrix",
        "agent_signal_context_matrix",
        "rf_coexistence_matrix",
        "signal_awareness_matrix",
        "agent_self_check_matrix" -> 0
        "wifi_channel_strength",
        "wifi_access_point_detail",
        "wifi_access_point_semantics",
        "wifi_channel_graph",
        "wifi_channel_rating",
        "wifi_channel_utilization",
        "wifi_band_coverage",
        "wifi_vendor_summary",
        "wifi_security_summary",
        "wifi_channel_width_summary",
        "wifi_standard_summary",
        "wifi_connection_link",
        "wifi_signal_history" -> 1
        "bluetooth_rssi",
        "bluetooth_device_detail",
        "bluetooth_metadata_summary",
        "bluetooth_signal_history" -> 2
        "radio_frequency_capability",
        "radio_signal_graph",
        "radio_receiver_bridge_schema",
        "radio_receiver_profile" -> 3
        "sensor_vector",
        "motion_sensor_history",
        "motion_pose_estimate",
        "motion_sensor_quality",
        "sensor_capability" -> 4
        "soc_backend_matrix",
        "soc_backend_policy_routes",
        "soc_backend_constraint_matrix",
        "gpu_backend_risk_matrix",
        "gpu_backend_risk_routes",
        "local_inference_compatibility_matrix",
        "runtime_backend_matrix",
        "runtime_stability_matrix" -> 5
        "agent_capability_matrix",
        "kai_parity_matrix",
        "agent_workflow_readiness",
        "kai_operations_matrix",
        "agent_tool_sandbox_matrix",
        "agent_self_check_routes",
        "agent_observation_matrix",
        "agent_observation_routes",
        "agent_card_manifest" -> 6
        else -> 10
    }
}

private fun graphRows(graphType: String?, rows: JSONArray): List<DiagnosticGraphRow> {
    return buildList {
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            val parsed = when (graphType) {
                "wifi_channel_strength" -> wifiRow(row)
                "wifi_access_point_detail" -> wifiAccessPointDetailRow(row)
                "wifi_access_point_semantics" -> wifiAccessPointSemanticRow(row)
                "wifi_channel_graph" -> wifiChannelGraphRow(row)
                "wifi_channel_rating" -> wifiChannelRatingRow(row)
                "wifi_channel_utilization" -> wifiChannelUtilizationRow(row)
                "wifi_band_coverage" -> wifiBandCoverageRow(row)
                "wifi_vendor_summary" -> wifiVendorSummaryRow(row)
                "wifi_security_summary" -> wifiSecuritySummaryRow(row)
                "wifi_channel_width_summary" -> wifiChannelWidthSummaryRow(row)
                "wifi_standard_summary" -> wifiStandardSummaryRow(row)
                "wifi_signal_history" -> wifiSignalHistoryRow(row)
                "bluetooth_rssi" -> bluetoothRow(row)
                "bluetooth_device_detail" -> bluetoothRow(row)
                "bluetooth_metadata_summary" -> bluetoothMetadataSummaryRow(row)
                "bluetooth_signal_history" -> bluetoothSignalHistoryRow(row)
                "radio_frequency_capability" -> radioRow(row)
                "radio_signal_graph" -> radioSignalGraphRow(row)
                "radio_receiver_bridge_schema" -> radioReceiverProfileRow(row)
                "radio_receiver_profile" -> radioReceiverProfileRow(row)
                "sensor_vector" -> sensorRow(row)
                "motion_sensor_history" -> motionSensorHistoryRow(row)
                "motion_pose_estimate" -> motionPoseEstimateRow(row)
                "motion_sensor_quality" -> capabilityMatrixRow(row)
                "sensor_capability" -> sensorCapabilityRow(row)
                "agent_capability_matrix", "kai_parity_matrix", "agent_workflow_readiness", "kai_operations_matrix", "agent_tool_sandbox_matrix",
                "agent_self_check_matrix", "agent_self_check_routes",
                "agent_observation_matrix", "agent_observation_routes", "agent_signal_context_matrix",
                "signal_evidence_matrix", "signal_evidence_routes",
                "rf_coexistence_matrix", "rf_coexistence_routes",
                "agent_card_manifest",
                "wifi_analyzer_feature_matrix", "wifi_analyzer_workflow_routes", "wifi_scan_policy_matrix", "wifi_connection_link",
                "wifi_filter_application",
                "bluetooth_analyzer_feature_matrix", "bluetooth_analyzer_workflow_routes", "bluetooth_scan_policy_matrix",
                "bluetooth_filter_application",
                "sensor_analyzer_feature_matrix", "sensor_analyzer_workflow_routes", "sensor_sampling_policy_matrix",
                "signal_awareness_matrix", "signal_workflow_routes", "signal_constraint_matrix",
                "radio_signal_feature_matrix", "radio_signal_workflow_routes", "radio_signal_constraint_matrix",
                "gpu_backend_risk_matrix", "gpu_backend_risk_routes",
                "mediatek_readiness_matrix",
                "local_inference_compatibility_matrix",
                "soc_backend_matrix", "soc_backend_policy_routes", "soc_backend_constraint_matrix",
                "runtime_backend_matrix", "runtime_stability_matrix" -> capabilityMatrixRow(row)
                else -> genericRow(row, index)
            }
            if (parsed != null) add(parsed)
        }
    }
}

private fun wifiRow(row: JSONObject): DiagnosticGraphRow? {
    val rssi = row.optNumber("rssi_dbm")?.toInt() ?: row.optNumber("level_dbm")?.toInt() ?: return null
    val ssid = row.optString("ssid").takeIf { it.isNotBlank() } ?: row.optString("bssid").ifBlank { "Wi-Fi" }
    val frequency = row.optNumber("frequency_mhz")?.toInt()
    val channel = row.opt("channel").takeUnless { it == null || it == JSONObject.NULL }?.toString()
    val distance = row.optNumber("estimated_distance_meters")?.toDouble()
    val detail = listOfNotNull(
        channel?.let { "ch $it" },
        frequency?.let { "$it MHz" },
        row.optString("band").takeIf { it.isNotBlank() },
        row.optString("channel_width").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("security_mode").takeIf { it.isNotBlank() },
        row.optString("bssid_vendor").takeIf { it.isNotBlank() && it != "Unknown vendor" },
        distance?.let { "~${formatDecimal(it, 1)} m" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = ssid,
        valueLabel = "$rssi dBm",
        detail = detail.ifBlank { "Wi-Fi signal" },
        fraction = dbmFraction(rssi),
    )
}

private fun wifiAccessPointDetailRow(row: JSONObject): DiagnosticGraphRow? {
    val rssi = row.optNumber("rssi_dbm")?.toInt()
    val ssid = row.optString("display_ssid").takeIf { it.isNotBlank() }
        ?: row.optString("ssid").takeIf { it.isNotBlank() }
        ?: row.optString("bssid").ifBlank { "Wi-Fi AP" }
    val distance = row.optNumber("estimated_distance_m")?.toDouble()
        ?: row.optNumber("estimated_distance_meters")?.toDouble()
    val channel = row.opt("channel").takeUnless { it == null || it == JSONObject.NULL }?.toString()
    val detail = listOfNotNull(
        channel?.let { "ch $it" },
        row.optNumber("frequency_mhz")?.toInt()?.let { "$it MHz" },
        row.optString("band").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("channel_width").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("wifi_standard").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("security_mode").takeIf { it.isNotBlank() },
        row.optString("bssid_vendor").takeIf { it.isNotBlank() && it != "Unknown vendor" },
        row.optString("bssid").takeIf { it.isNotBlank() },
        distance?.let { "~${formatDecimal(it, 1)} m" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = ssid,
        valueLabel = rssi?.let { "$it dBm" } ?: row.optString("signal_quality").ifBlank { "AP" },
        detail = detail.ifBlank { "Wi-Fi access point detail" },
        fraction = rssi?.let(::dbmFraction) ?: 0.25f,
    )
}

private fun wifiAccessPointSemanticRow(row: JSONObject): DiagnosticGraphRow? {
    val ssid = row.optString("display_ssid").takeIf { it.isNotBlank() }
        ?: row.optString("ssid").takeIf { it.isNotBlank() }
        ?: row.optString("bssid").ifBlank { "Wi-Fi AP" }
    val semanticLabel = row.optString("semantic_label").takeIf { it.isNotBlank() } ?: "nearby AP"
    val riskLabel = row.optString("security_risk_label").takeIf { it.isNotBlank() } ?: "unknown_security"
    val rssi = row.optNumber("rssi_dbm")?.toInt()
    val channel = row.opt("channel").takeUnless { it == null || it == JSONObject.NULL }?.toString()
    val detail = listOfNotNull(
        channel?.let { "ch $it" },
        row.optString("band").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("security_mode").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("wifi_standard").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("channel_width").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("bssid_vendor").takeIf { it.isNotBlank() && it != "Unknown vendor" },
        joinJsonStrings(row.optJSONArray("semantic_tags"), 4).takeIf { it.isNotBlank() },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = ssid,
        valueLabel = "$semanticLabel | ${riskLabel.replace('_', ' ')}",
        detail = detail.ifBlank { "Wi-Fi AP semantic label" },
        fraction = when (riskLabel) {
            "open_network", "legacy_weak_security", "wps_attention" -> 0.95f
            "unknown_security" -> 0.7f
            else -> rssi?.let(::dbmFraction) ?: 0.45f
        },
    )
}

private fun wifiChannelGraphRow(row: JSONObject): DiagnosticGraphRow? {
    val channel = row.optNumber("channel")?.toInt() ?: return null
    val rssi = row.optNumber("rssi_dbm")?.toInt() ?: row.optNumber("graph_y_dbm")?.toInt() ?: return null
    val ssid = row.optString("display_ssid").takeIf { it.isNotBlank() }
        ?: row.optString("ssid").takeIf { it.isNotBlank() }
        ?: row.optString("bssid").ifBlank { "Wi-Fi AP" }
    val band = row.optString("band").ifBlank { "Wi-Fi" }
    val spanStart = row.optNumber("channel_span_start")?.toInt()
    val spanEnd = row.optNumber("channel_span_end")?.toInt()
    val pressure = row.optNumber("overlap_pressure_score")?.toInt()
    val overlapCount = row.optNumber("overlap_network_count")?.toInt()
    val detail = listOfNotNull(
        "$band ch $channel",
        spanStart?.let { start -> spanEnd?.let { end -> "span $start-$end" } },
        row.optNumber("frequency_mhz")?.toInt()?.let { "$it MHz" },
        row.optString("channel_width").takeIf { it.isNotBlank() && it != "unknown" },
        pressure?.let { "$it% overlap pressure" },
        overlapCount?.let { "$it overlaps" },
        row.optString("security_mode").takeIf { it.isNotBlank() },
        row.optString("bssid_vendor").takeIf { it.isNotBlank() && it != "Unknown vendor" },
        joinJsonStrings(row.optJSONArray("overlap_sample_ssids"), 3).takeIf { it.isNotBlank() }?.let { "near $it" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = ssid,
        valueLabel = "$rssi dBm",
        detail = detail.ifBlank { "Wi-Fi channel graph envelope" },
        fraction = dbmFraction(rssi),
    )
}

private fun wifiChannelRatingRow(row: JSONObject): DiagnosticGraphRow? {
    val channel = row.optNumber("channel")?.toInt() ?: return null
    val score = row.optNumber("score")?.toInt()?.coerceIn(0, 100) ?: return null
    val band = row.optString("band").ifBlank { "Wi-Fi" }
    val sameChannelCount = row.optNumber("network_count")?.toInt() ?: 0
    val overlapCount = row.optNumber("overlap_count")?.toInt() ?: 0
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val rating = row.optString("rating_label").takeIf { it.isNotBlank() }
    val recommendation = row.optString("recommendation").takeIf { it.isNotBlank() }
    val detail = listOfNotNull(
        "$sameChannelCount same-channel",
        "$overlapCount overlapping",
        strongestRssi?.let { "strongest $it dBm" },
        recommendation,
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = "$band ch $channel",
        valueLabel = listOfNotNull("$score/100", rating).joinToString(" "),
        detail = detail.ifBlank { "Wi-Fi channel rating" },
        fraction = (score / 100f).coerceIn(0.05f, 1f),
    )
}

private fun wifiBandCoverageRow(row: JSONObject): DiagnosticGraphRow? {
    val band = row.optString("band").takeIf { it.isNotBlank() } ?: return null
    val count = row.optNumber("network_count")?.toInt() ?: 0
    val recommendedChannel = row.opt("recommended_channel").takeUnless { it == null || it == JSONObject.NULL }?.toString()
    val score = row.optNumber("recommended_score")?.toInt()
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val detail = listOfNotNull(
        joinJsonStrings(row.optJSONArray("visible_channels"), 6).takeIf { it.isNotBlank() }?.let { "ch $it" },
        joinJsonStrings(row.optJSONArray("observed_widths"), 4).takeIf { it.isNotBlank() },
        joinJsonStrings(row.optJSONArray("observed_standards"), 4).takeIf { it.isNotBlank() },
        strongestRssi?.let { "strongest $it dBm" },
        row.optNumber("security_attention_count")?.toInt()?.takeIf { it > 0 }?.let { "$it security attention" },
        row.optNumber("hidden_ssid_count")?.toInt()?.takeIf { it > 0 }?.let { "$it hidden" },
        recommendedChannel?.let { "best ch $it${score?.let { value -> " $value/100" } ?: ""}" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = band,
        valueLabel = if (count > 0) "$count AP${if (count == 1) "" else "s"} observed" else "not observed",
        detail = detail.ifBlank { "Wi-Fi band coverage" },
        fraction = when {
            count <= 0 -> 0.15f
            score != null -> (score / 100f).coerceIn(0.2f, 1f)
            strongestRssi != null -> dbmFraction(strongestRssi)
            else -> (count / 8f).coerceIn(0.2f, 1f)
        },
    )
}

private fun wifiChannelUtilizationRow(row: JSONObject): DiagnosticGraphRow? {
    val channel = row.optNumber("channel")?.toInt() ?: return null
    val score = row.optNumber("channel_pressure_score")?.toInt()?.coerceIn(0, 100)
        ?: row.optNumber("score")?.toInt()?.coerceIn(0, 100)
        ?: return null
    val band = row.optString("band").ifBlank { "Wi-Fi" }
    val sameChannelCount = row.optNumber("network_count")?.toInt() ?: 0
    val overlapCount = row.optNumber("overlap_count")?.toInt() ?: 0
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val averageRssi = row.optNumber("average_rssi_dbm")?.toInt()
    val maxWidth = row.optNumber("max_channel_width_mhz")?.toInt()
    val label = row.optString("utilization_label").takeIf { it.isNotBlank() }
    val detail = listOfNotNull(
        "$sameChannelCount same-channel",
        "$overlapCount visible overlap",
        strongestRssi?.let { "strongest $it dBm" },
        averageRssi?.let { "avg $it dBm" },
        maxWidth?.let { "${it}MHz max width" },
        joinJsonStrings(row.optJSONArray("security_modes"), 3).takeIf { it.isNotBlank() },
        joinJsonStrings(row.optJSONArray("sample_ssids"), 3).takeIf { it.isNotBlank() },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = "$band ch $channel",
        valueLabel = listOfNotNull("$score% busy", label?.replace('_', ' ')).joinToString(" "),
        detail = detail.ifBlank { "Wi-Fi channel utilization" },
        fraction = (score / 100f).coerceIn(0.05f, 1f),
    )
}

private fun wifiVendorSummaryRow(row: JSONObject): DiagnosticGraphRow? {
    val vendor = row.optString("vendor").takeIf { it.isNotBlank() } ?: return null
    val count = row.optNumber("network_count")?.toInt() ?: return null
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val ouiLabel = row.optJSONArray("bssid_ouis")
        ?.let { values ->
            buildList {
                for (index in 0 until minOf(values.length(), 3)) {
                    values.optString(index).takeIf { it.isNotBlank() }?.let(::add)
                }
            }.joinToString(", ")
        }
        .orEmpty()
    val detail = listOfNotNull(
        ouiLabel.takeIf { it.isNotBlank() }?.let { "OUI $it" },
        strongestRssi?.let { "strongest $it dBm" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = vendor,
        valueLabel = "$count AP${if (count == 1) "" else "s"}",
        detail = detail.ifBlank { "Wi-Fi vendor/OUI group" },
        fraction = strongestRssi?.let(::dbmFraction) ?: (count / 8f).coerceIn(0.1f, 1f),
    )
}

private fun wifiSecuritySummaryRow(row: JSONObject): DiagnosticGraphRow? {
    val security = row.optString("security_mode").takeIf { it.isNotBlank() } ?: return null
    val count = row.optNumber("network_count")?.toInt() ?: return null
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val detail = listOfNotNull(
        joinJsonStrings(row.optJSONArray("bands"), 3).takeIf { it.isNotBlank() },
        joinJsonStrings(row.optJSONArray("channels"), 4).takeIf { it.isNotBlank() }?.let { "ch $it" },
        strongestRssi?.let { "strongest $it dBm" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = security,
        valueLabel = "$count AP${if (count == 1) "" else "s"}",
        detail = detail.ifBlank { "Wi-Fi security group" },
        fraction = strongestRssi?.let(::dbmFraction) ?: (count / 8f).coerceIn(0.1f, 1f),
    )
}

private fun wifiChannelWidthSummaryRow(row: JSONObject): DiagnosticGraphRow? {
    val width = row.optString("channel_width").takeIf { it.isNotBlank() } ?: return null
    val count = row.optNumber("network_count")?.toInt() ?: return null
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val detail = listOfNotNull(
        row.optNumber("channel_width_mhz")?.toInt()?.let { "$it MHz effective" },
        joinJsonStrings(row.optJSONArray("bands"), 3).takeIf { it.isNotBlank() },
        joinJsonStrings(row.optJSONArray("channels"), 4).takeIf { it.isNotBlank() }?.let { "ch $it" },
        strongestRssi?.let { "strongest $it dBm" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = width,
        valueLabel = "$count AP${if (count == 1) "" else "s"}",
        detail = detail.ifBlank { "Wi-Fi channel width group" },
        fraction = strongestRssi?.let(::dbmFraction) ?: (count / 8f).coerceIn(0.1f, 1f),
    )
}

private fun wifiStandardSummaryRow(row: JSONObject): DiagnosticGraphRow? {
    val standard = row.optString("wifi_standard").takeIf { it.isNotBlank() } ?: return null
    val count = row.optNumber("network_count")?.toInt() ?: return null
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val detail = listOfNotNull(
        joinJsonStrings(row.optJSONArray("bands"), 3).takeIf { it.isNotBlank() },
        joinJsonStrings(row.optJSONArray("sample_widths"), 3).takeIf { it.isNotBlank() },
        strongestRssi?.let { "strongest $it dBm" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = standard,
        valueLabel = "$count AP${if (count == 1) "" else "s"}",
        detail = detail.ifBlank { "Wi-Fi standard group" },
        fraction = strongestRssi?.let(::dbmFraction) ?: (count / 8f).coerceIn(0.1f, 1f),
    )
}

private fun wifiSignalHistoryRow(row: JSONObject): DiagnosticGraphRow? {
    val currentRssi = row.optNumber("current_rssi_dbm")?.toInt() ?: return null
    val ssid = row.optString("ssid").takeIf { it.isNotBlank() } ?: row.optString("bssid").ifBlank { "Wi-Fi" }
    val sampleCount = row.optNumber("sample_count")?.toInt() ?: 1
    val averageRssi = row.optNumber("average_rssi_dbm")?.toInt()
    val minRssi = row.optNumber("min_rssi_dbm")?.toInt()
    val maxRssi = row.optNumber("max_rssi_dbm")?.toInt()
    val trendDb = row.optNumber("trend_db")?.toInt()
    val detail = listOfNotNull(
        row.opt("channel").takeUnless { it == null || it == JSONObject.NULL }?.toString()?.let { "ch $it" },
        row.optString("band").takeIf { it.isNotBlank() },
        row.optString("bssid_vendor").takeIf { it.isNotBlank() && it != "Unknown vendor" },
        "$sampleCount sample${if (sampleCount == 1) "" else "s"}",
        averageRssi?.let { "avg $it dBm" },
        if (minRssi != null && maxRssi != null) "range $minRssi..$maxRssi dBm" else null,
        trendDb?.let { "${row.optString("trend_label").ifBlank { "trend" }} ${if (it > 0) "+" else ""}$it dB" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = ssid,
        valueLabel = "$currentRssi dBm ${row.optString("trend_label").ifBlank { "stable" }}",
        detail = detail.ifBlank { "Wi-Fi signal history" },
        fraction = dbmFraction(currentRssi),
    )
}

private fun bluetoothRow(row: JSONObject): DiagnosticGraphRow? {
    val rssi = row.optNumber("rssi_dbm")?.toInt()
    val label = row.optString("display_label").takeIf { it.isNotBlank() && it != "<unnamed>" }
        ?: row.optString("advertised_name").takeIf { it.isNotBlank() && it != "<unnamed>" }
        ?: row.optString("device_name").takeIf { it.isNotBlank() && it != "<unnamed>" }
        ?: row.optString("address").ifBlank { "Bluetooth" }
    val detail = listOfNotNull(
        row.optString("semantic_label").takeIf { it.isNotBlank() && it != "bluetooth device" },
        row.optString("device_type").takeIf { it.isNotBlank() },
        row.optString("device_category").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("bond_state").takeIf { it.isNotBlank() },
        if (row.optBoolean("paired", false)) "paired" else null,
        row.optString("proximity_label").takeIf { it.isNotBlank() },
        row.optNumber("estimated_distance_meters")?.toDouble()?.let { "~${formatDecimal(it, 1)} m" },
        joinJsonStrings(row.optJSONArray("service_labels"), 2).takeIf { it.isNotBlank() }?.let { "services $it" },
        joinJsonStrings(row.optJSONArray("manufacturer_names"), 2).takeIf { it.isNotBlank() }?.let { "manufacturers $it" },
        row.optNumber("service_uuid_count")?.toInt()?.takeIf { it > 0 }?.let { "$it services" },
        row.optNumber("manufacturer_data_count")?.toInt()?.takeIf { it > 0 }?.let { "$it manufacturer records" },
        row.optNumber("scan_record_bytes")?.toInt()?.let { "$it scan bytes" },
        row.optNumber("metadata_completeness_score")?.toInt()?.let { "$it% metadata" },
        row.optString("evidence_summary").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = rssi?.let { "$it dBm" } ?: "paired",
        detail = detail.ifBlank { "Bluetooth device metadata" },
        fraction = rssi?.let(::dbmFraction) ?: 0.45f,
    )
}

private fun bluetoothMetadataSummaryRow(row: JSONObject): DiagnosticGraphRow? {
    val rawLabel = row.optString("label").takeIf { it.isNotBlank() } ?: return null
    val semanticLabel = row.optString("semantic_label").takeIf { it.isNotBlank() && it != "null" }
    val label = semanticLabel ?: rawLabel
    val count = row.optNumber("count")?.toInt() ?: return null
    val summaryType = row.optString("summary_type").ifBlank { "metadata" }
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val detail = listOfNotNull(
        summaryType.replace('_', ' '),
        semanticLabel?.let { "raw $rawLabel" },
        row.optNumber("paired_count")?.toInt()?.takeIf { it > 0 }?.let { "$it paired" },
        row.optNumber("connectable_count")?.toInt()?.takeIf { it > 0 }?.let { "$it connectable" },
        strongestRssi?.let { "strongest $it dBm" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = "$count device${if (count == 1) "" else "s"}",
        detail = detail.ifBlank { "Bluetooth metadata group" },
        fraction = strongestRssi?.let(::dbmFraction) ?: (count / 8f).coerceIn(0.1f, 1f),
    )
}

private fun bluetoothSignalHistoryRow(row: JSONObject): DiagnosticGraphRow? {
    val currentRssi = row.optNumber("current_rssi_dbm")?.toInt() ?: return null
    val sampleCount = row.optNumber("sample_count")?.toInt() ?: 0
    val label = row.optString("device_name").takeIf { it.isNotBlank() && it != "<unnamed>" }
        ?: row.optString("advertised_name").takeIf { it.isNotBlank() }
        ?: row.optString("address").ifBlank { "Bluetooth device" }
    val averageRssi = row.optNumber("average_rssi_dbm")?.toInt()
    val minRssi = row.optNumber("min_rssi_dbm")?.toInt()
    val maxRssi = row.optNumber("max_rssi_dbm")?.toInt()
    val trendDb = row.optNumber("trend_db")?.toInt()
    val detail = listOfNotNull(
        row.optString("device_type").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("device_category").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("proximity_label").takeIf { it.isNotBlank() },
        "$sampleCount sample${if (sampleCount == 1) "" else "s"}",
        averageRssi?.let { "avg $it dBm" },
        if (minRssi != null && maxRssi != null) "range $minRssi..$maxRssi dBm" else null,
        trendDb?.let { "${row.optString("trend_label").ifBlank { "trend" }} ${if (it > 0) "+" else ""}$it dB" },
        joinJsonStrings(row.optJSONArray("service_labels"), 2).takeIf { it.isNotBlank() }?.let { "services $it" },
        joinJsonStrings(row.optJSONArray("manufacturer_names"), 2).takeIf { it.isNotBlank() }?.let { "manufacturers $it" },
        joinJsonStrings(row.optJSONArray("service_uuids"), 2).takeIf { it.isNotBlank() }?.let { "services $it" },
        joinJsonStrings(row.optJSONArray("manufacturer_ids"), 2).takeIf { it.isNotBlank() }?.let { "manufacturers $it" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = "$currentRssi dBm ${row.optString("trend_label").ifBlank { "stable" }}",
        detail = detail.ifBlank { "Bluetooth signal history" },
        fraction = dbmFraction(currentRssi),
    )
}

private fun radioRow(row: JSONObject): DiagnosticGraphRow {
    val band = row.optString("band").ifBlank { "Radio band" }
    val supported = row.optBoolean("supported", false)
    val sampled = row.optBoolean("sampled", false)
    val publicAndroidScan = row.optBoolean("public_android_scan_supported", false)
    val hardwareHint = row.optBoolean("hardware_hint_supported", false)
    val requiresExternal = row.optBoolean("requires_external_hardware", false)
    val valueLabel = when {
        sampled -> "sampled"
        publicAndroidScan -> "Android API"
        hardwareHint -> "vendor hint"
        supported -> "available"
        requiresExternal -> "external"
        else -> "limited"
    }
    val range = radioRangeLabel(row)
    val reason = row.optString("reason").takeIf { it.isNotBlank() }
    val accessPath = row.optString("access_path").takeIf { it.isNotBlank() }
    val scanState = row.optString("scan_state").takeIf { it.isNotBlank() }
    return DiagnosticGraphRow(
        label = band,
        valueLabel = valueLabel,
        detail = listOfNotNull(range, accessPath, scanState, reason).joinToString(" | ").ifBlank { "Radio capability" },
        fraction = when {
            sampled -> 1f
            publicAndroidScan -> 0.85f
            supported -> 0.75f
            hardwareHint -> 0.55f
            requiresExternal -> 0.45f
            else -> 0.15f
        },
    )
}

private fun radioSignalGraphRow(row: JSONObject): DiagnosticGraphRow {
    val sampled = row.optBoolean("sampled", false)
    val label = row.optString("label").ifBlank {
        row.optString("station_label").ifBlank {
            row.optString("band").ifBlank { "Radio signal row" }
        }
    }
    val valueLabel = row.optString("value_label").ifBlank {
        when {
            sampled -> {
                val rssiDbm = row.optNumber("rssi_dbm")
                    ?: row.optNumber("signal_dbuv_or_rssi_dbm")
                val powerDb = row.optNumber("power_db")
                when {
                    rssiDbm != null -> rssiDbm.toDouble().let { "${formatDecimal(it, if (it % 1.0 == 0.0) 0 else 1)} dBm" }
                    powerDb != null -> powerDb.toDouble().let { "${formatDecimal(it, if (it % 1.0 == 0.0) 0 else 1)} dB" }
                    else -> "sample available"
                }
            }
            else -> row.optString("scan_state").ifBlank { "boundary" }
        }
    }
    val frequency = row.optString("frequency_label").takeIf { it.isNotBlank() }
        ?: radioPointFrequencyLabel(row)
        ?: radioRangeLabel(row)
    val detail = listOfNotNull(
        frequency,
        row.optString("band").takeIf { it.isNotBlank() },
        row.optString("source_type").takeIf { it.isNotBlank() },
        row.optString("receiver_id").takeIf { it.isNotBlank() }?.let { "receiver $it" },
        row.optString("modulation").takeIf { it.isNotBlank() && it != "unknown" }?.let { "modulation $it" },
        row.optString("rds_program_service").takeIf { it.isNotBlank() }?.let { "RDS $it" },
        row.optString("rds_radio_text").takeIf { it.isNotBlank() }?.let { "text $it" },
        row.optNumber("snr_db")?.toDouble()?.let { "SNR ${formatDecimal(it, if (it % 1.0 == 0.0) 0 else 1)} dB" },
        row.optNumber("bandwidth_hz")?.toDouble()?.let { "bandwidth ${formatDecimal(it, 0)} Hz" },
        row.optNumber("span_hz")?.toDouble()?.let { "span ${formatDecimal(it, 0)} Hz" },
        row.optNumber("sample_rate_hz")?.toDouble()?.let { "sample ${formatDecimal(it, 0)} Hz" },
        row.optString("scan_state").takeIf { it.isNotBlank() },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    val explicitFraction = row.optNumber("fraction")?.toFloat()
    val signalFraction = row.optNumber("rssi_dbm")?.toInt()?.let(::dbmFraction)
        ?: row.optNumber("signal_dbuv_or_rssi_dbm")?.toInt()?.let(::dbmFraction)
        ?: row.optNumber("power_db")?.toDouble()?.let { ((it + 110.0) / 80.0).toFloat().coerceIn(0.05f, 1f) }
    return DiagnosticGraphRow(
        label = label,
        valueLabel = valueLabel,
        detail = detail.ifBlank { if (sampled) "Radio bridge sample" else "Radio graph boundary" },
        fraction = explicitFraction?.coerceIn(0.05f, 1f) ?: signalFraction ?: if (sampled) 0.75f else 0.35f,
    )
}

private fun radioReceiverProfileRow(row: JSONObject): DiagnosticGraphRow {
    val label = row.optString("label").takeIf { it.isNotBlank() }
        ?: row.optString("receiver_id").ifBlank { "Radio receiver profile" }
    val publicAndroidScan = row.optBoolean("public_android_scan_supported", false)
    val vendorBridge = row.optBoolean("vendor_bridge_possible", false) || row.optBoolean("requires_vendor_bridge", false)
    val requiresExternal = row.optBoolean("requires_external_hardware", false)
    val scanState = row.optString("scan_state").takeIf { it.isNotBlank() }
    val valueLabel = when {
        publicAndroidScan -> "Android metadata"
        vendorBridge -> "vendor bridge"
        requiresExternal -> "external receiver"
        else -> row.optString("value_label").ifBlank { "schema only" }
    }
    val schemaFields = joinJsonStrings(row.optJSONArray("graph_row_schema"), 4)
    val stationFields = joinJsonStrings(row.optJSONArray("station_metadata_fields"), 3)
    val sampleFields = joinJsonStrings(row.optJSONArray("sample_fields"), 3)
    val directFields = joinJsonStrings(row.optJSONArray("direct_argument_fields"), 3)
    val jsonKeys = joinJsonStrings(row.optJSONArray("json_argument_keys"), 3)
    val detail = listOfNotNull(
        radioRangeLabel(row),
        row.optString("source_type").takeIf { it.isNotBlank() },
        row.optString("route_action").takeIf { it.isNotBlank() }?.let { "route $it" },
        row.optString("access_path").takeIf { it.isNotBlank() },
        scanState,
        schemaFields.takeIf { it.isNotBlank() }?.let { "schema $it" },
        stationFields.takeIf { it.isNotBlank() }?.let { "station $it" },
        sampleFields.takeIf { it.isNotBlank() }?.let { "samples $it" },
        directFields.takeIf { it.isNotBlank() }?.let { "args $it" },
        jsonKeys.takeIf { it.isNotBlank() }?.let { "json $it" },
        row.optString("recommendation").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    val fraction = row.optNumber("fraction")?.toFloat() ?: when {
        publicAndroidScan -> 0.9f
        vendorBridge -> 0.65f
        requiresExternal -> 0.45f
        else -> 0.3f
    }
    return DiagnosticGraphRow(
        label = label,
        valueLabel = valueLabel,
        detail = detail.ifBlank { "Radio receiver profile" },
        fraction = fraction.coerceIn(0.05f, 1f),
    )
}

private fun sensorRow(row: JSONObject): DiagnosticGraphRow {
    val label = row.optString("sensor_label").takeIf { it.isNotBlank() }
        ?: row.optString("sensor_type").ifBlank { "Sensor" }
    val sampled = row.optBoolean("sampled", false)
    val magnitude = row.optJSONArray("values")?.vectorMagnitude()
    val unit = row.optString("unit").takeIf { it.isNotBlank() }
    val detail = listOfNotNull(
        row.optString("sensor_name").takeIf { it.isNotBlank() },
        row.optString("vendor").takeIf { it.isNotBlank() },
        row.optString("accuracy_label").takeIf { it.isNotBlank() && it != "unknown" }?.let { "accuracy $it" },
        unit,
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = if (sampled && magnitude != null) {
            "${formatDecimal(magnitude, 2)}${unit?.let { " $it" }.orEmpty()}"
        } else {
            "unavailable"
        },
        detail = detail.ifBlank { if (sampled) "Sensor sample" else "Sensor unavailable" },
        fraction = if (sampled && magnitude != null) (magnitude / 20.0).toFloat().coerceIn(0.05f, 1f) else 0.08f,
    )
}

private fun motionSensorHistoryRow(row: JSONObject): DiagnosticGraphRow? {
    val currentMagnitude = row.optNumber("current_magnitude")?.toDouble() ?: return null
    val label = row.optString("sensor_label").takeIf { it.isNotBlank() }
        ?: row.optString("sensor_type").ifBlank { "Motion sensor" }
    val unit = row.optString("magnitude_unit")
        .ifBlank { row.optString("unit") }
        .takeIf { it.isNotBlank() }
    val sampleCount = row.optNumber("sample_count")?.toInt() ?: 0
    val averageMagnitude = row.optNumber("average_magnitude")?.toDouble()
    val minMagnitude = row.optNumber("min_magnitude")?.toDouble()
    val maxMagnitude = row.optNumber("max_magnitude")?.toDouble()
    val trendMagnitude = row.optNumber("trend_magnitude")?.toDouble()
    val trendLabel = row.optString("trend_label").ifBlank { "stable" }
    val stabilityLabel = row.optString("stability_label").takeIf { it.isNotBlank() }
    val vector = joinJsonStrings(row.optJSONArray("current_values"), 4)
    val detail = listOfNotNull(
        row.optString("sensor_name").takeIf { it.isNotBlank() },
        row.optString("vendor").takeIf { it.isNotBlank() },
        "$sampleCount sample${if (sampleCount == 1) "" else "s"}",
        stabilityLabel?.let { "stability $it" },
        averageMagnitude?.let { "avg ${formatDecimal(it, 2)}${unit?.let { value -> " $value" }.orEmpty()}" },
        if (minMagnitude != null && maxMagnitude != null) {
            "range ${formatDecimal(minMagnitude, 2)}..${formatDecimal(maxMagnitude, 2)}${unit?.let { " $it" }.orEmpty()}"
        } else {
            null
        },
        trendMagnitude?.let { "trend ${if (it > 0) "+" else ""}${formatDecimal(it, 2)} ${trendLabel}" },
        vector.takeIf { it.isNotBlank() }?.let { "vector $it" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = "${formatDecimal(currentMagnitude, 2)}${unit?.let { " $it" }.orEmpty()} $trendLabel",
        detail = detail.ifBlank { "Motion sensor history" },
        fraction = (currentMagnitude / 20.0).toFloat().coerceIn(0.05f, 1f),
    )
}

private fun motionPoseEstimateRow(row: JSONObject): DiagnosticGraphRow {
    val label = row.optString("label").takeIf { it.isNotBlank() }
        ?: row.optString("pose_type").ifBlank { "Motion pose" }.replace('_', ' ')
    val poseSource = row.optString("pose_source").takeIf { it.isNotBlank() }
    val confidence = row.optString("confidence_label").takeIf { it.isNotBlank() }
    val heading = row.optString("heading_label").takeIf { it.isNotBlank() }
    val motionState = row.optString("motion_state_label").takeIf { it.isNotBlank() }
    val roll = row.optNumber("roll_degrees")?.toDouble()
    val pitch = row.optNumber("pitch_degrees")?.toDouble()
    val tilt = row.optNumber("tilt_degrees")?.toDouble()
    val azimuth = row.optNumber("azimuth_degrees")?.toDouble()
    val angularVelocity = row.optNumber("angular_velocity_rad_s")?.toDouble()
    val accelerationDelta = row.optNumber("acceleration_delta_from_gravity")?.toDouble()
    val sourceSensors = joinJsonStrings(row.optJSONArray("source_sensors"), 4)
    val valueLabel = row.optString("value_label").takeIf { it.isNotBlank() }
        ?: row.optString("pose_label").takeIf { it.isNotBlank() }?.replace('_', ' ')
        ?: motionState?.replace('_', ' ')
        ?: "pose"
    val detail = listOfNotNull(
        poseSource?.let { "source $it" },
        sourceSensors.takeIf { it.isNotBlank() }?.let { "sensors $it" },
        confidence?.let { "confidence $it" },
        roll?.let { "roll ${formatDecimal(it, 1)} deg" },
        pitch?.let { "pitch ${formatDecimal(it, 1)} deg" },
        tilt?.let { "tilt ${formatDecimal(it, 1)} deg" },
        azimuth?.let { "azimuth ${formatDecimal(it, 1)} deg" },
        heading?.let { "heading $it" },
        angularVelocity?.let { "angular ${formatDecimal(it, 2)} rad/s" },
        accelerationDelta?.let { "movement ${formatDecimal(it, 2)} m/s^2" },
        motionState?.replace('_', ' '),
        row.optString("workflow_hint").takeIf { it.isNotBlank() },
    ).joinToString(" | ")
    val fraction = row.optNumber("fraction")?.toFloat()
        ?: angularVelocity?.let { (it / 2.0).toFloat() }
        ?: accelerationDelta?.let { (it / 4.0).toFloat() }
        ?: 0.55f
    return DiagnosticGraphRow(
        label = label,
        valueLabel = valueLabel,
        detail = detail.ifBlank { "Motion pose estimate" },
        fraction = fraction.coerceIn(0.05f, 1f),
    )
}

private fun sensorCapabilityRow(row: JSONObject): DiagnosticGraphRow {
    val label = row.optString("sensor_label").takeIf { it.isNotBlank() }
        ?: row.optString("sensor_type").ifBlank { "Sensor" }
    val available = row.optBoolean("available", false)
    val unit = row.optString("unit").takeIf { it.isNotBlank() }
    val maxRange = row.optNumber("maximum_range")?.toDouble()
    val resolution = row.optNumber("resolution")?.toDouble()
    val powerMa = row.optNumber("power_ma")?.toDouble()
    val minDelayUs = row.optNumber("min_delay_us")?.toInt()
    val reportingMode = row.optString("reporting_mode").takeIf { it.isNotBlank() && it != "unknown" }
    val detail = listOfNotNull(
        row.optString("sensor_name").takeIf { it.isNotBlank() },
        row.optString("vendor").takeIf { it.isNotBlank() },
        reportingMode?.replace('_', ' '),
        minDelayUs?.takeIf { it > 0 }?.let { "${formatDecimal(1_000_000.0 / it, 1)} Hz" },
        resolution?.takeIf { it > 0.0 }?.let { "res ${formatDecimal(it, 4)}${unit?.let { value -> " $value" }.orEmpty()}" },
        powerMa?.takeIf { it > 0.0 }?.let { "${formatDecimal(it, 2)} mA" },
        row.optBoolean("wake_up", false).takeIf { it }?.let { "wake-up" },
        row.optBoolean("dynamic_sensor", false).takeIf { it }?.let { "dynamic" },
        row.optBoolean("direct_channel_supported", false).takeIf { it }?.let { "direct channel" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = if (available && maxRange != null && maxRange > 0.0) {
            "range ${formatDecimal(maxRange, 2)}${unit?.let { " $it" }.orEmpty()}"
        } else if (available) {
            "available"
        } else {
            "unavailable"
        },
        detail = detail.ifBlank { if (available) "Sensor capability" else "Sensor unavailable" },
        fraction = when {
            !available -> 0.08f
            powerMa == null || powerMa <= 0.0 -> 0.55f
            powerMa <= 1.0 -> 0.9f
            powerMa <= 5.0 -> 0.65f
            else -> 0.35f
        },
    )
}

private fun capabilityMatrixRow(row: JSONObject): DiagnosticGraphRow {
    val label = row.optString("label").takeIf { it.isNotBlank() }
        ?: row.optString("category").ifBlank { "Capability" }
    val ready = row.optBoolean("ready", false)
    val category = row.optString("category").takeIf { it.isNotBlank() }
    val recommendation = row.optString("recommendation").takeIf { it.isNotBlank() }
    val detail = listOfNotNull(
        category?.replace('_', ' '),
        row.optString("detail").takeIf { it.isNotBlank() },
        recommendation,
    ).joinToString(" | ")
    val explicitFraction = row.optNumber("fraction")?.toFloat()
    val valueLabel = row.optString("value_label").ifBlank { if (ready) "ready" else "needs setup" }
    return DiagnosticGraphRow(
        label = label,
        valueLabel = valueLabel,
        detail = detail.ifBlank { if (ready) "Capability ready" else "Capability needs setup" },
        fraction = explicitFraction?.coerceIn(0.05f, 1f) ?: if (ready) 0.9f else 0.25f,
    )
}

private fun genericRow(row: JSONObject, index: Int): DiagnosticGraphRow {
    val label = row.optString("label").ifBlank { row.optString("name").ifBlank { "Row ${index + 1}" } }
    val value = row.optNumber("value")?.toDouble()
    return DiagnosticGraphRow(
        label = label,
        valueLabel = value?.let { formatDecimal(it, 2) }.orEmpty(),
        detail = row.optString("detail").ifBlank { row.toString().take(80) },
        fraction = value?.toFloat()?.coerceIn(0.05f, 1f) ?: 0.5f,
    )
}

private fun dbmFraction(dbm: Int): Float = ((dbm + 100f) / 70f).coerceIn(0.05f, 1f)

private fun joinJsonStrings(values: JSONArray?, limit: Int): String {
    if (values == null) return ""
    return buildList {
        for (index in 0 until minOf(values.length(), limit)) {
            values.optString(index).takeIf { it.isNotBlank() }?.let(::add)
        }
    }.joinToString(", ")
}

private fun radioRangeLabel(row: JSONObject): String? {
    val minKhz = row.optNumber("frequency_min_khz")?.toInt()
    val maxKhz = row.optNumber("frequency_max_khz")?.toInt()
    if (minKhz != null && maxKhz != null) return "$minKhz-$maxKhz kHz"
    val minMhz = row.optNumber("frequency_min_mhz")?.toDouble()
    val maxMhz = row.optNumber("frequency_max_mhz")?.toDouble()
    if (minMhz != null && maxMhz != null) return "${formatDecimal(minMhz, 1)}-${formatDecimal(maxMhz, 1)} MHz"
    return null
}

private fun radioPointFrequencyLabel(row: JSONObject): String? {
    val khz = row.optNumber("frequency_khz")?.toDouble()
    if (khz != null) return "${formatDecimal(khz, if (khz % 1.0 == 0.0) 0 else 1)} kHz"
    val mhz = row.optNumber("frequency_mhz")?.toDouble()
    if (mhz != null && mhz < 2.0) return "${formatDecimal(mhz * 1000.0, if ((mhz * 1000.0) % 1.0 == 0.0) 0 else 1)} kHz"
    if (mhz != null) return "${formatDecimal(mhz, if (mhz % 1.0 == 0.0) 0 else 1)} MHz"
    val hz = row.optNumber("frequency_hz")?.toDouble()
        ?: row.optNumber("center_frequency_hz")?.toDouble()
    return hz?.let { "${formatDecimal(it / 1_000_000.0, 3)} MHz" }
}

private fun JSONArray.vectorMagnitude(): Double {
    var sum = 0.0
    for (index in 0 until length()) {
        val value = optNumber(index)?.toDouble() ?: continue
        sum += value * value
    }
    return sqrt(sum)
}

private fun JSONObject.optNumber(key: String): Number? = opt(key) as? Number

private fun JSONArray.optNumber(index: Int): Number? = opt(index) as? Number

private fun formatDecimal(value: Double, places: Int): String = String.format(Locale.US, "%.${places}f", value)
