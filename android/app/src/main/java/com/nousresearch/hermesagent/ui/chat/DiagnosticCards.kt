package com.nousresearch.hermesagent.ui.chat

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

private fun graphRows(graphType: String?, rows: JSONArray): List<DiagnosticGraphRow> {
    return buildList {
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            val parsed = when (graphType) {
                "wifi_channel_strength" -> wifiRow(row)
                "wifi_access_point_detail" -> wifiAccessPointDetailRow(row)
                "wifi_channel_rating" -> wifiChannelRatingRow(row)
                "wifi_vendor_summary" -> wifiVendorSummaryRow(row)
                "wifi_security_summary" -> wifiSecuritySummaryRow(row)
                "wifi_channel_width_summary" -> wifiChannelWidthSummaryRow(row)
                "wifi_standard_summary" -> wifiStandardSummaryRow(row)
                "wifi_signal_history" -> wifiSignalHistoryRow(row)
                "bluetooth_rssi" -> bluetoothRow(row)
                "bluetooth_metadata_summary" -> bluetoothMetadataSummaryRow(row)
                "radio_frequency_capability" -> radioRow(row)
                "sensor_vector" -> sensorRow(row)
                "sensor_capability" -> sensorCapabilityRow(row)
                "agent_capability_matrix", "kai_parity_matrix", "agent_workflow_readiness",
                "wifi_analyzer_feature_matrix", "wifi_analyzer_workflow_routes", "wifi_scan_policy_matrix",
                "bluetooth_analyzer_feature_matrix", "bluetooth_analyzer_workflow_routes", "bluetooth_scan_policy_matrix",
                "sensor_analyzer_feature_matrix", "sensor_analyzer_workflow_routes", "sensor_sampling_policy_matrix",
                "signal_awareness_matrix", "signal_workflow_routes", "signal_constraint_matrix",
                "soc_backend_matrix", "soc_backend_policy_routes", "soc_backend_constraint_matrix" -> capabilityMatrixRow(row)
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
    val label = row.optString("device_name").takeIf { it.isNotBlank() && it != "<unnamed>" }
        ?: row.optString("address").ifBlank { "Bluetooth" }
    val detail = listOfNotNull(
        row.optString("device_type").takeIf { it.isNotBlank() },
        row.optString("device_category").takeIf { it.isNotBlank() && it != "unknown" },
        row.optString("bond_state").takeIf { it.isNotBlank() },
        if (row.optBoolean("paired", false)) "paired" else null,
        row.optString("proximity_label").takeIf { it.isNotBlank() },
        row.optNumber("estimated_distance_meters")?.toDouble()?.let { "~${formatDecimal(it, 1)} m" },
        row.optNumber("service_uuid_count")?.toInt()?.takeIf { it > 0 }?.let { "$it services" },
        row.optNumber("manufacturer_data_count")?.toInt()?.takeIf { it > 0 }?.let { "$it manufacturer records" },
        row.optNumber("scan_record_bytes")?.toInt()?.let { "$it scan bytes" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = rssi?.let { "$it dBm" } ?: "paired",
        detail = detail.ifBlank { "Bluetooth device metadata" },
        fraction = rssi?.let(::dbmFraction) ?: 0.45f,
    )
}

private fun bluetoothMetadataSummaryRow(row: JSONObject): DiagnosticGraphRow? {
    val label = row.optString("label").takeIf { it.isNotBlank() } ?: return null
    val count = row.optNumber("count")?.toInt() ?: return null
    val summaryType = row.optString("summary_type").ifBlank { "metadata" }
    val strongestRssi = row.optNumber("strongest_rssi_dbm")?.toInt()
    val detail = listOfNotNull(
        summaryType.replace('_', ' '),
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

private fun radioRow(row: JSONObject): DiagnosticGraphRow {
    val band = row.optString("band").ifBlank { "Radio band" }
    val supported = row.optBoolean("supported", false)
    val sampled = row.optBoolean("sampled", false)
    val requiresExternal = row.optBoolean("requires_external_hardware", false)
    val valueLabel = when {
        sampled -> "sampled"
        supported -> "available"
        requiresExternal -> "external"
        else -> "limited"
    }
    val range = radioRangeLabel(row)
    val reason = row.optString("reason").takeIf { it.isNotBlank() }
    return DiagnosticGraphRow(
        label = band,
        valueLabel = valueLabel,
        detail = listOfNotNull(range, reason).joinToString(" | ").ifBlank { "Radio capability" },
        fraction = when {
            sampled -> 1f
            supported -> 0.75f
            requiresExternal -> 0.45f
            else -> 0.15f
        },
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
