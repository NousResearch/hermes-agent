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
                "bluetooth_rssi" -> bluetoothRow(row)
                "radio_frequency_capability" -> radioRow(row)
                "sensor_vector" -> sensorRow(row)
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
        distance?.let { "~${formatDecimal(it, 1)} m" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = ssid,
        valueLabel = "$rssi dBm",
        detail = detail.ifBlank { "Wi-Fi signal" },
        fraction = dbmFraction(rssi),
    )
}

private fun bluetoothRow(row: JSONObject): DiagnosticGraphRow? {
    val rssi = row.optNumber("rssi_dbm")?.toInt()
    val label = row.optString("device_name").takeIf { it.isNotBlank() && it != "<unnamed>" }
        ?: row.optString("address").ifBlank { "Bluetooth" }
    val detail = listOfNotNull(
        row.optString("device_type").takeIf { it.isNotBlank() },
        row.optString("bond_state").takeIf { it.isNotBlank() },
        if (row.optBoolean("paired", false)) "paired" else null,
        row.optNumber("scan_record_bytes")?.toInt()?.let { "$it scan bytes" },
    ).joinToString(" | ")
    return DiagnosticGraphRow(
        label = label,
        valueLabel = rssi?.let { "$it dBm" } ?: "paired",
        detail = detail.ifBlank { "Bluetooth device metadata" },
        fraction = rssi?.let(::dbmFraction) ?: 0.45f,
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
