package com.nousresearch.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject

internal object OpenGuiActionCompat {
    private val functionCallPattern = Regex("""^([A-Za-z_][A-Za-z0-9_]*)\s*\(([\s\S]*)\)\s*[.;,]*\s*$""")
    private val numberPattern = Regex("""-?\d+(?:\.\d+)?""")
    private val interventionMessages = mapOf(
        "need_login" to "User must complete login, registration, or identity verification",
        "asset_risk" to "Involves assets, payment, transfer, refund, or major asset changes and requires user confirmation",
        "delete_confirm" to "Involves important data deletion or irreversible operations and requires user confirmation",
    )

    fun parse(rawText: String): ParsedOpenGuiAction {
        val call = extractFunctionCall(rawText)
            ?: throw IllegalArgumentException("OpenGUI action text must contain a function call such as click(...)")
        val normalizedCall = normalizeActionSyntax(call)
        val match = functionCallPattern.matchEntire(normalizedCall)
            ?: throw IllegalArgumentException("Unable to parse OpenGUI action call: ${call.take(120)}")
        val originalActionType = match.groupValues[1].trim()
        val normalizedOriginal = normalizeName(originalActionType)
        val args = parseArguments(match.groupValues[2])
        val canonicalType = canonicalActionType(normalizedOriginal)
        val interventionContent = interventionMessages[normalizedOriginal]
        val content = when {
            interventionContent != null -> "[${normalizedOriginal}] " + valueFrom(args, "content", "text", "message").ifBlank { interventionContent }
            canonicalType == "type" -> valueFrom(args, "content", "text", "value", "input")
            canonicalType == "call_user" -> valueFrom(args, "content", "text", "message")
            canonicalType == "update_working_memory" -> valueFrom(args, "content", "text", "memory", "value")
            else -> valueFrom(args, "content", "text", "value", "message")
        }
        val direction = directionFrom(normalizedOriginal, args)
        return ParsedOpenGuiAction(
            rawText = rawText.trim(),
            originalActionType = originalActionType,
            actionType = canonicalType,
            arguments = args,
            startCoords = coordinateFromArgs(args, "start_box", "box", "point", "start_point", "start", "coords", "coordinate"),
            endCoords = coordinateFromArgs(args, "end_box", "end_point", "end"),
            content = content,
            appName = valueFrom(args, "app_name", "app", "application", "label"),
            packageName = valueFrom(args, "package_name", "package", "packageName", "app_package"),
            direction = direction,
            durationMs = longFrom(args, "duration_ms", "duration", "wait_ms", "seconds")?.let { value ->
                if (args.containsKey("seconds")) value * 1000L else value
            },
            requiresUserIntervention = canonicalType == "call_user",
            terminal = canonicalType == "finished",
        )
    }

    private fun extractFunctionCall(rawText: String): String? {
        val stripped = normalizeActionSyntax(stripCodeFence(rawText).trim())
        val actionStart = Regex("""(?i)\bAction\s*[:：]""")
            .findAll(stripped)
            .lastOrNull()
            ?.range
            ?.last
            ?.plus(1)
        val text = if (actionStart != null) stripped.substring(actionStart).trim() else stripped
        val nameMatch = Regex("""[A-Za-z_][A-Za-z0-9_]*\s*\(""").find(text) ?: return null
        val openIndex = text.indexOf('(', nameMatch.range.first)
        val closeIndex = findMatchingParen(text, openIndex)
        return if (closeIndex > openIndex) {
            text.substring(nameMatch.range.first, closeIndex + 1).trim()
        } else {
            null
        }
    }

    private fun stripCodeFence(text: String): String {
        return text
            .replace(Regex("""^\s*```[A-Za-z0-9_-]*\s*"""), "")
            .replace(Regex("""\s*```\s*$"""), "")
    }

    private fun findMatchingParen(text: String, openIndex: Int): Int {
        var quote: Char? = null
        var depth = 0
        for (index in openIndex until text.length) {
            val ch = text[index]
            val previous = text.getOrNull(index - 1)
            if ((ch == '\'' || ch == '"') && previous != '\\') {
                quote = if (quote == ch) null else quote ?: ch
                continue
            }
            if (quote != null) continue
            when (ch) {
                '(' -> depth += 1
                ')' -> {
                    depth -= 1
                    if (depth == 0) return index
                }
            }
        }
        return -1
    }

    private fun normalizeActionSyntax(input: String): String {
        val builder = StringBuilder(input.length)
        var quote: Char? = null
        input.forEachIndexed { index, ch ->
            val previous = input.getOrNull(index - 1)
            if ((ch == '\'' || ch == '"') && previous != '\\') {
                quote = if (quote == ch) null else quote ?: ch
                builder.append(ch)
                return@forEachIndexed
            }
            builder.append(
                if (quote == null) {
                    when (ch) {
                        '（' -> '('
                        '）' -> ')'
                        '，' -> ','
                        else -> ch
                    }
                } else {
                    ch
                },
            )
        }
        return builder.toString()
    }

    private fun parseArguments(argsText: String): Map<String, String> {
        val args = linkedMapOf<String, String>()
        splitTopLevel(argsText).forEachIndexed { index, chunk ->
            val equalsIndex = chunk.indexOf('=')
            if (equalsIndex <= 0) {
                args["arg$index"] = cleanValue(chunk)
            } else {
                val key = normalizeName(chunk.substring(0, equalsIndex))
                args[key] = cleanValue(chunk.substring(equalsIndex + 1))
            }
        }
        return args
    }

    private fun splitTopLevel(argsText: String): List<String> {
        val chunks = mutableListOf<String>()
        val current = StringBuilder()
        var quote: Char? = null
        var depth = 0
        argsText.forEachIndexed { index, ch ->
            val previous = argsText.getOrNull(index - 1)
            if ((ch == '\'' || ch == '"') && previous != '\\') {
                quote = if (quote == ch) null else quote ?: ch
                current.append(ch)
                return@forEachIndexed
            }
            if (quote == null) {
                when (ch) {
                    '(', '[' -> depth += 1
                    ')', ']' -> depth = (depth - 1).coerceAtLeast(0)
                    ',' -> if (depth == 0) {
                        current.toString().trim().takeIf { it.isNotEmpty() }?.let { chunks += it }
                        current.clear()
                        return@forEachIndexed
                    }
                }
            }
            current.append(ch)
        }
        current.toString().trim().takeIf { it.isNotEmpty() }?.let { chunks += it }
        return chunks
    }

    private fun cleanValue(rawValue: String): String {
        var value = rawValue.trim()
        if (value.length >= 2 && ((value.first() == '\'' && value.last() == '\'') || (value.first() == '"' && value.last() == '"'))) {
            value = value.substring(1, value.length - 1)
        }
        value = value
            .replace("""\n""", "\n")
            .replace("<|box_start|>", "")
            .replace("<|box_end|>", "")
        value = Regex("""<bbox>([\s\S]*?)</bbox>""").replace(value) { match ->
            "(" + match.groupValues[1].trim().replace(Regex("""\s+"""), ",") + ")"
        }
        value = Regex("""<point>([\s\S]*?)</point>""").replace(value) { match ->
            "(" + match.groupValues[1].trim().replace(Regex("""\s+"""), ",") + ")"
        }
        return value.trim()
    }

    private fun canonicalActionType(actionName: String): String {
        return when (actionName) {
            "click", "tap", "touch" -> "click"
            "long_click", "long_press", "press" -> "long_press"
            "drag", "swipe" -> "swipe"
            "scroll", "scroll_up", "scroll_down", "scroll_left", "scroll_right" -> "scroll"
            "type", "input", "set_text", "enter_text" -> "type"
            "open_app", "launch_app" -> "open_app"
            "press_back", "back", "global_back" -> "press_back"
            "press_home", "home", "global_home" -> "press_home"
            "finished", "finish", "done" -> "finished"
            "wait", "sleep" -> "wait"
            "call_user", "need_login", "asset_risk", "delete_confirm" -> "call_user"
            "request_visual" -> "request_visual"
            "update_working_memory" -> "update_working_memory"
            "get_working_memory" -> "get_working_memory"
            else -> actionName
        }
    }

    private fun directionFrom(actionName: String, args: Map<String, String>): String {
        val requested = valueFrom(args, "direction", "scroll_direction").ifBlank {
            when (actionName) {
                "scroll_down" -> "down"
                "scroll_left" -> "left"
                "scroll_right" -> "right"
                else -> "up"
            }
        }
        return normalizeName(requested).ifBlank { "up" }
    }

    private fun coordinateFromArgs(args: Map<String, String>, vararg keys: String): NormalizedOpenGuiPoint? {
        keys.firstNotNullOfOrNull { key -> args[key]?.takeIf { it.isNotBlank() } }?.let { value ->
            return coordinateFromValue(value)
        }
        val x = doubleFrom(args, "x", "start_x")
        val y = doubleFrom(args, "y", "start_y")
        if (x != null && y != null && keys.any { it.startsWith("start") || it == "point" || it == "box" }) {
            return NormalizedOpenGuiPoint(normalizeCoordinate(x), normalizeCoordinate(y))
        }
        val x2 = doubleFrom(args, "x2", "end_x")
        val y2 = doubleFrom(args, "y2", "end_y")
        if (x2 != null && y2 != null && keys.any { it.startsWith("end") }) {
            return NormalizedOpenGuiPoint(normalizeCoordinate(x2), normalizeCoordinate(y2))
        }
        return null
    }

    private fun coordinateFromValue(value: String): NormalizedOpenGuiPoint? {
        val numbers = numberPattern.findAll(value).mapNotNull { it.value.toDoubleOrNull() }.toList()
        if (numbers.size < 2) return null
        val x = if (numbers.size >= 4) (numbers[0] + numbers[2]) / 2.0 else numbers[0]
        val y = if (numbers.size >= 4) (numbers[1] + numbers[3]) / 2.0 else numbers[1]
        return NormalizedOpenGuiPoint(normalizeCoordinate(x), normalizeCoordinate(y))
    }

    private fun normalizeCoordinate(value: Double): Double {
        return when {
            value.isNaN() || value.isInfinite() -> 0.0
            value in 0.0..1.0 -> value
            else -> (value / 1000.0).coerceIn(0.0, 1.0)
        }
    }

    private fun valueFrom(args: Map<String, String>, vararg keys: String): String {
        return keys.firstNotNullOfOrNull { key -> args[key]?.takeIf { it.isNotBlank() } }.orEmpty()
    }

    private fun doubleFrom(args: Map<String, String>, vararg keys: String): Double? {
        return keys.firstNotNullOfOrNull { key -> args[key]?.toDoubleOrNull() }
    }

    private fun longFrom(args: Map<String, String>, vararg keys: String): Long? {
        return keys.firstNotNullOfOrNull { key -> args[key]?.toLongOrNull() ?: args[key]?.toDoubleOrNull()?.toLong() }
    }

    private fun normalizeName(value: String): String {
        return value.trim()
            .replace(Regex("""([a-z0-9])([A-Z])"""), "$1_$2")
            .lowercase()
            .replace("-", "_")
            .replace(" ", "_")
    }
}

internal data class ParsedOpenGuiAction(
    val rawText: String,
    val originalActionType: String,
    val actionType: String,
    val arguments: Map<String, String>,
    val startCoords: NormalizedOpenGuiPoint?,
    val endCoords: NormalizedOpenGuiPoint?,
    val content: String,
    val appName: String,
    val packageName: String,
    val direction: String,
    val durationMs: Long?,
    val requiresUserIntervention: Boolean,
    val terminal: Boolean,
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("raw_action", rawText)
            .put("original_action_type", originalActionType)
            .put("action_type", actionType)
            .put("arguments", JSONObject(arguments))
            .put("content", content)
            .put("app_name", appName)
            .put("package_name", packageName)
            .put("direction", direction)
            .put("requires_user_intervention", requiresUserIntervention)
            .put("terminal", terminal)
            .also { json ->
                startCoords?.let { json.put("start_coords", it.toJsonArray()) }
                endCoords?.let { json.put("end_coords", it.toJsonArray()) }
                durationMs?.let { json.put("duration_ms", it) }
            }
    }
}

internal data class NormalizedOpenGuiPoint(
    val x: Double,
    val y: Double,
) {
    fun toJsonArray(): JSONArray = JSONArray().put(x).put(y)
}
