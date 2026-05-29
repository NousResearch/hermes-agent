package com.mobilefork.hermesagent.device

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.UUID
import kotlin.math.max

object HermesHindsightMemoryBridge {
    private const val PREFS_NAME = "hermes_hindsight_memory"
    private const val ENTRIES_KEY = "entries_json"
    private const val DEFAULT_LIMIT = 5
    private const val MAX_LIMIT = 20
    private const val MAX_ENTRIES = 160
    private const val PROMOTION_HIT_THRESHOLD = 5
    private const val PROMOTED_CONTEXT_LIMIT = 5
    private const val PROMOTED_CONTEXT_MAX_CHARS = 1200
    private const val RELEVANT_CONTEXT_LIMIT = 6
    private const val RELEVANT_CONTEXT_MAX_CHARS = 1600
    private val ACTIONS = listOf("status", "retain", "recall", "reflect", "relevant_context", "promoted_context", "clear")

    fun performActionJson(context: Context, rawAction: String, arguments: JSONObject = JSONObject()): String {
        val action = rawAction.trim().lowercase(Locale.US).ifBlank { "status" }
        return when (action) {
            "status", "read_status" -> statusJson(context)
            "retain", "remember", "store" -> retainJson(context, arguments)
            "recall", "search", "retrieve" -> recallJson(context, arguments)
            "reflect", "consolidate", "compact" -> reflectJson(context, arguments)
            "relevant_context", "rag_context", "context", "system_prompt_context", "prompt_context" -> relevantContextJson(context, arguments)
            "promoted_context" -> promotedContextJson(context, arguments)
            "clear", "reset" -> clearJson(context)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported hindsight memory action: $action")
                .put("available_actions", JSONArray(ACTIONS))
        }.toString()
    }

    fun statusJson(context: Context): JSONObject {
        val entries = readEntries(context)
        val reinforced = entries.count { it.optInt("hit_count", 0) > 0 }
        val promoted = entries.count(::isPromoted)
        return JSONObject()
            .put("success", true)
            .put("action", "status")
            .put("memory_count", entries.size)
            .put("reinforced_memory_count", reinforced)
            .put("promoted_memory_count", promoted)
            .put("promotion_hit_threshold", PROMOTION_HIT_THRESHOLD)
            .put("available_actions", JSONArray(ACTIONS))
            .put("cards", JSONArray().put(card("Hindsight Memory", "${entries.size} retained local memories, $reinforced reinforced, $promoted promoted.")))
    }

    private fun retainJson(context: Context, arguments: JSONObject): JSONObject {
        val entries = readEntries(context)
        val now = System.currentTimeMillis()
        val facts = factsFrom(arguments)
        val retained = JSONArray()
        val tags = stringArray(arguments.opt("tags"))
            .ifEmpty { listOfNotNull(arguments.optString("category").takeIf { it.isNotBlank() }) }
        val source = arguments.optString("source").ifBlank { "chat" }

        facts.forEach { fact ->
            val content = fact.trim()
            if (content.isBlank()) return@forEach
            val normalized = normalizeContent(content)
            val existing = entries.firstOrNull { it.optString("normalized_content") == normalized }
            val entry = existing ?: JSONObject()
                .put("id", UUID.randomUUID().toString())
                .put("created_at_ms", now)
                .put("hit_count", 0)
                .also { entries.add(it) }

            val hitCount = entry.optInt("hit_count", 0) + if (existing == null) 0 else 1
            val keywords = keywords(content)
            entry.put("content", content)
                .put("normalized_content", normalized)
                .put("source", source)
                .put("category", arguments.optString("category").ifBlank { "general" })
                .put("tags", JSONArray((jsonStringList(entry.optJSONArray("tags")) + tags).distinct()))
                .put("entities", JSONArray(extractEntities(content)))
                .put("semantic_keywords", JSONArray(keywords))
                .put("last_accessed_at_ms", now)
                .put("hit_count", hitCount)
                .put("salience", salienceFor(content, keywords.size, hitCount))
            maybePromote(entry, now)
            retained.put(compactEntry(entry))
        }

        trimAndSave(context, entries)
        return JSONObject()
            .put("success", true)
            .put("action", "retain")
            .put("retained_count", retained.length())
            .put("retained_memories", retained)
            .put("cards", JSONArray().put(card("Memory Retained", "${retained.length()} fact(s) retained with tags, entities, keywords, source, and reinforcement count.")))
    }

    private fun recallJson(context: Context, arguments: JSONObject): JSONObject {
        val query = arguments.optString("query").ifBlank {
            arguments.optString("text").ifBlank { arguments.optString("content") }
        }
        val limit = arguments.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_LIMIT)
        val entries = readEntries(context)
        val queryTerms = keywords(query)
        val now = System.currentTimeMillis()
        val scored = entries
            .map { entry -> entry to scoreEntry(entry, query, queryTerms, now) }
            .filter { (_, score) -> score > 0.0 || query.isBlank() }
            .sortedByDescending { (_, score) -> score }
            .take(limit)

        val memories = JSONArray()
        val returnedIds = scored.map { it.first.optString("id") }.toSet()
        entries.forEach { entry ->
            if (entry.optString("id") in returnedIds) {
                entry.put("hit_count", entry.optInt("hit_count", 0) + 1)
                    .put("last_accessed_at_ms", now)
                maybePromote(entry, now)
            }
        }
        trimAndSave(context, entries)
        scored.forEach { (entry, score) ->
            memories.put(compactEntry(entry).put("recall_score", score))
        }

        return JSONObject()
            .put("success", true)
            .put("action", "recall")
            .put("query", query)
            .put("result_count", memories.length())
            .put("memories", memories)
            .put("cards", JSONArray().put(card("Memory Recall", "${memories.length()} memory row(s) matched by keyword, entity, recency, and reinforcement signals.")))
    }

    private fun reflectJson(context: Context, arguments: JSONObject): JSONObject {
        val maxEntries = arguments.optInt("max_entries", MAX_ENTRIES).coerceIn(20, MAX_ENTRIES)
        val entries = readEntries(context)
        val before = entries.size
        val merged = LinkedHashMap<String, JSONObject>()
        entries.forEach { entry ->
            val key = entry.optString("normalized_content").ifBlank { normalizeContent(entry.optString("content")) }
            val existing = merged[key]
            if (existing == null) {
                merged[key] = entry
            } else {
                existing.put("hit_count", existing.optInt("hit_count", 0) + entry.optInt("hit_count", 0) + 1)
                    .put("last_accessed_at_ms", max(existing.optLong("last_accessed_at_ms", 0L), entry.optLong("last_accessed_at_ms", 0L)))
                    .put("salience", max(existing.optDouble("salience", 0.0), entry.optDouble("salience", 0.0)) + 0.25)
                    .put("tags", JSONArray((jsonStringList(existing.optJSONArray("tags")) + jsonStringList(entry.optJSONArray("tags"))).distinct()))
                if (isPromoted(entry)) {
                    existing.put("promoted", true)
                        .put("promoted_at_ms", max(existing.optLong("promoted_at_ms", 0L), entry.optLong("promoted_at_ms", 0L)))
                }
                maybePromote(existing, System.currentTimeMillis())
            }
        }
        val reflected = merged.values
            .sortedWith(compareByDescending<JSONObject> { it.optDouble("salience", 0.0) }.thenByDescending { it.optInt("hit_count", 0) })
            .take(maxEntries)
            .toMutableList()
        saveEntries(context, reflected)

        return JSONObject()
            .put("success", true)
            .put("action", "reflect")
            .put("before_count", before)
            .put("after_count", reflected.size)
            .put("merged_count", before - reflected.size)
            .put("max_entries", maxEntries)
            .put("cards", JSONArray().put(card("Memory Reflected", "Merged ${before - reflected.size} duplicate or low-priority memory row(s).")))
    }

    fun promotedContextJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val limit = arguments.optInt("limit", PROMOTED_CONTEXT_LIMIT).coerceIn(1, MAX_LIMIT)
        val maxChars = arguments.optInt("max_chars", PROMOTED_CONTEXT_MAX_CHARS).coerceIn(160, 4000)
        val promoted = promotedEntries(context).take(limit)
        val lines = promoted.mapIndexed { index, entry ->
            "${index + 1}. ${entry.optString("content")}"
        }
        val contextText = lines.joinToString("\n").let { text ->
            if (text.length <= maxChars) text else text.take(maxChars - 3).trimEnd() + "..."
        }
        return JSONObject()
            .put("success", true)
            .put("action", "promoted_context")
            .put("promotion_hit_threshold", PROMOTION_HIT_THRESHOLD)
            .put("promoted_memory_count", promoted.size)
            .put("system_prompt_context", contextText)
            .put("promoted_memories", JSONArray(promoted.map(::compactEntry)))
            .put("cards", JSONArray().put(card("Promoted Memory", "${promoted.size} high-reuse memory row(s) ready for prompt context.")))
    }

    fun relevantContextJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val query = arguments.optString("query").ifBlank {
            arguments.optString("text").ifBlank { arguments.optString("content") }
        }
        val limit = arguments.optInt("limit", RELEVANT_CONTEXT_LIMIT).coerceIn(1, MAX_LIMIT)
        val maxChars = arguments.optInt("max_chars", RELEVANT_CONTEXT_MAX_CHARS).coerceIn(240, 5000)
        val recall = recallJson(context, JSONObject().put("query", query).put("limit", limit))
        val recalled = jsonObjectList(recall.optJSONArray("memories"))
        val recalledIds = recalled.map { it.optString("id") }.toSet()
        val promoted = promotedEntries(context)
            .filterNot { it.optString("id") in recalledIds }
            .take(PROMOTED_CONTEXT_LIMIT)
            .map(::compactEntry)
        val lines = buildList {
            recalled.forEachIndexed { index, entry ->
                add("${index + 1}. ${entry.optString("content")}")
            }
            promoted.forEachIndexed { index, entry ->
                add("${recalled.size + index + 1}. ${entry.optString("content")}")
            }
        }
        val contextText = lines.joinToString("\n").let { text ->
            if (text.length <= maxChars) text else text.take(maxChars - 3).trimEnd() + "..."
        }
        return JSONObject()
            .put("success", true)
            .put("action", "relevant_context")
            .put("query", query)
            .put("retrieval_model", "local_keyword_entity_recency_salience_rag")
            .put("result_count", recalled.size + promoted.size)
            .put("recalled_memory_count", recalled.size)
            .put("promoted_memory_count", promoted.size)
            .put("system_prompt_context", contextText)
            .put("memories", JSONArray(recalled + promoted))
            .put("cards", JSONArray().put(card("Relevant Memory", "${recalled.size} recalled and ${promoted.size} promoted memory row(s) prepared as bounded prompt context.")))
    }

    private fun clearJson(context: Context): JSONObject {
        prefs(context).edit().remove(ENTRIES_KEY).apply()
        return JSONObject()
            .put("success", true)
            .put("action", "clear")
            .put("memory_count", 0)
            .put("cards", JSONArray().put(card("Memory Cleared", "Local hindsight memory store cleared.")))
    }

    private fun factsFrom(arguments: JSONObject): List<String> {
        val rawFacts = arguments.opt("facts")
        if (rawFacts is JSONArray) {
            return buildList {
                for (index in 0 until rawFacts.length()) {
                    val item = rawFacts.opt(index)
                    when (item) {
                        is JSONObject -> item.optString("content").ifBlank { item.optString("text") }.takeIf { it.isNotBlank() }?.let(::add)
                        else -> rawFacts.optString(index).takeIf { it.isNotBlank() }?.let(::add)
                    }
                }
            }
        }
        return listOf(
            arguments.optString("content")
                .ifBlank { arguments.optString("fact") }
                .ifBlank { arguments.optString("text") },
        ).filter { it.isNotBlank() }
    }

    private fun scoreEntry(entry: JSONObject, query: String, queryTerms: Set<String>, now: Long): Double {
        if (query.isBlank()) return entry.optDouble("salience", 0.0) + entry.optInt("hit_count", 0)
        val content = entry.optString("content").lowercase(Locale.US)
        val tags = jsonStringList(entry.optJSONArray("tags")).map { it.lowercase(Locale.US) }.toSet()
        val entities = jsonStringList(entry.optJSONArray("entities")).map { it.lowercase(Locale.US) }.toSet()
        val semantic = jsonStringList(entry.optJSONArray("semantic_keywords")).map { it.lowercase(Locale.US) }.toSet()
        val normalizedQuery = query.lowercase(Locale.US)
        var score = 0.0
        if (normalizedQuery in content) score += 5.0
        queryTerms.forEach { term ->
            if (term in content) score += 2.0
            if (term in tags) score += 3.0
            if (term in entities) score += 3.0
            if (term in semantic) score += 2.0
        }
        val ageDays = ((now - entry.optLong("last_accessed_at_ms", now)).coerceAtLeast(0L)).toDouble() / 86_400_000.0
        score += entry.optDouble("salience", 0.0)
        score += entry.optInt("hit_count", 0) * 0.35
        score += (3.0 - ageDays.coerceAtMost(3.0)) * 0.25
        return score
    }

    private fun compactEntry(entry: JSONObject): JSONObject {
        return JSONObject()
            .put("id", entry.optString("id"))
            .put("content", entry.optString("content"))
            .put("source", entry.optString("source"))
            .put("category", entry.optString("category"))
            .put("tags", entry.optJSONArray("tags") ?: JSONArray())
            .put("entities", entry.optJSONArray("entities") ?: JSONArray())
            .put("semantic_keywords", entry.optJSONArray("semantic_keywords") ?: JSONArray())
            .put("hit_count", entry.optInt("hit_count", 0))
            .put("salience", entry.optDouble("salience", 0.0))
            .put("promoted", isPromoted(entry))
            .put("promotion_hit_threshold", PROMOTION_HIT_THRESHOLD)
            .put("promoted_at_ms", entry.optLong("promoted_at_ms", 0L))
            .put("created_at_ms", entry.optLong("created_at_ms", 0L))
            .put("last_accessed_at_ms", entry.optLong("last_accessed_at_ms", 0L))
    }

    private fun promotedEntries(context: Context): List<JSONObject> {
        return readEntries(context)
            .filter(::isPromoted)
            .sortedWith(
                compareByDescending<JSONObject> { it.optDouble("salience", 0.0) }
                    .thenByDescending { it.optInt("hit_count", 0) }
                    .thenByDescending { it.optLong("last_accessed_at_ms", 0L) },
            )
    }

    private fun maybePromote(entry: JSONObject, now: Long) {
        if (entry.optInt("hit_count", 0) >= PROMOTION_HIT_THRESHOLD && !entry.optBoolean("promoted", false)) {
            entry.put("promoted", true)
                .put("promoted_at_ms", now)
                .put("tags", JSONArray((jsonStringList(entry.optJSONArray("tags")) + "promoted").distinct()))
        }
    }

    private fun isPromoted(entry: JSONObject): Boolean {
        return entry.optBoolean("promoted", false) || entry.optInt("hit_count", 0) >= PROMOTION_HIT_THRESHOLD
    }

    private fun salienceFor(content: String, keywordCount: Int, hitCount: Int): Double {
        val lengthScore = (content.length.coerceAtMost(240) / 240.0) * 2.0
        return 1.0 + lengthScore + keywordCount.coerceAtMost(10) * 0.15 + hitCount * 0.25
    }

    private fun keywords(text: String): Set<String> {
        return Regex("""[a-zA-Z0-9_./-]{3,}""")
            .findAll(text.lowercase(Locale.US))
            .map { it.value.trim('.', '/', '-') }
            .filter { it.length >= 3 && it !in STOPWORDS }
            .take(32)
            .toSet()
    }

    private fun extractEntities(text: String): List<String> {
        return Regex("""\b[A-Z][A-Za-z0-9_.-]{2,}\b""")
            .findAll(text)
            .map { it.value }
            .filterNot { it.lowercase(Locale.US) in STOPWORDS }
            .distinct()
            .take(16)
            .toList()
    }

    private fun normalizeContent(value: String): String {
        return value.lowercase(Locale.US)
            .replace(Regex("""\s+"""), " ")
            .trim()
    }

    private fun stringArray(value: Any?): List<String> {
        return when (value) {
            is JSONArray -> buildList {
                for (index in 0 until value.length()) {
                    value.optString(index).takeIf { it.isNotBlank() }?.let(::add)
                }
            }
            is String -> value.split(',').map { it.trim() }.filter { it.isNotBlank() }
            else -> emptyList()
        }
    }

    private fun jsonStringList(array: JSONArray?): List<String> {
        if (array == null) return emptyList()
        return buildList {
            for (index in 0 until array.length()) {
                array.optString(index).takeIf { it.isNotBlank() }?.let(::add)
            }
        }
    }

    private fun jsonObjectList(array: JSONArray?): List<JSONObject> {
        if (array == null) return emptyList()
        return buildList {
            for (index in 0 until array.length()) {
                array.optJSONObject(index)?.let(::add)
            }
        }
    }

    private fun readEntries(context: Context): MutableList<JSONObject> {
        val raw = prefs(context).getString(ENTRIES_KEY, "[]").orEmpty()
        val array = runCatching { JSONArray(raw) }.getOrDefault(JSONArray())
        return buildList {
            for (index in 0 until array.length()) {
                array.optJSONObject(index)?.let(::add)
            }
        }.toMutableList()
    }

    private fun trimAndSave(context: Context, entries: MutableList<JSONObject>) {
        val trimmed = entries
            .sortedWith(compareByDescending<JSONObject> { it.optDouble("salience", 0.0) }.thenByDescending { it.optLong("last_accessed_at_ms", 0L) })
            .take(MAX_ENTRIES)
            .toMutableList()
        saveEntries(context, trimmed)
    }

    private fun saveEntries(context: Context, entries: List<JSONObject>) {
        prefs(context).edit().putString(ENTRIES_KEY, JSONArray(entries).toString()).apply()
    }

    private fun prefs(context: Context) = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private fun card(title: String, body: String): JSONObject = JSONObject()
        .put("title", title)
        .put("body", body)

    private val STOPWORDS = setOf(
        "the", "and", "for", "with", "that", "this", "from", "into", "user", "chat", "tool", "should", "would",
        "could", "have", "has", "are", "was", "were", "you", "your", "they", "them", "their", "hermes",
    )
}
