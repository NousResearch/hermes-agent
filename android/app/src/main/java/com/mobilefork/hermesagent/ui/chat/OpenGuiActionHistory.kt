package com.mobilefork.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject

internal class OpenGuiActionHistory(
    private val maxSize: Int = OpenGuiExecutionReview.ACTION_WINDOW_SIZE,
) {
    private val actions = ArrayDeque<ParsedOpenGuiAction>()
    private val screenHashes = ArrayDeque<String>()

    fun recordAction(parsed: ParsedOpenGuiAction) {
        actions.addLast(parsed)
        while (actions.size > maxSize) {
            actions.removeFirst()
        }
    }

    fun rememberScreenHash(hash: String) {
        val clean = hash.trim().lowercase().filter { it in '0'..'9' || it in 'a'..'f' }
        if (clean.length < 8) {
            return
        }
        screenHashes.addLast(clean)
        while (screenHashes.size > maxSize) {
            screenHashes.removeFirst()
        }
    }

    fun actionsList(): List<ParsedOpenGuiAction> = actions.toList()

    fun screenHashList(): List<String> = screenHashes.toList()

    fun snapshotJson(): JSONObject {
        val actionList = actionsList()
        val hashList = screenHashList()
        return JSONObject()
            .put("success", true)
            .put("opengui_action_compat", true)
            .put("history_supported", true)
            .put("summary_style", "deterministic_local")
            .put("action_count", actionList.size)
            .put("screen_hash_count", hashList.size)
            .put("max_history_size", maxSize)
            .put("actions", actionsJson(actionList))
            .put("screen_hashes", JSONArray(hashList))
            .put("history_text", historyText(actionList))
            .put("history_summary", summaryText(actionList, hashList))
    }

    fun clearJson(): JSONObject {
        val clearedActionCount = actions.size
        val clearedScreenHashCount = screenHashes.size
        actions.clear()
        screenHashes.clear()
        return JSONObject()
            .put("success", true)
            .put("opengui_action_compat", true)
            .put("history_cleared", true)
            .put("cleared_action_count", clearedActionCount)
            .put("cleared_screen_hash_count", clearedScreenHashCount)
    }

    private fun actionsJson(actionList: List<ParsedOpenGuiAction>): JSONArray {
        val array = JSONArray()
        actionList.forEachIndexed { index, parsed ->
            array.put(
                JSONObject()
                    .put("loop_index", index + 1)
                    .put("channel", "gui")
                    .put("action_type", parsed.actionType)
                    .put("raw_action", parsed.rawText)
                    .put("summary", parsed.summary)
                    .put("thought", parsed.thought)
                    .put("reflection", parsed.reflection)
                    .put("action_summary", parsed.actionSummary)
                    .put("terminal", parsed.terminal)
                    .put("requires_user_intervention", parsed.requiresUserIntervention)
                    .put("parsed_action", parsed.toJson()),
            )
        }
        return array
    }

    private fun historyText(actionList: List<ParsedOpenGuiAction>): String {
        if (actionList.isEmpty()) {
            return ""
        }
        return actionList.mapIndexed { index, parsed ->
            val parts = mutableListOf<String>()
            parsed.summary.takeIf { it.isNotBlank() }?.let { parts += "Summary: $it" }
            parsed.thought.takeIf { it.isNotBlank() }?.let { parts += "Thought: $it" }
            parsed.reflection.takeIf { it.isNotBlank() }?.let { parts += "Reflection: $it" }
            parts += "Action: ${parsed.rawText}"
            "[Loop ${index + 1}] [GUI] ${parts.joinToString("\n")}"
        }.joinToString("\n")
    }

    private fun summaryText(actionList: List<ParsedOpenGuiAction>, hashList: List<String>): String {
        if (actionList.isEmpty()) {
            return "No OpenGUI-compatible mobile UI actions have been recorded in this chat session."
        }
        val actionPath = actionList.joinToString(" -> ") { it.actionType }
        val latest = actionList.last()
        val latestIntent = latest.actionSummary.ifBlank { latest.summary }.ifBlank { latest.content }
        val latestText = latestIntent.takeIf { it.isNotBlank() }?.let { " Latest intent: $it" }.orEmpty()
        val hashText = if (hashList.isNotEmpty()) {
            " Recent screen hashes: ${hashList.takeLast(3).joinToString(", ")}."
        } else {
            ""
        }
        return "Recorded ${actionList.size} OpenGUI-compatible GUI actions: $actionPath.$latestText$hashText"
    }
}
