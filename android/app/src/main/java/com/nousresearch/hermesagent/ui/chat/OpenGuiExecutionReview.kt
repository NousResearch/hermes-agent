package com.nousresearch.hermesagent.ui.chat

import org.json.JSONObject
import kotlin.math.abs

internal object OpenGuiExecutionReview {
    const val ACTION_WINDOW_SIZE = 10
    private const val ACTION_REPETITION_THRESHOLD = 5
    private const val COORD_SIMILARITY_THRESHOLD = 0.05
    private const val CYCLE_MIN_REPETITIONS = 3
    private const val MAX_CYCLE_LENGTH = 3
    private const val CONSECUTIVE_SCROLL_EXIT_THRESHOLD = 8
    private val passiveActionTypes = setOf("wait", "scroll", "finished", "press_back", "press_home")

    fun review(
        recentActions: List<ParsedOpenGuiAction>,
        nextAction: ParsedOpenGuiAction,
    ): OpenGuiExecutionReviewResult {
        val window = (recentActions + nextAction).takeLast(ACTION_WINDOW_SIZE)
        detectActionRepetition(window)?.let { return it }
        detectActionCycle(window)?.let { return it }
        detectConsecutiveScroll(window)?.let { return it }
        return OpenGuiExecutionReviewResult(
            detected = false,
            kind = "ok",
            reason = "",
            recentActionCount = window.size,
        )
    }

    fun blockedActionJson(
        parsed: ParsedOpenGuiAction,
        review: OpenGuiExecutionReviewResult,
    ): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("opengui_action_compat", true)
            .put("requires_replan", true)
            .put("error", "OpenGUI execution review blocked a likely repeated action loop. Request a fresh snapshot and choose a different next step.")
            .put("parsed_action", parsed.toJson())
            .put("execution_review", review.toJson())
    }

    private fun detectActionRepetition(window: List<ParsedOpenGuiAction>): OpenGuiExecutionReviewResult? {
        if (window.size < ACTION_REPETITION_THRESHOLD) {
            return null
        }
        val tail = window.takeLast(ACTION_REPETITION_THRESHOLD)
        val first = tail.first()
        if (!tail.all { isSimilarAction(first, it) }) {
            return null
        }
        return OpenGuiExecutionReviewResult(
            detected = true,
            kind = "action_repetition",
            reason = "Executed similar ${first.actionType} actions $ACTION_REPETITION_THRESHOLD times in a row; replan before continuing.",
            recentActionCount = window.size,
            threshold = ACTION_REPETITION_THRESHOLD,
        )
    }

    private fun detectActionCycle(window: List<ParsedOpenGuiAction>): OpenGuiExecutionReviewResult? {
        for (cycleLength in 2..MAX_CYCLE_LENGTH) {
            val needed = cycleLength * CYCLE_MIN_REPETITIONS
            if (window.size < needed) {
                continue
            }
            val tail = window.takeLast(needed)
            val isCycle = tail.indices.drop(cycleLength).all { index ->
                isSimilarActionStrict(tail[index % cycleLength], tail[index])
            }
            if (!isCycle) {
                continue
            }
            val cycleTypes = tail.take(cycleLength).map { it.actionType }.toSet()
            if (cycleTypes.size < 2) {
                continue
            }
            val cycleDescription = tail.take(cycleLength).joinToString(" -> ") { it.actionType }
            return OpenGuiExecutionReviewResult(
                detected = true,
                kind = "action_cycle",
                reason = "Detected a $cycleLength-step action cycle repeated $CYCLE_MIN_REPETITIONS times ($cycleDescription); replan before continuing.",
                recentActionCount = window.size,
                threshold = needed,
            )
        }
        return null
    }

    private fun detectConsecutiveScroll(window: List<ParsedOpenGuiAction>): OpenGuiExecutionReviewResult? {
        if (window.size < CONSECUTIVE_SCROLL_EXIT_THRESHOLD) {
            return null
        }
        val tail = window.takeLast(CONSECUTIVE_SCROLL_EXIT_THRESHOLD)
        if (!tail.all { it.actionType == "scroll" }) {
            return null
        }
        return OpenGuiExecutionReviewResult(
            detected = true,
            kind = "scroll_loop",
            reason = "Executed scroll $CONSECUTIVE_SCROLL_EXIT_THRESHOLD times in a row; the search strategy is not making progress.",
            recentActionCount = window.size,
            threshold = CONSECUTIVE_SCROLL_EXIT_THRESHOLD,
        )
    }

    private fun isSimilarAction(a: ParsedOpenGuiAction, b: ParsedOpenGuiAction): Boolean {
        if (a.actionType != b.actionType) {
            return false
        }
        val first = a.startCoords
        val second = b.startCoords
        if (first == null || second == null) {
            return a.actionType !in passiveActionTypes
        }
        return abs(first.x - second.x) + abs(first.y - second.y) <= COORD_SIMILARITY_THRESHOLD
    }

    private fun isSimilarActionStrict(a: ParsedOpenGuiAction, b: ParsedOpenGuiAction): Boolean {
        if (a.actionType != b.actionType) {
            return false
        }
        val first = a.startCoords
        val second = b.startCoords
        if (first == null || second == null) {
            return true
        }
        return abs(first.x - second.x) + abs(first.y - second.y) <= COORD_SIMILARITY_THRESHOLD
    }
}

internal data class OpenGuiExecutionReviewResult(
    val detected: Boolean,
    val kind: String,
    val reason: String,
    val recentActionCount: Int,
    val threshold: Int = 0,
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("detected", detected)
            .put("kind", kind)
            .put("reason", reason)
            .put("recent_action_count", recentActionCount)
            .put("window_size", OpenGuiExecutionReview.ACTION_WINDOW_SIZE)
            .also { json ->
                if (threshold > 0) {
                    json.put("threshold", threshold)
                }
            }
    }
}
