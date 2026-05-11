package com.nousresearch.hermesagent.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenGuiExecutionReviewTest {
    @Test
    fun allowsShortSimilarActionRuns() {
        val actions = List(4) { clickAt(0.5, 0.5) }

        val review = OpenGuiExecutionReview.review(actions.dropLast(1), actions.last())

        assertFalse(review.detected)
        assertEquals("ok", review.kind)
    }

    @Test
    fun detectsRepeatedCoordinateActionsBeforeExecutingFifthAction() {
        val actions = List(5) { clickAt(0.5 + it * 0.002, 0.5) }

        val review = OpenGuiExecutionReview.review(actions.dropLast(1), actions.last())

        assertTrue(review.detected)
        assertEquals("action_repetition", review.kind)
        assertTrue(review.reason.contains("similar click actions"))
    }

    @Test
    fun detectsTwoStepActionCycles() {
        val cycle = listOf(
            clickAt(0.2, 0.2),
            swipeFrom(0.6, 0.7),
            clickAt(0.21, 0.19),
            swipeFrom(0.59, 0.69),
            clickAt(0.2, 0.2),
            swipeFrom(0.6, 0.7),
        )

        val review = OpenGuiExecutionReview.review(cycle.dropLast(1), cycle.last())

        assertTrue(review.detected)
        assertEquals("action_cycle", review.kind)
    }

    @Test
    fun detectsLongScrollRunsWithoutCoordinates() {
        val scrolls = List(8) { OpenGuiActionCompat.parse("scroll(direction='down')") }

        val review = OpenGuiExecutionReview.review(scrolls.dropLast(1), scrolls.last())

        assertTrue(review.detected)
        assertEquals("scroll_loop", review.kind)
    }

    @Test
    fun emitsBlockedJsonForPlannerReplan() {
        val action = clickAt(0.5, 0.5)
        val review = OpenGuiExecutionReview.review(List(4) { clickAt(0.5, 0.5) }, action)
        val json = OpenGuiExecutionReview.blockedActionJson(action, review)

        assertFalse(json.getBoolean("success"))
        assertTrue(json.getBoolean("requires_replan"))
        assertEquals("action_repetition", json.getJSONObject("execution_review").getString("kind"))
    }

    private fun clickAt(x: Double, y: Double): ParsedOpenGuiAction {
        return OpenGuiActionCompat.parse("click(start_box='<point>${x * 1000} ${y * 1000}</point>')")
    }

    private fun swipeFrom(x: Double, y: Double): ParsedOpenGuiAction {
        return OpenGuiActionCompat.parse(
            "swipe(start_box='<point>${x * 1000} ${y * 1000}</point>', end_box='<point>${x * 1000} ${(y - 0.2) * 1000}</point>')",
        )
    }
}
