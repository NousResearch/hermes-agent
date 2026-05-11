package com.nousresearch.hermesagent.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenGuiActionCompatTest {
    @Test
    fun parsesOpenGuiPointClickFromVlmActionBlock() {
        val parsed = OpenGuiActionCompat.parse(
            """
            Summary: tap the search field
            Thought: use the visual coordinate.
            Action: click(start_box='<point>500 250</point>')
            """.trimIndent(),
        )

        assertEquals("click", parsed.actionType)
        assertEquals(0.5, parsed.startCoords!!.x, 0.0001)
        assertEquals(0.25, parsed.startCoords!!.y, 0.0001)
        assertFalse(parsed.terminal)
    }

    @Test
    fun parsesBoxDragAndFullWidthPunctuation() {
        val parsed = OpenGuiActionCompat.parse(
            "Action：drag（start_box='<bbox>100 200 200 400</bbox>'， end_box='(700,800)'）",
        )

        assertEquals("swipe", parsed.actionType)
        assertEquals(0.15, parsed.startCoords!!.x, 0.0001)
        assertEquals(0.3, parsed.startCoords!!.y, 0.0001)
        assertEquals(0.7, parsed.endCoords!!.x, 0.0001)
        assertEquals(0.8, parsed.endCoords!!.y, 0.0001)
    }

    @Test
    fun mapsUserInterventionActionsToCallUser() {
        val parsed = OpenGuiActionCompat.parse("delete_confirm(content='Delete the selected draft?')")

        assertEquals("call_user", parsed.actionType)
        assertTrue(parsed.requiresUserIntervention)
        assertTrue(parsed.content.contains("[delete_confirm]"))
        assertTrue(parsed.content.contains("Delete the selected draft?"))
    }

    @Test
    fun preservesOpenAppAndFinishedSignals() {
        val openApp = OpenGuiActionCompat.parse("open_app(app_name='Chrome')")
        val finished = OpenGuiActionCompat.parse("finished(content='Done')")

        assertEquals("open_app", openApp.actionType)
        assertEquals("Chrome", openApp.appName)
        assertEquals("finished", finished.actionType)
        assertTrue(finished.terminal)
    }

    @Test
    fun parsesParsedOpenGuiCoordinateNamesAndRecoveryAction() {
        val parsed = OpenGuiActionCompat.parse("swipe(start_coords=[0.25,0.5], end_coords=[0.75,0.2])")
        val recovery = OpenGuiActionCompat.parse("downgrade_to_a11y()")

        assertEquals("swipe", parsed.actionType)
        assertEquals(0.25, parsed.startCoords!!.x, 0.0001)
        assertEquals(0.5, parsed.startCoords!!.y, 0.0001)
        assertEquals(0.75, parsed.endCoords!!.x, 0.0001)
        assertEquals(0.2, parsed.endCoords!!.y, 0.0001)
        assertEquals("downgrade_to_a11y", recovery.actionType)
    }
}
