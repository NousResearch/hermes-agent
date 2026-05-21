package com.mobilefork.hermesagent.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenGuiActionHistoryTest {
    @Test
    fun recordsOpenGuiSemanticHistoryAndRecentHashes() {
        val history = OpenGuiActionHistory(maxSize = 3)

        history.rememberScreenHash("abcd1234")
        history.recordAction(
            OpenGuiActionCompat.parse(
                """
                Summary: open the app drawer
                Thought: use the visible launcher icon.
                Action: click(start_box='<point>500 250</point>')
                """.trimIndent(),
            ),
        )
        history.recordAction(
            OpenGuiActionCompat.parse(
                """
                Action_Summary: type the search term
                Action: type(content='Hermes')
                """.trimIndent(),
            ),
        )

        val json = history.snapshotJson()

        assertTrue(json.getBoolean("success"))
        assertTrue(json.getBoolean("history_supported"))
        assertEquals("deterministic_local", json.getString("summary_style"))
        assertEquals(2, json.getInt("action_count"))
        assertEquals(1, json.getInt("screen_hash_count"))
        assertTrue(json.getString("history_text").contains("[Loop 1] [GUI]"))
        assertTrue(json.getString("history_text").contains("Summary: open the app drawer"))
        assertTrue(json.getString("history_text").contains("Action: type(content='Hermes')"))
        assertTrue(json.getString("history_summary").contains("click -> type"))
        assertEquals("type", json.getJSONArray("actions").getJSONObject(1).getString("action_type"))
        assertEquals("type the search term", json.getJSONArray("actions").getJSONObject(1).getString("action_summary"))
    }

    @Test
    fun clearReportsAndResetsOpenGuiHistory() {
        val history = OpenGuiActionHistory(maxSize = 3)
        history.rememberScreenHash("abcd1234")
        history.recordAction(OpenGuiActionCompat.parse("Action: press_back()"))

        val cleared = history.clearJson()
        val snapshot = history.snapshotJson()

        assertTrue(cleared.getBoolean("history_cleared"))
        assertEquals(1, cleared.getInt("cleared_action_count"))
        assertEquals(1, cleared.getInt("cleared_screen_hash_count"))
        assertEquals(0, snapshot.getInt("action_count"))
        assertFalse(snapshot.getString("history_summary").isBlank())
    }
}
