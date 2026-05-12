package com.nousresearch.hermesagent.ui.chat

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenGuiSenseStatusTest {
    @Test
    fun prefersAccessibilityWhenSemanticSnapshotIsAvailable() {
        val history = OpenGuiActionHistory(maxSize = 3).apply {
            rememberScreenHash("abcd1234")
            recordAction(OpenGuiActionCompat.parse("Action: click(start_box='<point>500 500</point>')"))
        }.snapshotJson()

        val result = OpenGuiSenseStatus.build(
            accessibilityEnabled = true,
            accessibilityConnected = true,
            screenshotSupported = true,
            activePackage = "com.android.chrome",
            screenWidth = 1080,
            screenHeight = 2400,
            density = 3f,
            includeSnapshot = true,
            includeScreenshot = false,
            snapshot = JSONObject()
                .put("accessibility_connected", true)
                .put("ui_state_hash", "abcd1234")
                .put("nodes", org.json.JSONArray()),
            screenshot = null,
            history = history,
        )

        assertTrue(result.getBoolean("opengui_sense_compat"))
        assertEquals("a11y", result.getString("recommended_channel"))
        assertTrue(result.getBoolean("parallel_a11y_gui_probe_supported"))
        assertEquals(1, result.getInt("recent_action_count"))
        assertEquals(1, result.getInt("recent_screen_hash_count"))
        assertEquals("device:a11y_tree", result.getJSONArray("channels").getJSONObject(0).getString("event_alias"))
        assertTrue(result.getString("fallback_policy").contains("visual screenshot"))
    }

    @Test
    fun asksForPermissionSetupWhenNeitherChannelIsReady() {
        val result = OpenGuiSenseStatus.build(
            accessibilityEnabled = false,
            accessibilityConnected = false,
            screenshotSupported = true,
            activePackage = "",
            screenWidth = null,
            screenHeight = null,
            density = null,
            includeSnapshot = false,
            includeScreenshot = false,
            snapshot = null,
            screenshot = null,
            history = OpenGuiActionHistory().snapshotJson(),
        )

        assertEquals("permission_setup", result.getString("recommended_channel"))
        assertTrue(result.getString("recommended_next_action").contains("open_accessibility_settings"))
        assertFalse(result.getJSONArray("channels").getJSONObject(0).getBoolean("available"))
        assertTrue(result.getJSONArray("channels").getJSONObject(2).getBoolean("preferred"))
    }
}
