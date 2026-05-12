package com.nousresearch.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject

internal object OpenGuiSenseStatus {
    fun build(
        accessibilityEnabled: Boolean,
        accessibilityConnected: Boolean,
        screenshotSupported: Boolean,
        activePackage: String,
        screenWidth: Int?,
        screenHeight: Int?,
        density: Float?,
        includeSnapshot: Boolean,
        includeScreenshot: Boolean,
        snapshot: JSONObject?,
        screenshot: JSONObject?,
        history: JSONObject,
    ): JSONObject {
        val snapshotAvailable = snapshot?.optBoolean("accessibility_connected", false) == true &&
            !snapshot.has("error")
        val screenshotAvailable = screenshot?.optBoolean("accessibility_connected", false) == true &&
            !screenshot.has("error")
        val a11yAvailable = accessibilityConnected || snapshotAvailable
        val guiAvailable = screenshotSupported && (accessibilityConnected || screenshotAvailable)
        val recommendedChannel = when {
            snapshotAvailable || a11yAvailable -> "a11y"
            guiAvailable -> "gui"
            else -> "permission_setup"
        }

        return JSONObject()
            .put("success", true)
            .put("opengui_sense_compat", true)
            .put("sense_node_supported", true)
            .put("recommended_channel", recommendedChannel)
            .put("recommended_next_action", recommendedNextAction(recommendedChannel))
            .put("a11y_channel_supported", true)
            .put("gui_channel_supported", screenshotSupported)
            .put("parallel_a11y_gui_probe_supported", screenshotSupported)
            .put("accessibility_service_enabled", accessibilityEnabled)
            .put("accessibility_connected", accessibilityConnected)
            .put("screenshot_capture_supported", screenshotSupported)
            .put("active_package", activePackage)
            .put("current_app_name", activePackage)
            .put("coordinate_space", "absolute_px")
            .put("scale_factor", 1.0)
            .put("screen_width", screenWidth ?: JSONObject.NULL)
            .put("screen_height", screenHeight ?: JSONObject.NULL)
            .put("density", density?.toDouble() ?: JSONObject.NULL)
            .put("include_snapshot", includeSnapshot)
            .put("include_screenshot", includeScreenshot)
            .put("channels", channelsJson(a11yAvailable, guiAvailable, recommendedChannel))
            .put("history", history)
            .put("recent_action_count", history.optInt("action_count"))
            .put("recent_screen_hash_count", history.optInt("screen_hash_count"))
            .put("loop_review_supported", true)
            .put(
                "fallback_policy",
                "Prefer accessibility snapshots for text-side control, request a visual screenshot when semantic state is unavailable or ambiguous, and ask the user for permission setup when neither channel is available.",
            )
            .also { output ->
                snapshot?.let { output.put("snapshot", it) }
                screenshot?.let { output.put("screenshot", it) }
            }
    }

    private fun channelsJson(
        a11yAvailable: Boolean,
        guiAvailable: Boolean,
        recommendedChannel: String,
    ): JSONArray {
        return JSONArray(
            listOf(
                JSONObject()
                    .put("channel", "a11y")
                    .put("available", a11yAvailable)
                    .put("preferred", recommendedChannel == "a11y")
                    .put("request_action", "snapshot")
                    .put("event_alias", "device:a11y_tree")
                    .put("description", "Accessibility semantic tree, selectors, text input, and global navigation."),
                JSONObject()
                    .put("channel", "gui")
                    .put("available", guiAvailable)
                    .put("preferred", recommendedChannel == "gui")
                    .put("request_action", "screenshot")
                    .put("event_alias", "device:screenshot")
                    .put("description", "Visual screenshot fallback for VLM coordinate actions and screen hash review."),
                JSONObject()
                    .put("channel", "user")
                    .put("available", true)
                    .put("preferred", recommendedChannel == "permission_setup")
                    .put("request_action", "call_user")
                    .put("description", "Ask the user to enable AccessibilityService, screenshot access, login, or sensitive confirmation."),
            ),
        )
    }

    private fun recommendedNextAction(channel: String): String {
        return when (channel) {
            "a11y" -> "Use android_ui_tool action=snapshot or selector actions; request visual_snapshot only when semantic state is insufficient."
            "gui" -> "Use android_ui_tool action=visual_snapshot before coordinate actions."
            else -> "Use android_ui_tool action=open_accessibility_settings and ask the user to enable Hermes accessibility."
        }
    }
}
