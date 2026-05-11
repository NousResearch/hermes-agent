package com.nousresearch.hermesagent.ui.chat

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenGuiUserHandoffTest {
    @Test
    fun buildsUserVisibleNotificationPayloadForCallUser() {
        val parsed = OpenGuiActionCompat.parse("need_login(content='Please log in, then return to Hermes')")

        val payload = OpenGuiUserHandoff.notificationPayload(
            parsed = parsed,
            message = OpenGuiUserHandoff.userMessage(parsed),
            sessionId = "chat-123",
        )

        assertEquals("post", payload.getString("notification_action"))
        assertEquals("Hermes needs you", payload.getString("title"))
        assertEquals("OpenGUI user handoff", payload.getString("status_text"))
        assertTrue(payload.getBoolean("ongoing"))
        val visibleText = payload.getString("text")
            .substringAfter("] ")
            .ifBlank { payload.getString("text") }
        assertEquals("Please log in, then return to Hermes", visibleText)
        assertEquals(2, payload.getJSONArray("notification_buttons").length())
    }

    @Test
    fun callUserResultReportsPausedHandoffEvenWhenNotificationPermissionIsMissing() {
        val result = OpenGuiUserHandoff.resultJson(
            message = "Confirm the payment manually",
            notification = JSONObject()
                .put("success", false)
                .put("requires_permission", "android.permission.POST_NOTIFICATIONS"),
            toast = JSONObject().put("success", true),
            vibration = JSONObject().put("success", true),
        )

        assertTrue(result.getBoolean("success"))
        assertEquals("call_user", result.getString("status"))
        assertTrue(result.getBoolean("paused"))
        assertFalse(result.getBoolean("terminal"))
        assertTrue(result.getBoolean("requires_user_intervention"))
        assertTrue(result.getBoolean("surfaced_on_device"))
        assertTrue(result.getJSONArray("compatible_commands").toString().contains("/opengui resume"))
    }
}
