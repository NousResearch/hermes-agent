package com.nousresearch.hermesagent.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalIntelligenceQuickActionsTest {
    @Test
    fun quickActionsExposeDirectDiagnosticsForSignalCards() {
        val actionsById = SIGNAL_INTELLIGENCE_QUICK_ACTIONS.associateBy { it.id }

        assertEquals("signal_awareness_report", actionsById.getValue("signal_overview").diagnosticAction)
        assertEquals("agent_environment_report", actionsById.getValue("agent_environment").diagnosticAction)
        assertEquals("soc_compatibility_report", actionsById.getValue("soc_compatibility").diagnosticAction)
        assertEquals("wifi_analyzer_report", actionsById.getValue("wifi_analyzer").diagnosticAction)
        assertEquals("wifi_scan", actionsById.getValue("wifi_nearby").diagnosticAction)
        assertEquals("wifi_channel_utilization", actionsById.getValue("wifi_occupancy").diagnosticAction)
        assertEquals("bluetooth_analyzer_report", actionsById.getValue("bluetooth_analyzer").diagnosticAction)
        assertEquals("bluetooth_signal_history", actionsById.getValue("bluetooth_history").diagnosticAction)
        assertEquals("sensor_analyzer_report", actionsById.getValue("sensor_analyzer").diagnosticAction)
        assertEquals("radio_signal_status", actionsById.getValue("radio_limits").diagnosticAction)
        SIGNAL_INTELLIGENCE_QUICK_ACTIONS.forEach { action ->
            assertTrue(action.prompt.contains("android_device_diagnostics_tool action=${action.diagnosticAction}"))
            val parsed = requireNotNull(NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(action.prompt))
            assertEquals(action.diagnosticAction, parsed.getString("action"))
            if ("refresh=false" in action.prompt) {
                assertFalse(parsed.getBoolean("refresh"))
            }
        }
    }
}
