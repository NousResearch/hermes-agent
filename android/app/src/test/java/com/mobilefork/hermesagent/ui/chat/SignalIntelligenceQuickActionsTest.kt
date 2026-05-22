package com.mobilefork.hermesagent.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalIntelligenceQuickActionsTest {
    @Test
    fun quickActionsExposeDirectDiagnosticsForSignalCards() {
        val actionsById = SIGNAL_INTELLIGENCE_QUICK_ACTIONS.associateBy { it.id }

        assertEquals("signal_awareness_report", actionsById.getValue("signal_overview").diagnosticAction)
        assertEquals("agent_signal_evidence_report", actionsById.getValue("signal_evidence").diagnosticAction)
        assertEquals("agent_environment_report", actionsById.getValue("agent_environment").diagnosticAction)
        assertEquals("agent_observation_report", actionsById.getValue("agent_observation").diagnosticAction)
        assertEquals("agent_card_manifest_report", actionsById.getValue("card_manifest").diagnosticAction)
        assertEquals("soc_compatibility_report", actionsById.getValue("soc_compatibility").diagnosticAction)
        assertEquals("gpu_backend_risk_report", actionsById.getValue("backend_risk").diagnosticAction)
        assertEquals("local_inference_compatibility_report", actionsById.getValue("inference_compatibility").diagnosticAction)
        assertEquals("local_backend_runtime_report", actionsById.getValue("runtime_backend").diagnosticAction)
        assertEquals("wifi_analyzer_report", actionsById.getValue("wifi_analyzer").diagnosticAction)
        assertEquals("wifi_scan", actionsById.getValue("wifi_nearby").diagnosticAction)
        assertEquals("wifi_channel_utilization", actionsById.getValue("wifi_occupancy").diagnosticAction)
        assertEquals("bluetooth_analyzer_report", actionsById.getValue("bluetooth_analyzer").diagnosticAction)
        assertEquals("bluetooth_signal_history", actionsById.getValue("bluetooth_history").diagnosticAction)
        assertEquals("sensor_analyzer_report", actionsById.getValue("sensor_analyzer").diagnosticAction)
        assertEquals("motion_sensor_history", actionsById.getValue("motion_history").diagnosticAction)
        assertEquals("radio_signal_graph", actionsById.getValue("radio_limits").diagnosticAction)
        assertEquals("Runtime Backend", actionsById.getValue("runtime_backend").label)
        assertEquals("Backend Risk", actionsById.getValue("backend_risk").label)
        assertEquals("Inference Fit", actionsById.getValue("inference_compatibility").label)
        assertEquals("Evidence Bundle", actionsById.getValue("signal_evidence").label)
        assertEquals("Agent Observation", actionsById.getValue("agent_observation").label)
        assertEquals("Card Manifest", actionsById.getValue("card_manifest").label)
        assertEquals("Radio Signals", actionsById.getValue("radio_limits").label)
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
