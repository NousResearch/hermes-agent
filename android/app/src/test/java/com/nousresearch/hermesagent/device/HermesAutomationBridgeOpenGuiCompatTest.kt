package com.nousresearch.hermesagent.device

import org.json.JSONObject
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class HermesAutomationBridgeOpenGuiCompatTest {
    @Test
    fun openguiSlashLifecycleSubcommandsDoNotRequireExecutionId() {
        val context = RuntimeEnvironment.getApplication()

        listOf("cancel", "pause", "resume").forEach { subcommand ->
            val result = JSONObject(
                HermesAutomationBridge.performActionJson(
                    context,
                    "operator_command",
                    JSONObject()
                        .put("command", "/opengui")
                        .put("subcommand", subcommand),
                ),
            )

            assertTrue(result.toString(), result.getBoolean("success"))
            assertFalse(result.toString(), result.getBoolean("handled"))
            assertEquals("no_active_execution", result.getString("status"))
            assertEquals(subcommand, result.getJSONObject("parsed_command").getString("type"))
        }
    }

    @Test
    fun openguiStandbyHeartbeatUpdatesStatusSummary() {
        val context = RuntimeEnvironment.getApplication()

        val heartbeat = JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "operator_heartbeat",
                JSONObject()
                    .put("deviceId", "device-1")
                    .put("deviceName", "Hermes Test Phone")
                    .put("source", "discord"),
            ),
        )

        assertTrue(heartbeat.toString(), heartbeat.getBoolean("success"))
        assertTrue(heartbeat.getBoolean("accepted"))
        assertEquals("standby:heartbeat", heartbeat.getString("event"))
        assertEquals("device-1", heartbeat.getString("device_id"))
        val standby = heartbeat.getJSONObject("standby_dispatch")
        assertTrue(standby.getBoolean("standby_heartbeat_supported"))
        assertEquals("discord", standby.getString("last_heartbeat_source"))
        assertEquals("device-1", standby.getString("last_heartbeat_device_id"))

        val status = JSONObject(HermesAutomationBridge.performActionJson(context, "operator_standby_status"))
            .getJSONObject("standby_dispatch")
        assertEquals("discord", status.getString("last_heartbeat_source"))
        assertTrue(status.getJSONArray("compatible_dispatch_payloads").toString().contains("operator_heartbeat"))
    }

    @Test
    fun openguiBatchHeartbeatReportsTerminalAndMissingExecutions() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()
        store.addRunEvent(
            HermesAutomationRunEvent(
                id = "local-run-1",
                automationId = "automation-1",
                automationLabel = "OpenGUI batch heartbeat",
                actionType = ACTION_TYPE_SHELL,
                trigger = TRIGGER_REMOTE_DISPATCH,
                success = true,
                exitCode = 0,
                result = "ok",
                startedAtEpochMs = 1000,
                finishedAtEpochMs = 1500,
                dispatchSource = "opengui_standby",
                dispatchChannel = "discord",
                remoteExecutionId = "42",
                remoteTaskId = "7",
                remoteTaskName = "OpenGUI batch heartbeat",
            ),
        )

        val result = JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "operator_batch_heartbeat",
                JSONObject().put("executionIds", JSONArray(listOf("42", "99"))),
            ),
        )

        assertTrue(result.toString(), result.getBoolean("success"))
        assertEquals(30, result.getInt("heartbeatInterval"))
        assertTrue(result.getJSONArray("terminalExecutionIds").toString().contains("42"))
        assertTrue(result.getJSONArray("failedExecutionIds").toString().contains("99"))
        assertTrue(result.getJSONObject("standby_dispatch").getBoolean("batch_heartbeat_supported"))
    }

    @Test
    fun openguiModelRoutingCommandExposesPlannerAndVisionRoles() {
        val context = RuntimeEnvironment.getApplication()

        val result = JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "operator_command",
                JSONObject().put("command", "/opengui models"),
            ),
        )

        assertTrue(result.toString(), result.getBoolean("success"))
        assertTrue(result.getBoolean("role_routing_supported"))
        assertEquals("models", result.getJSONObject("parsed_command").getString("type"))
        assertTrue(result.getJSONArray("roles").toString().contains("planner"))
        assertTrue(result.getJSONArray("roles").toString().contains("executor_vlm"))
    }
}
