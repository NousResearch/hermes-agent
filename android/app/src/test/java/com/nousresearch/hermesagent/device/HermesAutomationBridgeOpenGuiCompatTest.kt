package com.nousresearch.hermesagent.device

import org.json.JSONObject
import org.junit.Assert.assertEquals
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
            assertEquals("recognized_not_active", result.getString("status"))
            assertEquals(subcommand, result.getJSONObject("parsed_command").getString("type"))
        }
    }
}
