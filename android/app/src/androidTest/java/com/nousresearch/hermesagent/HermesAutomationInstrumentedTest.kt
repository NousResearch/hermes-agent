package com.nousresearch.hermesagent

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.nousresearch.hermesagent.device.HermesAutomationBridge
import com.nousresearch.hermesagent.device.HermesAutomationStore
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class HermesAutomationInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesAutomationStore(app).clear()
    }

    @Test
    fun shellAutomationCanBeCreatedRunDisabledAndDeleted() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-automation-smoke.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shell_task",
                JSONObject()
                    .put("label", "Automation smoke")
                    .put("command", "printf automation-ok > \"\$HOME/hermes-automation-smoke.txt\"")
                    .put("enabled", false),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val id = created.getJSONObject("automation").getString("id")
        assertEquals("manual", created.getJSONObject("automation").getString("trigger_type"))

        val run = JSONObject(HermesAutomationBridge.performActionJson(app, "run", JSONObject().put("id", id)))
        assertTrue(run.toString(), run.getBoolean("success"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("automation-ok", target.readText())
        assertEquals(0, run.getJSONObject("automation").getInt("last_exit_code"))

        val enabled = JSONObject(HermesAutomationBridge.performActionJson(app, "enable", JSONObject().put("id", id)))
        assertTrue(enabled.toString(), enabled.getBoolean("success"))
        assertTrue(enabled.getJSONObject("automation").getBoolean("enabled"))

        val disabled = JSONObject(HermesAutomationBridge.performActionJson(app, "disable", JSONObject().put("id", id)))
        assertTrue(disabled.toString(), disabled.getBoolean("success"))
        assertFalse(disabled.getJSONObject("automation").getBoolean("enabled"))

        val deleted = JSONObject(HermesAutomationBridge.performActionJson(app, "delete", JSONObject().put("id", id)))
        assertTrue(deleted.toString(), deleted.getBoolean("success"))
    }

    @Test
    fun intervalAutomationRejectsTooFrequentSchedules() {
        val rejected = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shell_task",
                JSONObject()
                    .put("label", "Too fast")
                    .put("command", "printf nope")
                    .put("interval_minutes", 1),
            )
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
        assertTrue(rejected.getString("error").contains("at least 15"))
    }

    @Test
    fun triggerAutomationExpandsVariablesAndRunsShellTask() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-automation-trigger.txt").apply { delete() }

        val variable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%message")
                    .put("value", "trigger-ok"),
            )
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))
        assertEquals("MESSAGE", variable.getString("name"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shell_task",
                JSONObject()
                    .put("label", "Boot trigger smoke")
                    .put("command", "printf %MESSAGE > \"\$HOME/hermes-automation-trigger.txt\"")
                    .put("trigger", "boot"),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("boot", created.getJSONObject("automation").getString("trigger_type"))

        val triggered = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_trigger",
                JSONObject().put("trigger", "boot"),
            )
        )
        assertTrue(triggered.toString(), triggered.getBoolean("success"))
        assertEquals(1, triggered.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("trigger-ok", target.readText())
    }
}
