package com.nousresearch.hermesagent.device

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesAutomationStoreTest {
    @Test
    fun storeRoundTripsAndRemovesAutomationRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val record = HermesAutomationRecord(
            id = "auto-test",
            label = "Test automation",
            actionType = ACTION_TYPE_SHELL,
            command = "printf ok",
            useShizuku = false,
            triggerType = TRIGGER_INTERVAL,
            intervalMinutes = 15,
            enabled = true,
            createdAtEpochMs = 10L,
            updatedAtEpochMs = 20L,
        )

        store.upsert(record)
        val loaded = store.get("auto-test")
        assertEquals("Test automation", loaded?.label)
        assertEquals(TRIGGER_INTERVAL, loaded?.triggerType)
        assertEquals(15, loaded?.intervalMinutes)
        assertTrue(loaded?.enabled ?: false)

        store.upsert(record.copy(enabled = false, lastExitCode = 0, lastSuccess = true))
        val updated = store.get("auto-test")
        assertFalse(updated?.enabled ?: true)
        assertEquals(0, updated?.lastExitCode)
        assertEquals(true, updated?.lastSuccess)

        assertTrue(store.remove("auto-test"))
        assertNull(store.get("auto-test"))
        assertFalse(store.remove("auto-test"))
    }

    @Test
    fun storeNormalizesAndPersistsVariables() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        assertTrue(store.setVariable("%message", "hello"))
        assertEquals("hello", store.getVariable("MESSAGE"))
        assertEquals("hello", store.listVariables().getString("MESSAGE"))

        assertTrue(store.removeVariable("message"))
        assertNull(store.getVariable("MESSAGE"))
        assertFalse(store.setVariable("bad name", "nope"))
    }

    @Test
    fun bridgeCreatesPhoneStateTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shell_task",
                org.json.JSONObject()
                    .put("id", "auto-power")
                    .put("command", "printf ok")
                    .put("trigger", "charging"),
            ),
        )

        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(TRIGGER_POWER_CONNECTED, created.getJSONObject("automation").getString("trigger_type"))
    }

    @Test
    fun bridgeCreatesAndRunsAppForegroundTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_app_launch_task",
                org.json.JSONObject()
                    .put("id", "auto-foreground")
                    .put("package_name", "com.nousresearch.hermesagent.missing")
                    .put("trigger", "application")
                    .put("trigger_package_name", "com.example.foreground"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(TRIGGER_APP_FOREGROUND, created.getJSONObject("automation").getString("trigger_type"))
        assertEquals("com.example.foreground", created.getJSONObject("automation").getString("trigger_package_name"))

        val missed = org.json.JSONObject(HermesAutomationBridge.runAppForegroundTriggerJson(context, "com.example.other"))
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))

        val matched = org.json.JSONObject(HermesAutomationBridge.runAppForegroundTriggerJson(context, "com.example.foreground"))
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertFalse(matched.getJSONArray("results").getJSONObject(0).getBoolean("success"))

        val generic = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "app_foreground"),
            ),
        )
        assertFalse(generic.toString(), generic.getBoolean("success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shell_task",
                org.json.JSONObject()
                    .put("command", "printf no")
                    .put("trigger", "app_foreground"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCreatesFileAndSystemActionRecords() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val fileWrite = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_file_write_task",
                org.json.JSONObject()
                    .put("path", "tasker-file.txt")
                    .put("content", "ok")
                    .put("enabled", false),
            ),
        )
        assertTrue(fileWrite.toString(), fileWrite.getBoolean("success"))
        assertEquals(ACTION_TYPE_FILE_WRITE, fileWrite.getJSONObject("automation").getString("action_type"))

        val systemAction = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_system_action_task",
                org.json.JSONObject()
                    .put("system_action", "stop_background_runtime")
                    .put("enabled", false),
            ),
        )
        assertTrue(systemAction.toString(), systemAction.getBoolean("success"))
        assertEquals(ACTION_TYPE_SYSTEM_ACTION, systemAction.getJSONObject("automation").getString("action_type"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_system_action_task",
                org.json.JSONObject().put("system_action", "run_privileged_shell"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCreatesAndRunsUiActionRecordsThroughAccessibilityBoundary() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_ui_action_task",
                org.json.JSONObject()
                    .put("id", "auto-ui")
                    .put("ui_action", "back")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_UI_ACTION, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-ui"),
            ),
        )
        assertFalse(run.toString(), run.getBoolean("success"))
        assertEquals("back", run.getJSONObject("result").getString("action"))
        assertFalse(run.getJSONObject("result").getBoolean("accessibility_connected"))
        assertFalse(run.getJSONObject("automation").getBoolean("last_success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_ui_action_task",
                org.json.JSONObject()
                    .put("ui_action", "snapshot")
                    .put("enabled", false),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCreatesAndRunsAppLaunchRecordsSafely() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_app_launch_task",
                org.json.JSONObject()
                    .put("id", "auto-launch")
                    .put("package_name", "com.nousresearch.hermesagent.missing")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_APP_LAUNCH, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-launch"),
            ),
        )
        assertFalse(run.toString(), run.getBoolean("success"))
        assertEquals("launch_app", run.getJSONObject("result").getString("action"))
        assertEquals("com.nousresearch.hermesagent.missing", run.getJSONObject("result").getString("package_name"))
        assertFalse(run.getJSONObject("automation").getBoolean("last_success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_app_launch_task",
                org.json.JSONObject().put("package_name", ""),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }
}
