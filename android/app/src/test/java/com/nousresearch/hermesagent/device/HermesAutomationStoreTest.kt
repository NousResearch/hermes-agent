package com.nousresearch.hermesagent.device

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner
import java.util.Calendar

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
    fun bridgeCreatesAndRunsTimeTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_file_write_task",
                org.json.JSONObject()
                    .put("id", "auto-time")
                    .put("path", "time-trigger.txt")
                    .put("content", "%TIME:%TIME_DAY")
                    .put("trigger", "time")
                    .put("time", "08:30")
                    .put("days_of_week", org.json.JSONArray(listOf("mon", "wed"))),
            ),
        )

        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        assertEquals(TRIGGER_TIME, automation.getString("trigger_type"))
        assertEquals(510, automation.getInt("trigger_time_minutes"))
        assertEquals("MON,WED", automation.getString("trigger_days_of_week"))

        val triggered = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "time"),
            ),
        )
        assertTrue(triggered.toString(), triggered.getBoolean("success"))
        assertEquals(1, triggered.getInt("matched_count"))
        assertTrue(store.getVariable("TIME").orEmpty().matches(Regex("\\d{2}:\\d{2}")))
        assertTrue(store.getVariable("TIME_DAY").orEmpty() in setOf("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"))
    }

    @Test
    fun schedulerComputesNextTimeTriggerWithDayRestriction() {
        val now = Calendar.getInstance().apply {
            set(2026, Calendar.MAY, 4, 9, 0, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val next = HermesAutomationScheduler.nextTimeTriggerAtMillis(
            nowEpochMs = now.timeInMillis,
            timeMinutes = 8 * 60 + 30,
            daysOfWeekCsv = "MON,WED",
        )
        val expected = Calendar.getInstance().apply {
            set(2026, Calendar.MAY, 6, 8, 30, 0)
            set(Calendar.MILLISECOND, 0)
        }

        assertEquals(expected.timeInMillis, next)
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
    fun bridgeCreatesAndRunsNotificationPostedTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_app_launch_task",
                org.json.JSONObject()
                    .put("id", "auto-notification")
                    .put("package_name", "com.nousresearch.hermesagent.missing")
                    .put("trigger", "notification")
                    .put("trigger_package_name", "com.example.sender"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(TRIGGER_NOTIFICATION_POSTED, created.getJSONObject("automation").getString("trigger_type"))
        assertEquals("com.example.sender", created.getJSONObject("automation").getString("trigger_package_name"))

        val missed = org.json.JSONObject(
            HermesAutomationBridge.runNotificationPostedTriggerJson(
                context,
                "com.example.other",
                "Ignored",
                "No match",
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))

        val matched = org.json.JSONObject(
            HermesAutomationBridge.runNotificationPostedTriggerJson(
                context,
                "com.example.sender",
                "Tasker title",
                "Tasker body",
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertFalse(matched.getJSONArray("results").getJSONObject(0).getBoolean("success"))
        assertEquals("com.example.sender", store.getVariable("NOTIFICATION_PACKAGE"))
        assertEquals("Tasker title", store.getVariable("NOTIFICATION_TITLE"))
        assertEquals("Tasker body", store.getVariable("NOTIFICATION_TEXT"))

        val generic = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "notification_posted"),
            ),
        )
        assertFalse(generic.toString(), generic.getBoolean("success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shell_task",
                org.json.JSONObject()
                    .put("command", "printf no")
                    .put("trigger", "notification_posted"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCreatesAndRunsLocationTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_app_launch_task",
                org.json.JSONObject()
                    .put("id", "auto-location")
                    .put("package_name", "com.nousresearch.hermesagent.missing")
                    .put("trigger", "location")
                    .put("latitude", 37.7749)
                    .put("longitude", -122.4194)
                    .put("radius_meters", 200)
                    .put("location_provider", "gps")
                    .put("location_name", "office"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        val triggerData = org.json.JSONObject(automation.getString("trigger_data"))
        assertEquals(TRIGGER_LOCATION, automation.getString("trigger_type"))
        assertEquals("37.7749", triggerData.getString("latitude"))
        assertEquals("-122.4194", triggerData.getString("longitude"))
        assertEquals("gps", triggerData.getString("provider"))
        assertEquals("office", triggerData.getString("location_name"))

        val missed = org.json.JSONObject(
            HermesAutomationBridge.runLocationTriggerJson(
                context,
                org.json.JSONObject()
                    .put("latitude", 37.7849)
                    .put("longitude", -122.4194)
                    .put("location_provider", "gps")
                    .put("location_name", "Hermes Office"),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))

        val matched = org.json.JSONObject(
            HermesAutomationBridge.runLocationTriggerJson(
                context,
                org.json.JSONObject()
                    .put("latitude", 37.7750)
                    .put("longitude", -122.4195)
                    .put("accuracy_meters", 12.5)
                    .put("location_provider", "gps")
                    .put("location_name", "Hermes Office"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertFalse(matched.getJSONArray("results").getJSONObject(0).getBoolean("success"))
        assertEquals("37.775", store.getVariable("LAT"))
        assertEquals("-122.4195", store.getVariable("LON"))
        assertEquals("37.775,-122.4195", store.getVariable("LOC"))
        assertEquals("12.5", store.getVariable("LOCACC"))
        assertEquals("gps", store.getVariable("LOCPROVIDER"))
        assertEquals("Hermes Office", store.getVariable("LOCNAME"))

        val generic = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "location"),
            ),
        )
        assertFalse(generic.toString(), generic.getBoolean("success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_location_trigger",
                org.json.JSONObject().put("latitude", 37.7750),
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
