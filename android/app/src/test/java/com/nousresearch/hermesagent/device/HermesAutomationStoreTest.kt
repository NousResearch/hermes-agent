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
    fun bridgeCreatesAndRunsSensorTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_app_launch_task",
                org.json.JSONObject()
                    .put("id", "auto-sensor")
                    .put("package_name", "com.nousresearch.hermesagent.missing")
                    .put("trigger", "sensor")
                    .put("sensor_type", "accelerometer")
                    .put("sensor_event", "shake")
                    .put("value_name", "x")
                    .put("min_value", 1.5)
                    .put("max_value", 9.8),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        val triggerData = org.json.JSONObject(automation.getString("trigger_data"))
        assertEquals(TRIGGER_SENSOR, automation.getString("trigger_type"))
        assertEquals("accelerometer", triggerData.getString("sensor_type"))
        assertEquals("shake", triggerData.getString("sensor_event"))
        assertEquals("x", triggerData.getString("value_name"))
        assertEquals("1.5", triggerData.getString("min_value"))
        assertEquals("9.8", triggerData.getString("max_value"))

        val missed = org.json.JSONObject(
            HermesAutomationBridge.runSensorTriggerJson(
                context,
                org.json.JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_event", "shake")
                    .put("value_name", "x")
                    .put("sensor_value", 0.5),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))

        val matched = org.json.JSONObject(
            HermesAutomationBridge.runSensorTriggerJson(
                context,
                org.json.JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_event", "shake")
                    .put("value_name", "x")
                    .put("sensor_value", 2.25)
                    .put("sensor_unit", "m/s^2")
                    .put("sensor_accuracy", "high"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertFalse(matched.getJSONArray("results").getJSONObject(0).getBoolean("success"))
        assertEquals("accelerometer", store.getVariable("SENSOR_TYPE"))
        assertEquals("shake", store.getVariable("SENSOR_EVENT"))
        assertEquals("2.25", store.getVariable("SENSOR_VALUE"))
        assertEquals("x", store.getVariable("SENSOR_VALUE_NAME"))
        assertEquals("m/s^2", store.getVariable("SENSOR_UNIT"))
        assertEquals("high", store.getVariable("SENSOR_ACCURACY"))

        val generic = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "sensor"),
            ),
        )
        assertFalse(generic.toString(), generic.getBoolean("success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shell_task",
                org.json.JSONObject()
                    .put("command", "printf no")
                    .put("trigger", "sensor")
                    .put("min_value", 5)
                    .put("max_value", 1),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCreatesAndRunsLogcatEntryTriggerRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_file_write_task",
                org.json.JSONObject()
                    .put("id", "auto-logcat")
                    .put("path", "logcat-trigger.txt")
                    .put("content", "%LOGCAT_LEVEL/%LOGCAT_TAG/%LOGCAT_PID/%LOGCAT_PACKAGE/%LOGCAT_MESSAGE/%LOGCAT_TIME")
                    .put("trigger", "logcat_entry")
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message_contains", "ANR")
                    .put("logcat_level", "error")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "com.example.app"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        val triggerData = org.json.JSONObject(automation.getString("trigger_data"))
        assertEquals(TRIGGER_LOGCAT_ENTRY, automation.getString("trigger_type"))
        assertEquals("ActivityManager", triggerData.getString("tag"))
        assertEquals("ANR", triggerData.getString("message_contains"))
        assertEquals("error", triggerData.getString("level"))
        assertEquals("4242", triggerData.getString("pid"))
        assertEquals("com.example.app", triggerData.getString("package_name"))
        assertTrue(triggerData.getBoolean("requires_shizuku_for_background_watch"))

        val missed = org.json.JSONObject(
            HermesAutomationBridge.runLogcatEntryTriggerJson(
                context,
                org.json.JSONObject()
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message", "Process started cleanly")
                    .put("logcat_level", "E")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "com.example.app"),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))

        val matched = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_logcat_entry_trigger",
                org.json.JSONObject()
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message", "ANR in com.example.app")
                    .put("logcat_level", "E")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "com.example.app")
                    .put("logcat_timestamp", "05-07 12:34:56.789"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(TRIGGER_LOGCAT_ENTRY, matched.getString("trigger"))
        assertTrue(matched.getBoolean("requires_shizuku_for_background_watch"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue(matched.getJSONArray("results").getJSONObject(0).getBoolean("success"))
        assertEquals("ActivityManager", store.getVariable("LOGCAT_TAG"))
        assertEquals("ANR in com.example.app", store.getVariable("LOGCAT_MESSAGE"))
        assertEquals("E", store.getVariable("LOGCAT_LEVEL"))
        assertEquals("4242", store.getVariable("LOGCAT_PID"))
        assertEquals("com.example.app", store.getVariable("LOGCAT_PACKAGE"))
        assertEquals("05-07 12:34:56.789", store.getVariable("LOGCAT_TIME"))
        val filePath = matched
            .getJSONArray("results")
            .getJSONObject(0)
            .getJSONObject("result")
            .getString("path")
        assertEquals(
            "E/ActivityManager/4242/com.example.app/ANR in com.example.app/05-07 12:34:56.789",
            java.io.File(filePath).readText(),
        )

        val generic = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "logcat"),
            ),
        )
        assertFalse(generic.toString(), generic.getBoolean("success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shell_task",
                org.json.JSONObject()
                    .put("command", "printf no")
                    .put("trigger", "logcat_entry")
                    .put("logcat_level", "debug"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCreatesAndRunsExternalTriggerRecordsWithTokenGate() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_file_write_task",
                org.json.JSONObject()
                    .put("id", "auto-external")
                    .put("path", "external-trigger.txt")
                    .put("content", "%SA_TRIGGER_ID|%SA_TRIGGER_PACKAGE_NAME|%SA_REFERRER|%SA_EXTRAS")
                    .put("trigger", "external_trigger")
                    .put("trigger_id", "quick_tile")
                    .put("external_token", "secret-token")
                    .put("trigger_package_name", "com.example.trigger")
                    .put("referrer_contains", "tile"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        val triggerData = org.json.JSONObject(automation.getString("trigger_data"))
        assertEquals(TRIGGER_EXTERNAL, automation.getString("trigger_type"))
        assertEquals("quick_tile", triggerData.getString("trigger_id"))
        assertEquals("secret-token", triggerData.getString("external_token"))
        assertEquals("tile", triggerData.getString("referrer_contains"))

        val missed = org.json.JSONObject(
            HermesAutomationBridge.runExternalTriggerJson(
                context,
                org.json.JSONObject()
                    .put("trigger_id", "quick_tile")
                    .put("external_token", "wrong-token")
                    .put("trigger_package_name", "com.example.trigger")
                    .put("referrer", "tile://settings")
                    .put("extras", org.json.JSONObject().put("mode", "wrong")),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))

        val matched = org.json.JSONObject(
            HermesAutomationBridge.runExternalTriggerJson(
                context,
                org.json.JSONObject()
                    .put("trigger_id", "quick_tile")
                    .put("external_token", "secret-token")
                    .put("trigger_package_name", "com.example.trigger")
                    .put("referrer", "tile://settings")
                    .put("extras", org.json.JSONObject().put("mode", "focus")),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue(matched.getJSONArray("results").getJSONObject(0).getBoolean("success"))
        assertEquals("quick_tile", store.getVariable("SA_TRIGGER_ID"))
        assertEquals("com.example.trigger", store.getVariable("SA_TRIGGER_PACKAGE_NAME"))
        assertEquals("tile://settings", store.getVariable("SA_REFERRER"))
        assertEquals("""{"mode":"focus"}""", store.getVariable("SA_EXTRAS"))
        val filePath = matched
            .getJSONArray("results")
            .getJSONObject(0)
            .getJSONObject("result")
            .getString("path")
        assertEquals(
            """quick_tile|com.example.trigger|tile://settings|{"mode":"focus"}""",
            java.io.File(filePath).readText(),
        )

        val generic = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run_trigger",
                org.json.JSONObject().put("trigger", "external_trigger"),
            ),
        )
        assertFalse(generic.toString(), generic.getBoolean("success"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shell_task",
                org.json.JSONObject()
                    .put("command", "printf no")
                    .put("trigger", "external_trigger")
                    .put("trigger_id", "quick_tile"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
        assertTrue(rejected.getString("error").contains("external_token"))
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

    @Test
    fun bridgeCreatesAndRunsNotificationActionRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        assertTrue(store.setVariable("NOTICE_ID", "77"))
        assertTrue(store.setVariable("NOTICE_TAG", "hermes-test"))
        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_notification_task",
                org.json.JSONObject()
                    .put("id", "auto-notify")
                    .put("label", "Notify smoke")
                    .put("notification_title", "Hermes")
                    .put("notification_text", "Tasker-style notification")
                    .put("notification_id", "%NOTICE_ID")
                    .put("notification_tag", "%NOTICE_TAG")
                    .put("priority", "high")
                    .put("group_key", "hermes-group")
                    .put("only_alert_once", true)
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_NOTIFICATION_ACTION, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-notify"),
            ),
        )
        val result = run.getJSONObject("result")
        if (run.getBoolean("success")) {
            assertEquals("notification_post", result.getString("action"))
            assertEquals(77, result.getInt("notification_id"))
            assertEquals("hermes-test", result.getString("notification_tag"))
        } else {
            assertEquals("android.permission.POST_NOTIFICATIONS", result.getString("requires_permission"))
        }

        val cancel = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_notification_task",
                org.json.JSONObject()
                    .put("id", "auto-notify-cancel")
                    .put("notification_action", "cancel")
                    .put("notification_id", "%NOTICE_ID")
                    .put("notification_tag", "%NOTICE_TAG")
                    .put("enabled", false),
            ),
        )
        assertTrue(cancel.toString(), cancel.getBoolean("success"))
        val cancelRun = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-notify-cancel"),
            ),
        )
        assertTrue(cancelRun.toString(), cancelRun.getBoolean("success"))
        assertEquals("notification_cancel", cancelRun.getJSONObject("result").getString("action"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_notification_task",
                org.json.JSONObject().put("notification_action", "post"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
    }

    @Test
    fun bridgeCalculatesAndRunsSavedSunriseSunsetActions() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val direct = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "calculate_sunrise_sunset",
                org.json.JSONObject()
                    .put("latitude", 51.5074)
                    .put("longitude", -0.1278)
                    .put("date", "2026-06-21")
                    .put("timezone", "Europe/London"),
            ),
        )
        assertTrue(direct.toString(), direct.getBoolean("success"))
        assertEquals("calculate_sunrise_sunset", direct.getString("action"))
        assertEquals("2026-06-21", direct.getString("date"))
        assertEquals("Europe/London", direct.getString("timezone"))
        assertEquals("normal", direct.getString("sun_state"))
        assertTrue(direct.getString("sunrise").matches(Regex("\\d{2}:\\d{2}")))
        assertTrue(direct.getString("sunset").matches(Regex("\\d{2}:\\d{2}")))
        assertTrue(direct.getInt("daylight_minutes") in 900..1100)
        assertEquals(direct.getString("sunrise"), store.getVariable("SUNRISE"))
        assertEquals("51.5074", store.getVariable("SUN_LAT"))

        assertTrue(store.setVariable("LATITUDE", "51.5074"))
        assertTrue(store.setVariable("LONGITUDE", "-0.1278"))
        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_sunrise_sunset_task",
                org.json.JSONObject()
                    .put("id", "auto-sun")
                    .put("label", "Sun task")
                    .put("latitude", "%LATITUDE")
                    .put("longitude", "%LONGITUDE")
                    .put("date", "2026-12-21")
                    .put("timezone", "Europe/London")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_SUNRISE_SUNSET, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-sun"),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        val result = run.getJSONObject("result")
        assertEquals("sunrise_sunset", result.getString("action"))
        assertEquals("2026-12-21", result.getString("date"))
        assertTrue(result.getInt("daylight_minutes") in 400..600)
        assertEquals("2026-12-21", store.getVariable("SUN_DATE"))
        assertEquals("Europe/London", store.getVariable("SUN_TIMEZONE"))
        assertEquals(result.getString("sunset"), store.getVariable("SUNSET"))

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "calculate_sunrise_sunset",
                org.json.JSONObject()
                    .put("latitude", 91)
                    .put("longitude", 0)
                    .put("date", "2026-06-21")
                    .put("timezone", "UTC"),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
        assertTrue(rejected.getString("error").contains("latitude"))
    }

    @Test
    fun bridgeExportsAndImportsAutomationBundles() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val variable = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "set_variable",
                org.json.JSONObject()
                    .put("name", "%message")
                    .put("value", "bundle-ok"),
            ),
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_file_write_task",
                org.json.JSONObject()
                    .put("id", "auto-bundle")
                    .put("label", "Bundle smoke")
                    .put("path", "bundle-smoke.txt")
                    .put("content", "%MESSAGE")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val exported = org.json.JSONObject(HermesAutomationBridge.performActionJson(context, "export_automations"))
        assertTrue(exported.toString(), exported.getBoolean("success"))
        assertEquals("hermes_android_automation_bundle", exported.getString("kind"))
        assertEquals(1, exported.getInt("automation_count"))
        assertEquals("bundle-ok", exported.getJSONObject("variables").getString("MESSAGE"))

        store.clear()
        assertNull(store.get("auto-bundle"))
        assertNull(store.getVariable("MESSAGE"))

        val imported = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_automations",
                org.json.JSONObject()
                    .put("bundle", exported)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals(1, imported.getInt("imported_automation_count"))
        assertEquals(1, imported.getInt("imported_variable_count"))
        assertEquals("bundle-ok", store.getVariable("MESSAGE"))
        val record = store.get("auto-bundle")
        assertEquals(ACTION_TYPE_FILE_WRITE, record?.actionType)
        assertEquals("Bundle smoke", record?.label)
        assertFalse(record?.enabled ?: true)

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_automations",
                org.json.JSONObject().put(
                    "automations",
                    org.json.JSONArray().put(
                        org.json.JSONObject()
                            .put("id", "bad-import")
                            .put("action_type", "unsupported")
                            .put("command", "noop"),
                    ),
                ),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
        assertTrue(rejected.getString("error").contains("unsupported action_type"))
    }
}
