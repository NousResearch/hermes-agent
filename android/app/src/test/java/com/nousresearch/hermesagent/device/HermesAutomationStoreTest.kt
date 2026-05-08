package com.nousresearch.hermesagent.device

import android.content.ClipboardManager
import android.content.Context
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowToast
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
    fun bridgeCreatesAndRunsClipboardRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()
        store.setVariable("MESSAGE", "clipboard-ok")

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_clipboard_task",
                org.json.JSONObject()
                    .put("id", "auto-clipboard")
                    .put("clipboard_text", "Tasker %MESSAGE")
                    .put("clipboard_label", "Hermes test"),
            ),
        )

        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_CLIPBOARD_ACTION, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-clipboard"),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        assertEquals("Tasker clipboard-ok", clipboard.primaryClip?.getItemAt(0)?.coerceToText(context).toString())
    }

    @Test
    fun bridgeCreatesAndRunsVibrationRecords() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_vibration_task",
                org.json.JSONObject()
                    .put("id", "auto-vibration")
                    .put("vibration_pattern_ms", org.json.JSONArray(listOf(0, 15, 20, 25))),
            ),
        )

        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        assertEquals(ACTION_TYPE_VIBRATION_ACTION, automation.getString("action_type"))
        val command = org.json.JSONObject(automation.getString("command"))
        assertEquals(60, command.getLong("duration_ms"))
        assertEquals(4, command.getJSONArray("pattern_ms").length())

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-vibration"),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals("vibrate", run.getJSONObject("result").getString("action"))
        assertEquals(60, run.getJSONObject("result").getLong("duration_ms"))
    }

    @Test
    fun bridgeCreatesShizukuClearAppDataRecordsAndProtectsHermes() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_shizuku_action_task",
                org.json.JSONObject()
                    .put("id", "auto-clear-data")
                    .put("label", "Clear app data smoke")
                    .put("shizuku_action", "pm_clear")
                    .put("package_name", "com.example.target")
                    .put("enabled", false),
            ),
        )

        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        assertEquals(ACTION_TYPE_SHIZUKU_ACTION, automation.getString("action_type"))
        assertTrue(automation.getBoolean("use_shizuku"))
        val payload = org.json.JSONObject(automation.getString("command"))
        assertEquals("clear_app_data", payload.getString("shizuku_action"))
        assertEquals("com.example.target", payload.getString("package_name"))

        val selfClear = org.json.JSONObject(
            HermesPrivilegedAccessBridge.performStructuredActionJson(
                context,
                "clear_app_data",
                org.json.JSONObject().put("package_name", context.packageName),
            ),
        )
        assertFalse(selfClear.toString(), selfClear.getBoolean("success"))
        assertTrue(selfClear.getString("error").contains("Hermes"))
    }

    @Test
    fun bridgeCreatesAndRunsToastRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()
        store.setVariable("MESSAGE", "toast-ok")
        ShadowToast.reset()

        val direct = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "show_toast",
                org.json.JSONObject()
                    .put("toast_text", "Direct %MESSAGE")
                    .put("toast_long", true),
            ),
        )
        assertTrue(direct.toString(), direct.getBoolean("success"))
        assertEquals("show_toast", direct.getString("action"))
        assertTrue(direct.getBoolean("long"))
        assertEquals("Direct %MESSAGE", ShadowToast.getTextOfLatestToast())

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_toast_task",
                org.json.JSONObject()
                    .put("id", "auto-toast")
                    .put("toast_text", "Tasker %MESSAGE")
                    .put("toast_long", true),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_TOAST_ACTION, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-toast"),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals("show_toast", run.getJSONObject("result").getString("action"))
        assertEquals("Tasker toast-ok", ShadowToast.getTextOfLatestToast())
    }

    @Test
    fun bridgeCreatesAndRunsOverlaySceneRecords() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()
        store.setVariable("SCENE_MESSAGE", "overlay-ok")

        val payload = HermesOverlaySceneBridge.payloadFromArguments(
            org.json.JSONObject()
                .put("scene_id", "test-scene")
                .put("scene_title", "Hermes %SCENE_MESSAGE")
                .put("scene_text", "Tasker scene %SCENE_MESSAGE")
                .put("scene_button_text", "Close")
                .put("scene_position", "bottom")
                .put("scene_width_dp", 999)
                .put("scene_hide_after_ms", 9999999),
        )
        assertEquals("show", payload.getString("scene_action"))
        assertEquals("test-scene", payload.getString("scene_id"))
        assertEquals("Hermes %SCENE_MESSAGE", payload.getString("title"))
        assertEquals("bottom", payload.getString("position"))
        assertEquals(560, payload.getInt("width_dp"))
        assertEquals(600000L, payload.getLong("hide_after_ms"))

        val created = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_overlay_scene_task",
                org.json.JSONObject()
                    .put("id", "auto-scene")
                    .put("scene_title", "Hermes %SCENE_MESSAGE")
                    .put("scene_text", "Tasker scene %SCENE_MESSAGE")
                    .put("scene_button_text", "Close")
                    .put("scene_position", "bottom"),
            ),
        )

        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals(ACTION_TYPE_OVERLAY_SCENE, created.getJSONObject("automation").getString("action_type"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-scene"),
            ),
        )
        assertFalse(run.toString(), run.getBoolean("success"))
        assertEquals("show_overlay_scene", run.getJSONObject("result").getString("action"))
        assertTrue(run.getJSONObject("result").getBoolean("requires_overlay_permission"))

        val hideCreated = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_overlay_scene_task",
                org.json.JSONObject()
                    .put("id", "auto-hide-scene")
                    .put("scene_action", "hide")
                    .put("scene_id", "test-scene"),
            ),
        )
        assertTrue(hideCreated.toString(), hideCreated.getBoolean("success"))
        val hide = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", "auto-hide-scene"),
            ),
        )
        assertTrue(hide.toString(), hide.getBoolean("success"))
        assertEquals("hide_overlay_scene", hide.getJSONObject("result").getString("action"))
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
    fun bridgeExposesShizukuGatedLogcatWatcherActions() {
        val context = RuntimeEnvironment.getApplication()
        HermesAutomationStore(context).clear()
        HermesLogcatWatcherBridge.resetCursor(context)

        val parsed = HermesLogcatWatcherBridge.parseThreadtimeLogcatLinesJson(
            """
                05-07 12:34:56.789 10123  4242  777 E ActivityManager: ANR in com.example.app
                not a logcat threadtime line
                05-07 12:34:57.000  1000  1000 I Hermes: watcher ok
            """.trimIndent(),
        )
        assertEquals(2, parsed.length())
        assertEquals("05-07 12:34:56.789", parsed.getJSONObject(0).getString("logcat_timestamp"))
        assertEquals("10123", parsed.getJSONObject(0).getString("logcat_uid"))
        assertEquals("4242", parsed.getJSONObject(0).getString("logcat_pid"))
        assertEquals("E", parsed.getJSONObject(0).getString("logcat_level"))
        assertEquals("ActivityManager", parsed.getJSONObject(0).getString("logcat_tag"))
        assertEquals("ANR in com.example.app", parsed.getJSONObject(0).getString("logcat_message"))

        val packagesByUid = HermesLogcatWatcherBridge.parseUidPackageLines(
            """
                package:com.example.first uid:10123
                package:com.example.second uid:10123
                ignored
                package:android uid:1000
            """.trimIndent(),
        )
        assertEquals(listOf("com.example.first", "com.example.second"), packagesByUid["10123"])
        assertEquals(listOf("android"), packagesByUid["1000"])

        val list = org.json.JSONObject(HermesAutomationBridge.performActionJson(context, "list"))
        assertTrue(list.getJSONArray("available_actions").toString().contains("start_logcat_watcher"))
        assertTrue(list.getJSONArray("available_actions").toString().contains("scan_logcat_entries"))

        val status = org.json.JSONObject(HermesAutomationBridge.performActionJson(context, "logcat_watcher_status"))
        assertTrue(status.toString(), status.getBoolean("success"))
        assertFalse(status.getBoolean("running"))
        assertTrue(status.getBoolean("requires_shizuku"))
        assertTrue(status.getBoolean("durable_foreground_service"))
        assertFalse(status.getBoolean("foreground_service_running"))
        assertFalse(status.getBoolean("watcher_desired"))
        assertTrue(status.getBoolean("scan_cursor_enabled"))
        assertEquals(0, status.getInt("recent_event_signature_count"))
        assertEquals(0, status.getInt("enabled_logcat_record_count"))
        assertTrue(status.isNull("last_event_timestamp"))

        val scan = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "scan_logcat_entries",
                org.json.JSONObject().put("max_lines", 25),
            ),
        )
        assertFalse(scan.toString(), scan.getBoolean("success"))
        assertTrue(scan.getString("error").contains("Shizuku"))

        val start = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "start_logcat_watcher",
                org.json.JSONObject()
                    .put("scan_interval_seconds", 1)
                    .put("max_lines", 5),
            ),
        )
        assertFalse(start.toString(), start.getBoolean("success"))
        assertTrue(start.getString("error").contains("Shizuku"))
        assertFalse(
            org.json.JSONObject(HermesAutomationBridge.performActionJson(context, "logcat_watcher_status"))
                .getBoolean("watcher_desired"),
        )
    }

    @Test
    fun logcatWatcherCursorFiltersRepeatedEvents() {
        val context = RuntimeEnvironment.getApplication()
        HermesLogcatWatcherBridge.resetCursor(context)

        val firstBatch = HermesLogcatWatcherBridge.parseThreadtimeLogcatLines(
            """
                05-07 12:34:56.789 10123  4242  777 E ActivityManager: ANR in com.example.app
                05-07 12:34:57.000  1000  1000 I Hermes: watcher ok
            """.trimIndent(),
        )

        val firstFresh = HermesLogcatWatcherBridge.filterNewCursorEvents(context, firstBatch, cursorEnabled = true)
        assertEquals(2, firstFresh.size)
        assertEquals("05-07 12:34:57.000", HermesLogcatWatcherBridge.persistedLastEventTimestamp(context))
        assertEquals(2, HermesLogcatWatcherBridge.recentEventSignatureCount(context))

        val repeated = HermesLogcatWatcherBridge.filterNewCursorEvents(context, firstBatch, cursorEnabled = true)
        assertEquals(0, repeated.size)
        assertEquals(2, HermesLogcatWatcherBridge.recentEventSignatureCount(context))

        val secondBatch = HermesLogcatWatcherBridge.parseThreadtimeLogcatLines(
            """
                05-07 12:34:57.000  1000  1000 I Hermes: watcher ok
                05-07 12:34:58.111 10123  4243 W ActivityManager: New event
            """.trimIndent(),
        )
        val secondFresh = HermesLogcatWatcherBridge.filterNewCursorEvents(context, secondBatch, cursorEnabled = true)
        assertEquals(1, secondFresh.size)
        assertEquals("New event", secondFresh.single().message)
        assertEquals("05-07 12:34:58.111", HermesLogcatWatcherBridge.persistedLastEventTimestamp(context))
        assertEquals(3, HermesLogcatWatcherBridge.recentEventSignatureCount(context))

        val reset = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(context, "reset_logcat_watcher_cursor"),
        )
        assertTrue(reset.toString(), reset.getBoolean("success"))
        assertEquals(0, reset.getInt("recent_event_signature_count"))
        assertTrue(reset.isNull("last_event_timestamp"))

        val freshAfterReset = HermesLogcatWatcherBridge.filterNewCursorEvents(context, firstBatch, cursorEnabled = true)
        assertEquals(2, freshAfterReset.size)
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

    @Test
    fun bridgeImportsSafeTaskerXmlSubset() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val taskerXml = """
            <TaskerData sr="" dvi="1" tv="6.6.18">
              <Task sr="task1">
                <nme>Daily Import</nme>
                <Action sr="act0" ve="7">
                  <code>123</code>
                  <Str sr="arg0" ve="3">printf tasker-shell-ok</Str>
                  <Int sr="arg1" val="0"/>
                </Action>
                <Action sr="act1" ve="7">
                  <code>61</code>
                  <Int sr="arg0" val="30"/>
                </Action>
                <Action sr="act2" ve="7">
                  <code>62</code>
                  <Str sr="arg0" ve="3">0,10,20,30</Str>
                </Action>
                <Action sr="act3" ve="7">
                  <code>410</code>
                  <Str sr="arg0" ve="3">tasker-import.txt</Str>
                  <Str sr="arg1" ve="3">Tasker says %MESSAGE</Str>
                  <Int sr="arg2" val="1"/>
                </Action>
                <Action sr="act4" ve="7">
                  <code>104</code>
                  <Str sr="arg0" ve="3">https://nousresearch.com/</Str>
                </Action>
                <Action sr="act5" ve="7">
                  <code>105</code>
                  <Str sr="arg0" ve="3">Copy %MESSAGE</Str>
                </Action>
                <Action sr="act6" ve="7">
                  <code>548</code>
                  <Str sr="arg0" ve="3">Flash %MESSAGE</Str>
                  <Int sr="arg1" val="1"/>
                </Action>
                <Action sr="act7" ve="7">
                  <code>9999</code>
                </Action>
              </Task>
              <Variable sr="var1">
                <nme>%MESSAGE</nme>
                <val>hello</val>
              </Variable>
            </TaskerData>
        """.trimIndent()

        val imported = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_tasker_xml",
                org.json.JSONObject()
                    .put("tasker_xml", taskerXml)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals("tasker_xml", imported.getString("source"))
        assertEquals(1, imported.getInt("tasker_task_count"))
        assertEquals(7, imported.getInt("tasker_imported_action_count"))
        assertEquals(1, imported.getJSONArray("tasker_skipped_actions").length())
        assertEquals(7, imported.getInt("imported_automation_count"))
        assertEquals("hello", store.getVariable("MESSAGE"))

        val records = store.list()
        assertTrue(records.any { it.actionType == ACTION_TYPE_SHELL && it.command == "printf tasker-shell-ok" })
        assertTrue(records.any { it.actionType == ACTION_TYPE_INTENT && it.command.contains("nousresearch.com") })
        assertTrue(records.any { it.actionType == ACTION_TYPE_CLIPBOARD_ACTION && it.command.contains("Copy %MESSAGE") })
        assertTrue(records.any { it.actionType == ACTION_TYPE_TOAST_ACTION && it.command.contains("Flash %MESSAGE") })
        assertTrue(records.any { it.actionType == ACTION_TYPE_VIBRATION_ACTION && it.command.contains("pattern_ms") })
        assertTrue(records.none { it.enabled })

        val fileRecord = records.first { it.actionType == ACTION_TYPE_FILE_WRITE }
        val payload = org.json.JSONObject(fileRecord.command)
        assertEquals("tasker-import.txt", payload.getString("path"))
        assertEquals("Tasker says %MESSAGE", payload.getString("content"))
        assertTrue(payload.getBoolean("append"))

        val run = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", fileRecord.id),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertTrue(run.getJSONObject("result").getString("path").endsWith("tasker-import.txt"))
        assertTrue(run.getJSONObject("result").getBoolean("append"))

        ShadowToast.reset()
        val toastRecord = records.first { it.actionType == ACTION_TYPE_TOAST_ACTION }
        val toastRun = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "run",
                org.json.JSONObject().put("id", toastRecord.id),
            ),
        )
        assertTrue(toastRun.toString(), toastRun.getBoolean("success"))
        assertEquals("Flash hello", ShadowToast.getTextOfLatestToast())

        val rejected = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_tasker_xml",
                org.json.JSONObject().put(
                    "tasker_xml",
                    "<TaskerData><Task><nme>Unsafe</nme><Action><code>129</code></Action></Task></TaskerData>",
                ),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
        assertTrue(rejected.getString("error").contains("supported safe actions"))

        val malicious = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_tasker_xml",
                org.json.JSONObject().put(
                    "tasker_xml",
                    """
                        <!DOCTYPE TaskerData [
                          <!ENTITY xxe SYSTEM "file:///etc/passwd">
                        ]>
                        <TaskerData>
                          <Variable><nme>%MESSAGE</nme><val>&xxe;</val></Variable>
                        </TaskerData>
                    """.trimIndent(),
                ),
            ),
        )
        assertFalse(malicious.toString(), malicious.getBoolean("success"))
        assertTrue(malicious.getString("error").contains("DOCTYPE"))

        val dataUriXml = """
            <TaskerData>
              <Task>
                <nme>Data URI Import</nme>
                <Action>
                  <code>410</code>
                  <Str sr="arg0" ve="3">tasker-data-uri.txt</Str>
                  <Str sr="arg1" ve="3">uri-ok</Str>
                </Action>
              </Task>
            </TaskerData>
        """.trimIndent()
        val dataUri = "data:text/xml," + java.net.URLEncoder.encode(
            dataUriXml,
            java.nio.charset.StandardCharsets.UTF_8.name(),
        )
        val dataUriImported = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_tasker_data_uri",
                org.json.JSONObject()
                    .put("tasker_data_uri", dataUri)
                    .put("replace", true),
            ),
        )
        assertTrue(dataUriImported.toString(), dataUriImported.getBoolean("success"))
        assertEquals(1, dataUriImported.getInt("tasker_imported_action_count"))
        assertEquals("tasker_xml", dataUriImported.getString("source"))
        assertFalse(store.list().single().enabled)
    }

    @Test
    fun bridgeImportsTaskerGlobalUiAndSettingsActions() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val taskerXml = """
            <TaskerData sr="" dvi="1" tv="6.6.18">
              <Task sr="task1">
                <nme>Hermes Controls</nme>
                <Action><code>25</code></Action>
                <Action><code>245</code></Action>
                <Action><code>247</code></Action>
                <Action><code>219</code></Action>
                <Action><code>197</code></Action>
                <Action><code>201</code></Action>
                <Action><code>206</code></Action>
                <Action><code>218</code></Action>
                <Action><code>220</code></Action>
                <Action><code>236</code></Action>
                <Action><code>237</code></Action>
                <Action><code>956</code></Action>
              </Task>
            </TaskerData>
        """.trimIndent()

        val imported = org.json.JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "import_tasker_xml",
                org.json.JSONObject()
                    .put("tasker_xml", taskerXml)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals(12, imported.getInt("tasker_imported_action_count"))
        assertEquals(0, imported.getJSONArray("tasker_skipped_actions").length())
        assertEquals(12, imported.getInt("imported_automation_count"))

        val records = store.list()
        val uiActions = records
            .filter { it.actionType == ACTION_TYPE_UI_ACTION }
            .map { org.json.JSONObject(it.command).getString("ui_action") }
            .toSet()
        assertEquals(setOf("home", "back", "recents", "quick_settings"), uiActions)

        val systemActions = records
            .filter { it.actionType == ACTION_TYPE_SYSTEM_ACTION }
            .map { it.command }
            .toSet()
        assertEquals(
            setOf(
                "open_developer_options",
                "open_airplane_mode_settings",
                "open_wifi_panel",
                "open_bluetooth_settings",
                "open_mobile_network_settings",
                "open_accessibility_settings",
                "open_notification_listener_settings",
                "open_nfc_settings",
            ),
            systemActions,
        )
        assertTrue(records.none { it.enabled })
    }
}
