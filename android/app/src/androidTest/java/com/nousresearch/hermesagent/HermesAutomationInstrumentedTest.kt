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

    @Test
    fun fileAutomationWritesAndDeletesWorkspaceFile() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-file-action.txt").apply { delete() }

        val variable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%FILE_MESSAGE")
                    .put("value", "file-action-ok"),
            )
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))

        val writeTask = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Write file smoke")
                    .put("path", "hermes-file-action.txt")
                    .put("content", "%FILE_MESSAGE")
                    .put("enabled", false),
            )
        )
        assertTrue(writeTask.toString(), writeTask.getBoolean("success"))
        assertEquals("file_write", writeTask.getJSONObject("automation").getString("action_type"))

        val writeRun = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", writeTask.getJSONObject("automation").getString("id")),
            )
        )
        assertTrue(writeRun.toString(), writeRun.getBoolean("success"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("file-action-ok", target.readText())

        val deleteTask = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_delete_task",
                JSONObject()
                    .put("label", "Delete file smoke")
                    .put("path", "hermes-file-action.txt")
                    .put("enabled", false),
            )
        )
        assertTrue(deleteTask.toString(), deleteTask.getBoolean("success"))

        val deleteRun = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", deleteTask.getJSONObject("automation").getString("id")),
            )
        )
        assertTrue(deleteRun.toString(), deleteRun.getBoolean("success"))
        assertFalse("Expected ${target.absolutePath} to be deleted", target.exists())
    }

    @Test
    fun systemActionAutomationRunsSafeSystemAction() {
        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_system_action_task",
                JSONObject()
                    .put("label", "Stop runtime smoke")
                    .put("system_action", "stop_background_runtime")
                    .put("enabled", false),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("system_action", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            )
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals(0, run.getJSONObject("automation").getInt("last_exit_code"))
        assertEquals("stop_background_runtime", run.getJSONObject("result").getString("action"))
    }

    @Test
    fun uiActionAutomationPersistsAndFailsSafelyAtAccessibilityBoundary() {
        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_ui_action_task",
                JSONObject()
                    .put("label", "Missing UI smoke")
                    .put("ui_action", "click")
                    .put("text_contains", "HermesMissingNodeForAutomationSmoke")
                    .put("enabled", false),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("ui_action", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            )
        )
        assertFalse(run.toString(), run.getBoolean("success"))
        assertFalse(run.getJSONObject("automation").getBoolean("last_success"))
        assertTrue(
            run.toString(),
            run.getJSONObject("result").optString("error").contains("accessibility", ignoreCase = true),
        )
    }

    @Test
    fun appLaunchAutomationOpensHermesPackage() {
        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_app_launch_task",
                JSONObject()
                    .put("label", "Launch Hermes smoke")
                    .put("package_name", app.packageName)
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("app_launch", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals(0, run.getJSONObject("automation").getInt("last_exit_code"))
        assertEquals("launch_app", run.getJSONObject("result").getString("action"))
        assertEquals(app.packageName, run.getJSONObject("result").getString("package_name"))
    }

    @Test
    fun intentAutomationStartsExplicitHermesActivity() {
        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_intent_task",
                JSONObject()
                    .put("label", "Start activity smoke")
                    .put("intent_task_action", "start_activity")
                    .put("package_name", app.packageName)
                    .put("class_name", "com.nousresearch.hermesagent.MainActivity")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("intent", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        val result = run.getJSONObject("result")
        assertEquals("start_activity", result.getString("action"))
        assertEquals(app.packageName, result.getString("package_name"))
        assertEquals("com.nousresearch.hermesagent.MainActivity", result.getString("class_name"))
        assertEquals(0, run.getJSONObject("automation").getInt("last_exit_code"))
    }

    @Test
    fun intentAutomationSendsBroadcastAndExpandsVariables() {
        val packageVariable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%TARGET_PACKAGE")
                    .put("value", app.packageName),
            ),
        )
        assertTrue(packageVariable.toString(), packageVariable.getBoolean("success"))

        val actionVariable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%BROADCAST_SUFFIX")
                    .put("value", "AUTOMATION_SMOKE"),
            ),
        )
        assertTrue(actionVariable.toString(), actionVariable.getBoolean("success"))

        val messageVariable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%BROADCAST_MESSAGE")
                    .put("value", "broadcast-ok"),
            ),
        )
        assertTrue(messageVariable.toString(), messageVariable.getBoolean("success"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_broadcast_task",
                JSONObject()
                    .put("label", "Broadcast smoke")
                    .put("intent_action", app.packageName + ".%BROADCAST_SUFFIX")
                    .put("package_name", "%TARGET_PACKAGE")
                    .put(
                        "extras",
                        JSONObject()
                            .put("message", "%BROADCAST_MESSAGE")
                            .put("count", 7)
                            .put("enabled", true),
                    )
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("intent", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        val result = run.getJSONObject("result")
        assertEquals("send_broadcast", result.getString("action"))
        assertEquals(app.packageName + ".AUTOMATION_SMOKE", result.getString("intent_action"))
        assertEquals(app.packageName, result.getString("package_name"))
        assertEquals(3, result.getInt("extras_count"))
        assertEquals("broadcast-ok", result.getJSONObject("extras").getString("message"))
    }

    @Test
    fun intentAutomationRejectsUnsafeDefinitions() {
        val missingUri = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_intent_task",
                JSONObject().put("intent_task_action", "open_uri"),
            ),
        )
        assertFalse(missingUri.toString(), missingUri.getBoolean("success"))
        assertTrue(missingUri.getString("error").contains("data_uri"))

        val unsupported = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_intent_task",
                JSONObject().put("intent_task_action", "toggle_airplane_mode"),
            ),
        )
        assertFalse(unsupported.toString(), unsupported.getBoolean("success"))
        assertTrue(unsupported.getString("error").contains("Unsupported Android intent task action"))

        val badPackage = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_intent_task",
                JSONObject()
                    .put("intent_task_action", "start_activity")
                    .put("package_name", "bad\u0000package"),
            ),
        )
        assertFalse(badPackage.toString(), badPackage.getBoolean("success"))
        assertTrue(badPackage.getString("error").contains("NUL"))
    }

    @Test
    fun shizukuActionAutomationExpandsVariablesAndFailsSafelyAtPrivilegeBoundary() {
        val packageVariable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%TARGET_PACKAGE")
                    .put("value", app.packageName),
            ),
        )
        assertTrue(packageVariable.toString(), packageVariable.getBoolean("success"))

        val permissionVariable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "TARGET_PERMISSION")
                    .put("value", "android.permission.POST_NOTIFICATIONS"),
            ),
        )
        assertTrue(permissionVariable.toString(), permissionVariable.getBoolean("success"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shizuku_action_task",
                JSONObject()
                    .put("label", "Grant notification smoke")
                    .put("shizuku_action", "grant_runtime_permission")
                    .put("package_name", "%TARGET_PACKAGE")
                    .put("permission", "{{TARGET_PERMISSION}}")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        assertEquals("shizuku_action", automation.getString("action_type"))
        assertTrue(automation.getBoolean("use_shizuku"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", automation.getString("id")),
            ),
        )
        assertFalse(run.toString(), run.getBoolean("success"))
        assertFalse(run.getJSONObject("automation").getBoolean("last_success"))
        val result = run.getJSONObject("result")
        assertEquals("grant_runtime_permission", result.getString("action"))
        assertEquals(app.packageName, result.getString("package_name"))
        assertEquals("android.permission.POST_NOTIFICATIONS", result.getString("permission"))
        assertTrue(result.getString("adb_shell_command").contains("pm grant ${app.packageName} android.permission.POST_NOTIFICATIONS"))
        assertTrue(result.optString("error").contains("Shizuku", ignoreCase = true))
    }

    @Test
    fun shizukuActionAutomationRejectsUnsafeDefinitions() {
        val unsupported = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shizuku_action_task",
                JSONObject()
                    .put("shizuku_action", "toggle_airplane_mode")
                    .put("package_name", app.packageName),
            ),
        )
        assertFalse(unsupported.toString(), unsupported.getBoolean("success"))
        assertTrue(unsupported.getString("error").contains("Unsupported saved Shizuku action"))

        val missingPackage = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shizuku_action_task",
                JSONObject().put("shizuku_action", "force_stop_app"),
            ),
        )
        assertFalse(missingPackage.toString(), missingPackage.getBoolean("success"))
        assertTrue(missingPackage.getString("error").contains("package_name"))

        val missingPermission = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shizuku_action_task",
                JSONObject()
                    .put("shizuku_action", "grant_runtime_permission")
                    .put("package_name", app.packageName),
            ),
        )
        assertFalse(missingPermission.toString(), missingPermission.getBoolean("success"))
        assertTrue(missingPermission.getString("error").contains("permission"))
    }

    @Test
    fun shizukuStateTriggerRunsMatchingAutomationAndExposesVariables() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-shizuku-state-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Shizuku state smoke")
                    .put("path", "hermes-shizuku-state-trigger.txt")
                    .put(
                        "content",
                        "%SHIZUKU_AVAILABLE:%SHIZUKU_RUNNING:%SHIZUKU_PERMISSION_GRANTED:%SHIZUKU_PRIVILEGE_LABEL",
                    )
                    .put("trigger", "shizuku_unavailable"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("shizuku_unavailable", created.getJSONObject("automation").getString("trigger_type"))

        val triggered = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_shizuku_state_trigger",
                JSONObject().put("shizuku_state", "unavailable"),
            ),
        )
        assertTrue(triggered.toString(), triggered.getBoolean("success"))
        assertEquals("shizuku_unavailable", triggered.getString("trigger"))
        assertTrue(triggered.getJSONObject("shizuku_status").has("shizuku_installed"))
        assertTrue(triggered.getJSONObject("shizuku_status").has("sui_installed"))
        assertEquals(1, triggered.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertTrue(target.readText().matches(Regex("(true|false):(true|false):(true|false):\\S+")))

        val variables = JSONObject(HermesAutomationBridge.performActionJson(app, "list_variables"))
            .getJSONObject("variables")
        assertTrue(variables.has("SHIZUKU_AVAILABLE"))
        assertTrue(variables.has("SHIZUKU_RUNNING"))
        assertTrue(variables.has("SHIZUKU_PERMISSION_GRANTED"))
    }

    @Test
    fun appForegroundTriggerRunsMatchingAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-app-foreground-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Foreground app smoke")
                    .put("path", "hermes-app-foreground-trigger.txt")
                    .put("content", "foreground-ok")
                    .put("trigger", "app_foreground")
                    .put("trigger_package_name", app.packageName),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("app_foreground", created.getJSONObject("automation").getString("trigger_type"))
        assertEquals(app.packageName, created.getJSONObject("automation").getString("trigger_package_name"))

        val missed = JSONObject(HermesAutomationBridge.runAppForegroundTriggerJson(app, "com.example.other"))
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val matched = JSONObject(HermesAutomationBridge.runAppForegroundTriggerJson(app, app.packageName))
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("foreground-ok", target.readText())
    }

    @Test
    fun notificationPostedTriggerRunsMatchingAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-notification-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Notification smoke")
                    .put("path", "hermes-notification-trigger.txt")
                    .put("content", "%NOTIFICATION_PACKAGE:%NOTIFICATION_TITLE:%NOTIFICATION_TEXT")
                    .put("trigger", "notification_posted")
                    .put("trigger_package_name", app.packageName),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("notification_posted", created.getJSONObject("automation").getString("trigger_type"))
        assertEquals(app.packageName, created.getJSONObject("automation").getString("trigger_package_name"))

        val missed = JSONObject(
            HermesAutomationBridge.runNotificationPostedTriggerJson(
                app,
                "com.example.other",
                "Ignored",
                "No match",
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val matched = JSONObject(
            HermesAutomationBridge.runNotificationPostedTriggerJson(
                app,
                app.packageName,
                "Hermes title",
                "Hermes text",
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("${app.packageName}:Hermes title:Hermes text", target.readText())
    }

    @Test
    fun calendarEventTriggerRunsMatchingAutomationAndExposesVariables() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-calendar-trigger.txt").apply { delete() }

        val variable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%WORK_CAL")
                    .put("value", "Hermes Work"),
            ),
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Calendar event smoke")
                    .put("path", "hermes-calendar-trigger.txt")
                    .put("content", "%CALENDAR_NAME:%CALTITLE:%CALDESCR:%CALLOC")
                    .put("trigger", "calendar_event")
                    .put("calendar_name", "%WORK_CAL")
                    .put("title_contains", "Flight"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        val triggerData = JSONObject(automation.getString("trigger_data"))
        assertEquals("calendar_event", automation.getString("trigger_type"))
        assertEquals("%WORK_CAL", triggerData.getString("calendar_name"))
        assertEquals("Flight", triggerData.getString("title_contains"))

        val missed = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_calendar_event_trigger",
                JSONObject()
                    .put("calendar_name", "Hermes Work")
                    .put("calendar_title", "Lunch"),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val matched = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_calendar_event_trigger",
                JSONObject()
                    .put("calendar_name", "Hermes Work")
                    .put("calendar_title", "Flight to SF")
                    .put("calendar_description", "Boarding")
                    .put("calendar_location", "SFO"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals("calendar_event", matched.getString("trigger"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("Hermes Work:Flight to SF:Boarding:SFO", target.readText())

        val variables = JSONObject(HermesAutomationBridge.performActionJson(app, "list_variables"))
            .getJSONObject("variables")
        assertEquals("Flight to SF", variables.getString("CALTITLE"))
        assertEquals("Boarding", variables.getString("CALDESCR"))
        assertEquals("SFO", variables.getString("CALLOC"))
    }

    @Test
    fun locationTriggerRunsMatchingAutomationAndExposesVariables() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-location-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Location trigger smoke")
                    .put("path", "hermes-location-trigger.txt")
                    .put("content", "%LOCATION_PROVIDER:%LAT:%LON:%LOCNAME:%LOCACC")
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
        val triggerData = JSONObject(automation.getString("trigger_data"))
        assertEquals("location", automation.getString("trigger_type"))
        assertEquals("37.7749", triggerData.getString("latitude"))
        assertEquals("-122.4194", triggerData.getString("longitude"))
        assertEquals("office", triggerData.getString("location_name"))

        val missed = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_location_trigger",
                JSONObject()
                    .put("latitude", 37.7849)
                    .put("longitude", -122.4194)
                    .put("location_provider", "gps")
                    .put("location_name", "Hermes Office"),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val matched = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_location_trigger",
                JSONObject()
                    .put("latitude", 37.7750)
                    .put("longitude", -122.4195)
                    .put("accuracy_meters", 12.5)
                    .put("location_provider", "gps")
                    .put("location_name", "Hermes Office"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals("location", matched.getString("trigger"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("gps:37.775:-122.4195:Hermes Office:12.5", target.readText())

        val variables = JSONObject(HermesAutomationBridge.performActionJson(app, "list_variables"))
            .getJSONObject("variables")
        assertEquals("37.775", variables.getString("LOCATION_LATITUDE"))
        assertEquals("-122.4195", variables.getString("LOCATION_LONGITUDE"))
        assertEquals("Hermes Office", variables.getString("LOCATION_NAME"))
    }

    @Test
    fun sensorTriggerRunsMatchingAutomationAndExposesVariables() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-sensor-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Sensor trigger smoke")
                    .put("path", "hermes-sensor-trigger.txt")
                    .put("content", "%SENSOR_TYPE:%SENSOR_EVENT:%SENSOR_VALUE_NAME:%SENSOR_VALUE:%SENSOR_UNIT:%SENSOR_ACCURACY")
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
        val triggerData = JSONObject(automation.getString("trigger_data"))
        assertEquals("sensor", automation.getString("trigger_type"))
        assertEquals("accelerometer", triggerData.getString("sensor_type"))
        assertEquals("shake", triggerData.getString("sensor_event"))
        assertEquals("x", triggerData.getString("value_name"))

        val missed = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_sensor_trigger",
                JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_event", "shake")
                    .put("value_name", "x")
                    .put("sensor_value", 0.5),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val matched = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_sensor_trigger",
                JSONObject()
                    .put("sensor_type", "accelerometer")
                    .put("sensor_event", "shake")
                    .put("value_name", "x")
                    .put("sensor_value", 2.25)
                    .put("sensor_unit", "m/s^2")
                    .put("sensor_accuracy", "high"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals("sensor", matched.getString("trigger"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("accelerometer:shake:x:2.25:m/s^2:high", target.readText())

        val variables = JSONObject(HermesAutomationBridge.performActionJson(app, "list_variables"))
            .getJSONObject("variables")
        assertEquals("accelerometer", variables.getString("SENSOR_TYPE"))
        assertEquals("shake", variables.getString("SENSOR_EVENT"))
        assertEquals("2.25", variables.getString("SENSOR_VALUE"))
        assertEquals("x", variables.getString("SENSOR_VALUE_NAME"))
    }

    @Test
    fun timeTriggerRunsMatchingAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-time-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Time trigger smoke")
                    .put("path", "hermes-time-trigger.txt")
                    .put("content", "%TIME:%TIME_DAY")
                    .put("trigger", "time")
                    .put("time", "00:01")
                    .put("days_of_week", "weekday"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("time", created.getJSONObject("automation").getString("trigger_type"))
        assertEquals(1, created.getJSONObject("automation").getInt("trigger_time_minutes"))
        assertEquals("MON,TUE,WED,THU,FRI", created.getJSONObject("automation").getString("trigger_days_of_week"))

        val triggered = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_trigger",
                JSONObject().put("trigger", "time"),
            ),
        )
        assertTrue(triggered.toString(), triggered.getBoolean("success"))
        assertEquals(1, triggered.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertTrue(target.readText().matches(Regex("\\d{2}:\\d{2}:(MON|TUE|WED|THU|FRI|SAT|SUN)")))
    }
}
