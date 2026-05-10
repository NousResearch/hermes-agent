package com.nousresearch.hermesagent

import android.Manifest
import android.app.Application
import android.content.Intent
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nousresearch.hermesagent.device.HermesAutomationBridge
import com.nousresearch.hermesagent.device.HermesAutomationStore
import com.nousresearch.hermesagent.device.HermesAutomationWidgetBridge
import com.nousresearch.hermesagent.device.HermesExternalTriggerReceiver
import com.nousresearch.hermesagent.device.HermesLauncherShortcutBridge
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import com.nousresearch.hermesagent.device.HermesLogcatEvent
import com.nousresearch.hermesagent.device.HermesLogcatWatcherBridge
import com.nousresearch.hermesagent.device.HermesNotificationActionBridge
import com.nousresearch.hermesagent.device.HermesQuickSettingsTileBridge
import org.json.JSONArray
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

        val history = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_history",
                JSONObject().put("limit", 5),
            )
        )
        assertTrue(history.toString(), history.getBoolean("success"))
        assertTrue(history.getInt("run_count") >= 1)
        val latestRun = history.getJSONArray("runs").getJSONObject(0)
        assertEquals(id, latestRun.getString("automation_id"))
        assertEquals("Automation smoke", latestRun.getString("automation_label"))
        assertEquals("manual", latestRun.getString("trigger"))
        assertTrue(latestRun.getBoolean("success"))
        assertTrue(latestRun.getLong("duration_ms") >= 0L)

        val enabled = JSONObject(HermesAutomationBridge.performActionJson(app, "enable", JSONObject().put("id", id)))
        assertTrue(enabled.toString(), enabled.getBoolean("success"))
        assertTrue(enabled.getJSONObject("automation").getBoolean("enabled"))

        val standby = JSONObject(HermesAutomationBridge.performActionJson(app, "operator_standby_status"))
        assertTrue(standby.toString(), standby.getBoolean("success"))
        val standbyDispatch = standby.getJSONObject("standby_dispatch")
        assertTrue(standbyDispatch.toString(), standbyDispatch.getBoolean("ready"))
        assertEquals(1, standbyDispatch.getInt("enabled_automation_count"))
        assertEquals(1, standbyDispatch.getInt("recent_run_count"))
        assertEquals("Automation smoke", standbyDispatch.getString("last_run_label"))

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
    fun externalTriggerBroadcastRunsMatchingAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-external-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "External trigger smoke")
                    .put("path", "hermes-external-trigger.txt")
                    .put("content", "%SA_TRIGGER_ID|%SA_REFERRER|%SA_EXTRAS")
                    .put("trigger", "external_trigger")
                    .put("trigger_id", "broadcast-smoke")
                    .put("external_token", "token-smoke")
                    .put("referrer_contains", "smoke"),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        InstrumentationRegistry.getInstrumentation().waitForIdleSync()
        app.sendBroadcast(
            Intent(HermesExternalTriggerReceiver.ACTION_EXTERNAL_TRIGGER)
                .setPackage(app.packageName)
                .putExtra(HermesExternalTriggerReceiver.EXTRA_TRIGGER_ID, "broadcast-smoke")
                .putExtra(HermesExternalTriggerReceiver.EXTRA_TOKEN, "token-smoke")
                .putExtra(HermesExternalTriggerReceiver.EXTRA_REFERRER, "smoke://test")
                .putExtra("mode", "broadcast")
        )

        assertTrue("Expected ${target.absolutePath}", eventually(timeoutMs = 20_000L) { target.isFile })
        assertEquals("""broadcast-smoke|smoke://test|{"mode":"broadcast"}""", target.readText())
    }

    @Test
    fun openGuiStyleRemoteDispatchRunsMatchingEnabledAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-remote-dispatch.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_shell_task",
                JSONObject()
                    .put("label", "OpenGUI dispatch smoke")
                    .put("command", "printf '%DISPATCH_EXECUTION_ID|%DISPATCH_TASK_ID|%DISPATCH_TASK_NAME' > \"\$HOME/hermes-remote-dispatch.txt\"")
                    .put("trigger", "remote_dispatch"),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val dispatched = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_remote_dispatch",
                JSONObject()
                    .put("executionId", 42)
                    .put("taskId", 7)
                    .put("taskName", "OpenGUI dispatch smoke")
                    .put("dispatch_source", "opengui_standby")
                    .put("dispatch_channel", "discord"),
            )
        )
        assertTrue(dispatched.toString(), dispatched.getBoolean("success"))
        assertEquals(1, dispatched.getInt("matched_count"))
        assertEquals("remote_dispatch", dispatched.getString("trigger"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("42|7|OpenGUI dispatch smoke", target.readText())

        val history = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_history",
                JSONObject().put("limit", 1),
            )
        )
        val latestRun = history.getJSONArray("runs").getJSONObject(0)
        assertEquals("remote_dispatch", latestRun.getString("trigger"))
        assertEquals("opengui_standby", latestRun.getString("dispatch_source"))
        assertEquals("discord", latestRun.getString("dispatch_channel"))
        assertEquals("42", latestRun.getString("remote_execution_id"))
        assertEquals("7", latestRun.getString("remote_task_id"))
        assertEquals("OpenGUI dispatch smoke", latestRun.getString("remote_task_name"))

        val executionStatus = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "operator_execution_status",
                JSONObject().put("executionId", 42),
            )
        )
        assertTrue(executionStatus.toString(), executionStatus.getBoolean("success"))
        assertEquals("completed", executionStatus.getString("status"))
        assertEquals(1, executionStatus.getInt("matched_run_count"))
        val execution = executionStatus.getJSONObject("execution")
        assertEquals("42", execution.getString("execution_id"))
        assertEquals("7", execution.getString("task_id"))
        assertEquals("OpenGUI dispatch smoke", execution.getString("task_name"))
        assertEquals("opengui_standby", execution.getString("source"))

        val standby = JSONObject(HermesAutomationBridge.performActionJson(app, "operator_standby_status"))
            .getJSONObject("standby_dispatch")
        assertEquals(1, standby.getInt("remote_dispatch_count"))
        assertEquals("OpenGUI dispatch smoke", standby.getString("last_dispatch_task_name"))
        assertTrue(standby.getJSONArray("compatible_dispatch_payloads").toString().contains("OpenGUI standby:dispatch"))
        assertEquals("/standby", standby.getString("standby_namespace"))
        assertEquals("standby:heartbeat", standby.getString("standby_heartbeat_event"))
        assertTrue(standby.getJSONArray("compatible_dispatch_payloads").toString().contains("status"))

        val devices = JSONObject(HermesAutomationBridge.performActionJson(app, "operator_devices"))
        assertTrue(devices.toString(), devices.getBoolean("success"))
        assertEquals(1, devices.getInt("device_count"))
        assertEquals(1, devices.getInt("online_device_count"))
        val device = devices.getJSONArray("devices").getJSONObject(0)
        assertTrue(device.getBoolean("online"))
        assertEquals(1, device.getInt("remote_dispatch_count"))
        assertEquals("OpenGUI dispatch smoke", device.getString("last_dispatch_task_name"))
        assertTrue(devices.getJSONArray("compatible_device_queries").toString().contains("OpenGUI devices"))
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
    fun launcherShortcutRunsSavedAutomationFromShortcutIntent() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-launcher-shortcut.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Shortcut smoke")
                    .put("path", "hermes-launcher-shortcut.txt")
                    .put("content", "shortcut-ok")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automationId = created.getJSONObject("automation").getString("id")

        val shortcut = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_launcher_shortcut",
                JSONObject()
                    .put("automation_id", automationId)
                    .put("label", "Shortcut smoke")
                    .put("pin", false),
            ),
        )
        assertTrue(shortcut.toString(), shortcut.getBoolean("success"))
        assertEquals(automationId, shortcut.getString("automation_id"))
        assertTrue(shortcut.toString(), shortcut.getBoolean("dynamic_shortcut_created"))

        val listed = JSONObject(HermesAutomationBridge.performActionJson(app, "list_launcher_shortcuts"))
        assertTrue(listed.toString(), listed.getBoolean("success"))
        assertTrue(listed.getJSONArray("dynamic_shortcuts").toString().contains(shortcut.getString("shortcut_id")))

        val run = JSONObject(
            HermesLauncherShortcutBridge.handleShortcutIntentJson(
                app,
                HermesLauncherShortcutBridge.shortcutIntent(app, automationId),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals("launcher_shortcut", run.getString("trigger"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("shortcut-ok", target.readText())

        val removed = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "remove_launcher_shortcut",
                JSONObject().put("automation_id", automationId),
            ),
        )
        assertTrue(removed.toString(), removed.getBoolean("success"))
    }

    @Test
    fun quickSettingsTileRunsConfiguredSavedAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-quick-settings-tile.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Tile smoke")
                    .put("path", "hermes-quick-settings-tile.txt")
                    .put("content", "tile-ok")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automationId = created.getJSONObject("automation").getString("id")

        val configured = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_quick_settings_tile_automation",
                JSONObject()
                    .put("automation_id", automationId)
                    .put("label", "Tile smoke"),
            ),
        )
        assertTrue(configured.toString(), configured.getBoolean("success"))
        assertTrue(configured.getBoolean("configured"))
        assertEquals(automationId, configured.getString("automation_id"))

        val status = JSONObject(HermesAutomationBridge.performActionJson(app, "get_quick_settings_tile_automation"))
        assertTrue(status.toString(), status.getBoolean("success"))
        assertTrue(status.getBoolean("configured"))
        assertTrue(status.getBoolean("automation_exists"))
        assertEquals(automationId, status.getString("automation_id"))

        val run = JSONObject(HermesQuickSettingsTileBridge.runConfiguredAutomationJson(app))
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals("quick_settings_tile", run.getString("trigger"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("tile-ok", target.readText())

        val cleared = JSONObject(HermesAutomationBridge.performActionJson(app, "clear_quick_settings_tile_automation"))
        assertTrue(cleared.toString(), cleared.getBoolean("success"))
        assertFalse(cleared.getBoolean("configured"))
    }

    @Test
    fun homeScreenWidgetRunsConfiguredSavedAutomation() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-home-screen-widget.txt").apply { delete() }
        val appWidgetId = 1001

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Widget smoke")
                    .put("path", "hermes-home-screen-widget.txt")
                    .put("content", "widget-ok")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automationId = created.getJSONObject("automation").getString("id")

        val configured = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_home_screen_widget_automation",
                JSONObject()
                    .put("automation_id", automationId)
                    .put("app_widget_id", appWidgetId)
                    .put("label", "Widget smoke"),
            ),
        )
        assertTrue(configured.toString(), configured.getBoolean("success"))
        assertTrue(configured.getBoolean("configured"))
        assertEquals(automationId, configured.getString("automation_id"))
        assertEquals(appWidgetId, configured.getInt("app_widget_id"))

        val status = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "get_home_screen_widget_automation",
                JSONObject().put("app_widget_id", appWidgetId),
            ),
        )
        assertTrue(status.toString(), status.getBoolean("success"))
        assertTrue(status.getBoolean("configured"))
        assertTrue(status.getBoolean("automation_exists"))
        assertFalse(status.getBoolean("uses_default_config"))
        assertEquals(automationId, status.getString("automation_id"))

        val listed = JSONObject(HermesAutomationBridge.performActionJson(app, "list_home_screen_widgets"))
        assertTrue(listed.toString(), listed.getBoolean("success"))
        assertTrue(listed.getJSONArray("stored_widget_configs").toString().contains(automationId))

        val run = JSONObject(HermesAutomationWidgetBridge.runConfiguredAutomationJson(app, appWidgetId))
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals("home_screen_widget", run.getString("trigger"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("widget-ok", target.readText())

        val cleared = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "clear_home_screen_widget_automation",
                JSONObject().put("app_widget_id", appWidgetId),
            ),
        )
        assertTrue(cleared.toString(), cleared.getBoolean("success"))
        assertFalse(cleared.getBoolean("configured"))
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
    fun intentAutomationCanOpenBrowserUriWhenBrowserIsAvailable() {
        val uri = "https://example.com/#hermes-browser-smoke"

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_open_uri_task",
                JSONObject()
                    .put("label", "Open browser smoke")
                    .put("data_uri", uri)
                    .put("enabled", false),
            )
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            )
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        val result = run.getJSONObject("result")
        assertEquals("open_uri", result.getString("action"))
        assertEquals(uri, result.getString("data_uri"))
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
    fun notificationActionAutomationPostsUpdatesAndCancels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            runCatching {
                InstrumentationRegistry.getInstrumentation().uiAutomation.grantRuntimePermission(
                    app.packageName,
                    Manifest.permission.POST_NOTIFICATIONS,
                )
            }
        }

        val variable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%NOTICE_TEXT")
                    .put("value", "notification-action-ok"),
            ),
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_notification_task",
                JSONObject()
                    .put("label", "Notification action smoke")
                    .put("notification_title", "Hermes automation")
                    .put("notification_text", "%NOTICE_TEXT")
                    .put("notification_id", "9901")
                    .put("notification_tag", "hermes-instrumented")
                    .put("channel_id", "hermes_instrumented")
                    .put("priority", "high")
                    .put("only_alert_once", true)
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("notification_action", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            ),
        )
        val result = run.getJSONObject("result")
        if (!run.getBoolean("success") &&
            result.optString("requires_permission") == Manifest.permission.POST_NOTIFICATIONS
        ) {
            return
        }
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals("notification_post", result.getString("action"))
        assertEquals(9901, result.getInt("notification_id"))
        assertEquals("hermes-instrumented", result.getString("notification_tag"))

        val cancel = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_notification_task",
                JSONObject()
                    .put("label", "Cancel notification action smoke")
                    .put("notification_action", "cancel")
                    .put("notification_id", "9901")
                    .put("notification_tag", "hermes-instrumented")
                    .put("enabled", false),
            ),
        )
        assertTrue(cancel.toString(), cancel.getBoolean("success"))

        val cancelRun = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", cancel.getJSONObject("automation").getString("id")),
            ),
        )
        assertTrue(cancelRun.toString(), cancelRun.getBoolean("success"))
        assertEquals("notification_cancel", cancelRun.getJSONObject("result").getString("action"))
    }

    @Test
    fun notificationButtonCanRunSavedAutomation() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            runCatching {
                InstrumentationRegistry.getInstrumentation().uiAutomation.grantRuntimePermission(
                    app.packageName,
                    Manifest.permission.POST_NOTIFICATIONS,
                )
            }
        }

        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-notification-button.txt").apply { delete() }
        val targetId = "notification-button-target"

        val fileTask = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("id", targetId)
                    .put("label", "Notification button target")
                    .put("path", "hermes-notification-button.txt")
                    .put("content", "button:%BUTTON_VALUE")
                    .put("enabled", false),
            ),
        )
        assertTrue(fileTask.toString(), fileTask.getBoolean("success"))

        val variable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%BUTTON_VALUE")
                    .put("value", "ran"),
            ),
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_notification_task",
                JSONObject()
                    .put("label", "Notification button smoke")
                    .put("notification_title", "Hermes button")
                    .put("notification_text", "Run a saved automation")
                    .put("notification_id", "9910")
                    .put("notification_tag", "hermes-button")
                    .put(
                        "notification_buttons",
                        JSONArray().put(
                            JSONObject()
                                .put("title", "Run")
                                .put("action", "run_automation")
                                .put("automation_id", targetId)
                                .put("dismiss_on_tap", true),
                        ),
                    )
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val notificationPayload = JSONObject(created.getJSONObject("automation").getString("command"))
        assertEquals(1, notificationPayload.getJSONArray("notification_buttons").length())

        val runNotification = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", created.getJSONObject("automation").getString("id")),
            ),
        )
        val result = runNotification.getJSONObject("result")
        if (!runNotification.getBoolean("success") &&
            result.optString("requires_permission") == Manifest.permission.POST_NOTIFICATIONS
        ) {
            return
        }
        assertTrue(runNotification.toString(), runNotification.getBoolean("success"))
        assertEquals(1, result.getInt("notification_button_count"))

        val click = JSONObject(
            HermesNotificationActionBridge.handleNotificationButtonIntentJson(
                app,
                Intent(app, com.nousresearch.hermesagent.device.HermesAutomationReceiver::class.java).apply {
                    action = HermesNotificationActionBridge.ACTION_NOTIFICATION_BUTTON
                    putExtra(HermesNotificationActionBridge.EXTRA_BUTTON_ACTION, "run_automation")
                    putExtra(HermesNotificationActionBridge.EXTRA_AUTOMATION_ID, targetId)
                    putExtra(HermesNotificationActionBridge.EXTRA_NOTIFICATION_ID, 9910)
                    putExtra(HermesNotificationActionBridge.EXTRA_NOTIFICATION_TAG, "hermes-button")
                    putExtra(HermesNotificationActionBridge.EXTRA_DISMISS_ON_TAP, true)
                },
            ),
        )
        assertTrue(click.toString(), click.getBoolean("success"))
        assertEquals("run_automation", click.getString("notification_button_action"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("button:ran", target.readText())
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
    fun logcatEntryTriggerRunsMatchingAutomationAndExposesVariables() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-logcat-trigger.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Logcat entry smoke")
                    .put("path", "hermes-logcat-trigger.txt")
                    .put("content", "%LOGCAT_LEVEL:%LOGCAT_TAG:%LOGCAT_PID:%LOGCAT_PACKAGE:%LOGCAT_MESSAGE")
                    .put("trigger", "logcat_entry")
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message_contains", "ANR")
                    .put("logcat_level", "E")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "com.example.app"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        val automation = created.getJSONObject("automation")
        val triggerData = JSONObject(automation.getString("trigger_data"))
        assertEquals("logcat_entry", automation.getString("trigger_type"))
        assertEquals("ActivityManager", triggerData.getString("tag"))
        assertTrue(triggerData.getBoolean("requires_shizuku_for_background_watch"))

        val missed = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_logcat_entry_trigger",
                JSONObject()
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message", "Process started")
                    .put("logcat_level", "E")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "com.example.app"),
            ),
        )
        assertTrue(missed.toString(), missed.getBoolean("success"))
        assertEquals(0, missed.getInt("matched_count"))
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val matched = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_logcat_entry_trigger",
                JSONObject()
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message", "ANR in com.example.app")
                    .put("logcat_level", "error")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "com.example.app"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals("logcat_entry", matched.getString("trigger"))
        assertTrue(matched.getBoolean("requires_shizuku_for_background_watch"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("error:ActivityManager:4242:com.example.app:ANR in com.example.app", target.readText())

        val variables = JSONObject(HermesAutomationBridge.performActionJson(app, "list_variables"))
            .getJSONObject("variables")
        assertEquals("ActivityManager", variables.getString("LOGCAT_TAG"))
        assertEquals("ANR in com.example.app", variables.getString("LOGCAT_MESSAGE"))
        assertEquals("error", variables.getString("LOGCAT_LEVEL"))
        assertEquals("4242", variables.getString("LOGCAT_PID"))
        assertEquals("com.example.app", variables.getString("LOGCAT_PACKAGE"))
    }

    @Test
    fun logcatEntryTriggerMatchesSharedUidPackageCandidates() {
        val attributed = HermesLogcatWatcherBridge.attributePackages(
            HermesLogcatEvent(
                timestamp = "05-07 12:00:00.000",
                uid = "1000",
                pid = "4242",
                level = "E",
                tag = "ActivityManager",
                message = "ANR in com.example.app",
            ),
            listOf("android", "com.example.app", "bad package"),
        )
        assertEquals("com.example.app", attributed.packageName)
        assertEquals(listOf("android", "com.example.app"), attributed.packageCandidates)
        assertEquals("message", attributed.packageNameSource)

        val shared = HermesLogcatWatcherBridge.attributePackages(
            attributed.copy(message = "Shared uid event", packageName = "", packageCandidates = emptyList(), packageNameSource = ""),
            listOf("android", "com.example.app"),
        )
        assertEquals("android,com.example.app", shared.packageName)
        assertEquals("uid_shared", shared.packageNameSource)

        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-logcat-candidates.txt").apply { delete() }

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Logcat package candidates")
                    .put("path", "hermes-logcat-candidates.txt")
                    .put("content", "%LOGCAT_PACKAGE:%LOGCAT_PACKAGE_CANDIDATES:%LOGCAT_PACKAGE_SOURCE")
                    .put("trigger", "logcat_entry")
                    .put("trigger_package_name", "com.example.app")
                    .put("logcat_message_contains", "Shared uid"),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val matched = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run_logcat_entry_trigger",
                JSONObject()
                    .put("logcat_tag", "ActivityManager")
                    .put("logcat_message", "Shared uid event")
                    .put("logcat_level", "W")
                    .put("logcat_pid", "4242")
                    .put("logcat_package_name", "android,com.example.app")
                    .put("logcat_package_candidates", "android,com.example.app")
                    .put("logcat_package_source", "uid_shared"),
            ),
        )
        assertTrue(matched.toString(), matched.getBoolean("success"))
        assertEquals(1, matched.getInt("matched_count"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("android,com.example.app:android,com.example.app:uid_shared", target.readText())

        val variables = JSONObject(HermesAutomationBridge.performActionJson(app, "list_variables"))
            .getJSONObject("variables")
        assertEquals("android,com.example.app", variables.getString("LOGCAT_PACKAGE_CANDIDATES"))
        assertEquals("uid_shared", variables.getString("LOGCAT_PACKAGE_SOURCE"))
    }

    @Test
    fun logcatWatcherRequiresShizukuOrStopsCleanly() {
        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("label", "Logcat watcher smoke")
                    .put("path", "hermes-logcat-watcher.txt")
                    .put("content", "%LOGCAT_TAG:%LOGCAT_MESSAGE")
                    .put("trigger", "logcat_entry")
                    .put("logcat_tag", "Hermes")
                    .put("logcat_message_contains", "watcher")
                    .put("enabled", true),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val status = JSONObject(HermesAutomationBridge.performActionJson(app, "logcat_watcher_status"))
        assertTrue(status.toString(), status.getBoolean("success"))
        assertTrue(status.getBoolean("requires_shizuku"))

        val started = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "start_logcat_watcher",
                JSONObject()
                    .put("scan_interval_seconds", 5)
                    .put("max_lines", 25),
            ),
        )
        if (started.optBoolean("success", false)) {
            assertTrue(started.toString(), started.getBoolean("running"))
            val stopped = JSONObject(HermesAutomationBridge.performActionJson(app, "stop_logcat_watcher"))
            assertTrue(stopped.toString(), stopped.getBoolean("success"))
            assertTrue(stopped.toString(), stopped.getBoolean("stopped"))
        } else {
            assertTrue(started.toString(), started.getString("error").contains("Shizuku"))
            assertFalse(started.optBoolean("running", false))
        }
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

    @Test
    fun automationBundleImportRestoresVariablesAndRunsFileTask() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-bundle-import.txt").apply { delete() }

        val variable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%message")
                    .put("value", "bundle-restored"),
            ),
        )
        assertTrue(variable.toString(), variable.getBoolean("success"))

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_file_write_task",
                JSONObject()
                    .put("id", "auto-bundle-import")
                    .put("label", "Bundle import smoke")
                    .put("path", "hermes-bundle-import.txt")
                    .put("content", "import:%MESSAGE")
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val exported = JSONObject(HermesAutomationBridge.performActionJson(app, "export_automations"))
        HermesAutomationStore(app).clear()
        assertFalse("Expected ${target.absolutePath} to stay absent", target.exists())

        val imported = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "import_automations",
                JSONObject()
                    .put("bundle", exported)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals(1, imported.getInt("imported_automation_count"))
        assertEquals(1, imported.getInt("imported_variable_count"))

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", "auto-bundle-import"),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("import:bundle-restored", target.readText())
    }

    @Test
    fun taskerXmlImportCreatesDisabledSafeFileTask() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-tasker-import.txt").apply { delete() }

        val taskerXml = """
            <TaskerData sr="" dvi="1" tv="6.6.18">
              <Task sr="task1">
                <nme>Hermes Tasker Import</nme>
                <Action sr="act0" ve="7">
                  <code>410</code>
                  <Str sr="arg0" ve="3">hermes-tasker-import.txt</Str>
                  <Str sr="arg1" ve="3">tasker:%TASKER_MESSAGE</Str>
                  <Int sr="arg2" val="0"/>
                </Action>
                <Action sr="act1" ve="7">
                  <code>129</code>
                </Action>
              </Task>
              <Variable sr="var1">
                <nme>%TASKER_MESSAGE</nme>
                <val>xml-ok</val>
              </Variable>
            </TaskerData>
        """.trimIndent()

        val imported = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "import_tasker_xml",
                JSONObject()
                    .put("tasker_xml", taskerXml)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals(1, imported.getInt("tasker_imported_action_count"))
        assertEquals(1, imported.getJSONArray("tasker_skipped_actions").length())
        assertEquals("xml-ok", HermesAutomationStore(app).getVariable("TASKER_MESSAGE"))

        val record = HermesAutomationStore(app).list().single()
        assertFalse(record.enabled)
        assertEquals("file_write", record.actionType)

        val run = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", record.id),
            ),
        )
        assertTrue(run.toString(), run.getBoolean("success"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("tasker:xml-ok", target.readText())
    }

    @Test
    fun variableActionAutomationSetsAndClearsVariables() {
        val baseVariable = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "set_variable",
                JSONObject()
                    .put("name", "%BASE_MESSAGE")
                    .put("value", "expanded"),
            ),
        )
        assertTrue(baseVariable.toString(), baseVariable.getBoolean("success"))

        val setTask = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_variable_action_task",
                JSONObject()
                    .put("id", "auto-variable-set")
                    .put("label", "Variable set smoke")
                    .put("variable_action", "set")
                    .put("name", "%TASKER_DYNAMIC")
                    .put("value", "value:%BASE_MESSAGE")
                    .put("enabled", false),
            ),
        )
        assertTrue(setTask.toString(), setTask.getBoolean("success"))
        assertEquals("variable_action", setTask.getJSONObject("automation").getString("action_type"))

        val setRun = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", "auto-variable-set"),
            ),
        )
        assertTrue(setRun.toString(), setRun.getBoolean("success"))
        assertEquals("value:expanded", HermesAutomationStore(app).getVariable("TASKER_DYNAMIC"))

        val clearTask = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_variable_action_task",
                JSONObject()
                    .put("id", "auto-variable-clear")
                    .put("label", "Variable clear smoke")
                    .put("variable_action", "clear")
                    .put("name", "%TASKER_DYNAMIC")
                    .put("enabled", false),
            ),
        )
        assertTrue(clearTask.toString(), clearTask.getBoolean("success"))

        val clearRun = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "run",
                JSONObject().put("id", "auto-variable-clear"),
            ),
        )
        assertTrue(clearRun.toString(), clearRun.getBoolean("success"))
        assertFalse(HermesAutomationStore(app).listVariables().has("TASKER_DYNAMIC"))
    }

    @Test
    fun taskerXmlImportCreatesVariableSetAndClearActions() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val target = File(workspace, "hermes-tasker-variable-import.txt").apply { delete() }

        val taskerXml = """
            <TaskerData sr="" dvi="1" tv="6.6.18">
              <Task sr="task1">
                <nme>Hermes Tasker Variables</nme>
                <Action sr="act0" ve="7">
                  <code>547</code>
                  <Str sr="arg0" ve="3">%TASKER_DYNAMIC</Str>
                  <Str sr="arg1" ve="3">xml-action-ok</Str>
                </Action>
                <Action sr="act1" ve="7">
                  <code>410</code>
                  <Str sr="arg0" ve="3">hermes-tasker-variable-import.txt</Str>
                  <Str sr="arg1" ve="3">tasker:%TASKER_DYNAMIC</Str>
                  <Int sr="arg2" val="0"/>
                </Action>
                <Action sr="act2" ve="7">
                  <code>549</code>
                  <Str sr="arg0" ve="3">%TASKER_DYNAMIC</Str>
                </Action>
              </Task>
            </TaskerData>
        """.trimIndent()

        val imported = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "import_tasker_xml",
                JSONObject()
                    .put("tasker_xml", taskerXml)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals(3, imported.getInt("tasker_imported_action_count"))
        assertEquals(0, imported.getJSONArray("tasker_skipped_actions").length())

        val records = HermesAutomationStore(app).list()
        val setRecord = records.first { it.label.contains("Variable Set") }
        val fileRecord = records.first { it.actionType == "file_write" }
        val clearRecord = records.first { it.label.contains("Variable Clear") }
        assertFalse(setRecord.enabled)
        assertEquals("variable_action", setRecord.actionType)
        assertEquals("variable_action", clearRecord.actionType)

        val setRun = JSONObject(HermesAutomationBridge.performActionJson(app, "run", JSONObject().put("id", setRecord.id)))
        assertTrue(setRun.toString(), setRun.getBoolean("success"))
        assertEquals("xml-action-ok", HermesAutomationStore(app).getVariable("TASKER_DYNAMIC"))

        val fileRun = JSONObject(HermesAutomationBridge.performActionJson(app, "run", JSONObject().put("id", fileRecord.id)))
        assertTrue(fileRun.toString(), fileRun.getBoolean("success"))
        assertTrue("Expected ${target.absolutePath}", target.isFile)
        assertEquals("tasker:xml-action-ok", target.readText())

        val clearRun = JSONObject(HermesAutomationBridge.performActionJson(app, "run", JSONObject().put("id", clearRecord.id)))
        assertTrue(clearRun.toString(), clearRun.getBoolean("success"))
        assertFalse(HermesAutomationStore(app).listVariables().has("TASKER_DYNAMIC"))
    }

    @Test
    fun waitAutomationRunsBoundedDelay() {
        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_wait_task",
                JSONObject()
                    .put("id", "auto-wait")
                    .put("label", "Wait smoke")
                    .put("duration_ms", 5)
                    .put("enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))
        assertEquals("wait", created.getJSONObject("automation").getString("action_type"))

        val run = JSONObject(HermesAutomationBridge.performActionJson(app, "run", JSONObject().put("id", "auto-wait")))
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals(5L, run.getJSONObject("result").getLong("duration_ms"))

        val rejected = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "create_wait_task",
                JSONObject()
                    .put("id", "auto-wait-too-long")
                    .put("label", "Wait too long")
                    .put("duration_ms", 60_001),
            ),
        )
        assertFalse(rejected.toString(), rejected.getBoolean("success"))
        assertTrue(rejected.getString("error").contains("cannot exceed"))
    }

    @Test
    fun taskerXmlImportCreatesWaitAction() {
        HermesAutomationStore(app).clear()
        val taskerXml = """
            <TaskerData sr="" dvi="1" tv="6.6.18">
              <Task sr="task1">
                <nme>Hermes Tasker Wait</nme>
                <Action sr="act0" ve="7">
                  <code>30</code>
                  <Int sr="arg0" val="5"/>
                  <Int sr="arg1" val="0"/>
                  <Int sr="arg2" val="0"/>
                  <Int sr="arg3" val="0"/>
                  <Int sr="arg4" val="0"/>
                </Action>
              </Task>
            </TaskerData>
        """.trimIndent()

        val imported = JSONObject(
            HermesAutomationBridge.performActionJson(
                app,
                "import_tasker_xml",
                JSONObject()
                    .put("tasker_xml", taskerXml)
                    .put("replace", true),
            ),
        )
        assertTrue(imported.toString(), imported.getBoolean("success"))
        assertEquals(1, imported.getInt("tasker_imported_action_count"))
        assertEquals(0, imported.getJSONArray("tasker_skipped_actions").length())

        val record = HermesAutomationStore(app).list().single()
        assertFalse(record.enabled)
        assertEquals("wait", record.actionType)

        val run = JSONObject(HermesAutomationBridge.performActionJson(app, "run", JSONObject().put("id", record.id)))
        assertTrue(run.toString(), run.getBoolean("success"))
        assertEquals(5L, run.getJSONObject("result").getLong("duration_ms"))
    }

    private fun eventually(timeoutMs: Long = 5_000L, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (condition()) {
                return true
            }
            Thread.sleep(100L)
        }
        return condition()
    }
}
