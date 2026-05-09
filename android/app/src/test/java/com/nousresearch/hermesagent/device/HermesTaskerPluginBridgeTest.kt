package com.nousresearch.hermesagent.device

import android.os.Bundle
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesTaskerPluginBridgeTest {
    @Test
    fun taskerPluginResultBundleRequiresTokenBeforeRunningAutomation() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_variable_action_task",
                JSONObject()
                    .put("id", "auto-tasker-plugin")
                    .put("label", "Tasker plugin smoke")
                    .put("variable_action", "set")
                    .put("name", "%PLUGIN_RESULT")
                    .put("value", "ran via %TASKER_PLUGIN_AUTOMATION_ID")
                    .put("automation_enabled", false),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val missingBundle = JSONObject(HermesTaskerPluginBridge.runPluginBundleJson(context, null))
        assertFalse(missingBundle.toString(), missingBundle.getBoolean("success"))

        val unauthorized = JSONObject(
            HermesTaskerPluginBridge.runPluginBundleJson(
                context,
                Bundle().apply {
                    putString(HermesTaskerPluginBridge.KEY_AUTOMATION_ID, "auto-tasker-plugin")
                    putString(HermesTaskerPluginBridge.KEY_TOKEN, "not-authorized-token")
                },
            ),
        )
        assertFalse(unauthorized.toString(), unauthorized.getBoolean("success"))

        val resultIntent = HermesTaskerPluginBridge.buildResultIntent(
            context,
            automationId = "auto-tasker-plugin",
            label = "Plugin smoke",
        )
        assertEquals(
            "Run Hermes automation: Plugin smoke",
            resultIntent.getStringExtra(HermesTaskerPluginBridge.EXTRA_STRING_BLURB),
        )
        val bundle = HermesTaskerPluginBridge.bundleFromIntent(resultIntent)
        val fired = JSONObject(HermesTaskerPluginBridge.runPluginBundleJson(context, bundle, "net.dinglisch.android.taskerm"))

        assertTrue(fired.toString(), fired.getBoolean("success"))
        assertTrue(fired.getBoolean("tasker_plugin"))
        assertEquals(HermesTaskerPluginBridge.TRIGGER_TASKER_PLUGIN, fired.getString("trigger"))
        assertEquals("ran via auto-tasker-plugin", store.getVariable("PLUGIN_RESULT"))
        assertEquals("auto-tasker-plugin", store.getVariable("TASKER_PLUGIN_AUTOMATION_ID"))
        assertEquals("net.dinglisch.android.taskerm", store.getVariable("TASKER_PLUGIN_CALLER_PACKAGE"))
    }

    @Test
    fun taskerConditionPluginQueriesHermesVariableWithToken() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()
        store.setVariable("READY", "yes")

        val missingToken = HermesTaskerConditionBridge.queryCondition(
            context,
            Bundle().apply {
                putString(
                    HermesTaskerConditionBridge.KEY_CONDITION_TYPE,
                    HermesTaskerConditionBridge.CONDITION_VARIABLE_EQUALS,
                )
                putString(HermesTaskerConditionBridge.KEY_VARIABLE_NAME, "READY")
                putString(HermesTaskerConditionBridge.KEY_EXPECTED_VALUE, "yes")
            },
        )
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_UNKNOWN, missingToken.resultCode)

        val resultIntent = HermesTaskerConditionBridge.buildResultIntent(
            context = context,
            conditionType = HermesTaskerConditionBridge.CONDITION_VARIABLE_EQUALS,
            variableName = "%READY",
            expectedValue = "yes",
            label = "Hermes ready",
        )
        assertEquals("Hermes ready", resultIntent.getStringExtra(HermesTaskerConditionBridge.EXTRA_STRING_BLURB))
        val bundle = HermesTaskerConditionBridge.bundleFromIntent(resultIntent)
        assertNotNull(bundle)

        val satisfied = HermesTaskerConditionBridge.queryCondition(context, bundle)
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_SATISFIED, satisfied.resultCode)
        assertEquals("true", satisfied.variables.getString("%hermes_satisfied"))
        assertEquals("yes", satisfied.variables.getString("%hermes_variable_value"))

        store.setVariable("READY", "no")
        val unsatisfied = HermesTaskerConditionBridge.queryCondition(context, bundle)
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_UNSATISFIED, unsatisfied.resultCode)
        assertEquals("false", unsatisfied.variables.getString("%hermes_satisfied"))
    }

    @Test
    fun taskerConditionPluginQueriesAutomationState() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val created = JSONObject(
            HermesAutomationBridge.performActionJson(
                context,
                "create_variable_action_task",
                JSONObject()
                    .put("id", "condition-automation")
                    .put("label", "Condition automation")
                    .put("variable_action", "set")
                    .put("name", "%COND_RESULT")
                    .put("value", "ran")
                    .put("automation_enabled", true),
            ),
        )
        assertTrue(created.toString(), created.getBoolean("success"))

        val enabledIntent = HermesTaskerConditionBridge.buildResultIntent(
            context = context,
            conditionType = HermesTaskerConditionBridge.CONDITION_AUTOMATION_ENABLED,
            automationId = "condition-automation",
        )
        val enabled = HermesTaskerConditionBridge.queryCondition(
            context,
            HermesTaskerConditionBridge.bundleFromIntent(enabledIntent),
        )
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_SATISFIED, enabled.resultCode)

        val record = store.get("condition-automation")
        assertNotNull(record)
        store.upsert(record!!.copy(enabled = false, lastSuccess = false, lastRunEpochMs = 123L))

        val disabledIntent = HermesTaskerConditionBridge.buildResultIntent(
            context = context,
            conditionType = HermesTaskerConditionBridge.CONDITION_AUTOMATION_DISABLED,
            automationId = "condition-automation",
        )
        val disabled = HermesTaskerConditionBridge.queryCondition(
            context,
            HermesTaskerConditionBridge.bundleFromIntent(disabledIntent),
        )
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_SATISFIED, disabled.resultCode)

        val failedIntent = HermesTaskerConditionBridge.buildResultIntent(
            context = context,
            conditionType = HermesTaskerConditionBridge.CONDITION_AUTOMATION_LAST_FAILED,
            automationId = "condition-automation",
        )
        val failed = HermesTaskerConditionBridge.queryCondition(
            context,
            HermesTaskerConditionBridge.bundleFromIntent(failedIntent),
        )
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_SATISFIED, failed.resultCode)
    }

    @Test
    fun taskerEventPluginMatchesVerifiedAutomationUpdates() {
        val context = RuntimeEnvironment.getApplication()
        val store = HermesAutomationStore(context)
        store.clear()

        val record = HermesAutomationRecord(
            id = "event-automation",
            label = "Event automation",
            actionType = "variable_action",
            command = "{}",
            useShizuku = false,
            triggerType = "manual",
            intervalMinutes = null,
            enabled = true,
            createdAtEpochMs = 1L,
            updatedAtEpochMs = 1L,
        )
        val resultIntent = HermesTaskerEventBridge.buildResultIntent(
            context = context,
            eventType = HermesTaskerEventBridge.EVENT_AUTOMATION_SUCCEEDED,
            automationId = record.id,
            label = "Hermes success",
        )
        assertEquals("Hermes success", resultIntent.getStringExtra(HermesTaskerEventBridge.EXTRA_STRING_BLURB))

        val payload = JSONObject()
            .put(HermesTaskerEventBridge.PAYLOAD_EVENT_TYPE, HermesTaskerEventBridge.EVENT_AUTOMATION_FINISHED)
            .put(HermesTaskerEventBridge.PAYLOAD_AUTOMATION_ID, record.id)
            .put(HermesTaskerEventBridge.PAYLOAD_AUTOMATION_LABEL, record.label)
            .put(HermesTaskerEventBridge.PAYLOAD_TRIGGER, "manual")
            .put(HermesTaskerEventBridge.PAYLOAD_SUCCESS, true)
            .put(HermesTaskerEventBridge.PAYLOAD_RESULT, "done")
        val queryIntent = HermesTaskerEventBridge.buildRequestQueryIntent(context, payload)
            .putExtra(HermesTaskerEventBridge.EXTRA_BUNDLE, HermesTaskerEventBridge.bundleFromIntent(resultIntent))
        val event = HermesTaskerEventBridge.queryEvent(
            context,
            queryIntent,
            HermesTaskerEventBridge.bundleFromIntent(resultIntent),
        )

        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_SATISFIED, event.resultCode)
        assertEquals("true", event.variables.getString("%hermes_satisfied"))
        assertEquals("event-automation", event.variables.getString("%hermes_automation_id"))
        assertEquals("manual", event.variables.getString("%hermes_trigger"))

        val failedIntent = HermesTaskerEventBridge.buildResultIntent(
            context = context,
            eventType = HermesTaskerEventBridge.EVENT_AUTOMATION_FAILED,
            automationId = record.id,
        )
        val failed = HermesTaskerEventBridge.queryEvent(
            context,
            queryIntent,
            HermesTaskerEventBridge.bundleFromIntent(failedIntent),
        )
        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_UNSATISFIED, failed.resultCode)
    }

    @Test
    fun taskerEventPluginRejectsUnverifiedUpdates() {
        val context = RuntimeEnvironment.getApplication()
        val resultIntent = HermesTaskerEventBridge.buildResultIntent(
            context = context,
            eventType = HermesTaskerEventBridge.EVENT_SHIZUKU_AVAILABLE,
        )
        val unknown = HermesTaskerEventBridge.queryEvent(
            context,
            null,
            HermesTaskerEventBridge.bundleFromIntent(resultIntent),
        )

        assertEquals(HermesTaskerConditionBridge.RESULT_CONDITION_UNKNOWN, unknown.resultCode)
    }
}
