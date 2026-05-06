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
}
