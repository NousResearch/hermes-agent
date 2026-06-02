package com.mobilefork.hermesagent

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.mobilefork.hermesagent.device.HermesHyMemoryBridge
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HermesHyMemoryBridgeInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @Before
    fun clearBefore() {
        HermesHyMemoryBridge.performActionJson(app, "clear", JSONObject())
    }

    @After
    fun clearAfter() {
        HermesHyMemoryBridge.performActionJson(app, "clear", JSONObject())
    }

    @Test
    fun hyMemoryRetainsRecallsAndAnnotatesAndroidBackend() {
        val initialStatus = HermesHyMemoryBridge.statusJson(app)
        assertTrue(initialStatus.toString(), initialStatus.getBoolean("success"))
        assertEquals("hy_memory", initialStatus.getString("provider"))
        assertEquals("android_local_hy_memory", initialStatus.getString("backend"))
        assertEquals("hy_memory_tool", initialStatus.getString("tool_name"))
        assertEquals("hindsight_memory_tool", initialStatus.getString("compatibility_alias"))

        val sentinel = "Hermes hy memory validation sentinel violet-714 retained correctly"
        val retain = JSONObject(
            HermesHyMemoryBridge.performActionJson(
                app,
                "hy_memory_retain",
                JSONObject()
                    .put("content", sentinel)
                    .put("category", "validation")
                    .put("source", "androidTest")
                    .put("tags", JSONArray().put("hy-memory").put("instrumentation")),
            )
        )
        assertTrue(retain.toString(), retain.getBoolean("success"))
        assertEquals("hy_memory", retain.getString("provider"))
        assertEquals("android_local_hy_memory", retain.getString("backend"))
        assertEquals("retain", retain.getString("action"))
        assertEquals(retain.toString(), 1, retain.getInt("retained_count"))

        val recall = JSONObject(
            HermesHyMemoryBridge.performActionJson(
                app,
                "hy_memory_recall",
                JSONObject()
                    .put("query", "violet-714")
                    .put("limit", 3),
            )
        )
        assertTrue(recall.toString(), recall.getBoolean("success"))
        assertEquals("hy_memory", recall.getString("provider"))
        assertEquals("recall", recall.getString("action"))
        assertTrue(recall.toString(), recall.getInt("result_count") >= 1)
        assertTrue(recall.toString(), recall.getJSONArray("memories").toString().contains(sentinel))

        val context = JSONObject(
            HermesHyMemoryBridge.performActionJson(
                app,
                "relevant_context",
                JSONObject()
                    .put("query", "violet-714")
                    .put("limit", 3),
            )
        )
        assertTrue(context.toString(), context.getBoolean("success"))
        assertEquals("hy_memory", context.getString("provider"))
        assertTrue(context.toString(), context.getString("system_prompt_context").contains("violet-714"))
    }
}
