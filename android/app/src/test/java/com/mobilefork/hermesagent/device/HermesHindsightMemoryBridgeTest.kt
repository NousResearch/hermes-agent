package com.mobilefork.hermesagent.device

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesHindsightMemoryBridgeTest {
    private val context: Context = RuntimeEnvironment.getApplication()

    @Before
    fun clearStore() {
        HermesHindsightMemoryBridge.performActionJson(context, "clear")
    }

    @Test
    fun retainsRecallsAndReflectsStructuredMemories() {
        val retain = JSONObject(
            HermesHindsightMemoryBridge.performActionJson(
                context,
                "retain",
                JSONObject()
                    .put("content", "JDK 21 is the stable Android validation toolchain for native Hermes builds.")
                    .put("source", "test")
                    .put("category", "android")
                    .put("tags", JSONArray().put("validation").put("toolchain")),
            ),
        )

        assertTrue(retain.getBoolean("success"))
        assertEquals(1, retain.getInt("retained_count"))
        assertTrue(retain.getJSONArray("retained_memories").getJSONObject(0).getJSONArray("entities").toString().contains("JDK"))

        val recall = JSONObject(
            HermesHindsightMemoryBridge.performActionJson(
                context,
                "recall",
                JSONObject().put("query", "Android JDK validation").put("limit", 3),
            ),
        )

        assertTrue(recall.getBoolean("success"))
        assertEquals(1, recall.getInt("result_count"))
        assertTrue(recall.getJSONArray("memories").getJSONObject(0).getString("content").contains("JDK 21"))

        val reflect = JSONObject(HermesHindsightMemoryBridge.performActionJson(context, "reflect"))
        assertTrue(reflect.getBoolean("success"))
        assertEquals(1, reflect.getInt("after_count"))
    }

    @Test
    fun relevantContextBuildsBoundedPromptMemoryFromPriorTurns() {
        HermesHindsightMemoryBridge.performActionJson(
            context,
            "retain",
            JSONObject()
                .put("content", "Conversation phone-1: user asked about keyboard gaps | assistant answered: composer must follow IME insets.")
                .put("source", "chat")
                .put("category", "conversation")
                .put("tags", JSONArray().put("conversation").put("auto_recall")),
        )
        HermesHindsightMemoryBridge.performActionJson(
            context,
            "retain",
            JSONObject()
                .put("content", "Release validation should verify signed APK checksums and F-Droid metadata.")
                .put("source", "release")
                .put("category", "android"),
        )

        val contextJson = JSONObject(
            HermesHindsightMemoryBridge.performActionJson(
                context,
                "relevant_context",
                JSONObject()
                    .put("query", "keyboard composer IME")
                    .put("limit", 4)
                    .put("max_chars", 320),
            ),
        )

        assertTrue(contextJson.getBoolean("success"))
        assertEquals("local_keyword_entity_recency_salience_rag", contextJson.getString("retrieval_model"))
        assertTrue(contextJson.getString("system_prompt_context").contains("keyboard gaps"))
        assertTrue(contextJson.getString("system_prompt_context").length <= 320)
        assertTrue(contextJson.getInt("recalled_memory_count") >= 1)
    }

    @Test
    fun promotesHighReuseMemoriesIntoPromptContext() {
        HermesHindsightMemoryBridge.performActionJson(
            context,
            "retain",
            JSONObject()
                .put("content", "MediaTek phones should use SOC-neutral diagnostics without assuming Adreno GPU support.")
                .put("source", "kai_parity_research")
                .put("category", "compatibility")
                .put("tags", JSONArray().put("mediatek").put("soc")),
        )

        repeat(5) {
            HermesHindsightMemoryBridge.performActionJson(
                context,
                "recall",
                JSONObject().put("query", "MediaTek SOC diagnostics").put("limit", 1),
            )
        }

        val status = JSONObject(HermesHindsightMemoryBridge.performActionJson(context, "status"))
        assertEquals(1, status.getInt("promoted_memory_count"))
        assertEquals(5, status.getInt("promotion_hit_threshold"))

        val contextJson = JSONObject(HermesHindsightMemoryBridge.performActionJson(context, "promoted_context"))
        assertTrue(contextJson.getString("system_prompt_context").contains("MediaTek phones"))
        val promoted = contextJson.getJSONArray("promoted_memories").getJSONObject(0)
        assertTrue(promoted.getBoolean("promoted"))
        assertTrue(promoted.getInt("hit_count") >= 5)
        assertTrue(promoted.getJSONArray("tags").toString().contains("promoted"))
    }
}
