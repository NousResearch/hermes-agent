package com.mobilefork.hermesagent.device

import android.content.Context
import org.json.JSONObject
import java.util.Locale

object HermesHyMemoryBridge {
    private const val PROVIDER_NAME = "hy_memory"
    private const val ANDROID_BACKEND = "android_local_hy_memory"

    fun performActionJson(context: Context, rawAction: String, arguments: JSONObject = JSONObject()): String {
        val action = rawAction.trim().lowercase(Locale.US).ifBlank { "status" }
        val delegatedAction = when (action) {
            "hy_memory_status" -> "status"
            "hy_memory_retain" -> "retain"
            "hy_memory_recall" -> "recall"
            else -> action
        }
        return annotate(
            JSONObject(
                HermesHindsightMemoryBridge.performActionJson(
                    context = context.applicationContext,
                    rawAction = delegatedAction,
                    arguments = arguments,
                ),
            ),
        ).toString()
    }

    fun statusJson(context: Context): JSONObject {
        return annotate(HermesHindsightMemoryBridge.statusJson(context.applicationContext))
    }

    private fun annotate(payload: JSONObject): JSONObject {
        return payload
            .put("provider", PROVIDER_NAME)
            .put("backend", ANDROID_BACKEND)
            .put("tool_name", "hy_memory_tool")
            .put("compatibility_alias", "hindsight_memory_tool")
            .put("hy_memory_package", "hy-memory")
            .put("python_provider", "plugins.memory.hy_memory")
    }
}
