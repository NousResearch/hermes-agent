package com.nousresearch.hermesagent.device

import android.content.Context
import com.nousresearch.hermesagent.data.DeviceCapabilityStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

object DeviceStateWriter {
    private const val STATE_FILE_NAME = "android-device-state.json"

    fun workspaceDir(context: Context): File {
        return File(context.filesDir, "hermes-home/workspace").apply {
            mkdirs()
        }
    }

    private fun stateFile(context: Context): File {
        return File(context.filesDir, "hermes-home/$STATE_FILE_NAME").apply {
            parentFile?.mkdirs()
        }
    }

    fun write(context: Context) {
        val capabilities = DeviceCapabilityStore(context).load()
        val payload = JSONObject().apply {
            put("workspace_path", workspaceDir(context).absolutePath)
            put("shared_tree_uri", capabilities.sharedFolderUri)
            put("shared_tree_label", capabilities.sharedFolderLabel)
            put("accessibility_enabled", HermesAccessibilityController.isServiceEnabled(context))
            put("accessibility_connected", HermesAccessibilityController.isServiceConnected())
            put(
                "available_global_actions",
                JSONArray(HermesGlobalAction.values().map { action -> action.name.lowercase() }),
            )
        }
        stateFile(context).writeText(payload.toString(), Charsets.UTF_8)
    }
}
