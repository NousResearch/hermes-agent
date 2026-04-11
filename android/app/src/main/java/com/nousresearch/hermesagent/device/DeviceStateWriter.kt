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
        val linuxState = HermesLinuxSubsystemBridge.readState(context)
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
            put("linux_enabled", linuxState?.optBoolean("enabled") == true)
            put("linux_android_abi", linuxState?.optString("android_abi").orEmpty())
            put("linux_termux_arch", linuxState?.optString("termux_arch").orEmpty())
            put("linux_prefix_path", linuxState?.optString("prefix_path").orEmpty())
            put("linux_bash_path", linuxState?.optString("bash_path").orEmpty())
            put("linux_home_path", linuxState?.optString("home_path").orEmpty())
            put("linux_tmp_path", linuxState?.optString("tmp_path").orEmpty())
            put("linux_package_count", linuxState?.optJSONArray("packages")?.length() ?: 0)
        }
        stateFile(context).writeText(payload.toString(), Charsets.UTF_8)
    }
}
