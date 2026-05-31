package com.mobilefork.hermesagent.device

import android.content.Context
import com.mobilefork.hermesagent.data.DeviceCapabilityStore
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
            put("foreground_package_name", HermesAccessibilityController.currentForegroundPackageName())
            put("last_notification_package_name", HermesNotificationController.currentPackageName())
            put("last_notification_title", HermesNotificationController.currentTitle())
            put("last_notification_text", HermesNotificationController.currentText())
            put(
                "available_global_actions",
                JSONArray(HermesGlobalAction.values().map { action -> action.name.lowercase() }),
            )
            put("linux_enabled", false)
            put("linux_android_abi", "")
            put("linux_termux_arch", "")
            put("linux_prefix_path", "")
            put("linux_bash_path", "")
            put("linux_home_path", "")
            put("linux_tmp_path", "")
            put("linux_package_count", 0)
            put("wifi_enabled", false)
            put("active_network_label", "")
            put("airplane_mode_enabled", false)
            put("active_network_metered", false)
            put("data_saver_enabled", false)
            put("bluetooth_supported", false)
            put("bluetooth_enabled", false)
            put("bluetooth_permission_granted", false)
            put("paired_bluetooth_devices", JSONArray())
            put("usb_host_supported", false)
            put("usb_device_count", 0)
            put("usb_devices", JSONArray())
            put("nfc_supported", false)
            put("nfc_enabled", false)
            put("overlay_permission_granted", false)
            put("notification_permission_granted", false)
            put("notification_listener_enabled", false)
            put("notification_listener_connected", false)
            put("background_persistence_enabled", capabilities.backgroundPersistenceEnabled)
            put("runtime_service_running", false)
            put("floating_button_enabled", capabilities.floatingButtonEnabled)
            put("floating_button_running", false)
            put("floating_button_visible", false)
            put("floating_button_error", "")
            put("resizable_window_support", false)
            put("freeform_window_supported", false)
            put("available_system_actions", JSONArray())
            put("device_diagnostics_tool_available", true)
            put("hindsight_memory_tool_available", true)
            put("hindsight_memory_count", 0)
            put("hindsight_reinforced_memory_count", 0)
            put("hindsight_promoted_memory_count", 0)
            put("usage_access_granted", false)
            put("camera_supported", false)
            put("camera_permission_granted", false)
            put("wifi_scan_permission_status", JSONObject())
            put("bluetooth_scan_permission_status", JSONObject())
            put("android_soc_profile", JSONObject())
            put("android_device_performance_profile", JSONObject())
            put("available_diagnostic_sensor_types", JSONArray())
            put("available_diagnostics_actions", JSONArray())
        }
        stateFile(context).writeText(payload.toString(), Charsets.UTF_8)
    }
}
