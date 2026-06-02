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
        val linuxState = runCatching { HermesLinuxSubsystemBridge.readState(context) }.getOrNull()
        val systemStatus = runCatching { HermesSystemControlBridge.readStatus(context) }.getOrNull()
        val privilegedAccess = systemStatus?.privilegedAccess
        val memoryStatus = runCatching { HermesHyMemoryBridge.statusJson(context) }.getOrNull()
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
            put("linux_enabled", linuxState?.optBoolean("enabled", false) ?: false)
            put("linux_android_abi", linuxState?.optString("android_abi").orEmpty())
            put("linux_termux_arch", linuxState?.optString("termux_arch").orEmpty())
            put("linux_prefix_path", linuxState?.optString("prefix_path").orEmpty())
            put("linux_bash_path", linuxState?.optString("bash_path").orEmpty().ifBlank { linuxState?.optString("shell_path").orEmpty() })
            put("linux_home_path", linuxState?.optString("home_path").orEmpty())
            put("linux_tmp_path", linuxState?.optString("tmp_path").orEmpty())
            put("linux_package_count", linuxState?.optJSONArray("packages")?.length() ?: 0)
            put("wifi_enabled", systemStatus?.wifiEnabled ?: false)
            put("active_network_label", systemStatus?.activeNetworkLabel.orEmpty())
            put("airplane_mode_enabled", systemStatus?.airplaneModeEnabled ?: false)
            put("active_network_metered", systemStatus?.activeNetworkMetered ?: false)
            put("data_saver_enabled", systemStatus?.dataSaverEnabled ?: false)
            put("bluetooth_supported", systemStatus?.bluetoothSupported ?: false)
            put("bluetooth_enabled", systemStatus?.bluetoothEnabled ?: false)
            put("bluetooth_permission_granted", systemStatus?.bluetoothPermissionGranted ?: false)
            put("paired_bluetooth_devices", JSONArray(systemStatus?.pairedBluetoothDevices ?: emptyList<String>()))
            put("usb_host_supported", systemStatus?.usbHostSupported ?: false)
            put("usb_device_count", systemStatus?.usbDeviceCount ?: 0)
            put("usb_devices", JSONArray(systemStatus?.usbDevices ?: emptyList<String>()))
            put("nfc_supported", systemStatus?.nfcSupported ?: false)
            put("nfc_enabled", systemStatus?.nfcEnabled ?: false)
            put("overlay_permission_granted", systemStatus?.overlayPermissionGranted ?: false)
            put("notification_permission_granted", systemStatus?.notificationPermissionGranted ?: false)
            put("notification_listener_enabled", systemStatus?.notificationListenerEnabled ?: false)
            put("notification_listener_connected", systemStatus?.notificationListenerConnected ?: false)
            put("background_persistence_enabled", capabilities.backgroundPersistenceEnabled)
            put("runtime_service_running", systemStatus?.runtimeServiceRunning ?: false)
            put("floating_button_enabled", capabilities.floatingButtonEnabled)
            put("floating_button_running", systemStatus?.floatingButtonRunning ?: false)
            put("floating_button_visible", systemStatus?.floatingButtonVisible ?: false)
            put("floating_button_error", systemStatus?.floatingButtonError.orEmpty())
            put("resizable_window_support", systemStatus?.resizableWindowSupport ?: false)
            put("freeform_window_supported", systemStatus?.freeformWindowSupported ?: false)
            put(
                "privileged_access",
                privilegedAccess?.let { HermesPrivilegedAccessBridge.statusToJson(it) } ?: JSONObject(),
            )
            put("shizuku_installed", privilegedAccess?.shizukuInstalled ?: false)
            put("sui_installed", privilegedAccess?.suiInstalled ?: false)
            put("shizuku_binder_alive", privilegedAccess?.shizukuBinderAlive ?: false)
            put("shizuku_permission_granted", privilegedAccess?.shizukuPermissionGranted ?: false)
            put("shizuku_privilege_label", privilegedAccess?.shizukuPrivilegeLabel.orEmpty())
            put("shizuku_adb_start_command", privilegedAccess?.adbStartCommand.orEmpty())
            put("available_privileged_actions", JSONArray(privilegedAccess?.availablePrivilegedActions ?: emptyList<String>()))
            put("available_system_actions", JSONArray(systemStatus?.availableSystemActions ?: emptyList<String>()))
            put("device_diagnostics_tool_available", true)
            put("hy_memory_tool_available", true)
            put("hy_memory_backend", memoryStatus?.optString("backend").orEmpty())
            put("hy_memory_count", memoryStatus?.optInt("memory_count", 0) ?: 0)
            put("hy_memory_reinforced_memory_count", memoryStatus?.optInt("reinforced_memory_count", 0) ?: 0)
            put("hy_memory_promoted_memory_count", memoryStatus?.optInt("promoted_memory_count", 0) ?: 0)
            put("hindsight_memory_tool_available", true)
            put("hindsight_memory_count", memoryStatus?.optInt("memory_count", 0) ?: 0)
            put("hindsight_reinforced_memory_count", memoryStatus?.optInt("reinforced_memory_count", 0) ?: 0)
            put("hindsight_promoted_memory_count", memoryStatus?.optInt("promoted_memory_count", 0) ?: 0)
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
