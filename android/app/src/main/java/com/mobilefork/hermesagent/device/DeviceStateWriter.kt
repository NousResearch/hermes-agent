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
        val linuxState = HermesLinuxSubsystemBridge.readState(context)
        val systemStatus = HermesSystemControlBridge.readStatus(context)
        val diagnosticsStatus = HermesDeviceDiagnosticsBridge.statusJson(context)
        val hindsightStatus = HermesHindsightMemoryBridge.statusJson(context)
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
            put("linux_enabled", linuxState?.optBoolean("enabled") == true)
            put("linux_android_abi", linuxState?.optString("android_abi").orEmpty())
            put("linux_termux_arch", linuxState?.optString("termux_arch").orEmpty())
            put("linux_prefix_path", linuxState?.optString("prefix_path").orEmpty())
            put("linux_bash_path", linuxState?.optString("bash_path").orEmpty())
            put("linux_home_path", linuxState?.optString("home_path").orEmpty())
            put("linux_tmp_path", linuxState?.optString("tmp_path").orEmpty())
            put("linux_package_count", linuxState?.optJSONArray("packages")?.length() ?: 0)
            put("wifi_enabled", systemStatus.wifiEnabled)
            put("active_network_label", systemStatus.activeNetworkLabel)
            put("airplane_mode_enabled", systemStatus.airplaneModeEnabled)
            put("active_network_metered", systemStatus.activeNetworkMetered)
            put("data_saver_enabled", systemStatus.dataSaverEnabled)
            put("bluetooth_supported", systemStatus.bluetoothSupported)
            put("bluetooth_enabled", systemStatus.bluetoothEnabled)
            put("bluetooth_permission_granted", systemStatus.bluetoothPermissionGranted)
            put("paired_bluetooth_devices", JSONArray(systemStatus.pairedBluetoothDevices))
            put("usb_host_supported", systemStatus.usbHostSupported)
            put("usb_device_count", systemStatus.usbDeviceCount)
            put("usb_devices", JSONArray(systemStatus.usbDevices))
            put("nfc_supported", systemStatus.nfcSupported)
            put("nfc_enabled", systemStatus.nfcEnabled)
            put("overlay_permission_granted", systemStatus.overlayPermissionGranted)
            put("notification_permission_granted", systemStatus.notificationPermissionGranted)
            put("notification_listener_enabled", systemStatus.notificationListenerEnabled)
            put("notification_listener_connected", systemStatus.notificationListenerConnected)
            put("background_persistence_enabled", systemStatus.backgroundPersistenceEnabled)
            put("runtime_service_running", systemStatus.runtimeServiceRunning)
            put("floating_button_enabled", systemStatus.floatingButtonEnabled)
            put("floating_button_running", systemStatus.floatingButtonRunning)
            put("floating_button_visible", systemStatus.floatingButtonVisible)
            put("floating_button_error", systemStatus.floatingButtonError)
            put("resizable_window_support", systemStatus.resizableWindowSupport)
            put("freeform_window_supported", systemStatus.freeformWindowSupported)
            put("available_system_actions", JSONArray(systemStatus.availableSystemActions))
            put("device_diagnostics_tool_available", true)
            put("hindsight_memory_tool_available", true)
            put("hindsight_memory_count", hindsightStatus.optInt("memory_count", 0))
            put("hindsight_reinforced_memory_count", hindsightStatus.optInt("reinforced_memory_count", 0))
            put("hindsight_promoted_memory_count", hindsightStatus.optInt("promoted_memory_count", 0))
            put("usage_access_granted", diagnosticsStatus.optBoolean("usage_access_granted", false))
            put("camera_supported", diagnosticsStatus.optBoolean("camera_supported", false))
            put("camera_permission_granted", diagnosticsStatus.optBoolean("camera_permission_granted", false))
            put("wifi_scan_permission_status", diagnosticsStatus.optJSONObject("wifi_scan_permission_status") ?: JSONObject())
            put("bluetooth_scan_permission_status", diagnosticsStatus.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject())
            put("android_soc_profile", diagnosticsStatus.optJSONObject("soc_profile") ?: JSONObject())
            put("android_device_performance_profile", diagnosticsStatus.optJSONObject("device_performance_profile") ?: JSONObject())
            put("available_diagnostic_sensor_types", diagnosticsStatus.optJSONArray("available_sensor_types") ?: JSONArray())
            put("available_diagnostics_actions", diagnosticsStatus.optJSONArray("available_actions") ?: JSONArray())
        }
        stateFile(context).writeText(payload.toString(), Charsets.UTF_8)
    }
}
