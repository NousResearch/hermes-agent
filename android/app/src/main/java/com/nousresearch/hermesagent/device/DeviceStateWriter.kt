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
        val systemStatus = HermesSystemControlBridge.readStatus(context)
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
            put("background_persistence_enabled", systemStatus.backgroundPersistenceEnabled)
            put("runtime_service_running", systemStatus.runtimeServiceRunning)
            put("resizable_window_support", systemStatus.resizableWindowSupport)
            put("freeform_window_supported", systemStatus.freeformWindowSupported)
            put("available_system_actions", JSONArray(systemStatus.availableSystemActions))
        }
        stateFile(context).writeText(payload.toString(), Charsets.UTF_8)
    }
}
