package com.nousresearch.hermesagent.device

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.net.wifi.WifiManager
import android.nfc.NfcAdapter
import android.os.Build
import android.provider.Settings
import android.hardware.usb.UsbManager
import androidx.core.content.ContextCompat
import com.nousresearch.hermesagent.HermesApplication
import com.nousresearch.hermesagent.backend.HermesRuntimeService
import com.nousresearch.hermesagent.data.DeviceCapabilityStore
import org.json.JSONArray
import org.json.JSONObject

private val DEFAULT_SYSTEM_ACTIONS = listOf(
    "open_wifi_panel",
    "open_bluetooth_settings",
    "open_connected_devices_settings",
    "open_nfc_settings",
    "open_notification_settings",
    "open_overlay_settings",
    "open_accessibility_settings",
    "start_background_runtime",
    "stop_background_runtime",
)

data class HermesSystemStatus(
    val wifiEnabled: Boolean = false,
    val activeNetworkLabel: String = "Offline",
    val bluetoothSupported: Boolean = false,
    val bluetoothEnabled: Boolean = false,
    val bluetoothPermissionGranted: Boolean = false,
    val pairedBluetoothDevices: List<String> = emptyList(),
    val usbHostSupported: Boolean = false,
    val usbDeviceCount: Int = 0,
    val usbDevices: List<String> = emptyList(),
    val nfcSupported: Boolean = false,
    val nfcEnabled: Boolean = false,
    val overlayPermissionGranted: Boolean = false,
    val notificationPermissionGranted: Boolean = true,
    val backgroundPersistenceEnabled: Boolean = false,
    val runtimeServiceRunning: Boolean = false,
    val resizableWindowSupport: Boolean = true,
    val freeformWindowSupported: Boolean = false,
    val availableSystemActions: List<String> = DEFAULT_SYSTEM_ACTIONS,
)

data class HermesSystemActionResult(
    val success: Boolean,
    val action: String,
    val message: String,
)

object HermesSystemControlBridge {
    fun readStatus(context: Context): HermesSystemStatus {
        val appContext = context.applicationContext
        val capabilityStore = DeviceCapabilityStore(appContext)
        val stored = capabilityStore.load()
        val wifiManager = appContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
        val connectivityManager = appContext.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
        val bluetoothManager = appContext.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
        val bluetoothAdapter = bluetoothManager?.adapter ?: runCatching { BluetoothAdapter.getDefaultAdapter() }.getOrNull()
        val bluetoothPermissionGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ContextCompat.checkSelfPermission(appContext, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
        val pairedDevices = if (bluetoothAdapter != null && bluetoothPermissionGranted) {
            bluetoothAdapter.bondedDevices.orEmpty()
                .map { device -> device.name?.takeIf { it.isNotBlank() } ?: device.address }
                .sorted()
        } else {
            emptyList()
        }
        val usbManager = appContext.getSystemService(Context.USB_SERVICE) as? UsbManager
        val usbDevices = usbManager?.deviceList?.values
            ?.map { device ->
                buildString {
                    append(device.productName?.takeIf { !it.isNullOrBlank() } ?: device.deviceName)
                    append(" · ")
                    append("vid=")
                    append(device.vendorId)
                    append(" pid=")
                    append(device.productId)
                }
            }
            ?.sorted()
            .orEmpty()
        val nfcAdapter = runCatching { NfcAdapter.getDefaultAdapter(appContext) }.getOrNull()
        val notificationPermissionGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(appContext, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
        return HermesSystemStatus(
            wifiEnabled = wifiManager?.isWifiEnabled == true,
            activeNetworkLabel = activeNetworkLabel(connectivityManager),
            bluetoothSupported = bluetoothAdapter != null,
            bluetoothEnabled = bluetoothAdapter?.isEnabled == true,
            bluetoothPermissionGranted = bluetoothPermissionGranted,
            pairedBluetoothDevices = pairedDevices,
            usbHostSupported = appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_USB_HOST),
            usbDeviceCount = usbDevices.size,
            usbDevices = usbDevices,
            nfcSupported = nfcAdapter != null,
            nfcEnabled = nfcAdapter?.isEnabled == true,
            overlayPermissionGranted = Settings.canDrawOverlays(appContext),
            notificationPermissionGranted = notificationPermissionGranted,
            backgroundPersistenceEnabled = stored.backgroundPersistenceEnabled,
            runtimeServiceRunning = HermesRuntimeService.isRunning(),
            resizableWindowSupport = true,
            freeformWindowSupported = appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_FREEFORM_WINDOW_MANAGEMENT),
        )
    }

    fun performAction(context: Context, action: String): HermesSystemActionResult {
        val appContext = context.applicationContext
        return when (action) {
            "open_wifi_panel" -> launchIntent(appContext, action, wifiIntent(), "Opened Wi-Fi + internet controls")
            "open_bluetooth_settings" -> launchIntent(appContext, action, Intent(Settings.ACTION_BLUETOOTH_SETTINGS), "Opened Bluetooth settings")
            "open_connected_devices_settings" -> launchIntent(appContext, action, Intent(Settings.ACTION_WIRELESS_SETTINGS), "Opened connected-device settings")
            "open_nfc_settings" -> launchIntent(appContext, action, Intent(Settings.ACTION_NFC_SETTINGS), "Opened NFC settings")
            "open_notification_settings" -> launchIntent(appContext, action, notificationSettingsIntent(appContext), "Opened Hermes notification settings")
            "open_overlay_settings" -> launchIntent(
                appContext,
                action,
                Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${appContext.packageName}")),
                "Opened overlay permission settings",
            )
            "open_accessibility_settings" -> launchIntent(appContext, action, Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS), "Opened accessibility settings")
            "start_background_runtime" -> {
                DeviceCapabilityStore(appContext).saveBackgroundPersistenceEnabled(true)
                HermesRuntimeService.start(appContext)
                DeviceStateWriter.write(appContext)
                HermesSystemActionResult(success = true, action = action, message = "Started Hermes background runtime")
            }
            "stop_background_runtime" -> {
                DeviceCapabilityStore(appContext).saveBackgroundPersistenceEnabled(false)
                HermesRuntimeService.stop(appContext)
                DeviceStateWriter.write(appContext)
                HermesSystemActionResult(success = true, action = action, message = "Stopped Hermes background runtime persistence")
            }
            else -> HermesSystemActionResult(success = false, action = action, message = "Unsupported Android system action: $action")
        }
    }

    @JvmStatic
    fun statusJson(): String {
        return statusToJson(readStatus(HermesApplication.instance.applicationContext)).toString()
    }

    @JvmStatic
    fun performActionJson(action: String): String {
        return actionToJson(performAction(HermesApplication.instance.applicationContext, action)).toString()
    }

    private fun launchIntent(context: Context, action: String, intent: Intent, successMessage: String): HermesSystemActionResult {
        return runCatching {
            context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            DeviceStateWriter.write(context)
            HermesSystemActionResult(success = true, action = action, message = successMessage)
        }.getOrElse { error ->
            HermesSystemActionResult(
                success = false,
                action = action,
                message = error.message ?: error.javaClass.simpleName,
            )
        }
    }

    private fun wifiIntent(): Intent {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            Intent(Settings.Panel.ACTION_INTERNET_CONNECTIVITY)
        } else {
            Intent(Settings.ACTION_WIFI_SETTINGS)
        }
    }

    private fun notificationSettingsIntent(context: Context): Intent {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
            }
        } else {
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}"))
        }
    }

    private fun activeNetworkLabel(connectivityManager: ConnectivityManager?): String {
        val capabilities = connectivityManager?.getNetworkCapabilities(connectivityManager.activeNetwork) ?: return "Offline"
        return when {
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "Wi-Fi"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "Cellular"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "Ethernet"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_BLUETOOTH) -> "Bluetooth"
            else -> "Connected"
        }
    }

    private fun statusToJson(status: HermesSystemStatus): JSONObject {
        return JSONObject().apply {
            put("wifi_enabled", status.wifiEnabled)
            put("active_network_label", status.activeNetworkLabel)
            put("bluetooth_supported", status.bluetoothSupported)
            put("bluetooth_enabled", status.bluetoothEnabled)
            put("bluetooth_permission_granted", status.bluetoothPermissionGranted)
            put("paired_bluetooth_devices", JSONArray(status.pairedBluetoothDevices))
            put("usb_host_supported", status.usbHostSupported)
            put("usb_device_count", status.usbDeviceCount)
            put("usb_devices", JSONArray(status.usbDevices))
            put("nfc_supported", status.nfcSupported)
            put("nfc_enabled", status.nfcEnabled)
            put("overlay_permission_granted", status.overlayPermissionGranted)
            put("notification_permission_granted", status.notificationPermissionGranted)
            put("background_persistence_enabled", status.backgroundPersistenceEnabled)
            put("runtime_service_running", status.runtimeServiceRunning)
            put("resizable_window_support", status.resizableWindowSupport)
            put("freeform_window_supported", status.freeformWindowSupported)
            put("available_system_actions", JSONArray(status.availableSystemActions))
        }
    }

    private fun actionToJson(result: HermesSystemActionResult): JSONObject {
        return JSONObject().apply {
            put("success", result.success)
            put("action", result.action)
            put("message", result.message)
        }
    }
}
