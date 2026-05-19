package com.nousresearch.hermesagent.device

import android.Manifest
import android.app.ActivityManager
import android.app.AppOpsManager
import android.app.usage.StorageStatsManager
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult as BleScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.location.LocationManager
import android.net.Uri
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.os.storage.StorageManager
import android.provider.Settings
import android.text.format.Formatter
import androidx.core.content.ContextCompat
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.pow

object HermesDeviceDiagnosticsBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        return when (action.lowercase(Locale.US).ifBlank { "status" }) {
            "status", "diagnostics_status", "device_diagnostics_status" -> statusJson(appContext).toString()
            "top_apps", "top_resource_apps", "top_memory_apps", "top_storage_apps", "resource_apps" ->
                topAppsJson(appContext, arguments).toString()
            "wifi_scan", "wifi_analyzer", "scan_wifi", "nearby_wifi", "wifi_signals" ->
                wifiScanJson(appContext, arguments).toString()
            "bluetooth_scan", "bluetooth_scanner", "nearby_bluetooth", "ble_scan", "bluetooth_signals" ->
                bluetoothScanJson(appContext, arguments).toString()
            "sensor_snapshot", "sensors", "sensor_status", "sample_sensors", "motion_sensors" ->
                sensorSnapshotJson(appContext, arguments).toString()
            "camera_status", "camera", "camera_capabilities" -> cameraStatusJson(appContext).toString()
            "radio_signal_status", "radio_scan", "am_fm_radio_status", "am_radio", "fm_radio" ->
                radioSignalStatusJson(appContext).toString()
            "signal_capability_status", "rf_capabilities", "radio_status", "microwave_status" ->
                signalCapabilityStatusJson(appContext).toString()
            "social_gmail_goal_preflight", "social_gmail_preflight", "phone_goal_preflight", "end_to_end_goal_preflight" ->
                socialGmailGoalPreflightJson(appContext).toString()
            "show_active_overlay", "show_working_overlay", "active_overlay" ->
                showActiveOverlayJson(appContext, arguments).toString()
            "tool_catalog", "tools", "list_tools", "available_tools", "capabilities" ->
                toolCatalogJson().toString()
            "open_usage_access_settings" -> openSettingsJson(appContext, Settings.ACTION_USAGE_ACCESS_SETTINGS, "Opened usage access settings").toString()
            "open_camera_permission_settings", "open_diagnostics_permission_settings" ->
                openAppSettingsJson(appContext, "Opened Hermes app permission settings").toString()
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported device diagnostics action: $action")
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
    }

    fun statusJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val sensorManager = appContext.getSystemService(SensorManager::class.java)
        return JSONObject()
            .put("success", true)
            .put("action", "status")
            .put("usage_access_granted", hasUsageStatsAccess(appContext))
            .put("camera_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY))
            .put("camera_permission_granted", hasPermission(appContext, Manifest.permission.CAMERA))
            .put("wifi_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI))
            .put("wifi_scan_permission_status", wifiPermissionStatusJson(appContext))
            .put("bluetooth_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH))
            .put("bluetooth_le_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE))
            .put("bluetooth_scan_permission_status", bluetoothPermissionStatusJson(appContext))
            .put("soc_profile", socProfileJson())
            .put("location_enabled", isLocationEnabled(appContext))
            .put("sensor_count", sensorManager?.getSensorList(Sensor.TYPE_ALL)?.size ?: 0)
            .put("available_sensor_types", JSONArray(sensorTypeCatalog(appContext)))
            .put("overlay", JSONObject(HermesOverlaySceneBridge.statusJson(appContext)))
            .put("available_actions", JSONArray(ACTIONS))
            .put(
                "cards",
                JSONArray()
                    .put(card("Diagnostics", "Phone resource, Wi-Fi, Bluetooth, camera, sensor, SOC, and overlay diagnostics are available to the agent."))
                    .put(card("Radio Limits", "AM/FM and broad RF scanning require vendor radio APIs or external SDR hardware; Android phones expose Wi-Fi, Bluetooth, audio, camera, and built-in sensors.")),
            )
    }

    fun topAppsJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val limit = arguments.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_LIMIT)
        val appContext = context.applicationContext
        val usageAccess = hasUsageStatsAccess(appContext)
        val memorySummary = memorySummaryJson(appContext)
        val topMemoryApps = runningAppMemoryJson(appContext, limit)
        val topStorageApps = if (usageAccess) {
            topStorageAppsJson(appContext, limit)
        } else {
            JSONArray()
        }
        return JSONObject()
            .put("success", true)
            .put("action", "top_apps")
            .put("limit", limit)
            .put("memory_summary", memorySummary)
            .put("top_memory_apps", topMemoryApps)
            .put("top_storage_apps", topStorageApps)
            .put("usage_access_granted", usageAccess)
            .put("requires_usage_access_for_full_storage_rankings", !usageAccess)
            .put("usage_access_settings_action", "open_usage_access_settings")
            .put(
                "notes",
                JSONArray()
                    .put("Android limits per-app memory visibility; running process memory may include only processes visible to this app.")
                    .put("Full per-app storage rankings require Usage Access and Android StorageStats permission gates."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(card("Memory", "${topMemoryApps.length()} visible running app/process rows ranked by PSS memory."))
                    .put(card("Storage", if (usageAccess) "${topStorageApps.length()} installed apps ranked by app/data/cache bytes." else "Usage Access is required before Hermes can rank app storage usage.")),
            )
    }

    fun wifiScanJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val limit = arguments.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_WIFI_RESULTS)
        val refresh = arguments.optBoolean("refresh", false)
        val wifiManager = appContext.getSystemService(WifiManager::class.java)
        val permissionStatus = wifiPermissionStatusJson(appContext)
        val canReadScan = permissionStatus.optBoolean("can_read_scan_results", false)
        if (wifiManager == null) {
            return JSONObject()
                .put("success", false)
                .put("action", "wifi_scan")
                .put("error", "Wi-Fi service is unavailable on this device")
                .put("wifi_scan_permission_status", permissionStatus)
        }
        if (!canReadScan) {
            return JSONObject()
                .put("success", false)
                .put("action", "wifi_scan")
                .put("error", "Wi-Fi scan results require nearby Wi-Fi/location permissions and location services on supported Android versions")
                .put("wifi_scan_permission_status", permissionStatus)
                .put("settings_actions", JSONArray().put("open_location_settings").put("open_app_settings"))
        }
        val refreshAccepted = if (refresh) runCatching {
            @Suppress("DEPRECATION")
            wifiManager.startScan()
        }.getOrDefault(false) else false
        val scanResults = runCatching { wifiManager.scanResults.orEmpty() }.getOrElse { error ->
            return JSONObject()
                .put("success", false)
                .put("action", "wifi_scan")
                .put("error", error.message ?: "Android denied Wi-Fi scan results")
                .put("wifi_scan_permission_status", permissionStatus)
        }
        val networks = JSONArray()
        scanResults
            .sortedWith(compareByDescending<ScanResult> { it.level }.thenBy { it.SSID.orEmpty() })
            .take(limit)
            .forEach { result -> networks.put(scanResultJson(result)) }
        return JSONObject()
            .put("success", true)
            .put("action", "wifi_scan")
            .put("refresh_requested", refresh)
            .put("refresh_accepted", refreshAccepted)
            .put("result_count", networks.length())
            .put("wifi_enabled", wifiManager.isWifiEnabled)
            .put("wifi_scan_permission_status", permissionStatus)
            .put("wifi_networks", networks)
            .put(
                "cards",
                JSONArray().put(
                    graphCard(
                        title = "Wi-Fi Analyzer",
                        body = "${networks.length()} nearby Wi-Fi signals ranked by RSSI dBm with channel/frequency/width metadata.",
                        graphType = "wifi_channel_strength",
                        rows = networks,
                    ),
                ),
            )
    }

    fun bluetoothScanJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val limit = arguments.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_BLUETOOTH_RESULTS)
        val refresh = arguments.optBoolean("refresh", false)
        val timeoutMs = arguments.optLong("timeout_ms", DEFAULT_BLUETOOTH_TIMEOUT_MS).coerceIn(500L, MAX_BLUETOOTH_TIMEOUT_MS)
        val bluetoothManager = appContext.getSystemService(BluetoothManager::class.java)
        val adapter = bluetoothManager?.adapter ?: runCatching { BluetoothAdapter.getDefaultAdapter() }.getOrNull()
        val permissionStatus = bluetoothPermissionStatusJson(appContext)
        if (adapter == null) {
            return JSONObject()
                .put("success", false)
                .put("action", "bluetooth_scan")
                .put("error", "Bluetooth service is unavailable on this device")
                .put("bluetooth_scan_permission_status", permissionStatus)
        }
        val rows = ConcurrentHashMap<String, JSONObject>()
        if (permissionStatus.optBoolean("can_read_paired_devices", false)) {
            runCatching {
                adapter.bondedDevices.orEmpty().forEach { device ->
                    rows[device.address.orEmpty().ifBlank { bluetoothDeviceName(device) }] = bluetoothDeviceJson(device)
                }
            }
        }
        var refreshAccepted = false
        var scanError: String? = null
        if (refresh && permissionStatus.optBoolean("can_scan_nearby_devices", false)) {
            val scanner = runCatching { adapter.bluetoothLeScanner }.getOrNull()
            if (scanner == null) {
                scanError = "Bluetooth LE scanner is unavailable or Bluetooth is disabled"
            } else {
                val failedCode = AtomicInteger(0)
                val latch = CountDownLatch(1)
                val callback = object : ScanCallback() {
                    override fun onScanResult(callbackType: Int, result: BleScanResult) {
                        rows[result.device.address.orEmpty().ifBlank { "ble_${rows.size}" }] = bleScanResultJson(result, callbackType)
                    }

                    override fun onBatchScanResults(results: MutableList<BleScanResult>) {
                        results.forEach { result ->
                            rows[result.device.address.orEmpty().ifBlank { "ble_${rows.size}" }] = bleScanResultJson(result, 0)
                        }
                    }

                    override fun onScanFailed(errorCode: Int) {
                        failedCode.set(errorCode)
                        latch.countDown()
                    }
                }
                val settings = ScanSettings.Builder()
                    .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                    .build()
                refreshAccepted = runCatching {
                    scanner.startScan(null, settings, callback)
                    latch.await(timeoutMs, TimeUnit.MILLISECONDS)
                    scanner.stopScan(callback)
                    failedCode.get() == 0
                }.getOrElse { error ->
                    scanError = error.message ?: error.javaClass.simpleName
                    runCatching { scanner.stopScan(callback) }
                    false
                }
                if (failedCode.get() != 0) {
                    scanError = "Bluetooth LE scan failed with code ${failedCode.get()}"
                }
            }
        }
        val devices = JSONArray()
        rows.values
            .sortedWith(compareByDescending<JSONObject> { it.optInt("rssi_dbm", Int.MIN_VALUE) }.thenBy { it.optString("device_name") })
            .take(limit)
            .forEachIndexed { index, row -> devices.put(row.put("rank", index + 1)) }
        return JSONObject()
            .put("success", true)
            .put("action", "bluetooth_scan")
            .put("refresh_requested", refresh)
            .put("refresh_accepted", refreshAccepted)
            .put("scan_error", scanError ?: JSONObject.NULL)
            .put("bluetooth_enabled", runCatching { adapter.isEnabled }.getOrDefault(false))
            .put("bluetooth_device_count", devices.length())
            .put("bluetooth_scan_permission_status", permissionStatus)
            .put("bluetooth_devices", devices)
            .put(
                "cards",
                JSONArray().put(
                    graphCard(
                        title = "Bluetooth Nearby",
                        body = "${devices.length()} paired or scanned Bluetooth device row(s), with BLE RSSI when Android exposes it.",
                        graphType = "bluetooth_rssi",
                        rows = devices,
                    ),
                ),
            )
    }

    fun sensorSnapshotJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val sensorManager = appContext.getSystemService(SensorManager::class.java)
            ?: return JSONObject()
                .put("success", false)
                .put("action", "sensor_snapshot")
                .put("error", "Sensor service is unavailable")
        val requested = requestedSensorTypes(arguments)
        val timeoutMs = arguments.optLong("timeout_ms", DEFAULT_SENSOR_TIMEOUT_MS).coerceIn(150L, MAX_SENSOR_TIMEOUT_MS)
        val targets = requested.mapNotNull { key ->
            val androidType = androidSensorType(key)
            val sensor = androidType?.let { sensorManager.getDefaultSensor(it) }
            if (androidType != null && sensor != null) key to sensor else null
        }
        val samples = sampleSensors(sensorManager, requested, targets, timeoutMs)
        val available = sensorTypeCatalog(appContext)
        return JSONObject()
            .put("success", true)
            .put("action", "sensor_snapshot")
            .put("requested_sensor_types", JSONArray(requested))
            .put("sample_timeout_ms", timeoutMs)
            .put("available_sensor_types", JSONArray(available))
            .put("sensor_samples", samples)
            .put("supported_watcher_types", JSONArray(SENSOR_TYPE_LABELS.keys))
            .put(
                "cards",
                JSONArray().put(
                    graphCard(
                        title = "Motion Sensors",
                        body = "${samples.length()} one-shot accelerometer, gyroscope, magnetic, light, or proximity rows captured for the agent.",
                        graphType = "sensor_vector",
                        rows = samples,
                    ),
                ),
            )
    }

    fun cameraStatusJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val cameraManager = appContext.getSystemService(CameraManager::class.java)
        val cameras = JSONArray()
        if (cameraManager != null) {
            runCatching {
                cameraManager.cameraIdList.forEach { id ->
                    val chars = cameraManager.getCameraCharacteristics(id)
                    cameras.put(cameraJson(id, chars))
                }
            }
        }
        return JSONObject()
            .put("success", true)
            .put("action", "camera_status")
            .put("camera_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY))
            .put("camera_permission_granted", hasPermission(appContext, Manifest.permission.CAMERA))
            .put("requires_camera_permission_for_capture", !hasPermission(appContext, Manifest.permission.CAMERA))
            .put("camera_count", cameras.length())
            .put("cameras", cameras)
            .put("notes", JSONArray().put("Hermes can inspect camera capabilities without capture; photo/video capture still requires the Camera permission and UI flow."))
            .put("cards", JSONArray().put(card("Camera", "${cameras.length()} camera device entries available.")))
    }

    fun radioSignalStatusJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val packageManager = appContext.packageManager
        val vendorBroadcastRadioDeclared = BROADCAST_RADIO_FEATURE_NAMES.any { feature ->
            runCatching { packageManager.hasSystemFeature(feature) }.getOrDefault(false)
        }
        val bands = JSONArray()
            .put(
                JSONObject()
                    .put("band", "AM broadcast")
                    .put("frequency_min_khz", 530)
                    .put("frequency_max_khz", 1700)
                    .put("supported", false)
                    .put("sampled", false)
                    .put("reason", "Android public app APIs do not expose AM tuner scan results."),
            )
            .put(
                JSONObject()
                    .put("band", "FM broadcast")
                    .put("frequency_min_mhz", 87.5)
                    .put("frequency_max_mhz", 108.0)
                    .put("supported", false)
                    .put("sampled", false)
                    .put("reason", "Android public app APIs do not expose FM tuner scan results on normal phones."),
            )
            .put(
                JSONObject()
                    .put("band", "External SDR")
                    .put("supported", false)
                    .put("sampled", false)
                    .put("requires_external_hardware", true)
                    .put("reason", "Attach an SDR or vendor radio bridge over USB/Bluetooth/Wi-Fi for broad RF, AM, FM, or microwave spectrum data."),
            )
        return JSONObject()
            .put("success", true)
            .put("action", "radio_signal_status")
            .put("am_fm_public_android_scan_supported", false)
            .put("vendor_broadcast_radio_feature_declared", vendorBroadcastRadioDeclared)
            .put("general_radio_spectrum_supported", false)
            .put("microwave_spectrum_supported", false)
            .put("requires_external_sdr_for_broad_rf", true)
            .put("radio_bands", bands)
            .put("radio_scan_rows", JSONArray())
            .put(
                "cards",
                JSONArray().put(
                    graphCard(
                        title = "AM/FM Radio",
                        body = "No public Android AM/FM scan feed is exposed here; Hermes can show the capability card and use external SDR/vendor hardware when available.",
                        graphType = "radio_frequency_capability",
                        rows = bands,
                    ),
                ),
            )
    }

    fun signalCapabilityStatusJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val sensorTypes = sensorTypeCatalog(appContext)
        val radioStatus = radioSignalStatusJson(appContext)
        return JSONObject()
            .put("success", true)
            .put("action", "signal_capability_status")
            .put("audio_frequency_analysis_supported", hasPermission(appContext, Manifest.permission.RECORD_AUDIO))
            .put("wifi_signal_analysis_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI))
            .put("bluetooth_signal_access_supported", appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH))
            .put("bluetooth_scan_permission_status", bluetoothPermissionStatusJson(appContext))
            .put("soc_profile", socProfileJson())
            .put("magnetic_field_sensor_supported", "magnetic_field" in sensorTypes)
            .put("motion_sensor_analysis_supported", JSONArray(sensorTypes.filter { it in setOf("accelerometer", "gyroscope", "gravity", "linear_acceleration", "rotation_vector") }))
            .put("am_fm_public_android_scan_supported", radioStatus.optBoolean("am_fm_public_android_scan_supported"))
            .put("general_radio_spectrum_supported", radioStatus.optBoolean("general_radio_spectrum_supported"))
            .put("microwave_spectrum_supported", radioStatus.optBoolean("microwave_spectrum_supported"))
            .put("requires_external_sdr_for_broad_rf", radioStatus.optBoolean("requires_external_sdr_for_broad_rf"))
            .put("radio_bands", radioStatus.optJSONArray("radio_bands") ?: JSONArray())
            .put(
                "limits",
                JSONArray()
                    .put("Phones do not expose a general microwave/RF spectrum analyzer through Android APIs.")
                    .put("Feasible built-in analysis: microphone audio spectrum, Wi-Fi RSSI/frequency/channel data, Bluetooth metadata, magnetometer, accelerometer, gyroscope, light, pressure, proximity, and camera.")
                    .put("Broadband radio or microwave analysis requires external SDR/specialized radio hardware connected over USB/Bluetooth/Wi-Fi and a matching driver/tool."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(card("Signal Limits", "Built-in phone APIs cover Wi-Fi, Bluetooth, audio, camera, and sensors; microwave and broad RF need external hardware."))
                    .put(card("SOC Compatibility", "Diagnostics use Android SDK services, not Adreno-only GPU paths, so MediaTek and Snapdragon phones follow the same capability checks.")),
            )
    }

    fun socialGmailGoalPreflightJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val packageManager = appContext.packageManager
        val tikTok = packageStatusJson(packageManager, "TikTok", TIKTOK_PACKAGES)
        val instagram = packageStatusJson(packageManager, "Instagram", listOf(INSTAGRAM_PACKAGE))
        val gmail = packageStatusJson(packageManager, "Gmail", listOf(GMAIL_PACKAGE))
        val preferredModel = preferredLocalModelJson(appContext)
        val deviceIdentity = deviceIdentityJson()
        val physicalPhoneDetected = !isLikelyEmulatorDevice()
        val blockers = JSONArray()
        if (!physicalPhoneDetected) {
            blockers.put("Current Android target appears to be an emulator or virtual device; the requested end-to-end goal requires a physical phone")
        }
        if (!tikTok.optBoolean("installed")) {
            blockers.put("TikTok is not installed or visible to Hermes (${TIKTOK_PACKAGES.joinToString()})")
        }
        if (!instagram.optBoolean("installed")) {
            blockers.put("Instagram is not installed or visible to Hermes ($INSTAGRAM_PACKAGE)")
        }
        if (!gmail.optBoolean("installed")) {
            blockers.put("Gmail is not installed or visible to Hermes ($GMAIL_PACKAGE)")
        }
        if (!HermesAccessibilityController.isServiceEnabled(appContext)) {
            blockers.put("Hermes accessibility service is not enabled; UI snapshots, typing, scrolling, and send/post taps need it")
        } else if (!HermesAccessibilityController.isServiceConnected()) {
            blockers.put("Hermes accessibility service is enabled but not connected yet; open Hermes or toggle the service before UI actions")
        }
        if (!preferredModel.optBoolean("ready")) {
            blockers.put("No preferred local model file is ready; use the local model import button or download a model first")
        }
        return JSONObject()
            .put("success", true)
            .put("action", "social_gmail_goal_preflight")
            .put("objective", "TikTok video comment, TikTok hello DM, Instagram hello DM, Gmail latest-email summary, and Gmail self-email to adybag14@gmail.com using local Gemma/Qwen on the physical phone")
            .put("physical_phone_required", true)
            .put("physical_phone_detected", physicalPhoneDetected)
            .put("android_device_identity", deviceIdentity)
            .put("tiktok", tikTok)
            .put("instagram", instagram)
            .put("gmail", gmail)
            .put("accessibility_service_enabled", HermesAccessibilityController.isServiceEnabled(appContext))
            .put("accessibility_connected", HermesAccessibilityController.isServiceConnected())
            .put("foreground_package_name", HermesAccessibilityController.currentForegroundPackageName())
            .put("preferred_local_model", preferredModel)
            .put("local_model_import_button_supported", true)
            .put("local_model_import_entry", "Settings > Hugging Face local model downloads > Import model from phone files")
            .put("local_model_import_uses_android_open_document", true)
            .put("ready_for_social_ui_actions", physicalPhoneDetected && tikTok.optBoolean("installed") && instagram.optBoolean("installed") && HermesAccessibilityController.isServiceConnected())
            .put("ready_for_gmail_draft", gmail.optBoolean("installed"))
            .put("ready_for_gmail_latest_email_summary", physicalPhoneDetected && gmail.optBoolean("installed") && HermesAccessibilityController.isServiceConnected())
            .put("ready_for_local_model_run", preferredModel.optBoolean("ready"))
            .put("ready_for_full_goal", blockers.length() == 0)
            .put("blocking_items", blockers)
            .put(
                "gmail_latest_email_summary_strategy",
                JSONObject()
                    .put("method", "Open Gmail, use android_ui_tool snapshots to inspect the inbox/latest email UI, then summarize visible sender/subject/body text.")
                    .put("direct_mailbox_read_supported", false)
                    .put("requires_gmail_installed", true)
                    .put("requires_accessibility_snapshot", true)
                    .put("ready", physicalPhoneDetected && gmail.optBoolean("installed") && HermesAccessibilityController.isServiceConnected()),
            )
            .put("external_send_requires_visible_confirmation", true)
            .put(
                "external_send_safety_checks",
                JSONArray()
                    .put("Before posting the TikTok comment, confirm the current visible video/comment composer is the intended target.")
                    .put("Before each DM send, confirm the current visible TikTok or Instagram conversation recipient is the intended target.")
                    .put("Before tapping Gmail Send, confirm the composer is addressed to adybag14@gmail.com with the intended subject/body."),
            )
            .put(
                "suggested_phone_sequence",
                JSONArray()
                    .put("Run this preflight after the physical phone is attached.")
                    .put("Do not treat emulator-only app/model results as completion for this goal.")
                    .put("If local_model_import_button_supported is true but preferred_local_model.ready is false, use the import button to import a Gemma LiteRT-LM or Qwen GGUF model and mark it preferred.")
                    .put("For the latest Gmail summary, open the latest message in Gmail and summarize only text verified in the current UI snapshot.")
                    .put("Use android_ui_tool snapshots before every external send/post tap.")
                    .put("Use android_device_diagnostics_tool action=social_gmail_goal_preflight again after permissions, installs, or model changes."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(card("Phone Goal Preflight", if (blockers.length() == 0) "All required package/model/accessibility checks are ready." else "${blockers.length()} blocker(s) remain before the full phone run."))
                    .put(card("Send Safety", "External posts, DMs, and email sends require a fresh UI snapshot confirming target and content immediately before tapping send.")),
            )
    }

    private fun deviceIdentityJson(): JSONObject {
        return JSONObject()
            .put("manufacturer", Build.MANUFACTURER.orEmpty())
            .put("brand", Build.BRAND.orEmpty())
            .put("model", Build.MODEL.orEmpty())
            .put("device", Build.DEVICE.orEmpty())
            .put("product", Build.PRODUCT.orEmpty())
            .put("hardware", Build.HARDWARE.orEmpty())
            .put("fingerprint", Build.FINGERPRINT.orEmpty())
            .put("soc_profile", socProfileJson())
            .put("likely_emulator", isLikelyEmulatorDevice())
    }

    private fun showActiveOverlayJson(context: Context, arguments: JSONObject): JSONObject {
        val message = arguments.optString("message")
            .ifBlank { "Hermes is active and working on the current task." }
        val payload = JSONObject()
            .put("scene_id", arguments.optString("scene_id").ifBlank { "hermes-active-status" })
            .put("scene_title", arguments.optString("title").ifBlank { "Hermes Active" })
            .put("scene_text", message)
            .put("scene_button_text", arguments.optString("button_text").ifBlank { "Dismiss" })
            .put("position", arguments.optString("position").ifBlank { "top" })
            .put("width", arguments.optString("width").ifBlank { "92%" })
            .put("hide_after_ms", arguments.optLong("hide_after_ms", 0L))
        return JSONObject(HermesOverlaySceneBridge.showSceneJson(context, payload))
            .put("action", "show_active_overlay")
            .put("overlay_payload", payload)
    }

    private fun toolCatalogJson(): JSONObject {
        return JSONObject()
            .put("success", true)
            .put("action", "tool_catalog")
            .put(
                "native_tools",
                JSONArray()
                    .put(toolJson("terminal_tool", "Run short Android shell commands inside the Hermes workspace.", "command"))
                    .put(toolJson("file_write_tool", "Write UTF-8 text files inside the Hermes workspace.", "path, content, append"))
                    .put(toolJson("android_system_tool", "Read phone state and open settings or user-granted Shizuku/Sui actions.", "action, package_name, permission"))
                    .put(toolJson("android_ui_tool", "Inspect and control visible Android UI through accessibility and screenshots.", "action, selectors, coordinates"))
                    .put(toolJson("android_automation_tool", "Run/open/create saved automations, watcher tasks, overlays, notifications, widgets, and Tasker-style triggers.", "action, trigger, data_uri"))
                    .put(toolJson("android_device_diagnostics_tool", "Inspect resource-heavy apps, Wi-Fi signals, Bluetooth nearby devices, camera, sensors, SOC compatibility, overlay, radio/RF capability limits, and the social/Gmail end-to-end phone preflight.", "action, limit, refresh, sensor_types, timeout_ms"))
                    .put(toolJson("hindsight_memory_tool", "Retain, recall, and reflect local Hindsight-style memories with tags, entities, keywords, recency, and reinforcement.", "action, content, query, tags, category")),
            )
            .put("diagnostics_actions", JSONArray(ACTIONS))
            .put(
                "hindsight_memory_translation",
                JSONObject()
                    .put("retain", "Promote facts from chats/tool results into structured memories with source, category, timestamp, and reinforcement count.")
                    .put("recall", "Retrieve memories by semantic/keyword/time/entity signals before answering.")
                    .put("reflect", "Periodically consolidate repeated facts into fresher summaries and keep raw evidence links."),
            )
            .put("cards", JSONArray().put(card("Tools", "Hermes can inspect its tool catalog and pick the right native tool before acting.")))
    }

    private fun memorySummaryJson(context: Context): JSONObject {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val info = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(info)
        return JSONObject()
            .put("available_bytes", info.availMem)
            .put("available_label", formatBytes(context, info.availMem))
            .put("total_bytes", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) info.totalMem else 0L)
            .put("total_label", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) formatBytes(context, info.totalMem) else "unknown")
            .put("low_memory", info.lowMemory)
            .put("threshold_bytes", info.threshold)
            .put("threshold_label", formatBytes(context, info.threshold))
            .put("app_data_free_bytes", context.filesDir.freeSpace)
            .put("app_data_total_bytes", context.filesDir.totalSpace)
            .put("app_data_free_label", formatBytes(context, context.filesDir.freeSpace))
            .put("app_data_total_label", formatBytes(context, context.filesDir.totalSpace))
    }

    private fun runningAppMemoryJson(context: Context, limit: Int): JSONArray {
        val activityManager = context.getSystemService(ActivityManager::class.java) ?: return JSONArray()
        val running = activityManager.runningAppProcesses.orEmpty()
        if (running.isEmpty()) return JSONArray()
        val memoryInfos = runCatching { activityManager.getProcessMemoryInfo(running.map { it.pid }.toIntArray()) }
            .getOrNull()
            .orEmpty()
        val rows = running.mapIndexedNotNull { index, process ->
            val pssBytes = memoryInfos.getOrNull(index)?.totalPss?.toLong()?.times(1024L) ?: return@mapIndexedNotNull null
            val packageName = process.pkgList?.firstOrNull().orEmpty().ifBlank { process.processName }
            AppResourceRow(
                packageName = packageName,
                label = appLabel(context.packageManager, packageName),
                metricBytes = pssBytes,
                extra = JSONObject()
                    .put("process_name", process.processName)
                    .put("pid", process.pid)
                    .put("importance", process.importance),
            )
        }
        return rows
            .sortedByDescending { it.metricBytes }
            .take(limit)
            .toJsonArray(context, "memory_bytes", "memory_label")
    }

    private fun topStorageAppsJson(context: Context, limit: Int): JSONArray {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return JSONArray()
        val storageStatsManager = context.getSystemService(StorageStatsManager::class.java) ?: return JSONArray()
        val packageManager = context.packageManager
        val rows = mutableListOf<AppResourceRow>()
        installedApplications(packageManager).forEach { app ->
            val stats = runCatching {
                storageStatsManager.queryStatsForPackage(
                    app.storageUuid ?: StorageManager.UUID_DEFAULT,
                    app.packageName,
                    Process.myUserHandle(),
                )
            }.getOrNull() ?: return@forEach
            val total = stats.appBytes + stats.dataBytes + stats.cacheBytes
            if (total > 0L) {
                rows += AppResourceRow(
                    packageName = app.packageName,
                    label = appLabel(packageManager, app.packageName),
                    metricBytes = total,
                    extra = JSONObject()
                        .put("app_bytes", stats.appBytes)
                        .put("data_bytes", stats.dataBytes)
                        .put("cache_bytes", stats.cacheBytes)
                        .put("app_label", formatBytes(context, stats.appBytes))
                        .put("data_label", formatBytes(context, stats.dataBytes))
                        .put("cache_label", formatBytes(context, stats.cacheBytes)),
                )
            }
        }
        return rows
            .sortedByDescending { it.metricBytes }
            .take(limit)
            .toJsonArray(context, "storage_bytes", "storage_label")
    }

    private fun sampleSensors(
        sensorManager: SensorManager,
        requested: List<String>,
        targets: List<Pair<String, Sensor>>,
        timeoutMs: Long,
    ): JSONArray {
        val samples = JSONArray()
        if (targets.isEmpty()) {
            requested.forEach { key ->
                samples.put(
                    JSONObject()
                        .put("sensor_type", key)
                        .put("available", false)
                        .put("sampled", false),
                )
            }
            return samples
        }
        val readings = ConcurrentHashMap<Int, JSONObject>()
        val latch = CountDownLatch(targets.size)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                if (readings.containsKey(event.sensor.type)) return
                val values = JSONArray()
                event.values.forEach { values.put(it.toDouble()) }
                readings[event.sensor.type] = JSONObject()
                    .put("sensor_type", canonicalSensorType(event.sensor))
                    .put("sensor_name", event.sensor.name.orEmpty())
                    .put("vendor", event.sensor.vendor.orEmpty())
                    .put("values", values)
                    .put("unit", unitForSensorType(canonicalSensorType(event.sensor)))
                    .put("timestamp_nanos", event.timestamp)
                    .put("accuracy", event.accuracy)
                latch.countDown()
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        val thread = HandlerThread("HermesSensorSnapshot")
        thread.start()
        val handler = Handler(thread.looper)
        try {
            targets.forEach { (_, sensor) ->
                sensorManager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_NORMAL, handler)
            }
            latch.await(timeoutMs, TimeUnit.MILLISECONDS)
        } finally {
            sensorManager.unregisterListener(listener)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
                thread.quitSafely()
            } else {
                thread.quit()
            }
        }
        targets.forEach { (key, sensor) ->
            samples.put(
                readings[sensor.type] ?: JSONObject()
                    .put("sensor_type", key)
                    .put("sensor_name", sensor.name.orEmpty())
                    .put("vendor", sensor.vendor.orEmpty())
                    .put("available", true)
                    .put("sampled", false),
            )
        }
        val sampledKeys = targets.map { it.first }.toSet()
        requested.filterNot { it in sampledKeys }.forEach { key ->
            samples.put(
                JSONObject()
                    .put("sensor_type", key)
                    .put("available", false)
                    .put("sampled", false),
            )
        }
        return samples
    }

    private fun scanResultJson(result: ScanResult): JSONObject {
        val json = JSONObject()
            .put("ssid", result.SSID.orEmpty().ifBlank { "<hidden>" })
            .put("bssid", result.BSSID.orEmpty())
            .put("rssi_dbm", result.level)
            .put("frequency_mhz", result.frequency)
            .put("channel", channelForFrequencyMhz(result.frequency) ?: JSONObject.NULL)
            .put("band", wifiBandLabel(result.frequency))
            .put("estimated_distance_meters", estimateWifiDistanceMeters(result.level, result.frequency))
            .put("capabilities", result.capabilities.orEmpty())
            .put("timestamp_micros", result.timestamp)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            json.put("channel_width", channelWidthLabel(result.channelWidth))
            json.put("center_freq0_mhz", result.centerFreq0)
            json.put("center_freq1_mhz", result.centerFreq1)
            json.put("passpoint_network", result.isPasspointNetwork)
            json.put("80211mc_responder", result.is80211mcResponder)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            json.put("wifi_standard", wifiStandardLabel(result.wifiStandard))
        }
        return json
    }

    private fun bluetoothDeviceJson(device: BluetoothDevice): JSONObject {
        return JSONObject()
            .put("device_name", bluetoothDeviceName(device))
            .put("address", runCatching { device.address.orEmpty() }.getOrDefault(""))
            .put("device_type", bluetoothDeviceTypeLabel(runCatching { device.type }.getOrDefault(BluetoothDevice.DEVICE_TYPE_UNKNOWN)))
            .put("bond_state", bluetoothBondStateLabel(runCatching { device.bondState }.getOrDefault(BluetoothDevice.BOND_NONE)))
            .put("paired", runCatching { device.bondState == BluetoothDevice.BOND_BONDED }.getOrDefault(false))
            .put("connectable", true)
    }

    private fun bleScanResultJson(result: BleScanResult, callbackType: Int): JSONObject {
        return bluetoothDeviceJson(result.device)
            .put("rssi_dbm", result.rssi)
            .put("callback_type", callbackType)
            .put("connectable", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.isConnectable else JSONObject.NULL)
            .put("tx_power_dbm", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.txPower else JSONObject.NULL)
            .put("timestamp_nanos", result.timestampNanos)
            .put("scan_record_bytes", result.scanRecord?.bytes?.size ?: 0)
    }

    private fun bluetoothDeviceName(device: BluetoothDevice): String {
        return runCatching { device.name?.takeIf { it.isNotBlank() } }.getOrNull() ?: "<unnamed>"
    }

    private fun bluetoothDeviceTypeLabel(value: Int): String = when (value) {
        BluetoothDevice.DEVICE_TYPE_CLASSIC -> "classic"
        BluetoothDevice.DEVICE_TYPE_LE -> "le"
        BluetoothDevice.DEVICE_TYPE_DUAL -> "dual"
        else -> "unknown"
    }

    private fun bluetoothBondStateLabel(value: Int): String = when (value) {
        BluetoothDevice.BOND_BONDED -> "bonded"
        BluetoothDevice.BOND_BONDING -> "bonding"
        BluetoothDevice.BOND_NONE -> "none"
        else -> "unknown"
    }

    private fun cameraJson(id: String, chars: CameraCharacteristics): JSONObject {
        val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            ?.map(::cameraCapabilityLabel)
            .orEmpty()
        return JSONObject()
            .put("id", id)
            .put("lens_facing", lensFacingLabel(chars.get(CameraCharacteristics.LENS_FACING)))
            .put("flash_available", chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true)
            .put("sensor_orientation", chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: JSONObject.NULL)
            .put("capabilities", JSONArray(caps))
    }

    private fun wifiPermissionStatusJson(context: Context): JSONObject {
        val fineLocationGranted = hasPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
        val coarseLocationGranted = hasPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION)
        val nearbyWifiGranted = Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            hasPermission(context, Manifest.permission.NEARBY_WIFI_DEVICES)
        val locationEnabled = isLocationEnabled(context)
        val canRead = nearbyWifiGranted && fineLocationGranted && locationEnabled
        return JSONObject()
            .put("fine_location_granted", fineLocationGranted)
            .put("coarse_location_granted", coarseLocationGranted)
            .put("nearby_wifi_devices_granted", nearbyWifiGranted)
            .put("location_enabled", locationEnabled)
            .put("requires_fine_location", !fineLocationGranted)
            .put("requires_nearby_wifi_devices", Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && !nearbyWifiGranted)
            .put("can_read_scan_results", canRead)
    }

    private fun bluetoothPermissionStatusJson(context: Context): JSONObject {
        val bluetoothConnectGranted = Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
            hasPermission(context, Manifest.permission.BLUETOOTH_CONNECT)
        val bluetoothScanGranted = Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
            hasPermission(context, Manifest.permission.BLUETOOTH_SCAN)
        val fineLocationGranted = hasPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
        val locationEnabled = isLocationEnabled(context)
        val legacyScanReady = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S || (fineLocationGranted && locationEnabled)
        return JSONObject()
            .put("bluetooth_connect_granted", bluetoothConnectGranted)
            .put("bluetooth_scan_granted", bluetoothScanGranted)
            .put("fine_location_granted", fineLocationGranted)
            .put("location_enabled", locationEnabled)
            .put("requires_bluetooth_connect", Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !bluetoothConnectGranted)
            .put("requires_bluetooth_scan", Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !bluetoothScanGranted)
            .put("requires_location_for_legacy_scan", Build.VERSION.SDK_INT < Build.VERSION_CODES.S && !legacyScanReady)
            .put("can_read_paired_devices", bluetoothConnectGranted)
            .put("can_scan_nearby_devices", bluetoothScanGranted && legacyScanReady)
    }

    private fun hasUsageStatsAccess(context: Context): Boolean {
        val appOps = context.getSystemService(AppOpsManager::class.java) ?: return false
        val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            appOps.unsafeCheckOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)
        } else {
            @Suppress("DEPRECATION")
            appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)
        }
        return mode == AppOpsManager.MODE_ALLOWED
    }

    private fun openSettingsJson(context: Context, action: String, message: String): JSONObject {
        return runCatching {
            context.startActivity(Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            JSONObject()
                .put("success", true)
                .put("external_activity_handoff", true)
                .put("action", action)
                .put("message", message)
        }.getOrElse { error ->
            JSONObject()
                .put("success", false)
                .put("action", action)
                .put("error", error.message ?: "Unable to open Android settings")
        }
    }

    private fun openAppSettingsJson(context: Context, message: String): JSONObject {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}"))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return runCatching {
            context.startActivity(intent)
            JSONObject()
                .put("success", true)
                .put("external_activity_handoff", true)
                .put("action", "open_app_settings")
                .put("message", message)
        }.getOrElse { error ->
            JSONObject()
                .put("success", false)
                .put("action", "open_app_settings")
                .put("error", error.message ?: "Unable to open Hermes app settings")
        }
    }

    private fun requestedSensorTypes(arguments: JSONObject): List<String> {
        val raw = arguments.opt("sensor_types") ?: arguments.opt("sensors") ?: arguments.opt("sensor_type")
        val values = when (raw) {
            is JSONArray -> buildList {
                for (index in 0 until raw.length()) {
                    raw.optString(index).takeIf { it.isNotBlank() }?.let(::add)
                }
            }
            is String -> raw.split(',', ' ', ';').filter { it.isNotBlank() }
            else -> DEFAULT_SENSOR_TYPES
        }
        return values
            .map { canonicalSensorType(it) }
            .filter { it in SENSOR_TYPE_LABELS }
            .distinct()
            .ifEmpty { DEFAULT_SENSOR_TYPES }
            .take(MAX_SENSOR_TYPES_PER_SAMPLE)
    }

    private fun sensorTypeCatalog(context: Context): List<String> {
        val sensorManager = context.getSystemService(SensorManager::class.java) ?: return emptyList()
        return SENSOR_TYPE_LABELS.keys.filter { key ->
            androidSensorType(key)?.let { sensorManager.getDefaultSensor(it) != null } == true
        }
    }

    internal fun channelForFrequencyMhz(frequencyMhz: Int): Int? {
        return when {
            frequencyMhz == 2484 -> 14
            frequencyMhz in 2412..2472 -> (frequencyMhz - 2407) / 5
            frequencyMhz in 5160..5885 -> (frequencyMhz - 5000) / 5
            frequencyMhz in 5955..7115 -> (frequencyMhz - 5950) / 5
            else -> null
        }
    }

    internal fun isLikelyEmulatorDevice(
        fingerprint: String = Build.FINGERPRINT.orEmpty(),
        model: String = Build.MODEL.orEmpty(),
        manufacturer: String = Build.MANUFACTURER.orEmpty(),
        brand: String = Build.BRAND.orEmpty(),
        device: String = Build.DEVICE.orEmpty(),
        product: String = Build.PRODUCT.orEmpty(),
        hardware: String = Build.HARDWARE.orEmpty(),
    ): Boolean {
        val values = listOf(fingerprint, model, manufacturer, brand, device, product, hardware)
            .map { it.lowercase(Locale.US) }
        return values.any { value ->
            value.contains("generic") ||
                value.contains("emulator") ||
                value.contains("sdk_gphone") ||
                value.contains("goldfish") ||
                value.contains("ranchu") ||
                value.contains("robolectric")
        }
    }

    private fun androidSensorType(sensorType: String): Int? {
        return when (canonicalSensorType(sensorType)) {
            "accelerometer" -> Sensor.TYPE_ACCELEROMETER
            "gyroscope" -> Sensor.TYPE_GYROSCOPE
            "magnetic_field" -> Sensor.TYPE_MAGNETIC_FIELD
            "light" -> Sensor.TYPE_LIGHT
            "proximity" -> Sensor.TYPE_PROXIMITY
            "gravity" -> Sensor.TYPE_GRAVITY
            "linear_acceleration" -> Sensor.TYPE_LINEAR_ACCELERATION
            "rotation_vector" -> Sensor.TYPE_ROTATION_VECTOR
            "step_counter" -> Sensor.TYPE_STEP_COUNTER
            "pressure" -> Sensor.TYPE_PRESSURE
            "ambient_temperature" -> Sensor.TYPE_AMBIENT_TEMPERATURE
            "relative_humidity" -> Sensor.TYPE_RELATIVE_HUMIDITY
            else -> null
        }
    }

    private fun canonicalSensorType(sensor: Sensor): String = when (sensor.type) {
        Sensor.TYPE_ACCELEROMETER -> "accelerometer"
        Sensor.TYPE_GYROSCOPE -> "gyroscope"
        Sensor.TYPE_MAGNETIC_FIELD -> "magnetic_field"
        Sensor.TYPE_LIGHT -> "light"
        Sensor.TYPE_PROXIMITY -> "proximity"
        Sensor.TYPE_GRAVITY -> "gravity"
        Sensor.TYPE_LINEAR_ACCELERATION -> "linear_acceleration"
        Sensor.TYPE_ROTATION_VECTOR -> "rotation_vector"
        Sensor.TYPE_STEP_COUNTER -> "step_counter"
        Sensor.TYPE_PRESSURE -> "pressure"
        Sensor.TYPE_AMBIENT_TEMPERATURE -> "ambient_temperature"
        Sensor.TYPE_RELATIVE_HUMIDITY -> "relative_humidity"
        else -> "type_${sensor.type}"
    }

    private fun canonicalSensorType(value: String): String {
        return value.trim().lowercase(Locale.US)
            .replace('-', '_')
            .replace(' ', '_')
            .let { normalized ->
                when (normalized) {
                    "gyro", "gyrometer" -> "gyroscope"
                    "magnetometer", "compass" -> "magnetic_field"
                    "linear_accel" -> "linear_acceleration"
                    "rotation" -> "rotation_vector"
                    "temperature" -> "ambient_temperature"
                    "humidity" -> "relative_humidity"
                    else -> normalized
                }
            }
    }

    private fun unitForSensorType(sensorType: String): String = when (sensorType) {
        "accelerometer", "gravity", "linear_acceleration" -> "m/s^2"
        "gyroscope" -> "rad/s"
        "magnetic_field" -> "uT"
        "light" -> "lx"
        "proximity" -> "cm"
        "rotation_vector" -> "unitless"
        "step_counter" -> "steps"
        "pressure" -> "hPa"
        "ambient_temperature" -> "C"
        "relative_humidity" -> "%"
        else -> ""
    }

    private fun isLocationEnabled(context: Context): Boolean {
        val locationManager = context.getSystemService(LocationManager::class.java) ?: return false
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            locationManager.isLocationEnabled
        } else {
            @Suppress("DEPRECATION")
            Settings.Secure.getInt(context.contentResolver, Settings.Secure.LOCATION_MODE, Settings.Secure.LOCATION_MODE_OFF) != Settings.Secure.LOCATION_MODE_OFF
        }
    }

    private fun installedApplications(packageManager: PackageManager): List<ApplicationInfo> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            packageManager.getInstalledApplications(PackageManager.ApplicationInfoFlags.of(0))
        } else {
            @Suppress("DEPRECATION")
            packageManager.getInstalledApplications(0)
        }
    }

    private fun getApplicationInfo(packageManager: PackageManager, packageName: String): ApplicationInfo? {
        return runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                packageManager.getApplicationInfo(packageName, PackageManager.ApplicationInfoFlags.of(0))
            } else {
                @Suppress("DEPRECATION")
                packageManager.getApplicationInfo(packageName, 0)
            }
        }.getOrNull()
    }

    private fun appLabel(packageManager: PackageManager, packageName: String): String {
        val info = getApplicationInfo(packageManager, packageName) ?: return packageName
        return runCatching { packageManager.getApplicationLabel(info).toString() }.getOrDefault(packageName)
    }

    private fun packageStatusJson(packageManager: PackageManager, appName: String, packageNames: List<String>): JSONObject {
        val installed = packageNames.mapNotNull { packageName ->
            val info = getApplicationInfo(packageManager, packageName) ?: return@mapNotNull null
            JSONObject()
                .put("package_name", packageName)
                .put("label", runCatching { packageManager.getApplicationLabel(info).toString() }.getOrDefault(appName))
                .put("enabled", info.enabled)
        }
        return JSONObject()
            .put("app_name", appName)
            .put("installed", installed.isNotEmpty())
            .put("package_name", installed.firstOrNull()?.optString("package_name").orEmpty())
            .put("candidate_package_names", JSONArray(packageNames))
            .put("installed_candidates", JSONArray(installed))
    }

    private fun preferredLocalModelJson(context: Context): JSONObject {
        val store = LocalModelDownloadStore(context)
        val preferredId = store.preferredDownloadId()
        val preferred = store.loadDownloads().firstOrNull { it.id == preferredId }
        val file = preferred?.destinationPath?.takeIf { it.isNotBlank() }?.let { File(it) }
        return JSONObject()
            .put("preferred_download_id", preferredId)
            .put("ready", preferred != null && file?.isFile == true)
            .put("title", preferred?.title.orEmpty())
            .put("runtime_flavor", preferred?.runtimeFlavor.orEmpty())
            .put("destination_file_name", preferred?.destinationFileName.orEmpty())
            .put("destination_path", preferred?.destinationPath.orEmpty())
            .put("file_exists", file?.isFile == true)
            .put("file_bytes", if (file?.isFile == true) file.length() else 0L)
            .put("record_status", preferred?.status.orEmpty())
            .put("record_status_message", preferred?.statusMessage.orEmpty())
            .put("download_record_count", store.loadDownloads().size)
    }

    private fun hasPermission(context: Context, permission: String): Boolean {
        return ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun formatBytes(context: Context, bytes: Long): String = Formatter.formatShortFileSize(context, bytes.coerceAtLeast(0L))

    private fun List<AppResourceRow>.toJsonArray(context: Context, byteKey: String, labelKey: String): JSONArray {
        val array = JSONArray()
        forEachIndexed { index, row ->
            val json = JSONObject()
                .put("rank", index + 1)
                .put("package_name", row.packageName)
                .put("label", row.label)
                .put(byteKey, row.metricBytes)
                .put(labelKey, formatBytes(context, row.metricBytes))
            row.extra.keys().forEach { key -> json.put(key, row.extra.opt(key)) }
            array.put(json)
        }
        return array
    }

    private fun card(title: String, body: String): JSONObject {
        return JSONObject()
            .put("type", "diagnostic_card")
            .put("title", title)
            .put("body", body)
    }

    private fun graphCard(title: String, body: String, graphType: String, rows: JSONArray): JSONObject {
        return card(title, body)
            .put("type", "signal_graph_card")
            .put("graph_type", graphType)
            .put("row_count", rows.length())
            .put("rows", rows)
    }

    private fun toolJson(name: String, description: String, arguments: String): JSONObject {
        return JSONObject()
            .put("name", name)
            .put("description", description)
            .put("arguments", arguments)
    }

    private fun socProfileJson(): JSONObject {
        val socManufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MANUFACTURER.orEmpty() else ""
        val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL.orEmpty() else ""
        val values = listOf(
            socManufacturer,
            socModel,
            Build.HARDWARE.orEmpty(),
            Build.BOARD.orEmpty(),
            Build.DEVICE.orEmpty(),
            Build.PRODUCT.orEmpty(),
            Build.MANUFACTURER.orEmpty(),
            Build.BRAND.orEmpty(),
        )
        return JSONObject()
            .put("soc_manufacturer", socManufacturer)
            .put("soc_model", socModel)
            .put("hardware", Build.HARDWARE.orEmpty())
            .put("board", Build.BOARD.orEmpty())
            .put("likely_mediatek", isLikelyMediatekSoc(values))
            .put("likely_snapdragon", isLikelySnapdragonSoc(values))
            .put("compatibility_strategy", "Use Android SDK feature, permission, sensor, Wi-Fi, Bluetooth, camera, and storage APIs; avoid Adreno-only or Snapdragon-only assumptions.")
    }

    internal fun isLikelyMediatekSoc(values: List<String>): Boolean {
        val normalized = values.joinToString(" ").lowercase(Locale.US)
        return listOf("mediatek", "mtk", "dimensity", "helio").any { it in normalized } ||
            Regex("""\bmt[0-9]{4,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    internal fun isLikelySnapdragonSoc(values: List<String>): Boolean {
        val normalized = values.joinToString(" ").lowercase(Locale.US)
        return listOf("qualcomm", "snapdragon", "qcom", "msm", "sdm").any { it in normalized } ||
            Regex("""\bsm[0-9]{4,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    private fun wifiBandLabel(frequencyMhz: Int): String = when (frequencyMhz) {
        in 2400..2500 -> "2.4GHz"
        in 4900..5900 -> "5GHz"
        in 5925..7125 -> "6GHz"
        in 57000..71000 -> "60GHz"
        else -> "unknown"
    }

    private fun estimateWifiDistanceMeters(rssiDbm: Int, frequencyMhz: Int): Double {
        if (frequencyMhz <= 0 || rssiDbm >= 0) return 0.0
        val exponent = (27.55 - (20.0 * log10(frequencyMhz.toDouble())) + abs(rssiDbm).toDouble()) / 20.0
        return (10.0.pow(exponent) * 100.0).toInt() / 100.0
    }

    private fun channelWidthLabel(width: Int): String = when (width) {
        ScanResult.CHANNEL_WIDTH_20MHZ -> "20MHz"
        ScanResult.CHANNEL_WIDTH_40MHZ -> "40MHz"
        ScanResult.CHANNEL_WIDTH_80MHZ -> "80MHz"
        ScanResult.CHANNEL_WIDTH_160MHZ -> "160MHz"
        ScanResult.CHANNEL_WIDTH_80MHZ_PLUS_MHZ -> "80+80MHz"
        else -> "unknown"
    }

    private fun wifiStandardLabel(standard: Int): String = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        when (standard) {
            ScanResult.WIFI_STANDARD_LEGACY -> "legacy"
            ScanResult.WIFI_STANDARD_11N -> "802.11n"
            ScanResult.WIFI_STANDARD_11AC -> "802.11ac"
            ScanResult.WIFI_STANDARD_11AX -> "802.11ax"
            ScanResult.WIFI_STANDARD_11AD -> "802.11ad"
            ScanResult.WIFI_STANDARD_11BE -> "802.11be"
            else -> "unknown"
        }
    } else {
        "unknown"
    }

    private fun lensFacingLabel(value: Int?): String = when (value) {
        CameraCharacteristics.LENS_FACING_FRONT -> "front"
        CameraCharacteristics.LENS_FACING_BACK -> "back"
        CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
        else -> "unknown"
    }

    private fun cameraCapabilityLabel(value: Int): String = when (value) {
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE -> "backward_compatible"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR -> "manual_sensor"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING -> "manual_post_processing"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW -> "raw"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_PRIVATE_REPROCESSING -> "private_reprocessing"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_READ_SENSOR_SETTINGS -> "read_sensor_settings"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BURST_CAPTURE -> "burst_capture"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_YUV_REPROCESSING -> "yuv_reprocessing"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT -> "depth_output"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA -> "logical_multi_camera"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MONOCHROME -> "monochrome"
        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_SECURE_IMAGE_DATA -> "secure_image_data"
        else -> "capability_$value"
    }

    private data class AppResourceRow(
        val packageName: String,
        val label: String,
        val metricBytes: Long,
        val extra: JSONObject = JSONObject(),
    )

    private val ACTIONS = listOf(
        "status",
        "top_apps",
        "wifi_scan",
        "bluetooth_scan",
        "sensor_snapshot",
        "camera_status",
        "radio_signal_status",
        "signal_capability_status",
        "social_gmail_goal_preflight",
        "show_active_overlay",
        "tool_catalog",
        "open_usage_access_settings",
        "open_camera_permission_settings",
    )
    private val SENSOR_TYPE_LABELS = linkedMapOf(
        "accelerometer" to "Accelerometer",
        "gyroscope" to "Gyroscope",
        "magnetic_field" to "Magnetic field",
        "light" to "Light",
        "proximity" to "Proximity",
        "gravity" to "Gravity",
        "linear_acceleration" to "Linear acceleration",
        "rotation_vector" to "Rotation vector",
        "step_counter" to "Step counter",
        "pressure" to "Pressure",
        "ambient_temperature" to "Ambient temperature",
        "relative_humidity" to "Relative humidity",
    )
    private val DEFAULT_SENSOR_TYPES = listOf("accelerometer", "gyroscope", "magnetic_field", "light", "proximity")
    private const val DEFAULT_LIMIT = 5
    private const val MAX_LIMIT = 20
    private const val MAX_WIFI_RESULTS = 40
    private const val MAX_BLUETOOTH_RESULTS = 40
    private const val MAX_SENSOR_TYPES_PER_SAMPLE = 8
    private const val DEFAULT_SENSOR_TIMEOUT_MS = 800L
    private const val MAX_SENSOR_TIMEOUT_MS = 3_000L
    private const val DEFAULT_BLUETOOTH_TIMEOUT_MS = 2_500L
    private const val MAX_BLUETOOTH_TIMEOUT_MS = 8_000L
    private val BROADCAST_RADIO_FEATURE_NAMES = listOf(
        "android.hardware.broadcastradio",
        "android.hardware.radio",
        "android.hardware.fm",
        "android.hardware.fmradio",
    )
    private val TIKTOK_PACKAGES = listOf("com.zhiliaoapp.musically", "com.ss.android.ugc.trill")
    private const val INSTAGRAM_PACKAGE = "com.instagram.android"
    private const val GMAIL_PACKAGE = "com.google.android.gm"
}
