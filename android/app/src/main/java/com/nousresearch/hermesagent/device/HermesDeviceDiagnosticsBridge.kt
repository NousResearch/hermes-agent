package com.nousresearch.hermesagent.device

import android.Manifest
import android.app.ActivityManager
import android.app.AppOpsManager
import android.app.usage.StorageStatsManager
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothClass
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanRecord
import android.bluetooth.le.ScanResult as BleScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorDirectChannel
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
import android.os.SystemClock
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
import kotlin.math.roundToInt

object HermesDeviceDiagnosticsBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        return when (action.lowercase(Locale.US).ifBlank { "status" }) {
            "status", "diagnostics_status", "device_diagnostics_status" -> statusJson(appContext).toString()
            "top_apps", "top_resource_apps", "top_memory_apps", "top_storage_apps", "resource_apps" ->
                topAppsJson(appContext, arguments).toString()
            "wifi_scan", "wifi_analyzer", "scan_wifi", "nearby_wifi", "wifi_signals" ->
                wifiScanJson(appContext, arguments).toString()
            "wifi_channel_rating", "wifi_channels", "channel_rating", "best_wifi_channel", "wifi_congestion" ->
                wifiScanJson(appContext, arguments, "wifi_channel_rating").toString()
            "wifi_ap_details", "wifi_access_points", "wifi_access_point_details", "wifi_report" ->
                wifiScanJson(appContext, arguments, "wifi_ap_details").toString()
            "wifi_export", "wifi_analyzer_export", "wifi_access_point_export", "export_wifi" ->
                wifiScanJson(appContext, arguments, "wifi_export").toString()
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
                    .put(card("Diagnostics", "Phone resource, Wi-Fi vendor/OUI/channel, Bluetooth service/manufacturer/proximity, camera, sensor, SOC, and overlay diagnostics are available to the agent."))
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

    fun wifiScanJson(context: Context, arguments: JSONObject = JSONObject(), actionName: String = "wifi_scan"): JSONObject {
        val appContext = context.applicationContext
        val limit = arguments.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_WIFI_RESULTS)
        val refresh = arguments.optBoolean("refresh", false)
        val wifiManager = appContext.getSystemService(WifiManager::class.java)
        val permissionStatus = wifiPermissionStatusJson(appContext)
        val canReadScan = permissionStatus.optBoolean("can_read_scan_results", false)
        if (wifiManager == null) {
            return JSONObject()
                .put("success", false)
                .put("action", actionName)
                .put("error", "Wi-Fi service is unavailable on this device")
                .put("wifi_scan_permission_status", permissionStatus)
        }
        if (!canReadScan) {
            return JSONObject()
                .put("success", false)
                .put("action", actionName)
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
                .put("action", actionName)
                .put("error", error.message ?: "Android denied Wi-Fi scan results")
                .put("wifi_scan_permission_status", permissionStatus)
        }
        val observedAtMs = System.currentTimeMillis()
        val latestScanAgeMs = latestWifiScanAgeMs(scanResults)
        val allNetworks = JSONArray()
        val sortedScanResults = scanResults
            .sortedWith(compareByDescending<ScanResult> { it.level }.thenBy { it.SSID.orEmpty() })
        sortedScanResults.forEach { result -> allNetworks.put(scanResultJson(result)) }
        val networks = JSONArray()
        for (index in 0 until minOf(limit, allNetworks.length())) {
            networks.put(allNetworks.getJSONObject(index))
        }
        val channelRatings = wifiChannelRatingRowsForNetworks(allNetworks)
        val recommendedChannels = recommendedWifiChannels(channelRatings)
        val bandSummary = wifiBandSummaryJson(allNetworks, channelRatings)
        val vendorSummary = wifiVendorSummaryJson(allNetworks)
        val analyzerFilters = wifiAnalyzerFilterSummaryJson(allNetworks)
        val detailLimitDefault = if (actionName == "wifi_export" || actionName == "wifi_ap_details") MAX_WIFI_RESULTS else limit
        val detailLimit = arguments.optInt(
            "detail_limit",
            arguments.optInt("export_limit", detailLimitDefault),
        ).coerceIn(1, MAX_WIFI_RESULTS)
        val accessPointDetails = wifiAccessPointDetailRows(allNetworks, detailLimit)
        val securitySummary = wifiSecuritySummaryJson(allNetworks)
        val channelWidthSummary = wifiChannelWidthSummaryJson(allNetworks)
        val standardSummary = wifiStandardSummaryJson(allNetworks)
        val exportFormat = normalizedWifiExportFormat(arguments.optString("export_format").ifBlank {
            if (actionName == "wifi_export") "both" else "json"
        })
        val accessPointExport = wifiAccessPointExportJson(accessPointDetails, exportFormat, observedAtMs)
        val historyStore = updateWifiSignalHistory(appContext, allNetworks, observedAtMs)
        val signalHistory = wifiSignalHistoryRowsFromStore(historyStore)
        val wifiEnabled = wifiManager.isWifiEnabled
        val scanStatus = wifiScanStatusJson(
            refreshRequested = refresh,
            refreshAccepted = refreshAccepted,
            wifiEnabled = wifiEnabled,
            permissionStatus = permissionStatus,
            totalScanResultCount = allNetworks.length(),
            returnedNetworkCount = networks.length(),
            latestScanAgeMs = latestScanAgeMs,
        )
        return JSONObject()
            .put("success", true)
            .put("action", actionName)
            .put("refresh_requested", refresh)
            .put("refresh_accepted", refreshAccepted)
            .put("result_count", networks.length())
            .put("total_scan_result_count", allNetworks.length())
            .put("wifi_scan_age_ms", latestScanAgeMs ?: JSONObject.NULL)
            .put("wifi_vendor_count", vendorSummary.length())
            .put("wifi_filter_count", analyzerFilters.length())
            .put("wifi_history_network_count", signalHistory.length())
            .put("wifi_access_point_detail_count", accessPointDetails.length())
            .put("wifi_security_summary_count", securitySummary.length())
            .put("wifi_width_summary_count", channelWidthSummary.length())
            .put("wifi_standard_summary_count", standardSummary.length())
            .put("wifi_enabled", wifiEnabled)
            .put("wifi_scan_permission_status", permissionStatus)
            .put("wifi_scan_status", scanStatus)
            .put("wifi_networks", networks)
            .put("wifi_access_point_details", accessPointDetails)
            .put("wifi_access_point_export", accessPointExport)
            .put(
                "wifi_access_point_export_csv",
                if (exportFormat == "csv" || exportFormat == "both") wifiAccessPointCsv(accessPointDetails) else JSONObject.NULL,
            )
            .put("wifi_channel_ratings", channelRatings)
            .put("recommended_wifi_channels", recommendedChannels)
            .put("wifi_band_summary", bandSummary)
            .put("wifi_vendor_summary", vendorSummary)
            .put("wifi_analyzer_filters", analyzerFilters)
            .put("wifi_security_summary", securitySummary)
            .put("wifi_channel_width_summary", channelWidthSummary)
            .put("wifi_standard_summary", standardSummary)
            .put("wifi_signal_history", signalHistory)
            .put("privacy_note", "Vendor/OUI lookup uses local prefix hints from Android scan metadata; no internet lookup is performed.")
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Wi-Fi Analyzer",
                            body = "${networks.length()} nearby Wi-Fi signals ranked by RSSI dBm with channel/frequency/width/vendor/security metadata.",
                            graphType = "wifi_channel_strength",
                            rows = networks,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi AP Details",
                            body = "${accessPointDetails.length()} access point detail row(s) with export-ready SSID/BSSID, OUI, security, width, standard, distance, and channel metadata.",
                            graphType = "wifi_access_point_detail",
                            rows = accessPointDetails,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Channel Rating",
                            body = "${channelRatings.length()} channel ratings scored from nearby AP crowding, overlap, RSSI, and width metadata.",
                            graphType = "wifi_channel_rating",
                            rows = channelRatings,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Vendors",
                            body = "${vendorSummary.length()} vendor/OUI groups inferred locally from ${allNetworks.length()} scan rows.",
                            graphType = "wifi_vendor_summary",
                            rows = vendorSummary,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Security",
                            body = "${securitySummary.length()} security group(s) across the latest scan.",
                            graphType = "wifi_security_summary",
                            rows = securitySummary,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Widths",
                            body = "${channelWidthSummary.length()} channel-width group(s), including wide-channel interference context.",
                            graphType = "wifi_channel_width_summary",
                            rows = channelWidthSummary,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Standards",
                            body = "${standardSummary.length()} Wi-Fi standard group(s), including 802.11n/ac/ax/be metadata when Android exposes it.",
                            graphType = "wifi_standard_summary",
                            rows = standardSummary,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi History",
                            body = "${signalHistory.length()} AP signal history row(s) built from recent scan observations, including average RSSI and trend metadata.",
                            graphType = "wifi_signal_history",
                            rows = signalHistory,
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
        val metadataSummary = bluetoothMetadataSummaryRows(devices)
        val serviceUuidCount = bluetoothDistinctStringCount(devices, "service_uuids")
        val manufacturerIdCount = bluetoothDistinctStringCount(devices, "manufacturer_ids")
        return JSONObject()
            .put("success", true)
            .put("action", "bluetooth_scan")
            .put("refresh_requested", refresh)
            .put("refresh_accepted", refreshAccepted)
            .put("scan_error", scanError ?: JSONObject.NULL)
            .put("bluetooth_enabled", runCatching { adapter.isEnabled }.getOrDefault(false))
            .put("bluetooth_device_count", devices.length())
            .put("bluetooth_metadata_count", metadataSummary.length())
            .put("bluetooth_service_uuid_count", serviceUuidCount)
            .put("bluetooth_manufacturer_id_count", manufacturerIdCount)
            .put("bluetooth_scan_permission_status", permissionStatus)
            .put("bluetooth_devices", devices)
            .put("bluetooth_metadata_summary", metadataSummary)
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Bluetooth Nearby",
                            body = "${devices.length()} paired or scanned Bluetooth device row(s), with BLE RSSI, class, service UUID, manufacturer, and proximity metadata when Android exposes it.",
                            graphType = "bluetooth_rssi",
                            rows = devices,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Bluetooth Metadata",
                            body = "${metadataSummary.length()} Bluetooth class/service/manufacturer summary row(s) inferred from nearby and paired device metadata.",
                            graphType = "bluetooth_metadata_summary",
                            rows = metadataSummary,
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
        val capabilities = sensorCapabilityRows(sensorManager, requested)
        val available = sensorTypeCatalog(appContext)
        return JSONObject()
            .put("success", true)
            .put("action", "sensor_snapshot")
            .put("requested_sensor_types", JSONArray(requested))
            .put("sample_timeout_ms", timeoutMs)
            .put("available_sensor_types", JSONArray(available))
            .put("sensor_samples", samples)
            .put("sensor_capabilities", capabilities)
            .put("sensor_capability_count", capabilities.length())
            .put("motion_sensor_count", countSensorCapabilities(capabilities, MOTION_SENSOR_TYPES))
            .put("wake_up_sensor_count", countSensorCapabilityFlag(capabilities, "wake_up"))
            .put("supported_watcher_types", JSONArray(SENSOR_TYPE_LABELS.keys))
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Motion Sensors",
                            body = "${samples.length()} one-shot accelerometer, gyroscope, magnetic, light, or proximity rows captured for the agent.",
                            graphType = "sensor_vector",
                            rows = samples,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Sensor Hardware",
                            body = "${capabilities.length()} sensor capability row(s) with vendor, range, resolution, power, FIFO, wake-up, and sampling-rate metadata.",
                            graphType = "sensor_capability",
                            rows = capabilities,
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
                    .put(card("SOC Compatibility", "Diagnostics and LiteRT policy use ABI, SOC, and GPU capability probes across MediaTek/Mali/PowerVR, Snapdragon/Adreno, Tensor, Exynos, and generic ARM devices with CPU fallback.")),
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
                    .put(toolJson("android_device_diagnostics_tool", "Inspect resource-heavy apps, Wi-Fi signals/channel ratings/AP detail and export rows/vendor OUI/filter facets, Bluetooth nearby devices/service UUIDs/manufacturer/proximity, camera, sensors, SOC compatibility, overlay, radio/RF capability limits, and the social/Gmail end-to-end phone preflight.", "action, limit, detail_limit, export_format, refresh, sensor_types, timeout_ms"))
                    .put(toolJson("hindsight_memory_tool", "Retain, recall, reflect, and promote local Hindsight-style memories with tags, entities, keywords, recency, reinforcement, and reusable prompt context.", "action, content, query, tags, category")),
            )
            .put("diagnostics_actions", JSONArray(ACTIONS))
            .put(
                "hindsight_memory_translation",
                JSONObject()
                    .put("retain", "Promote facts from chats/tool results into structured memories with source, category, timestamp, and reinforcement count.")
                    .put("recall", "Retrieve memories by semantic/keyword/time/entity signals before answering.")
                    .put("reflect", "Periodically consolidate repeated facts into fresher summaries and keep raw evidence links.")
                    .put("promoted_context", "Expose high-reuse memories as compact prompt context after repeated recall/retention hits."),
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
                    unavailableSensorJson(key),
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
                val sensorType = canonicalSensorType(event.sensor)
                readings[event.sensor.type] = sensorMetadataJson(sensorType, event.sensor)
                    .put("sampled", true)
                    .put("values", values)
                    .put("timestamp_nanos", event.timestamp)
                    .put("accuracy", event.accuracy)
                    .put("accuracy_label", sensorAccuracyLabel(event.accuracy))
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
                    .put("sensor_label", SENSOR_TYPE_LABELS[key] ?: key)
                    .put("sensor_type", key)
                    .put("sensor_name", sensor.name.orEmpty())
                    .put("vendor", sensor.vendor.orEmpty())
                    .put("unit", unitForSensorType(key))
                    .put("available", true)
                    .put("sampled", false),
            )
        }
        val sampledKeys = targets.map { it.first }.toSet()
        requested.filterNot { it in sampledKeys }.forEach { key ->
            samples.put(
                unavailableSensorJson(key),
            )
        }
        return samples
    }

    private fun sensorCapabilityRows(sensorManager: SensorManager, requested: List<String>): JSONArray {
        val rows = JSONArray()
        requested.forEach { key ->
            val sensor = androidSensorType(key)?.let { sensorManager.getDefaultSensor(it) }
            rows.put(
                if (sensor != null) {
                    sensorMetadataJson(key, sensor)
                } else {
                    unavailableSensorJson(key)
                },
            )
        }
        return rows
    }

    private fun sensorMetadataJson(sensorType: String, sensor: Sensor): JSONObject {
        return JSONObject()
            .put("sensor_type", sensorType)
            .put("sensor_label", SENSOR_TYPE_LABELS[sensorType] ?: sensorType)
            .put("sensor_name", sensor.name.orEmpty())
            .put("vendor", sensor.vendor.orEmpty())
            .put("version", sensor.version)
            .put("unit", unitForSensorType(sensorType))
            .put("available", true)
            .put("sampled", false)
            .put("maximum_range", sensor.maximumRange.toDouble())
            .put("resolution", sensor.resolution.toDouble())
            .put("power_ma", sensor.power.toDouble())
            .put("min_delay_us", sensor.minDelay)
            .put("max_delay_us", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) sensor.maxDelay else 0)
            .put("fifo_reserved_event_count", sensor.fifoReservedEventCount)
            .put("fifo_max_event_count", sensor.fifoMaxEventCount)
            .put("reporting_mode", sensorReportingModeLabel(sensor))
            .put("wake_up", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) sensor.isWakeUpSensor else false)
            .put("dynamic_sensor", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) sensor.isDynamicSensor else false)
            .put("direct_channel_supported", sensorDirectChannelSupported(sensor))
            .put("highest_direct_report_rate_level", sensorHighestDirectReportRate(sensor))
    }

    private fun unavailableSensorJson(sensorType: String): JSONObject {
        return JSONObject()
            .put("sensor_type", sensorType)
            .put("sensor_label", SENSOR_TYPE_LABELS[sensorType] ?: sensorType)
            .put("unit", unitForSensorType(sensorType))
            .put("available", false)
            .put("sampled", false)
    }

    private fun countSensorCapabilities(rows: JSONArray, sensorTypes: Set<String>): Int {
        var count = 0
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            if (row.optBoolean("available", false) && row.optString("sensor_type") in sensorTypes) {
                count += 1
            }
        }
        return count
    }

    private fun countSensorCapabilityFlag(rows: JSONArray, flag: String): Int {
        var count = 0
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            if (row.optBoolean(flag, false)) {
                count += 1
            }
        }
        return count
    }

    private fun scanResultJson(result: ScanResult): JSONObject {
        val capabilities = result.capabilities.orEmpty()
        val oui = wifiBssidOui(result.BSSID.orEmpty())
        val rawSsid = result.SSID.orEmpty()
        val displaySsid = rawSsid.ifBlank { "<hidden>" }
        val estimatedDistanceMeters = estimateWifiDistanceMeters(result.level, result.frequency)
        val json = JSONObject()
            .put("ssid", displaySsid)
            .put("display_ssid", displaySsid)
            .put("hidden_ssid", rawSsid.isBlank())
            .put("bssid", result.BSSID.orEmpty())
            .put("bssid_oui", oui.ifBlank { JSONObject.NULL })
            .put("bssid_vendor", wifiOuiVendorLabel(oui))
            .put("rssi_dbm", result.level)
            .put("signal_quality", wifiSignalQualityLabel(result.level))
            .put("frequency_mhz", result.frequency)
            .put("channel", channelForFrequencyMhz(result.frequency) ?: JSONObject.NULL)
            .put("band", wifiBandLabel(result.frequency))
            .put("estimated_distance_meters", estimatedDistanceMeters)
            .put("estimated_distance_m", estimatedDistanceMeters)
            .put("security_mode", wifiSecurityLabel(capabilities))
            .put("capabilities", capabilities)
            .put("timestamp_micros", result.timestamp)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val channelWidth = channelWidthLabel(result.channelWidth)
            json.put("channel_width", channelWidth)
            json.put("channel_width_mhz", channelWidthMhz(channelWidth) ?: JSONObject.NULL)
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
        val bluetoothClass = runCatching { device.bluetoothClass }.getOrNull()
        val serviceUuids = runCatching {
            device.uuids
                ?.mapNotNull { it?.uuid?.toString() }
                ?.distinct()
                .orEmpty()
        }.getOrDefault(emptyList())
        val classJson = bluetoothClassJson(bluetoothClass)
        return JSONObject()
            .put("device_name", bluetoothDeviceName(device))
            .put("address", runCatching { device.address.orEmpty() }.getOrDefault(""))
            .put("device_type", bluetoothDeviceTypeLabel(runCatching { device.type }.getOrDefault(BluetoothDevice.DEVICE_TYPE_UNKNOWN)))
            .put("bond_state", bluetoothBondStateLabel(runCatching { device.bondState }.getOrDefault(BluetoothDevice.BOND_NONE)))
            .put("paired", runCatching { device.bondState == BluetoothDevice.BOND_BONDED }.getOrDefault(false))
            .put("connectable", true)
            .put("device_class", classJson.optString("device_class"))
            .put("major_device_class", classJson.optString("major_device_class"))
            .put("device_category", classJson.optString("device_category"))
            .put("service_uuids", JSONArray(serviceUuids.take(MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE)))
            .put("service_uuid_count", serviceUuids.size)
    }

    private fun bleScanResultJson(result: BleScanResult, callbackType: Int): JSONObject {
        val scanRecord = result.scanRecord
        val txPowerDbm = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.txPower else null
        val distanceMeters = estimateBluetoothDistanceMeters(result.rssi, txPowerDbm)
        return bluetoothDeviceJson(result.device)
            .put("advertised_name", scanRecord?.deviceName.orEmpty())
            .put("rssi_dbm", result.rssi)
            .put("proximity_label", bluetoothProximityLabel(result.rssi, distanceMeters))
            .put("estimated_distance_meters", distanceMeters ?: JSONObject.NULL)
            .put("callback_type", callbackType)
            .put("connectable", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.isConnectable else JSONObject.NULL)
            .put("tx_power_dbm", txPowerDbm ?: JSONObject.NULL)
            .put("legacy_advertisement", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.isLegacy else JSONObject.NULL)
            .put("primary_phy", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) bluetoothPhyLabel(result.primaryPhy) else JSONObject.NULL)
            .put("secondary_phy", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) bluetoothPhyLabel(result.secondaryPhy) else JSONObject.NULL)
            .put("advertising_sid", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.advertisingSid else JSONObject.NULL)
            .put("periodic_advertising_interval", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) result.periodicAdvertisingInterval else JSONObject.NULL)
            .put("advertising_flags", scanRecord?.advertiseFlags ?: JSONObject.NULL)
            .put("service_uuids", bluetoothServiceUuidsJson(scanRecord))
            .put("service_uuid_count", scanRecord?.serviceUuids?.size ?: 0)
            .put("service_data_uuids", bluetoothServiceDataUuidsJson(scanRecord))
            .put("manufacturer_ids", bluetoothManufacturerIdsJson(scanRecord))
            .put("manufacturer_data_count", scanRecord?.manufacturerSpecificData?.size() ?: 0)
            .put("manufacturer_data_bytes", bluetoothManufacturerDataBytes(scanRecord))
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

    private fun bluetoothPhyLabel(value: Int): String = when (value) {
        1 -> "le_1m"
        2 -> "le_2m"
        3 -> "le_coded"
        0 -> "unused"
        else -> "phy_$value"
    }

    private fun bluetoothClassJson(bluetoothClass: BluetoothClass?): JSONObject {
        if (bluetoothClass == null) {
            return JSONObject()
                .put("device_class", "unknown")
                .put("major_device_class", "unknown")
                .put("device_category", "unknown")
        }
        val majorClass = bluetoothClass.majorDeviceClass
        return JSONObject()
            .put("device_class", "0x${bluetoothClass.deviceClass.toString(16).uppercase(Locale.US)}")
            .put("major_device_class", bluetoothMajorDeviceClassLabel(majorClass))
            .put("device_category", bluetoothDeviceCategoryLabel(majorClass))
    }

    private fun bluetoothMajorDeviceClassLabel(value: Int): String = when (value) {
        BluetoothClass.Device.Major.COMPUTER -> "computer"
        BluetoothClass.Device.Major.PHONE -> "phone"
        BluetoothClass.Device.Major.NETWORKING -> "networking"
        BluetoothClass.Device.Major.AUDIO_VIDEO -> "audio_video"
        BluetoothClass.Device.Major.PERIPHERAL -> "peripheral"
        BluetoothClass.Device.Major.IMAGING -> "imaging"
        BluetoothClass.Device.Major.WEARABLE -> "wearable"
        BluetoothClass.Device.Major.TOY -> "toy"
        BluetoothClass.Device.Major.HEALTH -> "health"
        BluetoothClass.Device.Major.UNCATEGORIZED -> "uncategorized"
        else -> "unknown"
    }

    private fun bluetoothDeviceCategoryLabel(majorClass: Int): String = when (majorClass) {
        BluetoothClass.Device.Major.AUDIO_VIDEO -> "audio"
        BluetoothClass.Device.Major.PHONE,
        BluetoothClass.Device.Major.COMPUTER,
        BluetoothClass.Device.Major.NETWORKING -> "peer_device"
        BluetoothClass.Device.Major.PERIPHERAL -> "input_peripheral"
        BluetoothClass.Device.Major.IMAGING -> "imaging"
        BluetoothClass.Device.Major.WEARABLE,
        BluetoothClass.Device.Major.HEALTH -> "wearable_health"
        else -> bluetoothMajorDeviceClassLabel(majorClass)
    }

    private fun bluetoothServiceUuidsJson(scanRecord: ScanRecord?): JSONArray {
        val values = scanRecord?.serviceUuids
            ?.mapNotNull { it?.uuid?.toString() }
            ?.distinct()
            .orEmpty()
            .take(MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE)
        return JSONArray(values)
    }

    private fun bluetoothServiceDataUuidsJson(scanRecord: ScanRecord?): JSONArray {
        val values = scanRecord?.serviceData
            ?.keys
            ?.mapNotNull { it?.uuid?.toString() }
            ?.distinct()
            .orEmpty()
            .take(MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE)
        return JSONArray(values)
    }

    private fun bluetoothManufacturerIdsJson(scanRecord: ScanRecord?): JSONArray {
        val data = scanRecord?.manufacturerSpecificData ?: return JSONArray()
        val values = buildList {
            for (index in 0 until data.size()) {
                add("0x${data.keyAt(index).toString(16).uppercase(Locale.US).padStart(4, '0')}")
            }
        }
        return JSONArray(values.take(MAX_BLUETOOTH_MANUFACTURER_IDS_PER_DEVICE))
    }

    private fun bluetoothManufacturerDataBytes(scanRecord: ScanRecord?): Int {
        val data = scanRecord?.manufacturerSpecificData ?: return 0
        var bytes = 0
        for (index in 0 until data.size()) {
            bytes += data.valueAt(index)?.size ?: 0
        }
        return bytes
    }

    internal fun estimateBluetoothDistanceMeters(rssiDbm: Int, txPowerDbm: Int?): Double? {
        val tx = txPowerDbm?.takeIf { it in -127..20 } ?: return null
        if (rssiDbm >= 0) return null
        val distance = 10.0.pow((tx - rssiDbm).toDouble() / 20.0)
        return (distance * 100.0).roundToInt() / 100.0
    }

    internal fun bluetoothProximityLabel(rssiDbm: Int, distanceMeters: Double? = null): String = when {
        distanceMeters != null && distanceMeters <= 1.0 -> "immediate"
        rssiDbm >= -55 -> "near"
        distanceMeters != null && distanceMeters <= 5.0 -> "room"
        rssiDbm >= -75 -> "room"
        else -> "far"
    }

    internal fun bluetoothMetadataSummaryRows(devices: JSONArray): JSONArray {
        data class MetadataAccumulator(
            var count: Int = 0,
            var pairedCount: Int = 0,
            var connectableCount: Int = 0,
            var strongestRssiDbm: Int? = null,
            val sampleDevices: LinkedHashSet<String> = linkedSetOf(),
        )

        val summaries = linkedMapOf<String, MetadataAccumulator>()
        fun add(summaryType: String, label: String, row: JSONObject) {
            val key = "$summaryType|$label"
            val accumulator = summaries.getOrPut(key) { MetadataAccumulator() }
            accumulator.count += 1
            if (row.optBoolean("paired", false)) accumulator.pairedCount += 1
            if (row.optBoolean("connectable", false)) accumulator.connectableCount += 1
            jsonIntOrNull(row, "rssi_dbm")?.let { rssi ->
                accumulator.strongestRssiDbm = maxOf(accumulator.strongestRssiDbm ?: rssi, rssi)
            }
            row.optString("device_name")
                .takeIf { it.isNotBlank() && it != "<unnamed>" }
                ?.let { accumulator.sampleDevices.add(it) }
        }

        for (index in 0 until devices.length()) {
            val row = devices.optJSONObject(index) ?: continue
            val category = row.optString("device_category")
                .ifBlank { row.optString("major_device_class") }
                .ifBlank { row.optString("device_type") }
                .ifBlank { "unknown" }
            add("device_category", category, row)
            jsonStringList(row, "service_uuids").forEach { uuid ->
                add("service_uuid", uuid, row)
            }
            jsonStringList(row, "manufacturer_ids").forEach { manufacturerId ->
                add("manufacturer_id", manufacturerId, row)
            }
        }

        val rows = summaries.map { (key, accumulator) ->
            val parts = key.split('|', limit = 2)
            val summaryType = parts.firstOrNull().orEmpty()
            val label = parts.getOrNull(1).orEmpty()
            JSONObject()
                .put("summary_type", summaryType)
                .put("label", label)
                .put("count", accumulator.count)
                .put("paired_count", accumulator.pairedCount)
                .put("connectable_count", accumulator.connectableCount)
                .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                .put("sample_devices", JSONArray(accumulator.sampleDevices.take(MAX_BLUETOOTH_SUMMARY_SAMPLES)))
                .put(
                    "recommendation",
                    when (summaryType) {
                        "service_uuid" -> "BLE service UUID advertised nearby; use it to infer device capability before connecting."
                        "manufacturer_id" -> "Manufacturer data advertised nearby; useful for beacon or vendor-specific device identification."
                        else -> "Bluetooth device class group from paired or nearby scan metadata."
                    },
                )
        }
            .sortedWith(
                compareBy<JSONObject> { bluetoothSummarySortKey(it.optString("summary_type")) }
                    .thenByDescending { it.optInt("count") }
                    .thenByDescending { it.optInt("strongest_rssi_dbm", -100) }
                    .thenBy { it.optString("label") },
            )
            .take(MAX_BLUETOOTH_METADATA_SUMMARY_ROWS)
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    private fun bluetoothDistinctStringCount(devices: JSONArray, key: String): Int {
        val values = linkedSetOf<String>()
        for (index in 0 until devices.length()) {
            val row = devices.optJSONObject(index) ?: continue
            values.addAll(jsonStringList(row, key))
        }
        return values.size
    }

    private fun bluetoothSummarySortKey(summaryType: String): Int = when (summaryType) {
        "device_category" -> 0
        "service_uuid" -> 1
        "manufacturer_id" -> 2
        else -> 3
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
        return runCatching {
            val appOps = context.getSystemService(AppOpsManager::class.java) ?: return@runCatching false
            val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                appOps.unsafeCheckOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)
            } else {
                @Suppress("DEPRECATION")
                appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)
            }
            mode == AppOpsManager.MODE_ALLOWED
        }.getOrDefault(false)
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

    internal fun wifiChannelRatingRowsForNetworks(networks: JSONArray): JSONArray {
        val measurements = buildList {
            for (index in 0 until networks.length()) {
                val row = networks.optJSONObject(index) ?: continue
                val frequencyMhz = row.optInt("frequency_mhz", 0)
                val channel = row.opt("channel")
                    ?.takeUnless { it == JSONObject.NULL }
                    ?.let { value ->
                        when (value) {
                            is Number -> value.toInt()
                            else -> value.toString().toIntOrNull()
                        }
                    }
                    ?: channelForFrequencyMhz(frequencyMhz)
                    ?: continue
                val rssiDbm = row.opt("rssi_dbm")
                    ?.let { value ->
                        when (value) {
                            is Number -> value.toInt()
                            else -> value.toString().toIntOrNull()
                        }
                    }
                    ?: continue
                add(
                    WifiChannelMeasurement(
                        channel = channel,
                        band = canonicalWifiBandLabel(row.optString("band"), frequencyMhz),
                        rssiDbm = rssiDbm,
                        widthLabel = row.optString("channel_width").ifBlank { "20MHz" },
                        frequencyMhz = frequencyMhz,
                    ),
                )
            }
        }
        return wifiChannelRatingRowsForMeasurements(measurements)
    }

    internal fun wifiVendorSummaryJson(networks: JSONArray): JSONArray {
        data class VendorAccumulator(
            var networkCount: Int = 0,
            var strongestRssiDbm: Int? = null,
            val ssids: LinkedHashSet<String> = linkedSetOf(),
            val ouis: LinkedHashSet<String> = linkedSetOf(),
        )

        val vendors = linkedMapOf<String, VendorAccumulator>()
        for (index in 0 until networks.length()) {
            val row = networks.optJSONObject(index) ?: continue
            val bssid = row.optString("bssid")
            val oui = row.optString("bssid_oui").ifBlank { wifiBssidOui(bssid) }
            val vendor = row.optString("bssid_vendor").ifBlank { wifiOuiVendorLabel(oui) }
            val key = vendor.ifBlank { "Unknown vendor" }
            val accumulator = vendors.getOrPut(key) { VendorAccumulator() }
            accumulator.networkCount += 1
            jsonIntOrNull(row, "rssi_dbm")?.let { rssi ->
                accumulator.strongestRssiDbm = maxOf(accumulator.strongestRssiDbm ?: rssi, rssi)
            }
            row.optString("ssid").takeIf { it.isNotBlank() }?.let { accumulator.ssids.add(it) }
            oui.takeIf { it.isNotBlank() }?.let { accumulator.ouis.add(it) }
        }

        val rows = vendors.map { (vendor, accumulator) ->
            JSONObject()
                .put("vendor", vendor)
                .put("network_count", accumulator.networkCount)
                .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                .put("bssid_ouis", JSONArray(accumulator.ouis.take(MAX_WIFI_VENDOR_DETAILS)))
                .put("sample_ssids", JSONArray(accumulator.ssids.take(MAX_WIFI_VENDOR_DETAILS)))
                .put(
                    "recommendation",
                    when {
                        vendor.startsWith("Unknown", ignoreCase = true) -> "Unknown vendor: keep exact SSID/security and channel metadata visible for manual review."
                        vendor.startsWith("Locally administered", ignoreCase = true) -> "Likely randomized or locally administered BSSID: treat vendor attribution as unavailable."
                        (accumulator.strongestRssiDbm ?: -100) > -55 -> "Strong nearby vendor group: inspect channel overlap and security before choosing router placement."
                        else -> "Vendor group detected from OUI prefix; review channel and signal cards for impact."
                    },
                )
        }
            .sortedWith(
                compareByDescending<JSONObject> { it.optInt("network_count") }
                    .thenByDescending { it.optInt("strongest_rssi_dbm", -100) }
                    .thenBy { it.optString("vendor") },
            )
            .take(MAX_WIFI_VENDOR_ROWS)
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun wifiAnalyzerFilterSummaryJson(networks: JSONArray): JSONArray {
        val bandCounts = linkedMapOf<String, Int>()
        val signalCounts = linkedMapOf<String, Int>()
        val securityCounts = linkedMapOf<String, Int>()
        val ssidCounts = linkedMapOf<String, Int>()
        for (index in 0 until networks.length()) {
            val row = networks.optJSONObject(index) ?: continue
            val band = canonicalWifiBandLabel(row.optString("band"), row.optInt("frequency_mhz", 0))
            incrementCount(bandCounts, band.takeIf { it != "unknown" } ?: "unknown")
            val rssi = jsonIntOrNull(row, "rssi_dbm")
            incrementCount(signalCounts, rssi?.let(::wifiSignalQualityLabel) ?: row.optString("signal_quality").ifBlank { "unknown" })
            incrementCount(
                securityCounts,
                row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) },
            )
            incrementCount(ssidCounts, row.optString("ssid").ifBlank { "<hidden>" })
        }
        return JSONArray()
            .put(filterFacetJson("band", "Wi-Fi band", bandCounts, ::wifiBandSortKey))
            .put(filterFacetJson("signal_strength", "Signal strength", signalCounts, ::wifiSignalSortKey))
            .put(filterFacetJson("security", "Security", securityCounts, ::wifiSecuritySortKey))
            .put(filterFacetJson("ssid", "SSID", ssidCounts) { 0 })
    }

    internal fun wifiAccessPointDetailRows(networks: JSONArray, limit: Int = MAX_WIFI_RESULTS): JSONArray {
        val rows = buildList {
            for (index in 0 until networks.length()) {
                val row = networks.optJSONObject(index) ?: continue
                val frequencyMhz = jsonIntOrNull(row, "frequency_mhz") ?: 0
                val channel = jsonIntOrNull(row, "channel") ?: channelForFrequencyMhz(frequencyMhz)
                val rssiDbm = jsonIntOrNull(row, "rssi_dbm")
                val ssid = row.optString("display_ssid").ifBlank { row.optString("ssid").ifBlank { "<hidden>" } }
                val hiddenSsid = row.optBoolean("hidden_ssid", ssid == "<hidden>")
                val bssid = row.optString("bssid")
                val oui = row.optString("bssid_oui").ifBlank { wifiBssidOui(bssid) }
                val channelWidth = row.optString("channel_width").ifBlank { "unknown" }
                val widthMhz = jsonIntOrNull(row, "channel_width_mhz") ?: channelWidthMhz(channelWidth)
                val capabilities = row.optString("capabilities")
                val estimatedDistance = jsonDoubleOrNull(row, "estimated_distance_m")
                    ?: jsonDoubleOrNull(row, "estimated_distance_meters")
                val timestampMicros = jsonLongOrNull(row, "timestamp_micros")
                add(
                    JSONObject()
                        .put("ssid", ssid)
                        .put("display_ssid", ssid)
                        .put("hidden_ssid", hiddenSsid)
                        .put("bssid", bssid)
                        .put("bssid_oui", oui.ifBlank { JSONObject.NULL })
                        .put("bssid_vendor", row.optString("bssid_vendor").ifBlank { wifiOuiVendorLabel(oui) })
                        .put("rssi_dbm", rssiDbm ?: JSONObject.NULL)
                        .put("signal_quality", rssiDbm?.let(::wifiSignalQualityLabel) ?: row.optString("signal_quality").ifBlank { "unknown" })
                        .put("frequency_mhz", frequencyMhz.takeIf { it > 0 } ?: JSONObject.NULL)
                        .put("channel", channel ?: JSONObject.NULL)
                        .put("band", canonicalWifiBandLabel(row.optString("band"), frequencyMhz))
                        .put("channel_width", channelWidth)
                        .put("channel_width_mhz", widthMhz ?: JSONObject.NULL)
                        .put("center_freq0_mhz", jsonValueOrNull(row, "center_freq0_mhz"))
                        .put("center_freq1_mhz", jsonValueOrNull(row, "center_freq1_mhz"))
                        .put("wifi_standard", row.optString("wifi_standard").ifBlank { "unknown" })
                        .put("security_mode", row.optString("security_mode").ifBlank { wifiSecurityLabel(capabilities) })
                        .put("capabilities", capabilities)
                        .put("estimated_distance_m", estimatedDistance ?: JSONObject.NULL)
                        .put("estimated_distance_meters", estimatedDistance ?: JSONObject.NULL)
                        .put("passpoint_network", jsonValueOrNull(row, "passpoint_network"))
                        .put("80211mc_responder", jsonValueOrNull(row, "80211mc_responder"))
                        .put("timestamp_micros", timestampMicros ?: JSONObject.NULL)
                        .put("scan_age_ms", timestampMicros?.let(::wifiScanAgeMs) ?: JSONObject.NULL),
                )
            }
        }
        val sorted = rows
            .sortedWith(
                compareByDescending<JSONObject> { jsonIntOrNull(it, "rssi_dbm") ?: Int.MIN_VALUE }
                    .thenBy { it.optString("display_ssid") }
                    .thenBy { it.optString("bssid") },
            )
            .take(limit.coerceIn(1, MAX_WIFI_RESULTS))
        return JSONArray().also { array ->
            sorted.forEachIndexed { index, row ->
                array.put(row.put("rank", index + 1))
            }
        }
    }

    internal fun wifiAccessPointExportJson(details: JSONArray, format: String, generatedAtMs: Long = System.currentTimeMillis()): JSONObject {
        val normalizedFormat = normalizedWifiExportFormat(format)
        return JSONObject()
            .put("format", normalizedFormat)
            .put("row_count", details.length())
            .put("generated_at_ms", generatedAtMs)
            .put("json_array_key", "wifi_access_point_details")
            .put("csv_key", if (normalizedFormat == "csv" || normalizedFormat == "both") "wifi_access_point_export_csv" else JSONObject.NULL)
            .put("included_fields", JSONArray(WIFI_AP_EXPORT_FIELDS))
            .put("privacy_note", "Export rows are produced from Android's local Wi-Fi scan cache; Hermes does not perform internet vendor lookups.")
    }

    internal fun wifiAccessPointCsv(details: JSONArray): String {
        val header = WIFI_AP_EXPORT_FIELDS.joinToString(",")
        val rows = buildList {
            add(header)
            for (index in 0 until details.length()) {
                val row = details.optJSONObject(index) ?: continue
                add(WIFI_AP_EXPORT_FIELDS.joinToString(",") { field -> csvEscape(row.opt(field)) })
            }
        }
        return rows.joinToString("\n")
    }

    internal fun wifiSecuritySummaryJson(networks: JSONArray): JSONArray {
        val groups = linkedMapOf<String, WifiSummaryAccumulator>()
        for (index in 0 until networks.length()) {
            val row = networks.optJSONObject(index) ?: continue
            val security = row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) }
            val accumulator = groups.getOrPut(security) { WifiSummaryAccumulator() }
            accumulator.networkCount += 1
            appendWifiSummaryContext(accumulator, row)
        }
        val rows = groups.map { (security, accumulator) ->
            JSONObject()
                .put("security_mode", security)
                .put("network_count", accumulator.networkCount)
                .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                .put("bands", JSONArray(accumulator.bands.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("channels", JSONArray(accumulator.channels.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("sample_ssids", JSONArray(accumulator.sampleSsids.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("recommendation", wifiSecurityRecommendation(security, accumulator.networkCount, accumulator.strongestRssiDbm))
        }
            .sortedWith(
                compareBy<JSONObject> { wifiSecuritySortKey(it.optString("security_mode")) }
                    .thenByDescending { it.optInt("network_count") }
                    .thenBy { it.optString("security_mode") },
            )
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun wifiChannelWidthSummaryJson(networks: JSONArray): JSONArray {
        val groups = linkedMapOf<String, WifiSummaryAccumulator>()
        for (index in 0 until networks.length()) {
            val row = networks.optJSONObject(index) ?: continue
            val width = row.optString("channel_width").ifBlank { "unknown" }
            val accumulator = groups.getOrPut(width) { WifiSummaryAccumulator() }
            accumulator.networkCount += 1
            appendWifiSummaryContext(accumulator, row)
        }
        val rows = groups.map { (width, accumulator) ->
            val widthMhz = channelWidthMhz(width)
            JSONObject()
                .put("channel_width", width)
                .put("channel_width_mhz", widthMhz ?: JSONObject.NULL)
                .put("network_count", accumulator.networkCount)
                .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                .put("bands", JSONArray(accumulator.bands.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("channels", JSONArray(accumulator.channels.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("sample_ssids", JSONArray(accumulator.sampleSsids.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("recommendation", wifiChannelWidthRecommendation(width, accumulator.networkCount, accumulator.strongestRssiDbm))
        }
            .sortedWith(
                compareByDescending<JSONObject> { it.optInt("channel_width_mhz", 0) }
                    .thenByDescending { it.optInt("network_count") }
                    .thenBy { it.optString("channel_width") },
            )
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun wifiStandardSummaryJson(networks: JSONArray): JSONArray {
        val groups = linkedMapOf<String, WifiSummaryAccumulator>()
        for (index in 0 until networks.length()) {
            val row = networks.optJSONObject(index) ?: continue
            val standard = row.optString("wifi_standard").ifBlank { "unknown" }
            val accumulator = groups.getOrPut(standard) { WifiSummaryAccumulator() }
            accumulator.networkCount += 1
            appendWifiSummaryContext(accumulator, row)
            row.optString("channel_width").takeIf { it.isNotBlank() }?.let(accumulator.widths::add)
        }
        val rows = groups.map { (standard, accumulator) ->
            JSONObject()
                .put("wifi_standard", standard)
                .put("network_count", accumulator.networkCount)
                .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                .put("bands", JSONArray(accumulator.bands.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("sample_widths", JSONArray(accumulator.widths.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("sample_ssids", JSONArray(accumulator.sampleSsids.take(MAX_WIFI_SUMMARY_SAMPLES)))
                .put("recommendation", wifiStandardRecommendation(standard, accumulator.networkCount))
        }
            .sortedWith(
                compareBy<JSONObject> { wifiStandardSortKey(it.optString("wifi_standard")) }
                    .thenByDescending { it.optInt("network_count") }
                    .thenBy { it.optString("wifi_standard") },
            )
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun mergeWifiSignalHistory(existing: JSONObject, networks: JSONArray, observedAtMs: Long): JSONObject {
        val records = linkedMapOf<String, JSONObject>()
        val existingRecords = existing.optJSONArray("networks") ?: JSONArray()
        for (index in 0 until existingRecords.length()) {
            val record = existingRecords.optJSONObject(index) ?: continue
            val key = record.optString("key").ifBlank { wifiHistoryKey(record) }
            if (key.isNotBlank()) {
                records[key] = record.put("key", key)
            }
        }
        for (index in 0 until minOf(networks.length(), MAX_WIFI_HISTORY_NETWORKS_PER_SCAN)) {
            val network = networks.optJSONObject(index) ?: continue
            val key = wifiHistoryKey(network)
            if (key.isBlank()) continue
            val rssi = jsonIntOrNull(network, "rssi_dbm") ?: continue
            val record = records.getOrPut(key) { JSONObject().put("key", key) }
            record
                .put("ssid", network.optString("ssid").ifBlank { "<hidden>" })
                .put("bssid", network.optString("bssid"))
                .put("bssid_vendor", network.optString("bssid_vendor").ifBlank { "Unknown vendor" })
                .put("band", canonicalWifiBandLabel(network.optString("band"), network.optInt("frequency_mhz", 0)))
                .put("security_mode", network.optString("security_mode").ifBlank { wifiSecurityLabel(network.optString("capabilities")) })
                .put("frequency_mhz", network.optInt("frequency_mhz", 0))
                .put("channel", network.opt("channel") ?: JSONObject.NULL)
            val observations = record.optJSONArray("observations") ?: JSONArray()
            observations.put(JSONObject().put("observed_at_ms", observedAtMs).put("rssi_dbm", rssi))
            record.put("observations", trimWifiObservations(observations))
        }
        val ordered = records.values
            .filter { record -> (record.optJSONArray("observations")?.length() ?: 0) > 0 }
            .sortedWith(
                compareByDescending<JSONObject> { lastWifiObservationTime(it) }
                    .thenByDescending { currentWifiRssi(it) ?: Int.MIN_VALUE }
                    .thenBy { it.optString("ssid") },
            )
            .take(MAX_WIFI_HISTORY_NETWORKS)
        return JSONObject()
            .put("updated_at_ms", observedAtMs)
            .put("networks", JSONArray().also { array -> ordered.forEach(array::put) })
    }

    internal fun wifiSignalHistoryRowsFromStore(store: JSONObject, nowMs: Long = System.currentTimeMillis()): JSONArray {
        val records = store.optJSONArray("networks") ?: JSONArray()
        val rows = buildList {
            for (index in 0 until records.length()) {
                val record = records.optJSONObject(index) ?: continue
                val observations = record.optJSONArray("observations") ?: continue
                val rssiValues = buildList {
                    for (sampleIndex in 0 until observations.length()) {
                        jsonIntOrNull(observations.optJSONObject(sampleIndex) ?: continue, "rssi_dbm")?.let(::add)
                    }
                }
                if (rssiValues.isEmpty()) continue
                val firstRssi = rssiValues.first()
                val currentRssi = rssiValues.last()
                val averageRssi = (rssiValues.sum().toDouble() / rssiValues.size).roundToInt()
                val trendDb = currentRssi - firstRssi
                val lastSeenMs = (nowMs - lastWifiObservationTime(record)).coerceAtLeast(0L)
                add(
                    JSONObject()
                        .put("ssid", record.optString("ssid").ifBlank { "<hidden>" })
                        .put("bssid", record.optString("bssid"))
                        .put("bssid_vendor", record.optString("bssid_vendor").ifBlank { "Unknown vendor" })
                        .put("band", record.optString("band"))
                        .put("security_mode", record.optString("security_mode"))
                        .put("frequency_mhz", record.optInt("frequency_mhz", 0))
                        .put("channel", record.opt("channel") ?: JSONObject.NULL)
                        .put("sample_count", rssiValues.size)
                        .put("current_rssi_dbm", currentRssi)
                        .put("average_rssi_dbm", averageRssi)
                        .put("min_rssi_dbm", rssiValues.minOrNull() ?: currentRssi)
                        .put("max_rssi_dbm", rssiValues.maxOrNull() ?: currentRssi)
                        .put("trend_db", trendDb)
                        .put("trend_label", wifiSignalTrendLabel(trendDb))
                        .put("last_seen_ms", lastSeenMs)
                        .put("rssi_series", wifiObservationSeries(observations)),
                )
            }
        }
            .sortedWith(
                compareByDescending<JSONObject> { it.optInt("current_rssi_dbm", Int.MIN_VALUE) }
                    .thenByDescending { it.optInt("sample_count", 0) }
                    .thenBy { it.optString("ssid") },
            )
            .take(MAX_WIFI_HISTORY_ROWS)
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun wifiChannelRatingRowsForMeasurements(measurements: List<WifiChannelMeasurement>): JSONArray {
        val usable = measurements
            .filter { it.channel > 0 && it.rssiDbm < 0 }
            .map { measurement ->
                measurement.copy(
                    band = canonicalWifiBandLabel(measurement.band, measurement.frequencyMhz),
                    widthLabel = measurement.widthLabel.ifBlank { "20MHz" },
                )
            }
        val bands = usable.map { it.band }.filter { it != "unknown" }.distinct()
        val rows = bands.flatMap { band ->
            val bandMeasurements = usable.filter { it.band == band }
            val candidates = candidateWifiChannelsForBand(band, bandMeasurements.map { it.channel })
            candidates.map { channel ->
                wifiChannelRatingRow(band, channel, bandMeasurements)
            }
        }
        val sorted = rows
            .sortedWith(
                compareBy<JSONObject> { wifiBandSortKey(it.optString("band")) }
                    .thenByDescending { it.optInt("score") }
                    .thenBy { it.optInt("channel") },
            )
            .take(MAX_WIFI_CHANNEL_RATINGS)
        return JSONArray().also { array -> sorted.forEach(array::put) }
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

    private fun sensorAccuracyLabel(accuracy: Int): String = when (accuracy) {
        SensorManager.SENSOR_STATUS_ACCURACY_HIGH -> "high"
        SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> "medium"
        SensorManager.SENSOR_STATUS_ACCURACY_LOW -> "low"
        SensorManager.SENSOR_STATUS_UNRELIABLE -> "unreliable"
        else -> "unknown"
    }

    private fun sensorReportingModeLabel(sensor: Sensor): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) return "unknown"
        return when (sensor.reportingMode) {
            Sensor.REPORTING_MODE_CONTINUOUS -> "continuous"
            Sensor.REPORTING_MODE_ON_CHANGE -> "on_change"
            Sensor.REPORTING_MODE_ONE_SHOT -> "one_shot"
            Sensor.REPORTING_MODE_SPECIAL_TRIGGER -> "special_trigger"
            else -> "unknown"
        }
    }

    private fun sensorDirectChannelSupported(sensor: Sensor): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
        return sensor.isDirectChannelTypeSupported(SensorDirectChannel.TYPE_MEMORY_FILE) ||
            sensor.isDirectChannelTypeSupported(SensorDirectChannel.TYPE_HARDWARE_BUFFER)
    }

    private fun sensorHighestDirectReportRate(sensor: Sensor): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) sensor.highestDirectReportRateLevel else 0
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
        val supportedAbis = Build.SUPPORTED_ABIS.toList()
        val supported64BitAbis = Build.SUPPORTED_64_BIT_ABIS.toList()
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
        val hardwareProfile = HermesAndroidHardwareProfile.classify(values)
        return JSONObject()
            .put("soc_manufacturer", socManufacturer)
            .put("soc_model", socModel)
            .put("soc_family", hardwareProfile.socFamily)
            .put("soc_family_label", hardwareProfile.socLabel)
            .put("gpu_family_hint", hardwareProfile.gpuFamily)
            .put("gpu_family_label", hardwareProfile.gpuLabel)
            .put("hardware", Build.HARDWARE.orEmpty())
            .put("board", Build.BOARD.orEmpty())
            .put("android_sdk_int", Build.VERSION.SDK_INT)
            .put("android_release", Build.VERSION.RELEASE.orEmpty())
            .put("primary_abi", supportedAbis.firstOrNull().orEmpty())
            .put("native_abi_candidates", JSONArray(HermesAndroidHardwareProfile.nativeAbiCandidates(supportedAbis)))
            .put("supported_abis", JSONArray(supportedAbis))
            .put("supported_64_bit_abis", JSONArray(supported64BitAbis))
            .put("supports_64_bit_abi", supported64BitAbis.isNotEmpty())
            .put("supports_arm64", supportedAbis.any { it.contains("arm64", ignoreCase = true) })
            .put("supports_arm", supportedAbis.any { HermesAndroidHardwareProfile.isArmAbi(it) })
            .put("supports_x86_64", supportedAbis.any { it.contains("x86_64", ignoreCase = true) })
            .put("supports_x86", supportedAbis.any { HermesAndroidHardwareProfile.isX86Abi(it) })
            .put("likely_mediatek", isLikelyMediatekSoc(values))
            .put("likely_snapdragon", isLikelySnapdragonSoc(values))
            .put("likely_google_tensor", HermesAndroidHardwareProfile.isLikelyGoogleTensorSoc(values))
            .put("likely_exynos", HermesAndroidHardwareProfile.isLikelyExynosSoc(values))
            .put("likely_unisoc", HermesAndroidHardwareProfile.isLikelyUnisocSoc(values))
            .put("likely_adreno_gpu", hardwareProfile.gpuFamily == "adreno")
            .put("likely_mali_gpu", hardwareProfile.gpuFamily == "mali" || hardwareProfile.gpuFamily == "mali_immortalis")
            .put("likely_powervr_img_gpu", hardwareProfile.gpuFamily == "powervr_img")
            .put("likely_xclipse_gpu", hardwareProfile.gpuFamily == "xclipse")
            .put("litert_lm_acceleration_label", HermesAndroidHardwareProfile.accelerationLabel(hardwareProfile))
            .put("litert_lm_backend_strategy", "GPU-first on ARM devices when LiteRT-LM accepts the accelerator, then CPU fallback; CPU-only on x86 emulator/device builds.")
            .put("compatibility_strategy", "Use Android SDK feature, permission, sensor, Wi-Fi, Bluetooth, camera, and storage APIs; avoid Adreno-only or Snapdragon-only assumptions.")
            .put("native_abi_strategy", HermesAndroidHardwareProfile.nativeAbiStrategy(supportedAbis))
    }

    internal fun isLikelyMediatekSoc(values: List<String>): Boolean {
        return HermesAndroidHardwareProfile.isLikelyMediatekSoc(values)
    }

    internal fun isLikelySnapdragonSoc(values: List<String>): Boolean {
        return HermesAndroidHardwareProfile.isLikelySnapdragonSoc(values)
    }

    private fun wifiBandLabel(frequencyMhz: Int): String = when (frequencyMhz) {
        in 2400..2500 -> "2.4GHz"
        in 4900..5900 -> "5GHz"
        in 5925..7125 -> "6GHz"
        in 57000..71000 -> "60GHz"
        else -> "unknown"
    }

    private fun canonicalWifiBandLabel(label: String, frequencyMhz: Int = 0): String {
        if (frequencyMhz > 0) return wifiBandLabel(frequencyMhz)
        val normalized = label.trim().lowercase(Locale.US).replace(" ", "")
        return when {
            normalized.contains("2.4") || normalized == "24ghz" -> "2.4GHz"
            normalized.startsWith("5") -> "5GHz"
            normalized.startsWith("6") -> "6GHz"
            normalized.startsWith("60") -> "60GHz"
            else -> "unknown"
        }
    }

    private fun candidateWifiChannelsForBand(band: String, observedChannels: List<Int>): List<Int> {
        val observed = observedChannels.filter { it > 0 }
        val base = when (band) {
            "2.4GHz" -> listOf(1, 6, 11)
            "5GHz" -> listOf(36, 40, 44, 48, 149, 153, 157, 161)
            else -> emptyList()
        }
        return (base + observed)
            .distinct()
            .sorted()
            .take(MAX_WIFI_CANDIDATE_CHANNELS_PER_BAND)
    }

    private fun wifiChannelRatingRow(
        band: String,
        candidateChannel: Int,
        measurements: List<WifiChannelMeasurement>,
    ): JSONObject {
        var sameChannelCount = 0
        var overlapCount = 0
        var strongestRssi: Int? = null
        var interference = 0.0
        measurements.forEach { measurement ->
            val overlap = wifiChannelOverlapWeight(band, candidateChannel, measurement.channel)
            if (overlap <= 0.0) return@forEach
            overlapCount += 1
            if (measurement.channel == candidateChannel) sameChannelCount += 1
            strongestRssi = maxOf(strongestRssi ?: measurement.rssiDbm, measurement.rssiDbm)
            val signalWeight = ((measurement.rssiDbm + 100).coerceIn(0, 70)) / 70.0
            interference += signalWeight * overlap * wifiWidthWeight(measurement.widthLabel) * 34.0
        }
        val score = (100.0 - interference - (sameChannelCount * 8.0)).roundToInt().coerceIn(0, 100)
        return JSONObject()
            .put("band", band)
            .put("channel", candidateChannel)
            .put("frequency_hint_mhz", frequencyHintMhzForWifiChannel(band, candidateChannel) ?: JSONObject.NULL)
            .put("score", score)
            .put("rating_label", wifiRatingLabel(score))
            .put("network_count", sameChannelCount)
            .put("overlap_count", overlapCount)
            .put("strongest_rssi_dbm", strongestRssi ?: JSONObject.NULL)
            .put("recommendation", wifiChannelRecommendation(score, sameChannelCount, overlapCount, strongestRssi))
    }

    private fun wifiChannelOverlapWeight(band: String, candidateChannel: Int, observedChannel: Int): Double {
        val distance = abs(candidateChannel - observedChannel)
        return when (band) {
            "2.4GHz" -> if (distance >= 5) 0.0 else (5 - distance) / 5.0
            "5GHz", "6GHz" -> if (distance == 0) 1.0 else 0.0
            else -> if (distance == 0) 1.0 else 0.0
        }
    }

    private fun wifiWidthWeight(widthLabel: String): Double {
        val normalized = widthLabel.lowercase(Locale.US)
        return when {
            "320" in normalized -> 2.4
            "160" in normalized -> 2.0
            "80+80" in normalized -> 2.0
            "80" in normalized -> 1.6
            "40" in normalized -> 1.25
            else -> 1.0
        }
    }

    private fun frequencyHintMhzForWifiChannel(band: String, channel: Int): Int? {
        return when (band) {
            "2.4GHz" -> if (channel == 14) 2484 else 2407 + (channel * 5)
            "5GHz" -> 5000 + (channel * 5)
            "6GHz" -> 5950 + (channel * 5)
            else -> null
        }
    }

    private fun wifiRatingLabel(score: Int): String = when {
        score >= 85 -> "excellent"
        score >= 70 -> "good"
        score >= 50 -> "fair"
        else -> "congested"
    }

    private fun wifiChannelRecommendation(
        score: Int,
        sameChannelCount: Int,
        overlapCount: Int,
        strongestRssi: Int?,
    ): String {
        return when {
            overlapCount == 0 -> "Best current option: no overlapping APs in the latest scan."
            score >= 85 -> "Best current option: low overlap and weak competing signals."
            score >= 70 -> "Usable channel: some nearby overlap, but congestion is moderate."
            sameChannelCount > 2 || (strongestRssi ?: -100) > -55 -> "Congested channel: strong or repeated AP overlap detected."
            else -> "Crowded channel: prefer a higher-scored recommendation when available."
        }
    }

    private fun recommendedWifiChannels(ratings: JSONArray): JSONArray {
        val byBand = linkedMapOf<String, JSONObject>()
        for (index in 0 until ratings.length()) {
            val row = ratings.optJSONObject(index) ?: continue
            val band = row.optString("band")
            val current = byBand[band]
            if (current == null || row.optInt("score") > current.optInt("score")) {
                byBand[band] = row
            }
        }
        return JSONArray().also { array ->
            byBand.values
                .sortedBy { wifiBandSortKey(it.optString("band")) }
                .forEach { row ->
                    array.put(
                        JSONObject()
                            .put("band", row.optString("band"))
                            .put("channel", row.optInt("channel"))
                            .put("score", row.optInt("score"))
                            .put("rating_label", row.optString("rating_label"))
                            .put("recommendation", row.optString("recommendation")),
                    )
                }
        }
    }

    private fun wifiBandSummaryJson(networks: JSONArray, ratings: JSONArray): JSONArray {
        val networkCounts = linkedMapOf<String, Int>()
        for (index in 0 until networks.length()) {
            val band = canonicalWifiBandLabel(networks.optJSONObject(index)?.optString("band").orEmpty())
            if (band != "unknown") networkCounts[band] = (networkCounts[band] ?: 0) + 1
        }
        val ratingCounts = linkedMapOf<String, Int>()
        val bestRows = linkedMapOf<String, JSONObject>()
        for (index in 0 until ratings.length()) {
            val row = ratings.optJSONObject(index) ?: continue
            val band = row.optString("band")
            ratingCounts[band] = (ratingCounts[band] ?: 0) + 1
            val best = bestRows[band]
            if (best == null || row.optInt("score") > best.optInt("score")) bestRows[band] = row
        }
        val bands = (networkCounts.keys + ratingCounts.keys)
            .distinct()
            .sortedBy(::wifiBandSortKey)
        return JSONArray().also { array ->
            bands.forEach { band ->
                val best = bestRows[band]
                array.put(
                    JSONObject()
                        .put("band", band)
                        .put("network_count", networkCounts[band] ?: 0)
                        .put("rated_channel_count", ratingCounts[band] ?: 0)
                        .put("recommended_channel", best?.optInt("channel") ?: JSONObject.NULL)
                        .put("recommended_score", best?.optInt("score") ?: JSONObject.NULL)
                        .put("recommendation", best?.optString("recommendation").orEmpty()),
                )
            }
        }
    }

    private fun wifiBandSortKey(band: String): Int = when (band) {
        "2.4GHz" -> 0
        "5GHz" -> 1
        "6GHz" -> 2
        "60GHz" -> 3
        else -> 4
    }

    internal fun wifiBssidOui(bssid: String): String {
        val hex = bssid.filter { it.isDigit() || it.lowercaseChar() in 'a'..'f' }.uppercase(Locale.US)
        if (hex.length < 6) return ""
        return hex.take(6).chunked(2).joinToString(":")
    }

    internal fun wifiOuiVendorLabel(oui: String): String {
        val normalized = wifiBssidOui(oui)
        if (normalized.isBlank()) return "Unknown vendor"
        if (isLocallyAdministeredOui(normalized)) return "Locally administered / randomized"
        return WIFI_OUI_VENDOR_HINTS[normalized] ?: "Unknown vendor"
    }

    internal fun wifiSecurityLabel(capabilities: String): String {
        val normalized = capabilities.uppercase(Locale.US)
        return when {
            normalized.isBlank() -> "Open"
            "SAE" in normalized || "WPA3" in normalized -> "WPA3"
            "OWE" in normalized -> "Enhanced Open"
            "WPA2" in normalized || "RSN" in normalized -> "WPA2"
            "WPA" in normalized -> "WPA"
            "WEP" in normalized -> "WEP"
            else -> "Open"
        }
    }

    internal fun wifiSignalQualityLabel(rssiDbm: Int): String = when {
        rssiDbm >= -50 -> "excellent"
        rssiDbm >= -60 -> "good"
        rssiDbm >= -70 -> "fair"
        else -> "weak"
    }

    private fun isLocallyAdministeredOui(oui: String): Boolean {
        val firstOctet = oui.substringBefore(':').toIntOrNull(16) ?: return false
        return firstOctet and 0x02 != 0
    }

    private fun filterFacetJson(
        key: String,
        label: String,
        counts: Map<String, Int>,
        sortKey: (String) -> Int,
    ): JSONObject {
        val options = JSONArray()
        counts.entries
            .sortedWith(
                compareBy<Map.Entry<String, Int>> { sortKey(it.key) }
                    .thenByDescending { it.value }
                    .thenBy { it.key.lowercase(Locale.US) },
            )
            .take(MAX_WIFI_FILTER_OPTIONS)
            .forEach { entry ->
                options.put(
                    JSONObject()
                        .put("value", entry.key)
                        .put("count", entry.value),
                )
            }
        return JSONObject()
            .put("key", key)
            .put("label", label)
            .put("option_count", counts.size)
            .put("options", options)
    }

    private fun incrementCount(counts: MutableMap<String, Int>, key: String) {
        counts[key] = (counts[key] ?: 0) + 1
    }

    private fun updateWifiSignalHistory(context: Context, networks: JSONArray, observedAtMs: Long): JSONObject {
        val prefs = context.getSharedPreferences(WIFI_SIGNAL_HISTORY_PREFS, Context.MODE_PRIVATE)
        val existing = runCatching {
            JSONObject(prefs.getString(WIFI_SIGNAL_HISTORY_KEY, "{}").orEmpty().ifBlank { "{}" })
        }.getOrDefault(JSONObject())
        val updated = mergeWifiSignalHistory(existing, networks, observedAtMs)
        prefs.edit().putString(WIFI_SIGNAL_HISTORY_KEY, updated.toString()).apply()
        return updated
    }

    private fun wifiHistoryKey(row: JSONObject): String {
        val bssid = row.optString("bssid").trim().lowercase(Locale.US)
        if (bssid.isNotBlank()) return bssid
        val ssid = row.optString("ssid").trim().lowercase(Locale.US)
        val frequency = row.optInt("frequency_mhz", 0)
        val channel = row.opt("channel")?.toString().orEmpty()
        return listOf(ssid, frequency.toString(), channel).joinToString("|").takeIf { ssid.isNotBlank() }.orEmpty()
    }

    private fun trimWifiObservations(observations: JSONArray): JSONArray {
        val start = (observations.length() - MAX_WIFI_HISTORY_SAMPLES_PER_NETWORK).coerceAtLeast(0)
        val trimmed = JSONArray()
        for (index in start until observations.length()) {
            observations.optJSONObject(index)?.let(trimmed::put)
        }
        return trimmed
    }

    private fun lastWifiObservationTime(record: JSONObject): Long {
        val observations = record.optJSONArray("observations") ?: return 0L
        for (index in observations.length() - 1 downTo 0) {
            val time = observations.optJSONObject(index)?.optLong("observed_at_ms", 0L) ?: 0L
            if (time > 0L) return time
        }
        return 0L
    }

    private fun currentWifiRssi(record: JSONObject): Int? {
        val observations = record.optJSONArray("observations") ?: return null
        for (index in observations.length() - 1 downTo 0) {
            jsonIntOrNull(observations.optJSONObject(index) ?: continue, "rssi_dbm")?.let { return it }
        }
        return null
    }

    private fun wifiSignalTrendLabel(trendDb: Int): String = when {
        trendDb >= 5 -> "improving"
        trendDb <= -5 -> "fading"
        else -> "stable"
    }

    private fun wifiObservationSeries(observations: JSONArray): JSONArray {
        val series = JSONArray()
        val start = (observations.length() - MAX_WIFI_HISTORY_SERIES_POINTS).coerceAtLeast(0)
        for (index in start until observations.length()) {
            val observation = observations.optJSONObject(index) ?: continue
            series.put(
                JSONObject()
                    .put("observed_at_ms", observation.optLong("observed_at_ms", 0L))
                    .put("rssi_dbm", jsonIntOrNull(observation, "rssi_dbm") ?: JSONObject.NULL),
            )
        }
        return series
    }

    private fun jsonIntOrNull(row: JSONObject, key: String): Int? {
        return when (val value = row.opt(key)) {
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }
    }

    private fun jsonLongOrNull(row: JSONObject, key: String): Long? {
        return when (val value = row.opt(key)) {
            is Number -> value.toLong()
            is String -> value.toLongOrNull()
            else -> null
        }
    }

    private fun jsonDoubleOrNull(row: JSONObject, key: String): Double? {
        return when (val value = row.opt(key)) {
            is Number -> value.toDouble()
            is String -> value.toDoubleOrNull()
            else -> null
        }
    }

    private fun jsonValueOrNull(row: JSONObject, key: String): Any {
        return row.opt(key)?.takeUnless { it == JSONObject.NULL } ?: JSONObject.NULL
    }

    private fun appendWifiSummaryContext(accumulator: WifiSummaryAccumulator, row: JSONObject) {
        jsonIntOrNull(row, "rssi_dbm")?.let { rssi ->
            accumulator.strongestRssiDbm = maxOf(accumulator.strongestRssiDbm ?: rssi, rssi)
        }
        val frequencyMhz = jsonIntOrNull(row, "frequency_mhz") ?: 0
        canonicalWifiBandLabel(row.optString("band"), frequencyMhz)
            .takeIf { it != "unknown" }
            ?.let(accumulator.bands::add)
        val channel = jsonIntOrNull(row, "channel") ?: channelForFrequencyMhz(frequencyMhz)
        channel?.let { accumulator.channels.add(it.toString()) }
        row.optString("channel_width").takeIf { it.isNotBlank() }?.let(accumulator.widths::add)
        row.optString("display_ssid")
            .ifBlank { row.optString("ssid") }
            .ifBlank { "<hidden>" }
            .let(accumulator.sampleSsids::add)
    }

    private fun latestWifiScanAgeMs(scanResults: List<ScanResult>): Long? {
        val latestTimestampMicros = scanResults
            .map { it.timestamp }
            .filter { it > 0L }
            .maxOrNull()
            ?: return null
        return wifiScanAgeMs(latestTimestampMicros)
    }

    private fun wifiScanAgeMs(timestampMicros: Long): Long? {
        if (timestampMicros <= 0L) return null
        val nowMicros = SystemClock.elapsedRealtimeNanos() / 1_000L
        return ((nowMicros - timestampMicros) / 1_000L).coerceAtLeast(0L)
    }

    private fun wifiScanStatusJson(
        refreshRequested: Boolean,
        refreshAccepted: Boolean,
        wifiEnabled: Boolean,
        permissionStatus: JSONObject,
        totalScanResultCount: Int,
        returnedNetworkCount: Int,
        latestScanAgeMs: Long?,
    ): JSONObject {
        return JSONObject()
            .put("refresh_requested", refreshRequested)
            .put("refresh_accepted", refreshAccepted)
            .put("wifi_enabled", wifiEnabled)
            .put("scan_permission_ready", permissionStatus.optBoolean("can_read_scan_results", false))
            .put("location_enabled", permissionStatus.optBoolean("location_enabled", false))
            .put("scan_result_count", totalScanResultCount)
            .put("returned_network_count", returnedNetworkCount)
            .put("latest_scan_age_ms", latestScanAgeMs ?: JSONObject.NULL)
            .put(
                "android_scan_throttle_note",
                if (refreshRequested && !refreshAccepted) {
                    "Android did not accept the active refresh request; cached scan results may still be current enough for analysis."
                } else {
                    "Android may throttle active Wi-Fi scans; Hermes reports scan age when Android exposes timestamps."
                },
            )
    }

    private fun normalizedWifiExportFormat(format: String): String {
        return when (format.trim().lowercase(Locale.US)) {
            "csv" -> "csv"
            "both", "json+csv", "csv+json" -> "both"
            else -> "json"
        }
    }

    private fun csvEscape(value: Any?): String {
        val text = when (value) {
            null, JSONObject.NULL -> ""
            is JSONArray, is JSONObject -> value.toString()
            else -> value.toString()
        }
        val escaped = text.replace("\"", "\"\"")
        return if (escaped.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"$escaped\""
        } else {
            escaped
        }
    }

    private fun jsonStringList(row: JSONObject, key: String): List<String> {
        val array = row.optJSONArray(key) ?: return emptyList()
        return buildList {
            for (index in 0 until array.length()) {
                array.optString(index).takeIf { it.isNotBlank() }?.let(::add)
            }
        }
    }

    private fun wifiSignalSortKey(value: String): Int = when (value) {
        "excellent" -> 0
        "good" -> 1
        "fair" -> 2
        "weak" -> 3
        else -> 4
    }

    private fun wifiSecuritySortKey(value: String): Int = when (value) {
        "WPA3" -> 0
        "WPA2" -> 1
        "WPA" -> 2
        "Enhanced Open" -> 3
        "WEP" -> 4
        "Open" -> 5
        else -> 6
    }

    private fun wifiStandardSortKey(value: String): Int = when (value) {
        "802.11be" -> 0
        "802.11ax" -> 1
        "802.11ac" -> 2
        "802.11n" -> 3
        "802.11ad" -> 4
        "legacy" -> 5
        else -> 6
    }

    private fun channelWidthMhz(widthLabel: String): Int? {
        val normalized = widthLabel.lowercase(Locale.US).replace(" ", "")
        return when {
            "320" in normalized -> 320
            "160" in normalized -> 160
            "80+80" in normalized -> 160
            "80" in normalized -> 80
            "40" in normalized -> 40
            "20" in normalized -> 20
            else -> null
        }
    }

    private fun wifiSecurityRecommendation(security: String, networkCount: Int, strongestRssiDbm: Int?): String {
        return when (security) {
            "Open" -> "Open AP group: avoid sensitive work unless captive portal or VPN policy is explicitly acceptable."
            "WEP" -> "Legacy WEP AP group: treat as weak security even when signal is strong."
            "WPA3" -> "WPA3 AP group: prefer when channel score and signal are also healthy."
            "WPA2" -> "WPA2 AP group: broadly compatible; compare against WPA3 rows and channel congestion."
            else -> if ((strongestRssiDbm ?: -100) > -55 && networkCount > 1) {
                "Strong nearby security group: inspect SSID/BSSID details for channel and roaming behavior."
            } else {
                "Security group detected from Android capability metadata."
            }
        }
    }

    private fun wifiChannelWidthRecommendation(width: String, networkCount: Int, strongestRssiDbm: Int?): String {
        val widthMhz = channelWidthMhz(width) ?: return "Unknown width: Android did not expose enough channel-width metadata for this group."
        return when {
            widthMhz >= 160 -> "Very wide channel group: high throughput potential, but strong neighbors can create wide-band contention."
            widthMhz >= 80 && (strongestRssiDbm ?: -100) > -60 -> "Wide channel group with strong APs nearby: compare channel ratings before choosing placement."
            widthMhz >= 40 -> "Moderate-width channel group: useful throughput signal; watch overlap on crowded 2.4/5GHz channels."
            networkCount > 4 -> "Narrow channel group is crowded; prefer the highest-rated recommended channel."
            else -> "Narrow channel group with limited contention in the latest scan."
        }
    }

    private fun wifiStandardRecommendation(standard: String, networkCount: Int): String {
        return when (standard) {
            "802.11be" -> "Wi-Fi 7 metadata detected; verify client/router support and channel width before assuming throughput."
            "802.11ax" -> "Wi-Fi 6/6E metadata detected; compare against 6GHz availability and channel ratings."
            "802.11ac" -> "Wi-Fi 5 metadata detected; useful 5GHz baseline when channel congestion is acceptable."
            "802.11n" -> "Wi-Fi 4 metadata detected; prefer newer standard groups when signal and security are comparable."
            "legacy" -> "Legacy standard group: inspect security and channel details before relying on it."
            else -> "Android did not expose a Wi-Fi standard for this group; keep SSID/BSSID/channel metadata visible."
        }.let { recommendation ->
            if (networkCount > 1) "$recommendation $networkCount APs share this standard group." else recommendation
        }
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
        ScanResult.CHANNEL_WIDTH_320MHZ -> "320MHz"
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

    internal data class WifiChannelMeasurement(
        val channel: Int,
        val band: String,
        val rssiDbm: Int,
        val widthLabel: String = "20MHz",
        val frequencyMhz: Int = 0,
    )

    private data class WifiSummaryAccumulator(
        var networkCount: Int = 0,
        var strongestRssiDbm: Int? = null,
        val bands: LinkedHashSet<String> = linkedSetOf(),
        val channels: LinkedHashSet<String> = linkedSetOf(),
        val widths: LinkedHashSet<String> = linkedSetOf(),
        val sampleSsids: LinkedHashSet<String> = linkedSetOf(),
    )

    private val ACTIONS = listOf(
        "status",
        "top_apps",
        "wifi_scan",
        "wifi_channel_rating",
        "wifi_ap_details",
        "wifi_export",
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
    private val MOTION_SENSOR_TYPES = setOf("accelerometer", "gyroscope", "gravity", "linear_acceleration", "rotation_vector")
    private val DEFAULT_SENSOR_TYPES = listOf("accelerometer", "gyroscope", "magnetic_field", "light", "proximity")
    private const val DEFAULT_LIMIT = 5
    private const val MAX_LIMIT = 20
    private const val MAX_WIFI_RESULTS = 40
    private const val MAX_WIFI_CHANNEL_RATINGS = 24
    private const val MAX_WIFI_CANDIDATE_CHANNELS_PER_BAND = 12
    private const val MAX_WIFI_VENDOR_ROWS = 16
    private const val MAX_WIFI_VENDOR_DETAILS = 4
    private const val MAX_WIFI_FILTER_OPTIONS = 12
    private const val MAX_WIFI_SUMMARY_SAMPLES = 6
    private const val MAX_WIFI_HISTORY_NETWORKS_PER_SCAN = 40
    private const val MAX_WIFI_HISTORY_NETWORKS = 40
    private const val MAX_WIFI_HISTORY_ROWS = 16
    private const val MAX_WIFI_HISTORY_SAMPLES_PER_NETWORK = 12
    private const val MAX_WIFI_HISTORY_SERIES_POINTS = 8
    private const val WIFI_SIGNAL_HISTORY_PREFS = "hermes_wifi_signal_history"
    private const val WIFI_SIGNAL_HISTORY_KEY = "signal_history"
    private const val MAX_BLUETOOTH_RESULTS = 40
    private const val MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE = 8
    private const val MAX_BLUETOOTH_MANUFACTURER_IDS_PER_DEVICE = 8
    private const val MAX_BLUETOOTH_METADATA_SUMMARY_ROWS = 24
    private const val MAX_BLUETOOTH_SUMMARY_SAMPLES = 4
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
    private val WIFI_AP_EXPORT_FIELDS = listOf(
        "rank",
        "display_ssid",
        "hidden_ssid",
        "bssid",
        "bssid_oui",
        "bssid_vendor",
        "rssi_dbm",
        "signal_quality",
        "frequency_mhz",
        "channel",
        "band",
        "channel_width",
        "channel_width_mhz",
        "wifi_standard",
        "security_mode",
        "estimated_distance_m",
        "passpoint_network",
        "80211mc_responder",
    )
    private val TIKTOK_PACKAGES = listOf("com.zhiliaoapp.musically", "com.ss.android.ugc.trill")
    private const val INSTAGRAM_PACKAGE = "com.instagram.android"
    private const val GMAIL_PACKAGE = "com.google.android.gm"
    private val WIFI_OUI_VENDOR_HINTS = mapOf(
        "00:1A:A1" to "Cisco",
        "00:1B:54" to "Cisco",
        "00:25:9C" to "Cisco",
        "00:E0:FC" to "Huawei",
        "14:EB:B6" to "TP-Link",
        "20:4E:7F" to "Netgear",
        "24:A4:3C" to "Ubiquiti",
        "28:6C:07" to "Xiaomi",
        "3C:22:FB" to "Apple",
        "50:C7:BF" to "TP-Link",
        "60:F8:1D" to "Apple",
        "64:09:80" to "Xiaomi",
        "78:8A:20" to "Ubiquiti",
        "A0:04:60" to "Netgear",
        "A0:CC:2B" to "Samsung",
        "A0:F3:C1" to "TP-Link",
        "A4:5E:60" to "Apple",
        "AC:BC:32" to "Apple",
        "B8:27:EB" to "Raspberry Pi",
        "BC:52:B7" to "Apple",
        "C4:04:15" to "Netgear",
        "D4:25:8B" to "Xiaomi",
        "D8:BB:2C" to "Apple",
        "DC:A6:32" to "Raspberry Pi",
        "E0:63:DA" to "Ubiquiti",
        "E8:50:8B" to "Samsung",
        "E8:CD:2D" to "Huawei",
        "F0:18:98" to "Apple",
        "F4:7B:5E" to "Samsung",
        "F4:F2:6D" to "TP-Link",
        "F8:7B:20" to "Cisco",
        "F8:E8:11" to "Huawei",
    )
}
