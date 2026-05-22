package com.mobilefork.hermesagent.device

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
import android.content.IntentFilter
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
import android.os.BatteryManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.os.storage.StorageManager
import android.provider.Settings
import android.text.format.Formatter
import androidx.core.content.ContextCompat
import com.mobilefork.hermesagent.backend.BackendKind
import com.mobilefork.hermesagent.backend.LiteRtLmOpenAiProxy
import com.mobilefork.hermesagent.backend.LocalBackendStatus
import com.mobilefork.hermesagent.backend.OnDeviceBackendManager
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.LocalModelDownloadStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

object HermesDeviceDiagnosticsBridge {
    fun performActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        val appContext = context.applicationContext
        return when (action.lowercase(Locale.US).ifBlank { "status" }) {
            "status", "diagnostics_status", "device_diagnostics_status" -> statusJson(appContext).toString()
            "top_apps", "top_resource_apps", "top_memory_apps", "top_storage_apps", "resource_apps" ->
                topAppsJson(appContext, arguments).toString()
            "wifi_scan", "wifi_analyzer", "scan_wifi", "nearby_wifi", "wifi_signals",
            "wifi_filtered_scan", "wifi_filter", "wifi_filtered_ap_details" ->
                wifiScanJson(appContext, arguments).toString()
            "wifi_analyzer_report", "wifi_readiness_report", "wifi_feature_report", "wifi_scan_policy" ->
                wifiAnalyzerReportJson(appContext, arguments).toString()
            "wifi_channel_graph", "wifi_graph", "channel_graph", "wifi_signal_channel_graph" ->
                wifiScanJson(appContext, arguments, "wifi_channel_graph").toString()
            "wifi_channel_rating", "wifi_channels", "channel_rating", "best_wifi_channel", "wifi_congestion" ->
                wifiScanJson(appContext, arguments, "wifi_channel_rating").toString()
            "wifi_channel_utilization", "wifi_utilization", "wifi_spectrum", "wifi_interference", "wifi_band_occupancy" ->
                wifiScanJson(appContext, arguments, "wifi_channel_utilization").toString()
            "wifi_ap_details", "wifi_access_points", "wifi_access_point_details", "wifi_report" ->
                wifiScanJson(appContext, arguments, "wifi_ap_details").toString()
            "wifi_export", "wifi_analyzer_export", "wifi_access_point_export", "export_wifi" ->
                wifiScanJson(appContext, arguments, "wifi_export").toString()
            "bluetooth_analyzer_report", "bluetooth_readiness_report", "bluetooth_feature_report", "bluetooth_scan_policy", "nearby_bluetooth_report" ->
                bluetoothAnalyzerReportJson(appContext, arguments).toString()
            "bluetooth_signal_history", "bluetooth_history", "bluetooth_rssi_history", "bluetooth_trends", "bluetooth_trend" ->
                bluetoothScanJson(appContext, arguments, "bluetooth_signal_history").toString()
            "bluetooth_scan", "bluetooth_scanner", "nearby_bluetooth", "ble_scan", "bluetooth_signals" ->
                bluetoothScanJson(appContext, arguments).toString()
            "sensor_analyzer_report", "sensor_readiness_report", "sensor_feature_report", "sensor_sampling_policy", "motion_sensor_report" ->
                sensorAnalyzerReportJson(appContext, arguments).toString()
            "motion_sensor_history", "motion_history", "sensor_history", "imu_history", "imu_sensor_history", "accelerometer_history", "gyroscope_history", "sensor_trends", "motion_sensor_trends" ->
                motionSensorHistoryJson(appContext, arguments).toString()
            "motion_pose", "orientation_snapshot", "pose_snapshot", "motion_orientation" ->
                sensorSnapshotJson(appContext, motionPoseDefaultArguments(arguments)).toString()
            "sensor_snapshot", "sensors", "sensor_status", "sample_sensors", "motion_sensors" ->
                sensorSnapshotJson(appContext, arguments).toString()
            "camera_status", "camera", "camera_capabilities" -> cameraStatusJson(appContext).toString()
            "radio_signal_graph", "radio_graph", "am_fm_signal_graph", "am_fm_radio_graph", "broadcast_radio_graph",
            "radio_band_graph" -> radioSignalGraphJson(appContext, arguments).toString()
            "radio_signal_status", "radio_scan", "am_fm_radio_status", "am_radio", "fm_radio",
            "radio_analyzer_report", "broadcast_radio_report", "radio_feature_report", "radio_band_plan" ->
                radioSignalStatusJson(appContext).toString()
            "signal_capability_status", "rf_capabilities", "radio_status", "microwave_status" ->
                signalCapabilityStatusJson(appContext).toString()
            "local_backend_runtime_report", "runtime_backend_report", "backend_runtime_report", "litert_runtime_report", "model_backend_report" ->
                localBackendRuntimeReportJson(appContext).toString()
            "soc_compatibility_report", "soc_backend_report", "mediatek_compatibility_report", "litert_backend_report", "gpu_backend_report" ->
                socCompatibilityReportJson(appContext).toString()
            "gpu_backend_risk_report", "backend_risk_report", "accelerator_risk_report", "mediatek_backend_risk_report", "non_adreno_backend_risk_report" ->
                gpuBackendRiskReportJson(appContext).toString()
            "local_inference_compatibility_report", "mediatek_inference_compatibility_report", "non_adreno_compatibility_report", "local_model_compatibility_scorecard" ->
                localInferenceCompatibilityReportJson(appContext).toString()
            "device_performance_report", "thermal_status_report", "runtime_stability_report", "mediatek_stability_report", "power_stability_report" ->
                devicePerformanceReportJson(appContext).toString()
            "signal_awareness_report", "nearby_signal_report", "rf_sensor_fusion_report", "ambient_context_report" ->
                signalAwarenessReportJson(appContext).toString()
            "agent_signal_evidence_report", "signal_evidence_bundle", "current_signal_evidence", "gemma_signal_evidence" ->
                agentSignalEvidenceReportJson(appContext).toString()
            "agent_observation_report", "agent_signal_dashboard", "gemma_observation_report", "multimodal_signal_dashboard" ->
                agentObservationReportJson(appContext).toString()
            "agent_card_manifest_report", "card_manifest_report", "diagnostic_card_manifest", "graph_card_manifest" ->
                agentCardManifestReportJson(appContext).toString()
            "agent_environment_report", "environment_report", "capability_matrix", "system_capability_report", "kai_parity_report" ->
                agentEnvironmentReportJson(appContext).toString()
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
        val performanceProfile = devicePerformanceProfileJson(appContext)
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
            .put("selected_on_device_backend", BackendKind.fromPersistedValue(AppSettingsStore(appContext).load().onDeviceBackend).persistedValue)
            .put("current_local_backend", localBackendStatusJson(OnDeviceBackendManager.currentStatus()))
            .put("litert_runtime_health", liteRtRuntimeHealthJson())
            .put("soc_profile", socProfileJson())
            .put("device_performance_profile", performanceProfile)
            .put("thermal_status_label", performanceProfile.optString("thermal_status_label"))
            .put("power_save_mode", performanceProfile.optBoolean("power_save_mode", false))
            .put("low_ram_device", performanceProfile.optBoolean("low_ram_device", false))
            .put("media_performance_class", performanceProfile.optInt("media_performance_class", 0))
            .put("location_enabled", isLocationEnabled(appContext))
            .put("sensor_count", sensorManager?.getSensorList(Sensor.TYPE_ALL)?.size ?: 0)
            .put("available_sensor_types", JSONArray(sensorTypeCatalog(appContext)))
            .put("overlay", JSONObject(HermesOverlaySceneBridge.statusJson(appContext)))
            .put("available_actions", JSONArray(ACTIONS))
            .put(
                "cards",
                JSONArray()
                    .put(card("Diagnostics", "Phone resource, Wi-Fi vendor/OUI/channel, Bluetooth service labels/manufacturer names/proximity, camera, sensor, SOC, and overlay diagnostics are available to the agent."))
                    .put(card("Radio Signals", "AM/FM and broad RF scanning require vendor radio APIs or external SDR hardware; Android phones expose Wi-Fi, Bluetooth, audio, camera, and built-in sensors.")),
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
        val requestedRefresh = arguments.optBoolean("refresh", false)
        val scanMode = normalizedWifiScanMode(arguments)
        val refresh = effectiveWifiRefreshRequested(requestedRefresh, scanMode)
        val wifiManager = appContext.getSystemService(WifiManager::class.java)
        val permissionStatus = wifiPermissionStatusJson(appContext)
        val canReadScan = permissionStatus.optBoolean("can_read_scan_results", false)
        if (wifiManager == null) {
            return JSONObject()
                .put("success", false)
                .put("action", actionName)
                .put("error", "Wi-Fi service is unavailable on this device")
                .put("wifi_scan_permission_status", permissionStatus)
                .put("wifi_scan_control", wifiScanControlJson(scanMode, requestedRefresh, refresh, false))
                .put(
                    "wifi_scan_status",
                    wifiScanStatusJson(
                        refreshRequested = refresh,
                        refreshAccepted = false,
                        wifiEnabled = false,
                        permissionStatus = permissionStatus,
                        totalScanResultCount = 0,
                        returnedNetworkCount = 0,
                        latestScanAgeMs = null,
                        scanMode = scanMode,
                        userRefreshRequested = requestedRefresh,
                    ),
                )
        }
        if (!canReadScan) {
            return JSONObject()
                .put("success", false)
                .put("action", actionName)
                .put("error", "Wi-Fi scan results require nearby Wi-Fi/location permissions and location services on supported Android versions")
                .put("wifi_scan_permission_status", permissionStatus)
                .put("wifi_scan_control", wifiScanControlJson(scanMode, requestedRefresh, refresh, false))
                .put(
                    "wifi_scan_status",
                    wifiScanStatusJson(
                        refreshRequested = refresh,
                        refreshAccepted = false,
                        wifiEnabled = wifiManager.isWifiEnabled,
                        permissionStatus = permissionStatus,
                        totalScanResultCount = 0,
                        returnedNetworkCount = 0,
                        latestScanAgeMs = null,
                        scanMode = scanMode,
                        userRefreshRequested = requestedRefresh,
                    ),
                )
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
        val filterSpec = wifiScanFilterSpec(arguments)
        val filteredNetworks = wifiFilteredNetworkRows(allNetworks, filterSpec)
        val analysisNetworks = if (filterSpec.active) filteredNetworks else allNetworks
        val filterRows = wifiFilterApplicationRows(filterSpec, allNetworks.length(), analysisNetworks.length())
        val filterSummary = wifiFilterSummaryJson(filterSpec, allNetworks.length(), analysisNetworks.length())
        val networks = JSONArray()
        for (index in 0 until minOf(limit, analysisNetworks.length())) {
            networks.put(analysisNetworks.getJSONObject(index))
        }
        val channelRatings = wifiChannelRatingRowsForNetworks(analysisNetworks)
        val channelUtilization = wifiChannelUtilizationRowsForNetworks(analysisNetworks)
        val recommendedChannels = recommendedWifiChannels(channelRatings)
        val bandSummary = wifiBandSummaryJson(analysisNetworks, channelRatings)
        val vendorSummary = wifiVendorSummaryJson(analysisNetworks)
        val availableAnalyzerFilters = wifiAnalyzerFilterSummaryJson(allNetworks)
        val filteredAnalyzerFilters = wifiAnalyzerFilterSummaryJson(analysisNetworks)
        val detailLimitDefault = if (actionName == "wifi_export" || actionName == "wifi_ap_details") MAX_WIFI_RESULTS else limit
        val detailLimit = arguments.optInt(
            "detail_limit",
            arguments.optInt("export_limit", detailLimitDefault),
        ).coerceIn(1, MAX_WIFI_RESULTS)
        val channelGraph = wifiChannelGraphRows(analysisNetworks, detailLimit)
        val accessPointDetails = wifiAccessPointDetailRows(analysisNetworks, detailLimit)
        val accessPointSemantics = wifiAccessPointSemanticRows(accessPointDetails, detailLimit)
        val securitySummary = wifiSecuritySummaryJson(analysisNetworks)
        val channelWidthSummary = wifiChannelWidthSummaryJson(analysisNetworks)
        val standardSummary = wifiStandardSummaryJson(analysisNetworks)
        val bandCoverage = wifiBandCoverageRows(analysisNetworks, bandSummary, channelRatings)
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
            scanMode = scanMode,
            userRefreshRequested = requestedRefresh,
        )
        return JSONObject()
            .put("success", true)
            .put("action", actionName)
            .put("refresh_requested", requestedRefresh)
            .put("effective_refresh_requested", refresh)
            .put("refresh_accepted", refreshAccepted)
            .put("wifi_scan_mode", scanMode)
            .put("wifi_scan_control", wifiScanControlJson(scanMode, requestedRefresh, refresh, refreshAccepted))
            .put("result_count", networks.length())
            .put("total_scan_result_count", allNetworks.length())
            .put("filtered_scan_result_count", analysisNetworks.length())
            .put("wifi_filter_active", filterSpec.active)
            .put("wifi_active_filter_count", filterSpec.activeFilterCount)
            .put("wifi_filter_summary", filterSummary)
            .put("wifi_scan_age_ms", latestScanAgeMs ?: JSONObject.NULL)
            .put("wifi_vendor_count", vendorSummary.length())
            .put("wifi_filter_count", availableAnalyzerFilters.length())
            .put("applied_wifi_filter_count", filterRows.length())
            .put("wifi_history_network_count", signalHistory.length())
            .put("wifi_access_point_detail_count", accessPointDetails.length())
            .put("wifi_access_point_semantic_count", accessPointSemantics.length())
            .put("wifi_band_coverage_count", bandCoverage.length())
            .put("wifi_channel_graph_count", channelGraph.length())
            .put("wifi_channel_utilization_count", channelUtilization.length())
            .put("wifi_security_summary_count", securitySummary.length())
            .put("wifi_width_summary_count", channelWidthSummary.length())
            .put("wifi_standard_summary_count", standardSummary.length())
            .put("wifi_enabled", wifiEnabled)
            .put("wifi_scan_permission_status", permissionStatus)
            .put("wifi_scan_status", scanStatus)
            .put("wifi_networks", networks)
            .put("wifi_access_point_details", accessPointDetails)
            .put("wifi_access_point_semantics", accessPointSemantics)
            .put("wifi_access_point_export", accessPointExport)
            .put(
                "wifi_access_point_export_csv",
                if (exportFormat == "csv" || exportFormat == "both") wifiAccessPointCsv(accessPointDetails) else JSONObject.NULL,
            )
            .put("wifi_channel_ratings", channelRatings)
            .put("wifi_channel_utilization", channelUtilization)
            .put("wifi_channel_graph", channelGraph)
            .put("recommended_wifi_channels", recommendedChannels)
            .put("wifi_band_summary", bandSummary)
            .put("wifi_band_coverage", bandCoverage)
            .put("wifi_vendor_summary", vendorSummary)
            .put("wifi_analyzer_filters", availableAnalyzerFilters)
            .put("available_wifi_analyzer_filters", availableAnalyzerFilters)
            .put("filtered_wifi_analyzer_filters", filteredAnalyzerFilters)
            .put("applied_wifi_filters", filterRows)
            .put("wifi_security_summary", securitySummary)
            .put("wifi_channel_width_summary", channelWidthSummary)
            .put("wifi_standard_summary", standardSummary)
            .put("wifi_signal_history", signalHistory)
            .put("privacy_note", "Vendor/OUI lookup uses local prefix hints from Android scan metadata; no internet lookup is performed.")
            .put(
                "cards",
                JSONArray().also { cards ->
                    if (filterRows.length() > 0) {
                        cards.put(
                            graphCard(
                                title = "Wi-Fi Applied Filters",
                                body = "${analysisNetworks.length()} of ${allNetworks.length()} AP row(s) matched the requested Wi-Fi Analyzer filter(s).",
                                graphType = "wifi_filter_application",
                                rows = filterRows,
                            ),
                        )
                    }
                }
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
                            title = "Wi-Fi AP Semantics",
                            body = "${accessPointSemantics.length()} access point semantic row(s) with router, guest, hotspot, hidden, passpoint, security-risk, and agent-routing labels.",
                            graphType = "wifi_access_point_semantics",
                            rows = accessPointSemantics,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Channel Graph",
                            body = "${channelGraph.length()} access point channel envelope row(s), including dBm, channel width, channel span, frequency span, and overlap pressure.",
                            graphType = "wifi_channel_graph",
                            rows = channelGraph,
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
                            title = "Wi-Fi Channel Utilization",
                            body = "${channelUtilization.length()} observed channel utilization row(s), inferred from visible AP crowding, overlap, RSSI pressure, width, security, and SSID samples.",
                            graphType = "wifi_channel_utilization",
                            rows = channelUtilization,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Wi-Fi Band Coverage",
                            body = "${bandCoverage.length()} 2.4/5/6GHz coverage row(s) showing observed AP counts, channels, widths, standards, and security attention per band.",
                            graphType = "wifi_band_coverage",
                            rows = bandCoverage,
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

    fun wifiAnalyzerReportJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val wifiManager = appContext.getSystemService(WifiManager::class.java)
        val permissionStatus = wifiPermissionStatusJson(appContext)
        val canReadScan = wifiManager != null && permissionStatus.optBoolean("can_read_scan_results", false)
        val passiveArguments = JSONObject(arguments.toString()).put("refresh", false)
        val scanResult = if (canReadScan) {
            wifiScanJson(appContext, passiveArguments, "wifi_analyzer_report")
        } else {
            null
        }
        val scanSucceeded = scanResult?.optBoolean("success", false) == true
        val networks = scanResult?.optJSONArray("wifi_networks") ?: JSONArray()
        val accessPointDetails = scanResult?.optJSONArray("wifi_access_point_details") ?: JSONArray()
        val channelGraph = scanResult?.optJSONArray("wifi_channel_graph") ?: wifiChannelGraphRows(networks)
        val channelRatings = scanResult?.optJSONArray("wifi_channel_ratings") ?: JSONArray()
        val channelUtilization = scanResult?.optJSONArray("wifi_channel_utilization") ?: JSONArray()
        val recommendedChannels = scanResult?.optJSONArray("recommended_wifi_channels") ?: JSONArray()
        val bandSummary = scanResult?.optJSONArray("wifi_band_summary") ?: JSONArray()
        val bandCoverage = scanResult?.optJSONArray("wifi_band_coverage") ?: wifiBandCoverageRows(networks, bandSummary, channelRatings)
        val vendorSummary = scanResult?.optJSONArray("wifi_vendor_summary") ?: JSONArray()
        val filters = scanResult?.optJSONArray("wifi_analyzer_filters") ?: JSONArray()
        val availableFilters = scanResult?.optJSONArray("available_wifi_analyzer_filters") ?: filters
        val filteredFilters = scanResult?.optJSONArray("filtered_wifi_analyzer_filters") ?: filters
        val appliedFilters = scanResult?.optJSONArray("applied_wifi_filters") ?: JSONArray()
        val filterSummary = scanResult?.optJSONObject("wifi_filter_summary")
            ?: wifiFilterSummaryJson(wifiScanFilterSpec(arguments), networks.length(), networks.length())
        val securitySummary = scanResult?.optJSONArray("wifi_security_summary") ?: JSONArray()
        val widthSummary = scanResult?.optJSONArray("wifi_channel_width_summary") ?: JSONArray()
        val standardSummary = scanResult?.optJSONArray("wifi_standard_summary") ?: JSONArray()
        val accessPointSemantics = scanResult?.optJSONArray("wifi_access_point_semantics")
            ?: wifiAccessPointSemanticRows(accessPointDetails, accessPointDetails.length().coerceIn(1, MAX_WIFI_RESULTS))
        val cachedHistory = scanResult?.optJSONArray("wifi_signal_history")
            ?: wifiSignalHistoryRowsFromStore(readWifiSignalHistory(appContext))
        val scanStatus = scanResult?.optJSONObject("wifi_scan_status") ?: wifiScanStatusJson(
            refreshRequested = false,
            refreshAccepted = false,
            wifiEnabled = wifiManager?.isWifiEnabled == true,
            permissionStatus = permissionStatus,
            totalScanResultCount = 0,
            returnedNetworkCount = 0,
            latestScanAgeMs = null,
        )
        val scanControl = scanStatus.optJSONObject("wifi_scan_control")
            ?: scanResult?.optJSONObject("wifi_scan_control")
            ?: wifiScanControlJson(WIFI_SCAN_MODE_AUTO, false, false, false)
        val featureRows = wifiAnalyzerFeatureRows(
            wifiAvailable = wifiManager != null,
            permissionStatus = permissionStatus,
            networkCount = scanResult?.optInt("total_scan_result_count", networks.length()) ?: 0,
            returnedNetworkCount = networks.length(),
            accessPointDetailCount = accessPointDetails.length(),
            channelRatingCount = channelRatings.length(),
            channelGraphCount = channelGraph.length(),
            channelUtilizationCount = channelUtilization.length(),
            vendorCount = vendorSummary.length(),
            filterCount = filters.length(),
            historyCount = cachedHistory.length(),
            semanticCount = accessPointSemantics.length(),
            bandCoverageCount = bandCoverage.length(),
            observedBandCount = countWifiObservedBands(bandCoverage),
            widthCount = widthSummary.length(),
            standardCount = standardSummary.length(),
            exportReady = accessPointDetails.length() > 0,
        )
        val routeRows = wifiAnalyzerWorkflowRows(
            permissionStatus = permissionStatus,
            scanStatus = scanStatus,
            historyCount = cachedHistory.length(),
            channelGraphCount = channelGraph.length(),
            channelRatingCount = channelRatings.length(),
            accessPointDetailCount = accessPointDetails.length(),
        )
        val policyRows = wifiScanPolicyRows(
            wifiAvailable = wifiManager != null,
            permissionStatus = permissionStatus,
            scanStatus = scanStatus,
            scanSucceeded = scanSucceeded,
        )
        val result = if (scanSucceeded) JSONObject(scanResult.toString()) else JSONObject()
        val cards = JSONArray()
            .put(
                graphCard(
                    title = "Wi-Fi Analyzer Readiness",
                    body = "${featureRows.length()} feature row(s) covering AP discovery, graphs, filters, OUI lookup, export, and scan-history parity.",
                    graphType = "wifi_analyzer_feature_matrix",
                    rows = featureRows,
                ),
            )
            .put(
                graphCard(
                    title = "Wi-Fi Analyzer Routes",
                    body = "${routeRows.length()} route row(s) for choosing live scan, channel rating, AP detail, export, or cached-history flows.",
                    graphType = "wifi_analyzer_workflow_routes",
                    rows = routeRows,
                ),
            )
            .put(
                graphCard(
                    title = "Wi-Fi Scan Policy",
                    body = "${policyRows.length()} permission, throttling, passive-report, and privacy policy row(s) for honest Wi-Fi analysis.",
                    graphType = "wifi_scan_policy_matrix",
                    rows = policyRows,
                ),
            )
        scanResult?.optJSONArray("cards")?.let { scanCards ->
            for (index in 0 until scanCards.length()) {
                cards.put(scanCards.getJSONObject(index))
            }
        }
        return result
            .put("success", true)
            .put("action", "wifi_analyzer_report")
            .put("report_scope", "WiFiAnalyzer-style readiness and routing report for AP discovery, signal/channel graph envelopes, AP history, channel rating, channel utilization/occupancy inference, filter facets, OUI/vendor lookup, export, scan throttling, and privacy boundaries.")
            .put("wifi_scan_permission_status", permissionStatus)
            .put("wifi_scan_status", scanStatus)
            .put("wifi_scan_control", scanControl)
            .put("wifi_networks", networks)
            .put("wifi_access_point_details", accessPointDetails)
            .put("wifi_channel_ratings", channelRatings)
            .put("wifi_channel_utilization", channelUtilization)
            .put("wifi_channel_graph", channelGraph)
            .put("recommended_wifi_channels", recommendedChannels)
            .put("wifi_band_summary", bandSummary)
            .put("wifi_band_coverage", bandCoverage)
            .put("wifi_vendor_summary", vendorSummary)
            .put("wifi_analyzer_filters", filters)
            .put("available_wifi_analyzer_filters", availableFilters)
            .put("filtered_wifi_analyzer_filters", filteredFilters)
            .put("applied_wifi_filters", appliedFilters)
            .put("wifi_filter_summary", filterSummary)
            .put("wifi_filter_active", filterSummary.optBoolean("active", false))
            .put("filtered_scan_result_count", filterSummary.optInt("matched_network_count", networks.length()))
            .put("wifi_security_summary", securitySummary)
            .put("wifi_channel_width_summary", widthSummary)
            .put("wifi_standard_summary", standardSummary)
            .put("wifi_access_point_semantics", accessPointSemantics)
            .put("wifi_signal_history", cachedHistory)
            .put("wifi_analyzer_feature_matrix", featureRows)
            .put("wifi_analyzer_workflow_routes", routeRows)
            .put("wifi_scan_policy_matrix", policyRows)
            .put("wifi_analyzer_feature_count", featureRows.length())
            .put("ready_wifi_analyzer_feature_count", countReadyRows(featureRows))
            .put("wifi_analyzer_workflow_route_count", routeRows.length())
            .put("wifi_channel_graph_count", channelGraph.length())
            .put("wifi_channel_utilization_count", channelUtilization.length())
            .put("wifi_access_point_semantic_count", accessPointSemantics.length())
            .put("wifi_band_coverage_count", bandCoverage.length())
            .put("wifi_scan_policy_count", policyRows.length())
            .put("applied_wifi_filter_count", appliedFilters.length())
            .put("cards", cards)
    }

    fun bluetoothScanJson(context: Context, arguments: JSONObject = JSONObject(), actionName: String = "bluetooth_scan"): JSONObject {
        val appContext = context.applicationContext
        val limit = arguments.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_BLUETOOTH_RESULTS)
        val requestedRefresh = arguments.optBoolean("refresh", false)
        val scanMode = normalizedBluetoothScanMode(arguments)
        val refresh = effectiveBluetoothRefreshRequested(requestedRefresh, scanMode)
        val timeoutMs = arguments.optLong("timeout_ms", DEFAULT_BLUETOOTH_TIMEOUT_MS).coerceIn(500L, MAX_BLUETOOTH_TIMEOUT_MS)
        val bluetoothManager = appContext.getSystemService(BluetoothManager::class.java)
        val adapter = bluetoothManager?.adapter ?: runCatching { BluetoothAdapter.getDefaultAdapter() }.getOrNull()
        val permissionStatus = bluetoothPermissionStatusJson(appContext)
        if (adapter == null) {
            return JSONObject()
                .put("success", false)
                .put("action", actionName)
                .put("error", "Bluetooth service is unavailable on this device")
                .put("bluetooth_scan_permission_status", permissionStatus)
                .put("bluetooth_scan_control", bluetoothScanControlJson(scanMode, requestedRefresh, refresh, false))
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
        val allDevices = JSONArray()
        rows.values
            .sortedWith(compareByDescending<JSONObject> { it.optInt("rssi_dbm", Int.MIN_VALUE) }.thenBy { it.optString("device_name") })
            .forEach { row -> allDevices.put(row) }
        val filterSpec = bluetoothScanFilterSpec(arguments)
        val filteredDeviceRows = bluetoothFilteredDeviceRowsForSpec(allDevices, filterSpec)
        val devices = JSONArray()
        for (index in 0 until minOf(filteredDeviceRows.length(), limit)) {
            devices.put(filteredDeviceRows.getJSONObject(index).put("rank", index + 1))
        }
        val filterSummary = bluetoothFilterSummaryJson(filterSpec, allDevices.length(), filteredDeviceRows.length())
        val filterRows = bluetoothFilterApplicationRows(filterSpec, allDevices.length(), filteredDeviceRows.length())
        val availableAnalyzerFilters = bluetoothAnalyzerFilterSummaryJson(allDevices)
        val filteredAnalyzerFilters = bluetoothAnalyzerFilterSummaryJson(filteredDeviceRows)
        val metadataSummary = bluetoothMetadataSummaryRows(devices)
        val serviceUuidCount = bluetoothDistinctStringCount(devices, "service_uuids")
        val serviceLabelCount = bluetoothDistinctStringCount(devices, "service_labels")
        val manufacturerIdCount = bluetoothDistinctStringCount(devices, "manufacturer_ids")
        val manufacturerNameCount = bluetoothDistinctStringCount(devices, "manufacturer_names")
        val observedAtMs = System.currentTimeMillis()
        val historyStore = updateBluetoothSignalHistory(appContext, allDevices, observedAtMs)
        val signalHistory = bluetoothSignalHistoryRowsFromStore(historyStore)
        return JSONObject()
            .put("success", true)
            .put("action", actionName)
            .put("refresh_requested", requestedRefresh)
            .put("effective_refresh_requested", refresh)
            .put("refresh_accepted", refreshAccepted)
            .put("bluetooth_scan_mode", scanMode)
            .put("bluetooth_scan_control", bluetoothScanControlJson(scanMode, requestedRefresh, refresh, refreshAccepted))
            .put("scan_error", scanError ?: JSONObject.NULL)
            .put("bluetooth_enabled", runCatching { adapter.isEnabled }.getOrDefault(false))
            .put("bluetooth_total_device_count", allDevices.length())
            .put("bluetooth_device_count", devices.length())
            .put("bluetooth_filter_active", filterSpec.active)
            .put("bluetooth_active_filter_count", filterSpec.activeFilterCount)
            .put("applied_bluetooth_filter_count", filterRows.length())
            .put("bluetooth_filter_summary", filterSummary)
            .put("bluetooth_metadata_count", metadataSummary.length())
            .put("bluetooth_service_uuid_count", serviceUuidCount)
            .put("bluetooth_service_label_count", serviceLabelCount)
            .put("bluetooth_manufacturer_id_count", manufacturerIdCount)
            .put("bluetooth_manufacturer_name_count", manufacturerNameCount)
            .put("bluetooth_signal_history_count", signalHistory.length())
            .put("bluetooth_scan_permission_status", permissionStatus)
            .put("bluetooth_devices", devices)
            .put("bluetooth_metadata_summary", metadataSummary)
            .put("bluetooth_analyzer_filters", availableAnalyzerFilters)
            .put("available_bluetooth_analyzer_filters", availableAnalyzerFilters)
            .put("filtered_bluetooth_analyzer_filters", filteredAnalyzerFilters)
            .put("applied_bluetooth_filters", filterRows)
            .put("bluetooth_signal_history", signalHistory)
            .put(
                "cards",
                JSONArray().also { cards ->
                    if (filterSpec.active) {
                        cards.put(
                            graphCard(
                                title = "Bluetooth Applied Filters",
                                body = "${filteredDeviceRows.length()} of ${allDevices.length()} Bluetooth row(s) matched the requested nearby-device filter(s).",
                                graphType = "bluetooth_filter_application",
                                rows = filterRows,
                            ),
                        )
                    }
                }
                    .put(
                        graphCard(
                            title = "Bluetooth Nearby",
                            body = "${devices.length()} paired or scanned Bluetooth device row(s), with BLE RSSI, class, service UUID labels, manufacturer names, and proximity metadata when Android exposes it.",
                            graphType = "bluetooth_rssi",
                            rows = devices,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Bluetooth Metadata",
                            body = "${metadataSummary.length()} Bluetooth class/service/manufacturer summary row(s) inferred from nearby and paired device metadata, with semantic labels where assigned numbers are recognized.",
                            graphType = "bluetooth_metadata_summary",
                            rows = metadataSummary,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Bluetooth Signal History",
                            body = "${signalHistory.length()} Bluetooth RSSI history row(s) built from recent BLE scan observations, including average, min/max, trend, and last-seen metadata.",
                            graphType = "bluetooth_signal_history",
                            rows = signalHistory,
                        ),
                    ),
            )
    }

    fun bluetoothAnalyzerReportJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val bluetoothManager = appContext.getSystemService(BluetoothManager::class.java)
        val adapter = bluetoothManager?.adapter ?: runCatching { BluetoothAdapter.getDefaultAdapter() }.getOrNull()
        val packageManager = appContext.packageManager
        val permissionStatus = bluetoothPermissionStatusJson(appContext)
        val bluetoothAvailable = adapter != null || packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH)
        val bluetoothLeSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)
        val bluetoothEnabled = adapter?.let { runCatching { it.isEnabled }.getOrDefault(false) } ?: false
        val canReadAnyBluetoothRows = adapter != null &&
            (permissionStatus.optBoolean("can_read_paired_devices", false) || permissionStatus.optBoolean("can_scan_nearby_devices", false))
        val passiveArguments = JSONObject(arguments.toString()).put("refresh", false)
        val scanResult = if (canReadAnyBluetoothRows) {
            bluetoothScanJson(appContext, passiveArguments)
        } else {
            null
        }
        val scanSucceeded = scanResult?.optBoolean("success", false) == true
        val devices = scanResult?.optJSONArray("bluetooth_devices") ?: JSONArray()
        val metadataSummary = scanResult?.optJSONArray("bluetooth_metadata_summary") ?: JSONArray()
        val filters = scanResult?.optJSONArray("bluetooth_analyzer_filters") ?: JSONArray()
        val availableFilters = scanResult?.optJSONArray("available_bluetooth_analyzer_filters") ?: filters
        val filteredFilters = scanResult?.optJSONArray("filtered_bluetooth_analyzer_filters") ?: filters
        val appliedFilters = scanResult?.optJSONArray("applied_bluetooth_filters") ?: JSONArray()
        val filterSummary = scanResult?.optJSONObject("bluetooth_filter_summary")
            ?: bluetoothFilterSummaryJson(bluetoothScanFilterSpec(arguments), devices.length(), devices.length())
        val cachedHistory = scanResult?.optJSONArray("bluetooth_signal_history")
            ?: bluetoothSignalHistoryRowsFromStore(readBluetoothSignalHistory(appContext))
        val serviceUuidCount = scanResult?.optInt("bluetooth_service_uuid_count", bluetoothDistinctStringCount(devices, "service_uuids"))
            ?: bluetoothDistinctStringCount(devices, "service_uuids")
        val serviceLabelCount = scanResult?.optInt("bluetooth_service_label_count", bluetoothDistinctStringCount(devices, "service_labels"))
            ?: bluetoothDistinctStringCount(devices, "service_labels")
        val manufacturerIdCount = scanResult?.optInt("bluetooth_manufacturer_id_count", bluetoothDistinctStringCount(devices, "manufacturer_ids"))
            ?: bluetoothDistinctStringCount(devices, "manufacturer_ids")
        val manufacturerNameCount = scanResult?.optInt("bluetooth_manufacturer_name_count", bluetoothDistinctStringCount(devices, "manufacturer_names"))
            ?: bluetoothDistinctStringCount(devices, "manufacturer_names")
        val rssiDeviceCount = bluetoothRssiDeviceCount(devices)
        val scanStatus = bluetoothScanStatusJson(
            bluetoothAvailable = bluetoothAvailable,
            bluetoothLeSupported = bluetoothLeSupported,
            bluetoothEnabled = bluetoothEnabled,
            permissionStatus = permissionStatus,
            scanResult = scanResult,
            returnedDeviceCount = devices.length(),
            metadataCount = metadataSummary.length(),
            serviceUuidCount = serviceUuidCount,
            manufacturerIdCount = manufacturerIdCount,
            rssiDeviceCount = rssiDeviceCount,
        )
        val scanControl = scanStatus.optJSONObject("bluetooth_scan_control")
            ?: scanResult?.optJSONObject("bluetooth_scan_control")
            ?: bluetoothScanControlJson(BLUETOOTH_SCAN_MODE_AUTO, false, false, false)
        val featureRows = bluetoothAnalyzerFeatureRows(
            bluetoothAvailable = bluetoothAvailable,
            bluetoothLeSupported = bluetoothLeSupported,
            bluetoothEnabled = bluetoothEnabled,
            permissionStatus = permissionStatus,
            deviceCount = devices.length(),
            metadataCount = metadataSummary.length(),
            serviceUuidCount = serviceUuidCount,
            manufacturerIdCount = manufacturerIdCount,
            rssiDeviceCount = rssiDeviceCount,
            historyCount = cachedHistory.length(),
            categoryCount = bluetoothDistinctCategoryCount(devices),
        )
        val routeRows = bluetoothAnalyzerWorkflowRows(
            permissionStatus = permissionStatus,
            deviceCount = devices.length(),
            serviceUuidCount = serviceUuidCount,
            manufacturerIdCount = manufacturerIdCount,
            rssiDeviceCount = rssiDeviceCount,
            historyCount = cachedHistory.length(),
        )
        val policyRows = bluetoothScanPolicyRows(
            bluetoothAvailable = bluetoothAvailable,
            bluetoothLeSupported = bluetoothLeSupported,
            bluetoothEnabled = bluetoothEnabled,
            permissionStatus = permissionStatus,
            scanStatus = scanStatus,
            scanSucceeded = scanSucceeded,
        )
        val result = if (scanSucceeded) JSONObject(scanResult.toString()) else JSONObject()
        val cards = JSONArray()
            .put(
                graphCard(
                    title = "Bluetooth Analyzer Readiness",
                    body = "${featureRows.length()} feature row(s) covering paired inventory, BLE nearby scans, RSSI proximity/history, service UUID labels, manufacturer names, category hints, card rendering, and safety boundaries.",
                    graphType = "bluetooth_analyzer_feature_matrix",
                    rows = featureRows,
                ),
            )
            .put(
                graphCard(
                    title = "Bluetooth Analyzer Routes",
                    body = "${routeRows.length()} route row(s) for choosing nearby scan, paired inventory, proximity explanation, service label/manufacturer name metadata, or scan-policy flows.",
                    graphType = "bluetooth_analyzer_workflow_routes",
                    rows = routeRows,
                ),
            )
            .put(
                graphCard(
                    title = "Bluetooth Scan Policy",
                    body = "${policyRows.length()} service, enablement, permission, legacy-location, scan-cadence, and privacy row(s) for honest Bluetooth analysis.",
                    graphType = "bluetooth_scan_policy_matrix",
                    rows = policyRows,
                ),
            )
        if (!scanSucceeded && cachedHistory.length() > 0) {
            cards.put(
                graphCard(
                    title = "Bluetooth Signal History",
                    body = "${cachedHistory.length()} cached Bluetooth RSSI trend row(s), preserving recent scan context for Gemma even when the analyzer report stays passive.",
                    graphType = "bluetooth_signal_history",
                    rows = cachedHistory,
                ),
            )
        }
        scanResult?.optJSONArray("cards")?.let { scanCards ->
            for (index in 0 until scanCards.length()) {
                cards.put(scanCards.getJSONObject(index))
            }
        }
        return result
            .put("success", true)
            .put("action", "bluetooth_analyzer_report")
            .put("report_scope", "Bluetooth Analyzer readiness and routing report for paired devices, nearby BLE devices, RSSI proximity graphs, signal history/trends, Bluetooth SIG service labels, manufacturer names, category metadata, scan cadence, permissions, and privacy boundaries.")
            .put("bluetooth_scan_permission_status", permissionStatus)
            .put("bluetooth_scan_status", scanStatus)
            .put("bluetooth_scan_control", scanControl)
            .put("bluetooth_devices", devices)
            .put("bluetooth_metadata_summary", metadataSummary)
            .put("bluetooth_analyzer_filters", filters)
            .put("available_bluetooth_analyzer_filters", availableFilters)
            .put("filtered_bluetooth_analyzer_filters", filteredFilters)
            .put("applied_bluetooth_filters", appliedFilters)
            .put("bluetooth_filter_summary", filterSummary)
            .put("bluetooth_filter_active", filterSummary.optBoolean("active", false))
            .put("bluetooth_signal_history", cachedHistory)
            .put("bluetooth_analyzer_feature_matrix", featureRows)
            .put("bluetooth_analyzer_workflow_routes", routeRows)
            .put("bluetooth_scan_policy_matrix", policyRows)
            .put("bluetooth_device_count", devices.length())
            .put("bluetooth_total_device_count", scanResult?.optInt("bluetooth_total_device_count", devices.length()) ?: devices.length())
            .put("applied_bluetooth_filter_count", appliedFilters.length())
            .put("bluetooth_metadata_count", metadataSummary.length())
            .put("bluetooth_service_uuid_count", serviceUuidCount)
            .put("bluetooth_service_label_count", serviceLabelCount)
            .put("bluetooth_manufacturer_id_count", manufacturerIdCount)
            .put("bluetooth_manufacturer_name_count", manufacturerNameCount)
            .put("bluetooth_signal_history_count", cachedHistory.length())
            .put("bluetooth_analyzer_feature_count", featureRows.length())
            .put("ready_bluetooth_analyzer_feature_count", countReadyRows(featureRows))
            .put("bluetooth_analyzer_workflow_route_count", routeRows.length())
            .put("bluetooth_scan_policy_count", policyRows.length())
            .put("cards", cards)
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
        val observedAtMs = System.currentTimeMillis()
        val motionHistoryStore = updateMotionSensorHistory(appContext, samples, observedAtMs)
        val motionHistory = motionSensorHistoryRowsFromStore(motionHistoryStore, observedAtMs)
        val motionPoseEstimates = motionPoseEstimateRows(samples, motionHistory)
        val sampledSensorCount = countSampledSensors(samples)
        return JSONObject()
            .put("success", true)
            .put("action", "sensor_snapshot")
            .put("requested_sensor_types", JSONArray(requested))
            .put("sample_timeout_ms", timeoutMs)
            .put("available_sensor_types", JSONArray(available))
            .put("sensor_samples", samples)
            .put("sampled_sensor_count", sampledSensorCount)
            .put("sensor_capabilities", capabilities)
            .put("sensor_capability_count", capabilities.length())
            .put("motion_sensor_count", countSensorCapabilities(capabilities, MOTION_SENSOR_TYPES))
            .put("motion_sensor_history", motionHistory)
            .put("motion_sensor_history_count", motionHistory.length())
            .put("motion_pose_estimates", motionPoseEstimates)
            .put("motion_pose_estimate_count", motionPoseEstimates.length())
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
                            title = "Motion Pose Estimate",
                            body = "${motionPoseEstimates.length()} fused pose, heading, angular-motion, or acceleration row(s) derived from available IMU samples.",
                            graphType = "motion_pose_estimate",
                            rows = motionPoseEstimates,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Motion Sensor History",
                            body = "${motionHistory.length()} cached accelerometer, gyroscope, linear-acceleration, rotation, or magnetic trend row(s) for motion-aware agent context.",
                            graphType = "motion_sensor_history",
                            rows = motionHistory,
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

    fun motionSensorHistoryJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val sensorManager = appContext.getSystemService(SensorManager::class.java)
        val requested = requestedMotionHistorySensorTypes(arguments)
        val timeoutMs = arguments.optLong("timeout_ms", DEFAULT_SENSOR_TIMEOUT_MS).coerceIn(150L, MAX_SENSOR_TIMEOUT_MS)
        val sampleRequested = arguments.optBoolean("sample", arguments.optBoolean("refresh", true))
        val sampleResult = if (sampleRequested && sensorManager != null) {
            val sampleArguments = JSONObject(arguments.toString())
                .put("sensor_types", requested.joinToString(","))
                .put("timeout_ms", timeoutMs)
            sensorSnapshotJson(appContext, sampleArguments)
        } else {
            null
        }
        val historyRows = sampleResult?.optJSONArray("motion_sensor_history")
            ?: motionSensorHistoryRowsFromStore(readMotionSensorHistory(appContext))
        val motionPoseEstimates = sampleResult?.optJSONArray("motion_pose_estimates")
            ?: motionPoseEstimateRows(JSONArray(), historyRows)
        val capabilities = if (sensorManager != null) {
            sensorCapabilityRows(sensorManager, requested)
        } else {
            unavailableSensorRows(requested)
        }
        val available = sensorTypeCatalog(appContext)
        return JSONObject()
            .put("success", sensorManager != null || historyRows.length() > 0)
            .put("action", "motion_sensor_history")
            .put("sensor_service_available", sensorManager != null)
            .put("sample_requested", sampleRequested)
            .put("sample_timeout_ms", timeoutMs)
            .put("requested_sensor_types", JSONArray(requested))
            .put("available_sensor_types", JSONArray(available))
            .put("sampled_sensor_count", sampleResult?.optInt("sampled_sensor_count") ?: 0)
            .put("motion_sensor_history", historyRows)
            .put("motion_sensor_history_count", historyRows.length())
            .put("motion_pose_estimates", motionPoseEstimates)
            .put("motion_pose_estimate_count", motionPoseEstimates.length())
            .put("sensor_capabilities", capabilities)
            .put("sensor_capability_count", capabilities.length())
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Motion Pose Estimate",
                            body = "${motionPoseEstimates.length()} fused pose, heading, angular-motion, or acceleration row(s) derived from current or cached IMU data.",
                            graphType = "motion_pose_estimate",
                            rows = motionPoseEstimates,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Motion Sensor History",
                            body = "${historyRows.length()} cached motion/IMU trend row(s) with magnitude, trend, stability, and current vector values.",
                            graphType = "motion_sensor_history",
                            rows = historyRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Sensor Hardware",
                            body = "${capabilities.length()} motion sensor capability row(s) with vendor, range, resolution, power, FIFO, wake-up, and sampling-rate metadata.",
                            graphType = "sensor_capability",
                            rows = capabilities,
                        ),
                    ),
            )
    }

    fun sensorAnalyzerReportJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val sensorManager = appContext.getSystemService(SensorManager::class.java)
        val requested = sensorAnalyzerRequestedSensorTypes(arguments)
        val available = sensorTypeCatalog(appContext)
        val availableSet = available.toSet()
        val capabilities = if (sensorManager != null) {
            sensorCapabilityRows(sensorManager, requested)
        } else {
            unavailableSensorRows(requested)
        }
        val includeSnapshot = arguments.optBoolean("include_snapshot", false) || arguments.optBoolean("sample", false)
        val timeoutMs = arguments.optLong("timeout_ms", DEFAULT_SENSOR_TIMEOUT_MS).coerceIn(150L, MAX_SENSOR_TIMEOUT_MS)
        val snapshot = if (includeSnapshot && sensorManager != null) sensorSnapshotJson(appContext, arguments) else null
        val samples = snapshot?.optJSONArray("sensor_samples") ?: JSONArray()
        val motionHistory = snapshot?.optJSONArray("motion_sensor_history")
            ?: motionSensorHistoryRowsFromStore(readMotionSensorHistory(appContext))
        val motionPoseEstimates = snapshot?.optJSONArray("motion_pose_estimates")
            ?: motionPoseEstimateRows(samples, motionHistory)
        val motionSensorCount = countSensorCapabilities(capabilities, MOTION_SENSOR_TYPES)
        val ambientSensorCount = countSensorCapabilities(capabilities, AMBIENT_SENSOR_TYPES)
        val wakeUpSensorCount = countSensorCapabilityFlag(capabilities, "wake_up")
        val directChannelSensorCount = countSensorCapabilityFlag(capabilities, "direct_channel_supported")
        val requestedAvailableCount = requested.count { it in availableSet }
        val samplingStatus = sensorSamplingStatusJson(
            sensorServiceAvailable = sensorManager != null,
            requestedSensorCount = requested.size,
            availableSensorCount = available.size,
            requestedAvailableCount = requestedAvailableCount,
            sampledSensorCount = samples.length(),
            activeSampleRequested = includeSnapshot,
            timeoutMs = timeoutMs,
        )
        val featureRows = sensorAnalyzerFeatureRows(
            sensorServiceAvailable = sensorManager != null,
            availableSensors = available,
            capabilityCount = capabilities.length(),
            motionSensorCount = motionSensorCount,
            ambientSensorCount = ambientSensorCount,
            wakeUpSensorCount = wakeUpSensorCount,
            directChannelSensorCount = directChannelSensorCount,
            sampledSensorCount = samples.length(),
            motionHistoryCount = motionHistory.length(),
            motionPoseEstimateCount = motionPoseEstimates.length(),
        )
        val routeRows = sensorAnalyzerWorkflowRows(
            availableSensors = available,
            motionSensorCount = motionSensorCount,
            ambientSensorCount = ambientSensorCount,
            capabilityCount = capabilities.length(),
            motionHistoryCount = motionHistory.length(),
            motionPoseEstimateCount = motionPoseEstimates.length(),
        )
        val policyRows = sensorSamplingPolicyRows(
            sensorServiceAvailable = sensorManager != null,
            requestedSensorCount = requested.size,
            requestedAvailableCount = requestedAvailableCount,
            wakeUpSensorCount = wakeUpSensorCount,
            directChannelSensorCount = directChannelSensorCount,
            activeSampleRequested = includeSnapshot,
            timeoutMs = timeoutMs,
        )
        val cards = JSONArray()
            .put(
                graphCard(
                    title = "Sensor Analyzer Readiness",
                    body = "${featureRows.length()} feature row(s) covering accelerometer, gyroscope, orientation, ambient context, hardware metadata, watcher automation, card rendering, and privacy boundaries.",
                    graphType = "sensor_analyzer_feature_matrix",
                    rows = featureRows,
                ),
            )
            .put(
                graphCard(
                    title = "Sensor Analyzer Routes",
                    body = "${routeRows.length()} route row(s) for choosing one-shot samples, orientation context, ambient context, hardware metadata, watcher automation, or sampling-policy flows.",
                    graphType = "sensor_analyzer_workflow_routes",
                    rows = routeRows,
                ),
            )
            .put(
                graphCard(
                    title = "Sensor Sampling Policy",
                    body = "${policyRows.length()} service, availability, sampling-cadence, timeout, power, and privacy row(s) for honest motion and ambient analysis.",
                    graphType = "sensor_sampling_policy_matrix",
                    rows = policyRows,
                ),
            )
            .put(
                graphCard(
                    title = "Sensor Hardware",
                    body = "${capabilities.length()} sensor capability row(s) with vendor, range, resolution, power, FIFO, wake-up, direct-channel, and sampling-rate metadata.",
                    graphType = "sensor_capability",
                    rows = capabilities,
                ),
            )
        if (snapshot == null && motionHistory.length() > 0) {
            cards.put(
                graphCard(
                    title = "Motion Pose Estimate",
                    body = "${motionPoseEstimates.length()} cached pose, heading, angular-motion, or acceleration row(s) available without forcing another sample.",
                    graphType = "motion_pose_estimate",
                    rows = motionPoseEstimates,
                ),
            )
            cards.put(
                graphCard(
                    title = "Motion Sensor History",
                    body = "${motionHistory.length()} cached motion/IMU trend row(s) available without forcing another sample.",
                    graphType = "motion_sensor_history",
                    rows = motionHistory,
                ),
            )
        }
        snapshot?.optJSONArray("cards")?.let { snapshotCards ->
            for (index in 0 until snapshotCards.length()) {
                cards.put(snapshotCards.getJSONObject(index))
            }
        }
        return JSONObject()
            .put("success", true)
            .put("action", "sensor_analyzer_report")
            .put("report_scope", "Sensor Analyzer readiness and routing report for accelerometer, gyroscope, rotation/orientation, ambient sensors, sensor hardware metadata, one-shot sampling, watcher automation, sampling cadence, power, and privacy boundaries.")
            .put("sensor_service_available", sensorManager != null)
            .put("requested_sensor_types", JSONArray(requested))
            .put("available_sensor_types", JSONArray(available))
            .put("sensor_sampling_status", samplingStatus)
            .put("sensor_samples", samples)
            .put("sensor_capabilities", capabilities)
            .put("cached_motion_sensor_history", motionHistory)
            .put("cached_motion_sensor_history_count", motionHistory.length())
            .put("motion_sensor_history", motionHistory)
            .put("motion_sensor_history_count", motionHistory.length())
            .put("motion_pose_estimates", motionPoseEstimates)
            .put("motion_pose_estimate_count", motionPoseEstimates.length())
            .put("supported_watcher_types", JSONArray(SENSOR_TYPE_LABELS.keys))
            .put("sensor_analyzer_feature_matrix", featureRows)
            .put("sensor_analyzer_workflow_routes", routeRows)
            .put("sensor_sampling_policy_matrix", policyRows)
            .put("sensor_catalog_count", available.size)
            .put("sensor_capability_count", capabilities.length())
            .put("requested_available_sensor_count", requestedAvailableCount)
            .put("sampled_sensor_count", samples.length())
            .put("motion_sensor_count", motionSensorCount)
            .put("ambient_sensor_count", ambientSensorCount)
            .put("wake_up_sensor_count", wakeUpSensorCount)
            .put("direct_channel_sensor_count", directChannelSensorCount)
            .put("sensor_analyzer_feature_count", featureRows.length())
            .put("ready_sensor_analyzer_feature_count", countReadyRows(featureRows))
            .put("sensor_analyzer_workflow_route_count", routeRows.length())
            .put("sensor_sampling_policy_count", policyRows.length())
            .put("cards", cards)
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
        val wifiSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI)
        val bluetoothSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH) ||
            packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)
        val bands = radioBandPlanRows(
            vendorBroadcastRadioDeclared = vendorBroadcastRadioDeclared,
            wifiSupported = wifiSupported,
            bluetoothSupported = bluetoothSupported,
        )
        val featureRows = radioSignalFeatureRows(
            vendorBroadcastRadioDeclared = vendorBroadcastRadioDeclared,
            wifiSupported = wifiSupported,
            bluetoothSupported = bluetoothSupported,
        )
        val routeRows = radioSignalWorkflowRows(
            vendorBroadcastRadioDeclared = vendorBroadcastRadioDeclared,
            wifiSupported = wifiSupported,
            bluetoothSupported = bluetoothSupported,
        )
        val constraintRows = radioSignalConstraintRows(vendorBroadcastRadioDeclared)
        val receiverProfiles = radioReceiverProfileRows(
            vendorBroadcastRadioDeclared = vendorBroadcastRadioDeclared,
            wifiSupported = wifiSupported,
            bluetoothSupported = bluetoothSupported,
        )
        val graphRows = radioSignalGraphRows(JSONObject(), vendorBroadcastRadioDeclared)
        val graphSampleRows = sampledRadioSignalGraphRows(graphRows)
        return JSONObject()
            .put("success", true)
            .put("action", "radio_signal_status")
            .put("am_fm_public_android_scan_supported", false)
            .put("vendor_broadcast_radio_feature_declared", vendorBroadcastRadioDeclared)
            .put("vendor_broadcast_radio_feature_names", JSONArray(BROADCAST_RADIO_FEATURE_NAMES))
            .put("wifi_radio_metadata_supported", wifiSupported)
            .put("bluetooth_radio_metadata_supported", bluetoothSupported)
            .put("general_radio_spectrum_supported", false)
            .put("microwave_spectrum_supported", false)
            .put("requires_external_sdr_for_broad_rf", true)
            .put("radio_bands", bands)
            .put("radio_band_plan_count", bands.length())
            .put("radio_signal_graph_rows", graphRows)
            .put("radio_signal_graph_row_count", graphRows.length())
            .put("radio_signal_graph_sample_rows", graphSampleRows)
            .put("radio_signal_graph_sample_count", graphSampleRows.length())
            .put("radio_signal_graph_bridge_ready", graphSampleRows.length() > 0)
            .put("radio_receiver_profiles", receiverProfiles)
            .put("radio_receiver_profile_count", receiverProfiles.length())
            .put("ready_radio_receiver_profile_count", countReadyRows(receiverProfiles))
            .put("radio_signal_feature_matrix", featureRows)
            .put("radio_signal_workflow_routes", routeRows)
            .put("radio_signal_constraint_matrix", constraintRows)
            .put("radio_signal_feature_count", featureRows.length())
            .put("ready_radio_signal_feature_count", countReadyRows(featureRows))
            .put("radio_signal_workflow_route_count", routeRows.length())
            .put("radio_signal_constraint_count", constraintRows.length())
            .put("radio_scan_rows", JSONArray())
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Radio Band Plan",
                            body = "${bands.length()} radio band row(s) covering AM/FM broadcast boundaries, built-in Wi-Fi/Bluetooth radio metadata, and external SDR-only spectrum paths.",
                            graphType = "radio_frequency_capability",
                            rows = bands,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "AM/FM Signal Graph",
                            body = "${graphRows.length()} AM/FM graph row(s) covering bridge-provided samples plus public Android tuner limitations.",
                            graphType = "radio_signal_graph",
                            rows = graphRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Receiver Profiles",
                            body = "${receiverProfiles.length()} receiver profile row(s) covering AM/FM vendor bridge expectations, Wi-Fi/Bluetooth public metadata routes, and external SDR scan schemas.",
                            graphType = "radio_receiver_profile",
                            rows = receiverProfiles,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Radio Signal Routes",
                            body = "${routeRows.length()} route row(s) for AM/FM explanations, Wi-Fi spectrum graphs, Bluetooth proximity, and external radio hardware.",
                            graphType = "radio_signal_workflow_routes",
                            rows = routeRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Radio Scan Boundaries",
                            body = "${constraintRows.length()} Android API, vendor HAL, permission, and external receiver constraint row(s) for honest RF claims.",
                            graphType = "radio_signal_constraint_matrix",
                            rows = constraintRows,
                        ),
                    ),
            )
    }

    fun radioSignalGraphJson(context: Context, arguments: JSONObject = JSONObject()): JSONObject {
        val appContext = context.applicationContext
        val packageManager = appContext.packageManager
        val vendorBroadcastRadioDeclared = BROADCAST_RADIO_FEATURE_NAMES.any { feature ->
            runCatching { packageManager.hasSystemFeature(feature) }.getOrDefault(false)
        }
        val wifiSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI)
        val bluetoothSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH) ||
            packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)
        val graphRows = radioSignalGraphRows(arguments, vendorBroadcastRadioDeclared)
        val graphSampleRows = sampledRadioSignalGraphRows(graphRows)
        val receiverProfiles = radioReceiverProfileRows(
            vendorBroadcastRadioDeclared = vendorBroadcastRadioDeclared,
            wifiSupported = wifiSupported,
            bluetoothSupported = bluetoothSupported,
        )
        val constraintRows = radioSignalConstraintRows(vendorBroadcastRadioDeclared)
        return JSONObject()
            .put("success", true)
            .put("action", "radio_signal_graph")
            .put("am_fm_public_android_scan_supported", false)
            .put("native_android_public_tuner_scan_supported", false)
            .put("vendor_broadcast_radio_feature_declared", vendorBroadcastRadioDeclared)
            .put("vendor_broadcast_radio_feature_names", JSONArray(BROADCAST_RADIO_FEATURE_NAMES))
            .put("requires_vendor_or_external_receiver_for_am_fm_samples", true)
            .put("radio_signal_graph_bridge_ready", graphSampleRows.length() > 0)
            .put("radio_signal_graph_rows", graphRows)
            .put("radio_signal_graph_row_count", graphRows.length())
            .put("radio_signal_graph_sample_rows", graphSampleRows)
            .put("radio_signal_graph_sample_count", graphSampleRows.length())
            .put("radio_receiver_profiles", receiverProfiles)
            .put("radio_receiver_profile_count", receiverProfiles.length())
            .put("ready_radio_receiver_profile_count", countReadyRows(receiverProfiles))
            .put("radio_signal_constraint_matrix", constraintRows)
            .put("radio_signal_constraint_count", constraintRows.length())
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "AM/FM Signal Graph",
                            body = "${graphSampleRows.length()} bridge sample row(s) and ${graphRows.length() - graphSampleRows.length()} AM/FM boundary row(s) available for graphable radio cards.",
                            graphType = "radio_signal_graph",
                            rows = graphRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Radio Receiver Source Readiness",
                            body = "${receiverProfiles.length()} receiver profile row(s) explain where AM/FM, Wi-Fi, Bluetooth, and SDR graph samples can come from.",
                            graphType = "radio_receiver_profile",
                            rows = receiverProfiles,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Radio Graph Constraints",
                            body = "${constraintRows.length()} Android API, vendor bridge, permission, and external receiver constraint row(s).",
                            graphType = "radio_signal_constraint_matrix",
                            rows = constraintRows,
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
            .put("vendor_broadcast_radio_feature_declared", radioStatus.optBoolean("vendor_broadcast_radio_feature_declared"))
            .put("wifi_radio_metadata_supported", radioStatus.optBoolean("wifi_radio_metadata_supported"))
            .put("bluetooth_radio_metadata_supported", radioStatus.optBoolean("bluetooth_radio_metadata_supported"))
            .put("general_radio_spectrum_supported", radioStatus.optBoolean("general_radio_spectrum_supported"))
            .put("microwave_spectrum_supported", radioStatus.optBoolean("microwave_spectrum_supported"))
            .put("requires_external_sdr_for_broad_rf", radioStatus.optBoolean("requires_external_sdr_for_broad_rf"))
            .put("radio_bands", radioStatus.optJSONArray("radio_bands") ?: JSONArray())
            .put("radio_band_plan_count", radioStatus.optInt("radio_band_plan_count", 0))
            .put("radio_signal_graph_rows", radioStatus.optJSONArray("radio_signal_graph_rows") ?: JSONArray())
            .put("radio_signal_graph_row_count", radioStatus.optInt("radio_signal_graph_row_count", 0))
            .put("radio_signal_graph_sample_rows", radioStatus.optJSONArray("radio_signal_graph_sample_rows") ?: JSONArray())
            .put("radio_signal_graph_sample_count", radioStatus.optInt("radio_signal_graph_sample_count", 0))
            .put("radio_signal_graph_bridge_ready", radioStatus.optBoolean("radio_signal_graph_bridge_ready", false))
            .put("radio_receiver_profiles", radioStatus.optJSONArray("radio_receiver_profiles") ?: JSONArray())
            .put("radio_receiver_profile_count", radioStatus.optInt("radio_receiver_profile_count", 0))
            .put("ready_radio_receiver_profile_count", radioStatus.optInt("ready_radio_receiver_profile_count", 0))
            .put("radio_signal_feature_matrix", radioStatus.optJSONArray("radio_signal_feature_matrix") ?: JSONArray())
            .put("radio_signal_workflow_routes", radioStatus.optJSONArray("radio_signal_workflow_routes") ?: JSONArray())
            .put("radio_signal_constraint_matrix", radioStatus.optJSONArray("radio_signal_constraint_matrix") ?: JSONArray())
            .put("radio_signal_feature_count", radioStatus.optInt("radio_signal_feature_count", 0))
            .put("ready_radio_signal_feature_count", radioStatus.optInt("ready_radio_signal_feature_count", 0))
            .put("radio_signal_workflow_route_count", radioStatus.optInt("radio_signal_workflow_route_count", 0))
            .put("radio_signal_constraint_count", radioStatus.optInt("radio_signal_constraint_count", 0))
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

    fun localBackendRuntimeReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val settings = AppSettingsStore(appContext).load()
        val selectedBackend = BackendKind.fromPersistedValue(settings.onDeviceBackend)
        val currentBackend = OnDeviceBackendManager.currentStatus()
        val runtimeHealth = liteRtRuntimeHealthJson()
        val socProfile = socProfileJson()
        val performanceProfile = devicePerformanceProfileJson(appContext)
        val preferredModel = preferredLocalModelJson(appContext)
        val rows = runtimeBackendMatrixRows(
            selectedBackend = selectedBackend,
            currentBackend = currentBackend,
            runtimeHealth = runtimeHealth,
            socProfile = socProfile,
            preferredModel = preferredModel,
            offlineAirplaneMode = settings.offlineAirplaneMode,
        )
        val stabilityRows = devicePerformanceMatrixRows(performanceProfile, socProfile)
        return JSONObject()
            .put("success", true)
            .put("action", "local_backend_runtime_report")
            .put("report_scope", "Passive local backend runtime health report for LiteRT-LM/AICore/llama.cpp readiness, accelerator visibility, and non-Snapdragon fallback policy.")
            .put("selected_on_device_backend", selectedBackend.persistedValue)
            .put("offline_airplane_mode", settings.offlineAirplaneMode)
            .put("current_local_backend", localBackendStatusJson(currentBackend))
            .put("litert_runtime_health", runtimeHealth)
            .put("soc_profile", socProfile)
            .put("device_performance_profile", performanceProfile)
            .put("preferred_local_model", preferredModel)
            .put("runtime_backend_matrix", rows)
            .put("runtime_stability_matrix", stabilityRows)
            .put("runtime_backend_feature_count", rows.length())
            .put("ready_runtime_backend_feature_count", countReadyRows(rows))
            .put("runtime_stability_feature_count", stabilityRows.length())
            .put("ready_runtime_stability_feature_count", countReadyRows(stabilityRows))
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Runtime Backend Health",
                            body = "${rows.length()} selected-backend, current-runtime, accelerator, artifact, multimodal, and MediaTek fallback row(s) visible without starting a model.",
                            graphType = "runtime_backend_matrix",
                            rows = rows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Thermal & Memory Guardrails",
                            body = "${stabilityRows.length()} thermal, low-RAM, battery, power-saver, performance-class, and non-Adreno stability row(s) for local inference.",
                            graphType = "runtime_stability_matrix",
                            rows = stabilityRows,
                        ),
                    ),
            )
    }

    fun socCompatibilityReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val socProfile = socProfileJson()
        val performanceProfile = devicePerformanceProfileJson(appContext)
        val preferredModel = preferredLocalModelJson(appContext)
        val settings = AppSettingsStore(appContext).load()
        val selectedBackend = BackendKind.fromPersistedValue(settings.onDeviceBackend)
        val currentBackend = OnDeviceBackendManager.currentStatus()
        val runtimeHealth = liteRtRuntimeHealthJson()
        val backendRows = socBackendMatrixRows(socProfile, preferredModel)
        val routeRows = socBackendRouteRows(socProfile, preferredModel)
        val constraintRows = socBackendConstraintRows(socProfile)
        val runtimeRows = runtimeBackendMatrixRows(
            selectedBackend = selectedBackend,
            currentBackend = currentBackend,
            runtimeHealth = runtimeHealth,
            socProfile = socProfile,
            preferredModel = preferredModel,
            offlineAirplaneMode = settings.offlineAirplaneMode,
        )
        val stabilityRows = devicePerformanceMatrixRows(performanceProfile, socProfile)
        return JSONObject()
            .put("success", true)
            .put("action", "soc_compatibility_report")
            .put("report_scope", "Dedicated SOC, GPU, ABI, and LiteRT-LM backend compatibility report for non-Snapdragon Android devices.")
            .put("android_device_identity", deviceIdentityJson())
            .put("soc_profile", socProfile)
            .put("device_performance_profile", performanceProfile)
            .put("preferred_local_model", preferredModel)
            .put("selected_on_device_backend", selectedBackend.persistedValue)
            .put("offline_airplane_mode", settings.offlineAirplaneMode)
            .put("current_local_backend", localBackendStatusJson(currentBackend))
            .put("litert_runtime_health", runtimeHealth)
            .put("likely_mediatek", socProfile.optBoolean("likely_mediatek", false))
            .put("likely_snapdragon", socProfile.optBoolean("likely_snapdragon", false))
            .put("likely_mali_gpu", socProfile.optBoolean("likely_mali_gpu", false))
            .put("likely_powervr_img_gpu", socProfile.optBoolean("likely_powervr_img_gpu", false))
            .put("likely_adreno_gpu", socProfile.optBoolean("likely_adreno_gpu", false))
            .put("supports_arm64", socProfile.optBoolean("supports_arm64", false))
            .put("supports_x86_64", socProfile.optBoolean("supports_x86_64", false))
            .put("soc_backend_matrix", backendRows)
            .put("soc_backend_policy_routes", routeRows)
            .put("soc_backend_constraint_matrix", constraintRows)
            .put("runtime_backend_matrix", runtimeRows)
            .put("runtime_stability_matrix", stabilityRows)
            .put("soc_backend_feature_count", backendRows.length())
            .put("ready_soc_backend_feature_count", countReadyRows(backendRows))
            .put("soc_backend_route_count", routeRows.length())
            .put("soc_backend_constraint_count", constraintRows.length())
            .put("runtime_backend_feature_count", runtimeRows.length())
            .put("ready_runtime_backend_feature_count", countReadyRows(runtimeRows))
            .put("runtime_stability_feature_count", stabilityRows.length())
            .put("ready_runtime_stability_feature_count", countReadyRows(stabilityRows))
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "SOC Compatibility",
                            body = "${backendRows.length()} SOC/GPU/ABI row(s) covering MediaTek, Mali, PowerVR/IMG, Snapdragon, Tensor, Exynos, Unisoc, and CPU fallback policy.",
                            graphType = "soc_backend_matrix",
                            rows = backendRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Backend Routes",
                            body = "${routeRows.length()} route row(s) for choosing diagnostics, model readiness, and cross-signal context before local inference.",
                            graphType = "soc_backend_policy_routes",
                            rows = routeRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Native ABI Policy",
                            body = "${constraintRows.length()} constraint row(s) for ABI coverage, GPU probing, CPU fallback, and emulator/device separation.",
                            graphType = "soc_backend_constraint_matrix",
                            rows = constraintRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Runtime Backend Health",
                            body = "${runtimeRows.length()} passive runtime row(s) covering selected backend, current local state, /health accelerator fields, artifact compatibility, and MediaTek fallback guidance.",
                            graphType = "runtime_backend_matrix",
                            rows = runtimeRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Thermal & Memory Guardrails",
                            body = "${stabilityRows.length()} stability row(s) for throttling, low-RAM, battery, power saver, and Android media performance class on MediaTek/non-Adreno devices.",
                            graphType = "runtime_stability_matrix",
                            rows = stabilityRows,
                        ),
                    ),
            )
    }

    fun gpuBackendRiskReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val socProfile = socProfileJson()
        val performanceProfile = devicePerformanceProfileJson(appContext)
        val preferredModel = preferredLocalModelJson(appContext)
        val settings = AppSettingsStore(appContext).load()
        val selectedBackend = BackendKind.fromPersistedValue(settings.onDeviceBackend)
        val currentBackend = OnDeviceBackendManager.currentStatus()
        val runtimeHealth = liteRtRuntimeHealthJson()
        val deviceIdentity = deviceIdentityJson()
        val riskRows = gpuBackendRiskMatrixRows(
            socProfile = socProfile,
            performanceProfile = performanceProfile,
            runtimeHealth = runtimeHealth,
            currentBackend = currentBackend,
            preferredModel = preferredModel,
            selectedBackend = selectedBackend,
            offlineAirplaneMode = settings.offlineAirplaneMode,
            deviceIdentity = deviceIdentity,
        )
        val routeRows = gpuBackendRiskRouteRows()
        val maxRiskScore = maxRiskScore(riskRows)
        return JSONObject()
            .put("success", true)
            .put("action", "gpu_backend_risk_report")
            .put("report_scope", "Operational GPU/backend risk matrix for MediaTek, Mali, PowerVR/IMG, Xclipse, Adreno, Tensor, Exynos, Unisoc, and CPU fallback decisions.")
            .put("android_device_identity", deviceIdentity)
            .put("soc_profile", socProfile)
            .put("device_performance_profile", performanceProfile)
            .put("preferred_local_model", preferredModel)
            .put("selected_on_device_backend", selectedBackend.persistedValue)
            .put("offline_airplane_mode", settings.offlineAirplaneMode)
            .put("current_local_backend", localBackendStatusJson(currentBackend))
            .put("litert_runtime_health", runtimeHealth)
            .put("gpu_backend_risk_level", riskLevelForScore(maxRiskScore))
            .put("gpu_backend_risk_score", maxRiskScore)
            .put("gpu_backend_risk_matrix", riskRows)
            .put("gpu_backend_risk_routes", routeRows)
            .put("gpu_backend_risk_count", riskRows.length())
            .put("high_gpu_backend_risk_count", countHighRiskRows(riskRows))
            .put("ready_gpu_backend_risk_count", countReadyRows(riskRows))
            .put("gpu_backend_risk_route_count", routeRows.length())
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "GPU Backend Risk",
                            body = "${riskRows.length()} accelerator, SOC/GPU, thermal, memory, power, model, and validation-scope risk row(s) for non-Adreno local inference.",
                            graphType = "gpu_backend_risk_matrix",
                            rows = riskRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Backend Risk Routes",
                            body = "${routeRows.length()} route row(s) for moving from risk triage into runtime, SOC, stability, or phone preflight diagnostics.",
                            graphType = "gpu_backend_risk_routes",
                            rows = routeRows,
                        ),
                    ),
            )
    }

    fun localInferenceCompatibilityReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val socProfile = socProfileJson()
        val performanceProfile = devicePerformanceProfileJson(appContext)
        val preferredModel = preferredLocalModelJson(appContext)
        val settings = AppSettingsStore(appContext).load()
        val selectedBackend = BackendKind.fromPersistedValue(settings.onDeviceBackend)
        val currentBackend = OnDeviceBackendManager.currentStatus()
        val runtimeHealth = liteRtRuntimeHealthJson()
        val deviceIdentity = deviceIdentityJson()
        val riskRows = gpuBackendRiskMatrixRows(
            socProfile = socProfile,
            performanceProfile = performanceProfile,
            runtimeHealth = runtimeHealth,
            currentBackend = currentBackend,
            preferredModel = preferredModel,
            selectedBackend = selectedBackend,
            offlineAirplaneMode = settings.offlineAirplaneMode,
            deviceIdentity = deviceIdentity,
        )
        val compatibilityRows = localInferenceCompatibilityRows(
            socProfile = socProfile,
            performanceProfile = performanceProfile,
            runtimeHealth = runtimeHealth,
            currentBackend = currentBackend,
            preferredModel = preferredModel,
            selectedBackend = selectedBackend,
            offlineAirplaneMode = settings.offlineAirplaneMode,
            deviceIdentity = deviceIdentity,
            riskRows = riskRows,
        )
        val compatibilityScore = averageCapabilityValue(compatibilityRows)
        val maxRiskScore = maxRiskScore(riskRows)
        return JSONObject()
            .put("success", true)
            .put("action", "local_inference_compatibility_report")
            .put("report_scope", "Fused local inference compatibility scorecard for MediaTek, Mali, PowerVR/IMG, Xclipse, Adreno, Tensor, Exynos, Unisoc, and CPU fallback decisions.")
            .put("source_report_actions", JSONArray().put("soc_compatibility_report").put("gpu_backend_risk_report").put("local_backend_runtime_report").put("device_performance_report"))
            .put("android_device_identity", deviceIdentity)
            .put("soc_profile", socProfile)
            .put("device_performance_profile", performanceProfile)
            .put("preferred_local_model", preferredModel)
            .put("selected_on_device_backend", selectedBackend.persistedValue)
            .put("offline_airplane_mode", settings.offlineAirplaneMode)
            .put("current_local_backend", localBackendStatusJson(currentBackend))
            .put("litert_runtime_health", runtimeHealth)
            .put("gpu_backend_risk_level", riskLevelForScore(maxRiskScore))
            .put("gpu_backend_risk_score", maxRiskScore)
            .put("gpu_backend_risk_matrix", riskRows)
            .put("local_inference_compatibility_score", compatibilityScore)
            .put("local_inference_compatibility_level", compatibilityLevelForScore(compatibilityScore))
            .put("local_inference_compatibility_matrix", compatibilityRows)
            .put("local_inference_compatibility_count", compatibilityRows.length())
            .put("ready_local_inference_compatibility_count", countReadyRows(compatibilityRows))
            .put(
                "gemma_observation_directives",
                JSONArray()
                    .put("Read local_inference_compatibility_matrix before promising local GPU acceleration or offline multimodal readiness.")
                    .put("For MediaTek, Mali, PowerVR/IMG, Xclipse, or other non-Adreno phones, treat GPU acceptance as unproven until /health reports gpu without CPU fallback.")
                    .put("Use source_report_actions to drill into SOC policy, GPU risk, runtime health, or thermal stability rows when a scorecard row is weak."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Local Inference Compatibility",
                            body = "${compatibilityRows.length()} scorecard row(s) fusing SOC/GPU policy, model artifact fit, runtime accelerator state, thermal/memory runway, and phone-validation scope.",
                            graphType = "local_inference_compatibility_matrix",
                            rows = compatibilityRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "GPU Backend Risk",
                            body = "${riskRows.length()} risk row(s) backing the compatibility scorecard.",
                            graphType = "gpu_backend_risk_matrix",
                            rows = riskRows,
                        ),
                    ),
            )
    }

    fun devicePerformanceReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val socProfile = socProfileJson()
        val performanceProfile = devicePerformanceProfileJson(appContext)
        val rows = devicePerformanceMatrixRows(performanceProfile, socProfile)
        return JSONObject()
            .put("success", true)
            .put("action", "device_performance_report")
            .put("report_scope", "Passive thermal, memory, battery, power-saver, and Android performance-class guardrails for stable local inference on MediaTek, Mali, PowerVR/IMG, and other non-Adreno phones.")
            .put("soc_profile", socProfile)
            .put("device_performance_profile", performanceProfile)
            .put("thermal_status", performanceProfile.optInt("thermal_status", -1))
            .put("thermal_status_label", performanceProfile.optString("thermal_status_label"))
            .put("thermal_api_supported", performanceProfile.optBoolean("thermal_api_supported", false))
            .put("power_save_mode", performanceProfile.optBoolean("power_save_mode", false))
            .put("low_ram_device", performanceProfile.optBoolean("low_ram_device", false))
            .put("memory_class_mb", performanceProfile.optInt("memory_class_mb", 0))
            .put("large_memory_class_mb", performanceProfile.optInt("large_memory_class_mb", 0))
            .put("media_performance_class", performanceProfile.optInt("media_performance_class", 0))
            .put("battery_status_label", performanceProfile.optString("battery_status_label"))
            .put("runtime_stability_matrix", rows)
            .put("runtime_stability_feature_count", rows.length())
            .put("ready_runtime_stability_feature_count", countReadyRows(rows))
            .put(
                "notes",
                JSONArray()
                    .put("This report is passive: it reads Android thermal, memory, power, and battery state and does not start a model or scan hardware.")
                    .put("Use it with SOC compatibility and local backend reports before judging MediaTek/Mali/PowerVR runtime stability."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Thermal & Memory Guardrails",
                            body = "${rows.length()} passive stability row(s) covering thermal throttling, memory class, low-RAM, battery/power state, Android media performance class, and non-Adreno local inference guardrails.",
                            graphType = "runtime_stability_matrix",
                            rows = rows,
                        ),
                    ),
            )
    }

    fun signalAwarenessReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val settings = AppSettingsStore(appContext).load()
        val diagnostics = statusJson(appContext)
        val signalStatus = signalCapabilityStatusJson(appContext)
        val radioStatus = radioSignalStatusJson(appContext)
        val preferredModel = preferredLocalModelJson(appContext)
        val backendRiskReport = gpuBackendRiskReportJson(appContext)
        val socProfile = diagnostics.optJSONObject("soc_profile") ?: socProfileJson()
        val availableSensors = diagnostics.optJSONArray("available_sensor_types") ?: JSONArray()
        val cachedWifiHistory = wifiSignalHistoryRowsFromStore(readWifiSignalHistory(appContext))
        val cachedBluetoothHistory = bluetoothSignalHistoryRowsFromStore(readBluetoothSignalHistory(appContext))
        val cachedMotionHistory = motionSensorHistoryRowsFromStore(readMotionSensorHistory(appContext))
        val cachedMotionPoseEstimates = motionPoseEstimateRows(JSONArray(), cachedMotionHistory)
        val awarenessRows = signalAwarenessRows(
            diagnostics = diagnostics,
            signalStatus = signalStatus,
            radioStatus = radioStatus,
            preferredModel = preferredModel,
            socProfile = socProfile,
            availableSensors = availableSensors,
            cachedWifiHistory = cachedWifiHistory,
            cachedBluetoothHistory = cachedBluetoothHistory,
            cachedMotionHistory = cachedMotionHistory,
            cachedMotionPoseEstimates = cachedMotionPoseEstimates,
            backendRiskReport = backendRiskReport,
        )
        val workflowRows = signalWorkflowRouteRows(
            diagnostics = diagnostics,
            signalStatus = signalStatus,
            preferredModel = preferredModel,
            availableSensors = availableSensors,
            cachedWifiHistory = cachedWifiHistory,
            cachedBluetoothHistory = cachedBluetoothHistory,
            cachedMotionHistory = cachedMotionHistory,
            cachedMotionPoseEstimates = cachedMotionPoseEstimates,
            backendRiskReport = backendRiskReport,
        )
        val constraintRows = signalConstraintRows(diagnostics, signalStatus, radioStatus)
        val radioBandCount = radioStatus.optJSONArray("radio_bands")?.length() ?: 0
        return JSONObject()
            .put("success", true)
            .put("action", "signal_awareness_report")
            .put("report_scope", "Cross-signal situational awareness across Wi-Fi, Bluetooth, AM/FM/RF limits, sensors, SOC/backend risk, and local multimodal readiness.")
            .put("android_device_identity", deviceIdentityJson())
            .put("soc_profile", socProfile)
            .put("preferred_local_model", preferredModel)
            .put("gpu_backend_risk_level", backendRiskReport.optString("gpu_backend_risk_level"))
            .put("gpu_backend_risk_score", backendRiskReport.optInt("gpu_backend_risk_score", 0))
            .put("gpu_backend_risk_matrix", backendRiskReport.optJSONArray("gpu_backend_risk_matrix") ?: JSONArray())
            .put("gpu_backend_risk_routes", backendRiskReport.optJSONArray("gpu_backend_risk_routes") ?: JSONArray())
            .put("gpu_backend_risk_count", backendRiskReport.optInt("gpu_backend_risk_count", 0))
            .put("high_gpu_backend_risk_count", backendRiskReport.optInt("high_gpu_backend_risk_count", 0))
            .put("ready_gpu_backend_risk_count", backendRiskReport.optInt("ready_gpu_backend_risk_count", 0))
            .put("gpu_backend_risk_route_count", backendRiskReport.optInt("gpu_backend_risk_route_count", 0))
            .put("signal_capability_status", compactSignalCapabilityJson(signalStatus))
            .put("wifi_scan_permission_status", diagnostics.optJSONObject("wifi_scan_permission_status") ?: JSONObject())
            .put("bluetooth_scan_permission_status", diagnostics.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject())
            .put("cached_wifi_signal_history", cachedWifiHistory)
            .put("cached_wifi_history_network_count", cachedWifiHistory.length())
            .put("cached_bluetooth_signal_history", cachedBluetoothHistory)
            .put("cached_bluetooth_history_device_count", cachedBluetoothHistory.length())
            .put("cached_motion_sensor_history", cachedMotionHistory)
            .put("cached_motion_sensor_history_count", cachedMotionHistory.length())
            .put("cached_motion_pose_estimates", cachedMotionPoseEstimates)
            .put("cached_motion_pose_estimate_count", cachedMotionPoseEstimates.length())
            .put("radio_bands", radioStatus.optJSONArray("radio_bands") ?: JSONArray())
            .put("radio_band_plan_count", radioStatus.optInt("radio_band_plan_count", 0))
            .put("radio_receiver_profiles", radioStatus.optJSONArray("radio_receiver_profiles") ?: JSONArray())
            .put("radio_receiver_profile_count", radioStatus.optInt("radio_receiver_profile_count", 0))
            .put("ready_radio_receiver_profile_count", radioStatus.optInt("ready_radio_receiver_profile_count", 0))
            .put("radio_signal_feature_matrix", radioStatus.optJSONArray("radio_signal_feature_matrix") ?: JSONArray())
            .put("radio_signal_workflow_routes", radioStatus.optJSONArray("radio_signal_workflow_routes") ?: JSONArray())
            .put("radio_signal_constraint_matrix", radioStatus.optJSONArray("radio_signal_constraint_matrix") ?: JSONArray())
            .put("radio_signal_feature_count", radioStatus.optInt("radio_signal_feature_count", 0))
            .put("ready_radio_signal_feature_count", radioStatus.optInt("ready_radio_signal_feature_count", 0))
            .put("radio_signal_workflow_route_count", radioStatus.optInt("radio_signal_workflow_route_count", 0))
            .put("radio_signal_constraint_count", radioStatus.optInt("radio_signal_constraint_count", 0))
            .put("signal_awareness_matrix", awarenessRows)
            .put("signal_workflow_routes", workflowRows)
            .put("signal_constraint_matrix", constraintRows)
            .put("signal_awareness_count", awarenessRows.length())
            .put("ready_signal_awareness_count", countReadyRows(awarenessRows))
            .put("signal_workflow_route_count", workflowRows.length())
            .put("signal_constraint_count", constraintRows.length())
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Signal Awareness",
                            body = "${awarenessRows.length()} fused row(s) covering Wi-Fi, Bluetooth, radio limits, sensors, SOC, and local model context.",
                            graphType = "signal_awareness_matrix",
                            rows = awarenessRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Signal Routes",
                            body = "${workflowRows.length()} next-tool route row(s) for choosing the right scanner or sensor path.",
                            graphType = "signal_workflow_routes",
                            rows = workflowRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Signal Constraints",
                            body = "${constraintRows.length()} Android permission, hardware, and public-API constraints for honest radio/sensor reasoning.",
                            graphType = "signal_constraint_matrix",
                            rows = constraintRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "GPU Backend Risk",
                            body = "${backendRiskReport.optInt("gpu_backend_risk_count", 0)} backend/SOC risk row(s) at ${backendRiskReport.optString("gpu_backend_risk_level").ifBlank { "unknown" }} level for MediaTek, Mali, PowerVR, and fallback decisions.",
                            graphType = "gpu_backend_risk_matrix",
                            rows = backendRiskReport.optJSONArray("gpu_backend_risk_matrix") ?: JSONArray(),
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Radio Band Plan",
                            body = "$radioBandCount AM/FM, Wi-Fi, Bluetooth, and SDR band row(s) available to the signal-awareness report.",
                            graphType = "radio_frequency_capability",
                            rows = radioStatus.optJSONArray("radio_bands") ?: JSONArray(),
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Receiver Profiles",
                            body = "${radioStatus.optInt("radio_receiver_profile_count", 0)} receiver profile row(s) describing AM/FM vendor bridge, Wi-Fi/Bluetooth metadata, and SDR scan schemas.",
                            graphType = "radio_receiver_profile",
                            rows = radioStatus.optJSONArray("radio_receiver_profiles") ?: JSONArray(),
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Motion Sensor History",
                            body = "${cachedMotionHistory.length()} cached motion/IMU trend row(s) available to the signal-awareness report.",
                            graphType = "motion_sensor_history",
                            rows = cachedMotionHistory,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Motion Pose Estimate",
                            body = "${cachedMotionPoseEstimates.length()} cached pose, heading, angular-motion, or acceleration row(s) available to the signal-awareness report.",
                            graphType = "motion_pose_estimate",
                            rows = cachedMotionPoseEstimates,
                        ),
                    ),
            )
    }

    fun agentEnvironmentReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val settings = AppSettingsStore(appContext).load()
        val diagnostics = statusJson(appContext)
        val signalStatus = signalCapabilityStatusJson(appContext)
        val radioStatus = radioSignalStatusJson(appContext)
        val preferredModel = preferredLocalModelJson(appContext)
        val hindsightStatus = HermesHindsightMemoryBridge.statusJson(appContext)
        val automationStatus = runCatching {
            JSONObject(HermesAutomationBridge.performActionJson(appContext, "operator_standby_status"))
        }.getOrDefault(JSONObject())
        val modelRouting = runCatching {
            JSONObject(HermesAutomationBridge.performActionJson(appContext, "operator_model_routing"))
        }.getOrDefault(JSONObject())
        val personaStatus = agentPersonaStatusJson(settings)
        val socProfile = diagnostics.optJSONObject("soc_profile") ?: socProfileJson()
        val availableSensors = diagnostics.optJSONArray("available_sensor_types") ?: JSONArray()
        val capabilityRows = agentCapabilityMatrixRows(
            context = appContext,
            diagnostics = diagnostics,
            signalStatus = signalStatus,
            radioStatus = radioStatus,
            preferredModel = preferredModel,
            hindsightStatus = hindsightStatus,
            automationStatus = automationStatus,
            modelRouting = modelRouting,
            socProfile = socProfile,
            availableSensors = availableSensors,
        )
        val kaiParityRows = kaiParityMatrixRows(preferredModel, hindsightStatus, automationStatus, modelRouting, personaStatus)
        val kaiOperationsRows = kaiOperationsMatrixRows(automationStatus, modelRouting, personaStatus)
        val readinessRows = workflowReadinessRows(diagnostics, signalStatus, preferredModel, automationStatus, modelRouting, availableSensors)
        return JSONObject()
            .put("success", true)
            .put("action", "agent_environment_report")
            .put("report_scope", "Hermes agent environment, Kai parity and operations, wireless/radio/sensor inputs, and SOC/backend compatibility context.")
            .put("android_device_identity", deviceIdentityJson())
            .put("soc_profile", socProfile)
            .put("preferred_local_model", preferredModel)
            .put("model_routing", compactModelRoutingJson(modelRouting))
            .put("agent_persona_status", personaStatus)
            .put("hindsight_memory_status", compactHindsightStatusJson(hindsightStatus))
            .put("automation_standby_status", compactAutomationStandbyStatusJson(automationStatus))
            .put("signal_capability_status", compactSignalCapabilityJson(signalStatus))
            .put("agent_capability_matrix", capabilityRows)
            .put("kai_parity_matrix", kaiParityRows)
            .put("kai_operations_matrix", kaiOperationsRows)
            .put("workflow_readiness_matrix", readinessRows)
            .put("agent_capability_count", capabilityRows.length())
            .put("ready_capability_count", countReadyRows(capabilityRows))
            .put("kai_parity_count", kaiParityRows.length())
            .put("kai_operations_count", kaiOperationsRows.length())
            .put("ready_kai_operations_count", countReadyRows(kaiOperationsRows))
            .put("workflow_readiness_count", readinessRows.length())
            .put(
                "ai_experience_elevation_plan",
                JSONArray()
                    .put("Use wifi_scan/wifi_ap_details/wifi_export when the agent needs channel, security, vendor, distance, or exportable AP metadata for a nearby network decision.")
                    .put("Use bluetooth_scan when the agent needs nearby device, service UUID, manufacturer, RSSI, or proximity context.")
                    .put("Use sensor_snapshot or sensor watchers for motion, orientation, ambient, and workflow-trigger context.")
                    .put("Use radio_signal_status/signal_capability_status to explain AM/FM/RF limits honestly and route broad RF work to external SDR/vendor hardware.")
                    .put("Use SOC and LiteRT backend policy fields to avoid Snapdragon-only assumptions and keep MediaTek/Mali/PowerVR devices on GPU-first with CPU fallback when available.")
                    .put("Use hindsight_memory_tool and operator heartbeat/status rows to retain durable context and expose autonomous task readiness.")
                    .put("Use Settings Agent persona plus secret-free app settings export/import for Kai-style customizable soul/system prompt behavior.")
                    .put("Use kai_operations_matrix to route Kai-style provider fallback, tool bridge, configurable persona, encrypted storage, secret-free settings backup, automation backup, TTS, image, and shell-boundary work through native Hermes surfaces."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Agent Environment",
                            body = "${capabilityRows.length()} capability row(s) covering wireless, radio, sensors, local model, UI, automation, memory, and SOC backend policy.",
                            graphType = "agent_capability_matrix",
                            rows = capabilityRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Kai Parity",
                            body = "${kaiParityRows.length()} Kai-inspired parity row(s) for memory, customizable persona, on-device inference, tools, heartbeat, app settings backup/restore, and multimodal UX.",
                            graphType = "kai_parity_matrix",
                            rows = kaiParityRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Kai Operations",
                            body = "${kaiOperationsRows.length()} operation row(s) mapping Kai-style provider fallback, configurable persona, tool bridge, secure storage, settings/automation backup, TTS, and shell capabilities onto Hermes Android routes.",
                            graphType = "kai_operations_matrix",
                            rows = kaiOperationsRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Workflow Readiness",
                            body = "${readinessRows.length()} readiness row(s) that tell Gemma which native capability to call next.",
                            graphType = "agent_workflow_readiness",
                            rows = readinessRows,
                        ),
                    ),
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

    private fun agentCapabilityMatrixRows(
        context: Context,
        diagnostics: JSONObject,
        signalStatus: JSONObject,
        radioStatus: JSONObject,
        preferredModel: JSONObject,
        hindsightStatus: JSONObject,
        automationStatus: JSONObject,
        modelRouting: JSONObject,
        socProfile: JSONObject,
        availableSensors: JSONArray,
    ): JSONArray {
        val wifiPermission = diagnostics.optJSONObject("wifi_scan_permission_status") ?: JSONObject()
        val bluetoothPermission = diagnostics.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject()
        val motionSensors = jsonStringList(signalStatus.optJSONArray("motion_sensor_analysis_supported"))
        val packageManager = context.packageManager
        val wifiSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_WIFI)
        val bluetoothSupported = packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH)
        val screenshotSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
        return JSONArray()
            .put(
                capabilityRow(
                    category = "local_model",
                    label = "On-device inference",
                    ready = preferredModel.optBoolean("ready"),
                    valueLabel = preferredModel.optString("title").ifBlank { "model import needed" },
                    detail = listOf(
                        preferredModel.optString("runtime_flavor").ifBlank { "unknown runtime" },
                        preferredModel.optString("record_status").ifBlank { "no preferred record" },
                    ).joinToString(" | "),
                    recommendation = "Use a preferred Gemma 4 LiteRT-LM or Qwen GGUF model before treating the phone as fully autonomous offline.",
                    fraction = if (preferredModel.optBoolean("ready")) 1f else 0.35f,
                ),
            )
            .put(
                capabilityRow(
                    category = "multimodal",
                    label = "Vision and image attachments",
                    ready = modelRouting.optBoolean("vision_capable", false) && screenshotSupported,
                    valueLabel = if (modelRouting.optBoolean("vision_capable", false)) "vision-ready" else "text-first",
                    detail = "Chat image attachments and accessibility screenshots are available for Gemma/VLM workflows when the active model supports images.",
                    recommendation = "Use android_ui_tool visual_snapshot or chat image attachments when semantic UI text is insufficient.",
                    fraction = if (modelRouting.optBoolean("vision_capable", false)) 0.9f else 0.45f,
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend",
                    label = "SOC and LiteRT backend policy",
                    ready = true,
                    valueLabel = socProfile.optString("litert_lm_acceleration_label").ifBlank { socProfile.optString("soc_family_label").ifBlank { "Android SOC" } },
                    detail = listOf(
                        socProfile.optString("soc_family_label").ifBlank { "unknown SOC" },
                        socProfile.optString("gpu_family_label").ifBlank { "unknown GPU" },
                        socProfile.optString("primary_abi").ifBlank { "unknown ABI" },
                    ).joinToString(" | "),
                    recommendation = socProfile.optString("litert_lm_backend_strategy").ifBlank { "Prefer SOC-neutral GPU probing with CPU fallback; do not assume Snapdragon/Adreno only." },
                    fraction = 0.95f,
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi",
                    label = "Wi-Fi analyzer",
                    ready = wifiSupported && wifiPermission.optBoolean("can_read_scan_results", false),
                    valueLabel = if (wifiSupported) {
                        if (wifiPermission.optBoolean("can_read_scan_results", false)) "scan-ready" else "permission needed"
                    } else {
                        "no Wi-Fi feature"
                    },
                    detail = "Supports AP details, channel ratings, security/width/standard summaries, signal history, and exportable rows.",
                    recommendation = "Call wifi_scan, wifi_ap_details, or wifi_export for nearby signal/channel decisions.",
                    fraction = if (wifiSupported && wifiPermission.optBoolean("can_read_scan_results", false)) 1f else if (wifiSupported) 0.55f else 0.1f,
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth",
                    label = "Bluetooth scanner",
                    ready = bluetoothSupported && bluetoothPermission.optBoolean("can_scan_nearby_devices", false),
                    valueLabel = if (bluetoothSupported) {
                        if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) "scan-ready" else "permission needed"
                    } else {
                        "no Bluetooth feature"
                    },
                    detail = "Supports paired/BLE rows, RSSI proximity, service UUID labels, manufacturer names, and metadata cards.",
                    recommendation = "Call bluetooth_scan when nearby device identity, proximity, or BLE advertisement context matters.",
                    fraction = if (bluetoothSupported && bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) 1f else if (bluetoothSupported) 0.55f else 0.1f,
                ),
            )
            .put(
                capabilityRow(
                    category = "radio",
                    label = "AM/FM and broad RF",
                    ready = radioStatus.optBoolean("am_fm_public_android_scan_supported", false),
                    valueLabel = if (radioStatus.optBoolean("requires_external_sdr_for_broad_rf", true)) "external SDR needed" else "built-in scan available",
                    detail = "Android public APIs expose Wi-Fi/Bluetooth/audio/sensors, but not a general AM/FM/microwave scan feed on normal phones.",
                    recommendation = "Use radio_signal_status for honest capability cards and route broad RF work to vendor APIs or external SDR hardware.",
                    fraction = if (radioStatus.optBoolean("am_fm_public_android_scan_supported", false)) 0.8f else 0.25f,
                ),
            )
            .put(
                capabilityRow(
                    category = "sensors",
                    label = "Motion and environmental sensors",
                    ready = motionSensors.isNotEmpty(),
                    valueLabel = "${motionSensors.size} motion type(s)",
                    detail = "Available sensors: ${jsonStringList(availableSensors).take(8).joinToString(", ").ifBlank { "none reported" }}",
                    recommendation = "Call sensor_snapshot or start_sensor_watcher for accelerometer, gyroscope, magnetic, ambient, and workflow-trigger context.",
                    fraction = (motionSensors.size / 5f).coerceIn(0.1f, 1f),
                ),
            )
            .put(
                capabilityRow(
                    category = "ui_control",
                    label = "UI perception and control",
                    ready = HermesAccessibilityController.isServiceConnected(),
                    valueLabel = if (HermesAccessibilityController.isServiceConnected()) "accessibility connected" else "enable accessibility",
                    detail = "Accessibility snapshots, screenshot hashes, OpenGUI-style VLM actions, and guarded UI actions are available when the service is enabled.",
                    recommendation = "Call android_ui_tool status/sense/visual_snapshot before sensitive UI actions or external sends.",
                    fraction = if (HermesAccessibilityController.isServiceConnected()) 1f else if (HermesAccessibilityController.isServiceEnabled(context)) 0.65f else 0.35f,
                ),
            )
            .put(
                capabilityRow(
                    category = "automation",
                    label = "Autonomous heartbeat and dispatch",
                    ready = automationStatus.optBoolean("standby_heartbeat_supported", false),
                    valueLabel = if (automationStatus.optBoolean("standby_heartbeat_supported", false)) "heartbeat-ready" else "not configured",
                    detail = "${automationStatus.optInt("enabled_automation_count", 0)} enabled automation(s), ${automationStatus.optInt("recent_run_count", 0)} recent run(s).",
                    recommendation = "Use android_automation_tool operator_standby_status/operator_heartbeat for Kai-style autonomous status and remote dispatch.",
                    fraction = if (automationStatus.optBoolean("standby_heartbeat_supported", false)) 0.9f else 0.4f,
                ),
            )
            .put(
                capabilityRow(
                    category = "memory",
                    label = "Persistent hindsight memory",
                    ready = true,
                    valueLabel = "${hindsightStatus.optInt("memory_count", 0)} memories",
                    detail = "${hindsightStatus.optInt("reinforced_memory_count", 0)} reinforced, ${hindsightStatus.optInt("promoted_memory_count", 0)} promoted for prompt context.",
                    recommendation = "Use hindsight_memory_tool retain/recall/reflect/promoted_context around complex work.",
                    fraction = ((hindsightStatus.optInt("promoted_memory_count", 0) + 1) / 5f).coerceIn(0.25f, 1f),
                ),
            )
    }

    fun agentObservationReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val wifiReport = wifiAnalyzerReportJson(appContext, JSONObject().put("refresh", false))
        val bluetoothReport = bluetoothAnalyzerReportJson(appContext, JSONObject().put("refresh", false))
        val sensorReport = sensorAnalyzerReportJson(appContext, JSONObject().put("include_snapshot", false))
        val radioReport = radioSignalStatusJson(appContext)
        val signalReport = signalAwarenessReportJson(appContext)
        val backendRiskReport = gpuBackendRiskReportJson(appContext)
        val environmentReport = agentEnvironmentReportJson(appContext)
        val signalContextRows = agentSignalContextFusionRows(
            wifiReport = wifiReport,
            bluetoothReport = bluetoothReport,
            sensorReport = sensorReport,
            radioReport = radioReport,
            signalReport = signalReport,
            backendRiskReport = backendRiskReport,
        )
        val cardManifestSources = agentCardManifestSources(
            wifiReport = wifiReport,
            bluetoothReport = bluetoothReport,
            sensorReport = sensorReport,
            radioReport = radioReport,
            backendRiskReport = backendRiskReport,
            signalReport = signalReport,
            environmentReport = environmentReport,
        )
        val cardManifestRows = agentCardManifestRows(cardManifestSources)
        val observationRows = agentObservationMatrixRows(
            wifiReport = wifiReport,
            bluetoothReport = bluetoothReport,
            sensorReport = sensorReport,
            radioReport = radioReport,
            signalReport = signalReport,
            backendRiskReport = backendRiskReport,
            environmentReport = environmentReport,
            signalContextRows = signalContextRows,
            cardManifestRows = cardManifestRows,
        )
        val routeRows = agentObservationRouteRows()
        return JSONObject()
            .put("success", true)
            .put("action", "agent_observation_report")
            .put("report_scope", "Gemma-visible dashboard for Wi-Fi, Bluetooth, sensor, radio, SOC/backend risk, local model, Kai operations, and expandable top-card routes.")
            .put("source_report_actions", agentCardManifestSourceActions(cardManifestSources))
            .put("wifi_observation_summary", observationSummaryJson(wifiReport, "wifi_analyzer_report"))
            .put("bluetooth_observation_summary", observationSummaryJson(bluetoothReport, "bluetooth_analyzer_report"))
            .put("sensor_observation_summary", observationSummaryJson(sensorReport, "sensor_analyzer_report"))
            .put("radio_observation_summary", observationSummaryJson(radioReport, "radio_signal_status"))
            .put("backend_risk_observation_summary", observationSummaryJson(backendRiskReport, "gpu_backend_risk_report"))
            .put("signal_observation_summary", observationSummaryJson(signalReport, "signal_awareness_report"))
            .put("agent_environment_observation_summary", observationSummaryJson(environmentReport, "agent_environment_report"))
            .put("agent_signal_context_matrix", signalContextRows)
            .put("agent_signal_context_count", signalContextRows.length())
            .put("ready_agent_signal_context_count", countReadyRows(signalContextRows))
            .put("agent_card_manifest", cardManifestRows)
            .put("agent_card_manifest_count", cardManifestRows.length())
            .put("ready_agent_card_manifest_count", countReadyRows(cardManifestRows))
            .put("agent_card_graph_types", cardGraphTypeList(cardManifestRows))
            .put("agent_observation_matrix", observationRows)
            .put("agent_observation_routes", routeRows)
            .put("agent_observation_count", observationRows.length())
            .put("ready_agent_observation_count", countReadyRows(observationRows))
            .put("agent_observation_route_count", routeRows.length())
            .put(
                "gemma_observation_directives",
                JSONArray()
                    .put("Read agent_observation_matrix first to decide which signal, sensor, radio, SOC, or Kai surface is actually available on this device.")
                    .put("Read agent_signal_context_matrix before summarizing nearby RF context so Wi-Fi channel/band rows, Bluetooth RSSI/history rows, motion pose rows, and radio hardware limits stay fused.")
                    .put("Read agent_card_manifest to choose the exact expandable graph card and refresh policy before asking for live Wi-Fi, Bluetooth, radio, sensor, or backend data.")
                    .put("Read gpu_backend_risk_report context before promising MediaTek, Mali, PowerVR, Xclipse, or non-Adreno local model acceleration.")
                    .put("Use agent_observation_routes for the next precise android_device_diagnostics_tool action instead of guessing from text.")
                    .put("Open the expandable cards before explaining Wi-Fi channels, Bluetooth proximity, radio limits, motion context, backend risk, or local model readiness.")
                    .put("Prefer passive analyzer reports for planning; request refresh or active sampling only when the user needs current live data."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Agent Observation",
                            body = "${observationRows.length()} Gemma-visible row(s) summarizing Wi-Fi, Bluetooth, sensors, radio, SOC/backend risk, local model, Kai operations, and card coverage.",
                            graphType = "agent_observation_matrix",
                            rows = observationRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Signal Context Fusion",
                            body = "${signalContextRows.length()} fused context row(s) combining Wi-Fi channel/band coverage, Bluetooth RSSI/history metadata, motion pose context, radio hardware limits, and backend risk for Gemma.",
                            graphType = "agent_signal_context_matrix",
                            rows = signalContextRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Agent Card Manifest",
                            body = "${cardManifestRows.length()} source card row(s) mapping graph types to tool actions, refresh policy, and permission gates for Gemma.",
                            graphType = "agent_card_manifest",
                            rows = cardManifestRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Observation Routes",
                            body = "${routeRows.length()} direct route row(s) for opening the right analyzer or dashboard card next.",
                            graphType = "agent_observation_routes",
                            rows = routeRows,
                        ),
                    ),
            )
    }

    fun agentSignalEvidenceReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val wifiReport = wifiAnalyzerReportJson(appContext, JSONObject().put("refresh", false))
        val bluetoothReport = bluetoothAnalyzerReportJson(appContext, JSONObject().put("refresh", false))
        val sensorReport = sensorAnalyzerReportJson(appContext, JSONObject().put("include_snapshot", false))
        val radioReport = radioSignalStatusJson(appContext)
        val signalReport = signalAwarenessReportJson(appContext)
        val backendRiskReport = gpuBackendRiskReportJson(appContext)
        val inferenceReport = localInferenceCompatibilityReportJson(appContext)
        val evidenceRows = agentSignalEvidenceRows(
            wifiReport = wifiReport,
            bluetoothReport = bluetoothReport,
            sensorReport = sensorReport,
            radioReport = radioReport,
            signalReport = signalReport,
            backendRiskReport = backendRiskReport,
            inferenceReport = inferenceReport,
        )
        val routeRows = agentSignalEvidenceRouteRows()
        val graphTypes = signalEvidenceGraphTypes()
        return JSONObject()
            .put("success", true)
            .put("action", "agent_signal_evidence_report")
            .put("report_scope", "Compact Gemma-readable evidence bundle for what Hermes can currently view across Wi-Fi, Bluetooth, motion sensors, AM/FM/RF boundaries, and local inference readiness.")
            .put("source_report_actions", signalEvidenceSourceActions())
            .put("wifi_evidence_summary", observationSummaryJson(wifiReport, "wifi_analyzer_report"))
            .put("bluetooth_evidence_summary", observationSummaryJson(bluetoothReport, "bluetooth_analyzer_report"))
            .put("sensor_evidence_summary", observationSummaryJson(sensorReport, "sensor_analyzer_report"))
            .put("radio_evidence_summary", observationSummaryJson(radioReport, "radio_signal_status"))
            .put("signal_awareness_evidence_summary", observationSummaryJson(signalReport, "signal_awareness_report"))
            .put("backend_risk_evidence_summary", observationSummaryJson(backendRiskReport, "gpu_backend_risk_report"))
            .put("local_inference_evidence_summary", observationSummaryJson(inferenceReport, "local_inference_compatibility_report"))
            .put("signal_evidence_matrix", evidenceRows)
            .put("signal_evidence_routes", routeRows)
            .put("signal_evidence_graph_types", graphTypes)
            .put("signal_evidence_count", evidenceRows.length())
            .put("ready_signal_evidence_count", countReadyRows(evidenceRows))
            .put("signal_evidence_route_count", routeRows.length())
            .put("signal_evidence_graph_type_count", graphTypes.length())
            .put(
                "gemma_observation_directives",
                JSONArray()
                    .put("Read signal_evidence_matrix before answering what Hermes can currently view from nearby signals, motion sensors, radio limits, or local inference readiness.")
                    .put("Treat analyzer summaries as passive evidence; request refresh=true only when the user needs live Wi-Fi, Bluetooth, or motion samples.")
                    .put("Use signal_evidence_routes and signal_evidence_graph_types to open the exact expandable card before turning evidence into a user-facing explanation.")
                    .put("Use local_inference_compatibility_report evidence before promising Gemma 4 multimodal or non-Adreno local acceleration behavior."),
            )
            .put(
                "cards",
                JSONArray()
                    .put(
                        graphCard(
                            title = "Signal Evidence Bundle",
                            body = "${evidenceRows.length()} evidence row(s) fusing Wi-Fi AP/channel graphs, Bluetooth proximity/history, motion sensors, AM/FM/RF boundaries, and local inference compatibility.",
                            graphType = "signal_evidence_matrix",
                            rows = evidenceRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Signal Evidence Routes",
                            body = "${routeRows.length()} route row(s) for opening the right analyzer, graph, or scorecard behind the current evidence.",
                            graphType = "signal_evidence_routes",
                            rows = routeRows,
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Local Inference Compatibility",
                            body = "${inferenceReport.optInt("local_inference_compatibility_count", 0)} local inference compatibility row(s) backing Gemma-readable signal reasoning.",
                            graphType = "local_inference_compatibility_matrix",
                            rows = inferenceReport.optJSONArray("local_inference_compatibility_matrix") ?: JSONArray(),
                        ),
                    )
                    .put(
                        graphCard(
                            title = "Signal Awareness",
                            body = "${signalReport.optInt("signal_awareness_count", 0)} cross-signal context row(s) backing the evidence bundle.",
                            graphType = "signal_awareness_matrix",
                            rows = signalReport.optJSONArray("signal_awareness_matrix") ?: JSONArray(),
                        ),
                    ),
            )
    }

    fun agentCardManifestReportJson(context: Context): JSONObject {
        val appContext = context.applicationContext
        val wifiReport = wifiAnalyzerReportJson(appContext, JSONObject().put("refresh", false))
        val bluetoothReport = bluetoothAnalyzerReportJson(appContext, JSONObject().put("refresh", false))
        val sensorReport = sensorAnalyzerReportJson(appContext, JSONObject().put("include_snapshot", false))
        val radioReport = radioSignalStatusJson(appContext)
        val backendRiskReport = gpuBackendRiskReportJson(appContext)
        val signalReport = signalAwarenessReportJson(appContext)
        val environmentReport = agentEnvironmentReportJson(appContext)
        val cardManifestSources = agentCardManifestSources(
            wifiReport = wifiReport,
            bluetoothReport = bluetoothReport,
            sensorReport = sensorReport,
            radioReport = radioReport,
            backendRiskReport = backendRiskReport,
            signalReport = signalReport,
            environmentReport = environmentReport,
        )
        val cardManifestRows = agentCardManifestRows(cardManifestSources)
        return JSONObject()
            .put("success", true)
            .put("action", "agent_card_manifest_report")
            .put("report_scope", "Direct Gemma-readable manifest that maps expandable diagnostic cards to graph types, source actions, refresh policies, and permission gates.")
            .put("source_report_actions", agentCardManifestSourceActions(cardManifestSources))
            .put("agent_card_manifest", cardManifestRows)
            .put("agent_card_manifest_count", cardManifestRows.length())
            .put("ready_agent_card_manifest_count", countReadyRows(cardManifestRows))
            .put("agent_card_graph_types", cardGraphTypeList(cardManifestRows))
            .put(
                "gemma_observation_directives",
                JSONArray()
                    .put("Read agent_card_manifest before opening Wi-Fi, Bluetooth, sensor, radio, backend, or Kai environment graph cards.")
                    .put("Use each row's graph_type, source_action, refresh_policy, and permission_gate fields to choose the next android_device_diagnostics_tool action.")
                    .put("Prefer passive analyzer reports first; request active Wi-Fi, Bluetooth, motion, or radio refresh only when the user needs live data."),
            )
            .put(
                "cards",
                JSONArray().put(
                    graphCard(
                        title = "Agent Card Manifest",
                        body = "${cardManifestRows.length()} source card row(s) mapping graph types to tool actions, refresh policy, and permission gates for Gemma.",
                        graphType = "agent_card_manifest",
                        rows = cardManifestRows,
                    ),
                ),
            )
    }

    private fun kaiParityMatrixRows(
        preferredModel: JSONObject,
        hindsightStatus: JSONObject,
        automationStatus: JSONObject,
        modelRouting: JSONObject,
        personaStatus: JSONObject,
    ): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Persistent memory",
                    ready = true,
                    valueLabel = "${hindsightStatus.optInt("memory_count", 0)} local row(s)",
                    detail = "Hermes retains, recalls, reflects, and promotes local memories into compact prompt context.",
                    recommendation = "Parallels Kai persistent memory and promotion behavior.",
                    fraction = 0.9f,
                    extra = JSONObject().put("parity_source", "Kai persistent memory"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Customizable soul / system prompt",
                    ready = true,
                    valueLabel = if (personaStatus.optBoolean("custom_system_prompt_enabled", false)) "custom persona enabled" else "default persona",
                    detail = "Hermes stores a bounded user-editable custom_system_prompt in app settings and appends it to native chat system prompts without exporting provider secrets.",
                    recommendation = "Use Settings > Agent persona or import_app_settings to carry a Kai-style custom system prompt between installs.",
                    fraction = if (personaStatus.optBoolean("custom_system_prompt_enabled", false)) 0.95f else 0.8f,
                    extra = JSONObject()
                        .put("parity_source", "Kai customizable soul")
                        .put("settings_key", "custom_system_prompt")
                        .put("custom_prompt_chars", personaStatus.optInt("custom_system_prompt_chars", 0))
                        .put("max_custom_prompt_chars", personaStatus.optInt("max_custom_system_prompt_chars", AppSettings.MAX_CUSTOM_SYSTEM_PROMPT_CHARS)),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "On-device LiteRT inference",
                    ready = preferredModel.optBoolean("ready"),
                    valueLabel = preferredModel.optString("runtime_flavor").ifBlank { "LiteRT/GGUF capable" },
                    detail = "Hermes supports local LiteRT-LM/GGUF model records, Gemma multimodal routing, and SOC-aware backend fallback.",
                    recommendation = "Keep model readiness and backend health visible before offline work.",
                    fraction = if (preferredModel.optBoolean("ready")) 1f else 0.55f,
                    extra = JSONObject().put("parity_source", "Kai on-device inference"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Multi-provider priority and fallback",
                    ready = modelRouting.optBoolean("role_routing_supported", false) || modelRouting.optBoolean("single_runtime_fallback", false),
                    valueLabel = modelRouting.optString("active_provider_label").ifBlank { modelRouting.optString("active_provider").ifBlank { "provider route" } },
                    detail = "Hermes exposes active provider/model routing plus local LiteRT fallback and Android-native action execution roles.",
                    recommendation = "Use android_automation_tool operator_model_routing before choosing remote provider, local LiteRT, or Android-native execution.",
                    fraction = if (modelRouting.optBoolean("role_routing_supported", false) || modelRouting.optBoolean("single_runtime_fallback", false)) 0.9f else 0.45f,
                    extra = JSONObject().put("parity_source", "Kai multi-service fallback"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Tool execution",
                    ready = true,
                    valueLabel = "native Android tools",
                    detail = "Hermes exposes terminal, file, Android system, UI, automation, diagnostics, and memory tools to the local agent.",
                    recommendation = "Prefer native Android APIs first, then Linux/shell tooling when the task truly needs it.",
                    fraction = 0.95f,
                    extra = JSONObject().put("parity_source", "Kai tool execution and Linux sandbox"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "MCP and external tool equivalents",
                    ready = true,
                    valueLabel = "native tool bridge",
                    detail = "Hermes maps Kai-style tool-server work to terminal, file, system, UI, automation, diagnostics, and memory tools while MCP-server parity remains an explicit future bridge.",
                    recommendation = "Call tool_catalog first, then choose the narrow native tool before adding an external MCP server dependency.",
                    fraction = 0.8f,
                    extra = JSONObject().put("parity_source", "Kai MCP server support"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Autonomous heartbeat",
                    ready = automationStatus.optBoolean("standby_heartbeat_supported", false),
                    valueLabel = "${automationStatus.optInt("heartbeat_interval_seconds", 30)}s interval",
                    detail = "Operator heartbeat/status rows expose standby state, model routing, recent runs, and remote dispatch compatibility.",
                    recommendation = "Use this for Kai-style self-checks and status surfacing.",
                    fraction = if (automationStatus.optBoolean("standby_heartbeat_supported", false)) 0.9f else 0.45f,
                    extra = JSONObject().put("parity_source", "Kai autonomous heartbeat"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Encrypted credentials and local storage",
                    ready = true,
                    valueLabel = "secure secrets store",
                    detail = "Hermes stores provider credentials in AndroidX encrypted preferences with sealed integrity envelopes and keeps app state local unless a provider route is selected.",
                    recommendation = "Use provider auth/session stores for credentials and keep diagnostic exports free of raw secrets.",
                    fraction = 0.9f,
                    extra = JSONObject().put("parity_source", "Kai encrypted storage"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "App settings and automation backup",
                    ready = true,
                    valueLabel = "settings + automation export",
                    detail = "Hermes can export/import a secret-free app settings bundle plus automation bundles with records and variables.",
                    recommendation = "Use android_automation_tool export_app_settings/import_app_settings for app preferences and export_automations/import_automations for workflow migration.",
                    fraction = 0.9f,
                    extra = JSONObject().put("parity_source", "Kai settings export/import"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Text to speech",
                    ready = true,
                    valueLabel = "speak last reply",
                    detail = "Hermes exposes Android TextToSpeech through chat speak buttons and the /speak last command.",
                    recommendation = "Use the chat TTS route for read-aloud responses and keep automation audio actions separate from assistant speech.",
                    fraction = 0.8f,
                    extra = JSONObject().put("parity_source", "Kai text to speech"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_parity",
                    label = "Image attachments and screen vision",
                    ready = modelRouting.optBoolean("vision_capable", false),
                    valueLabel = if (modelRouting.optBoolean("vision_capable", false)) "vision-ready" else "model dependent",
                    detail = "Chat attachments, screenshot capture, and OpenGUI visual fallback feed image context into multimodal Gemma-capable paths.",
                    recommendation = "Use a vision-capable LiteRT model before relying on image attachments for local-only reasoning.",
                    fraction = if (modelRouting.optBoolean("vision_capable", false)) 0.9f else 0.5f,
                    extra = JSONObject().put("parity_source", "Kai image attachments"),
                ),
            )
    }

    private fun kaiOperationsMatrixRows(
        automationStatus: JSONObject,
        modelRouting: JSONObject,
        personaStatus: JSONObject,
    ): JSONArray {
        val providerLabel = modelRouting.optString("active_provider_label").ifBlank {
            modelRouting.optString("active_provider").ifBlank { "provider route" }
        }
        val activeModel = modelRouting.optString("active_model").ifBlank { "model route" }
        val heartbeatReady = automationStatus.optBoolean("standby_heartbeat_supported", false)
        val routingReady = modelRouting.optBoolean("role_routing_supported", false) || modelRouting.optBoolean("single_runtime_fallback", false)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Provider priority and fallback route",
                    ready = routingReady,
                    valueLabel = providerLabel,
                    detail = "$providerLabel -> $activeModel; role routing keeps planning, supervision, VLM, summarization, and action execution explicit.",
                    recommendation = "Call android_automation_tool operator_model_routing before model/provider-sensitive work.",
                    fraction = if (routingReady) 0.9f else 0.45f,
                    extra = JSONObject()
                        .put("tool_action", "android_automation_tool:operator_model_routing")
                        .put("source_surface", "operator_model_routing"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Tool and MCP bridge route",
                    ready = true,
                    valueLabel = "tool_catalog",
                    detail = "Terminal, file, Android system, UI, automation, diagnostics, and hindsight memory tools cover the first Kai-style tool bridge before external MCP endpoints are added.",
                    recommendation = "Call android_device_diagnostics_tool action=tool_catalog, then invoke the narrow native tool for the requested operation.",
                    fraction = 0.85f,
                    extra = JSONObject()
                        .put("tool_action", "android_device_diagnostics_tool:tool_catalog")
                        .put("source_surface", "native_tool_catalog"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "System behavior and tool permissions",
                    ready = true,
                    valueLabel = "settings and guards",
                    detail = "Provider/model settings, accessibility state, app permissions, Shizuku/Sui gates, and external-send review checks keep agent behavior editable and bounded.",
                    recommendation = "Check settings/auth status plus android_ui_tool or social preflight rows before external sends, privileged actions, or UI control.",
                    fraction = 0.85f,
                    extra = JSONObject()
                        .put("tool_action", "android_ui_tool:status")
                        .put("source_surface", "settings_permissions_guards"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Persona and system prompt route",
                    ready = true,
                    valueLabel = if (personaStatus.optBoolean("custom_system_prompt_enabled", false)) "custom prompt active" else "default prompt active",
                    detail = "The native chat prompt merges Hermes tool instructions, the user's custom agent persona, and promoted local memory context in that order.",
                    recommendation = "Use Settings Agent persona for behavior changes and export_app_settings/import_app_settings to migrate the prompt without secrets.",
                    fraction = if (personaStatus.optBoolean("custom_system_prompt_enabled", false)) 0.95f else 0.8f,
                    extra = JSONObject()
                        .put("tool_action", "settings:agent_persona")
                        .put("source_surface", "custom_system_prompt")
                        .put("custom_prompt_chars", personaStatus.optInt("custom_system_prompt_chars", 0)),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Encrypted credentials and backup route",
                    ready = true,
                    valueLabel = "secure store + settings JSON",
                    detail = "Provider secrets use AndroidX encrypted preferences; app settings export/import is secret-free, and automations/variables have separate JSON migration bundles.",
                    recommendation = "Use secure auth/session stores for credentials, export_app_settings/import_app_settings for preferences, and export_automations/import_automations for workflows.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("tool_action", "android_automation_tool:export_app_settings")
                        .put("source_surface", "secure_secrets_store"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Heartbeat and dispatch route",
                    ready = heartbeatReady,
                    valueLabel = if (heartbeatReady) "${automationStatus.optInt("heartbeat_interval_seconds", 30)}s heartbeat" else "standby status only",
                    detail = "${automationStatus.optInt("enabled_automation_count", 0)} enabled automation(s), ${automationStatus.optInt("recent_run_count", 0)} recent run(s), batch heartbeat supported=${automationStatus.optBoolean("batch_heartbeat_supported", false)}.",
                    recommendation = "Use operator_standby_status/operator_heartbeat for Kai-style self-check visibility and remote task dispatch.",
                    fraction = if (heartbeatReady) 0.9f else 0.45f,
                    extra = JSONObject()
                        .put("tool_action", "android_automation_tool:operator_heartbeat")
                        .put("source_surface", "operator_standby"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Scheduled task compatibility route",
                    ready = true,
                    valueLabel = "schedule_task/list_tasks/cancel_task",
                    detail = "Hermes maps Kai-style scheduled task tool names to native Android automation notification records with time/day, interval, or explicit phone triggers.",
                    recommendation = "Use schedule_task for reminder-like Android automations, list_tasks to inspect them, and cancel_task with task_id to remove them; background AI prompt execution remains explicit and is not implied.",
                    fraction = 0.85f,
                    extra = JSONObject()
                        .put("tool_action", "schedule_task")
                        .put("list_action", "list_tasks")
                        .put("cancel_action", "cancel_task")
                        .put("source_surface", "kai_task_compat")
                        .put("kai_task_compat", true)
                        .put("background_ai_prompt_execution", false),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "TTS and image conversation route",
                    ready = true,
                    valueLabel = if (modelRouting.optBoolean("vision_capable", false)) "speech + vision" else "speech + image route",
                    detail = "Chat supports TextToSpeech read-aloud, image attachments, and screenshot visual snapshots when the active model can consume vision inputs.",
                    recommendation = "Use /speak last for read-aloud and prefer a vision-capable LiteRT model before depending on local image reasoning.",
                    fraction = if (modelRouting.optBoolean("vision_capable", false)) 0.9f else 0.65f,
                    extra = JSONObject()
                        .put("tool_action", "chat:/speak last")
                        .put("source_surface", "chat_tts_image_attachments"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_operations",
                    label = "Android shell boundary route",
                    ready = true,
                    valueLabel = "terminal workspace",
                    detail = "Hermes routes shell work through the terminal tool and app workspace while keeping Android host, root, and permission boundaries explicit.",
                    recommendation = "Use terminal_tool for short workspace commands and prefer Android-native tools when device APIs provide structured data.",
                    fraction = 0.8f,
                    extra = JSONObject()
                        .put("tool_action", "terminal_tool")
                        .put("source_surface", "android_terminal_workspace"),
                ),
            )
    }

    private fun agentObservationMatrixRows(
        wifiReport: JSONObject,
        bluetoothReport: JSONObject,
        sensorReport: JSONObject,
        radioReport: JSONObject,
        signalReport: JSONObject,
        backendRiskReport: JSONObject,
        environmentReport: JSONObject,
        signalContextRows: JSONArray,
        cardManifestRows: JSONArray,
    ): JSONArray {
        val wifiNetworkCount = wifiReport.optInt("total_scan_result_count", wifiReport.optJSONArray("wifi_networks")?.length() ?: 0)
        val wifiDetailCount = wifiReport.optJSONArray("wifi_access_point_details")?.length() ?: 0
        val wifiChannelCount = wifiReport.optJSONArray("wifi_channel_ratings")?.length() ?: 0
        val bluetoothDeviceCount = bluetoothReport.optInt("bluetooth_device_count", bluetoothReport.optJSONArray("bluetooth_devices")?.length() ?: 0)
        val bluetoothMetadataCount = bluetoothReport.optInt("bluetooth_metadata_count", bluetoothReport.optJSONArray("bluetooth_metadata_summary")?.length() ?: 0)
        val sensorCapabilityCount = sensorReport.optInt("sensor_capability_count", sensorReport.optJSONArray("sensor_capabilities")?.length() ?: 0)
        val motionPoseCount = sensorReport.optInt("motion_pose_estimate_count", sensorReport.optJSONArray("motion_pose_estimates")?.length() ?: 0)
        val radioBandCount = radioReport.optInt("radio_band_plan_count", radioReport.optJSONArray("radio_bands")?.length() ?: 0)
        val signalRouteCount = signalReport.optInt("signal_workflow_route_count", signalReport.optJSONArray("signal_workflow_routes")?.length() ?: 0)
        val backendRiskCount = backendRiskReport.optInt("gpu_backend_risk_count", backendRiskReport.optJSONArray("gpu_backend_risk_matrix")?.length() ?: 0)
        val highBackendRiskCount = backendRiskReport.optInt("high_gpu_backend_risk_count", 0)
        val backendRiskLevel = backendRiskReport.optString("gpu_backend_risk_level").ifBlank { "unknown" }
        val backendRiskScore = backendRiskReport.optInt("gpu_backend_risk_score", 0)
        val kaiOperationsCount = environmentReport.optInt("kai_operations_count", environmentReport.optJSONArray("kai_operations_matrix")?.length() ?: 0)
        val capabilityCount = environmentReport.optInt("agent_capability_count", environmentReport.optJSONArray("agent_capability_matrix")?.length() ?: 0)
        val signalContextCount = signalContextRows.length()
        val readySignalContextCount = countReadyRows(signalContextRows)
        val cardManifestCount = cardManifestRows.length()
        return JSONArray()
            .put(
                capabilityRow(
                    category = "agent_observation",
                    label = "Gemma signal dashboard",
                    ready = true,
                    valueLabel = "single observation report",
                    detail = "Combines passive Wi-Fi, Bluetooth, sensor, radio, SOC, local model, and Kai operation summaries before the model chooses a next tool.",
                    recommendation = "Use this first when the user asks what Hermes can currently see about nearby signals or device readiness.",
                    fraction = 0.95f,
                    extra = JSONObject().put("tool_action", "agent_observation_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Signal context fusion matrix",
                    ready = signalContextCount > 0,
                    valueLabel = "$readySignalContextCount/$signalContextCount fused row(s)",
                    detail = "Machine-readable rows bind Wi-Fi channel/band coverage, Bluetooth RSSI/history metadata, motion pose context, radio hardware limits, and source-card routes before Gemma explains nearby signals.",
                    recommendation = "Open the Signal Context Fusion card before producing a nearby-signal summary or choosing a scanner.",
                    fraction = if (signalContextCount > 0) (readySignalContextCount / signalContextCount.toFloat()).coerceIn(0.35f, 0.95f) else 0.25f,
                    extra = JSONObject()
                        .put("tool_action", "agent_observation_report")
                        .put("graph_type", "agent_signal_context_matrix"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_card_manifest",
                    label = "Agent-readable card manifest",
                    ready = cardManifestCount > 0,
                    valueLabel = "$cardManifestCount card route row(s)",
                    detail = "Each source report card is indexed with graph_type, source_action, row_count, refresh_policy, and permission_gate so Gemma can choose the exact expandable card before asking for live data.",
                    recommendation = "Read agent_card_manifest before explaining source evidence or requesting active Wi-Fi, Bluetooth, sensor, radio, or backend refreshes.",
                    fraction = if (cardManifestCount > 0) 0.95f else 0.25f,
                    extra = JSONObject()
                        .put("tool_action", "agent_observation_report")
                        .put("graph_type", "agent_card_manifest")
                        .put("agent_card_manifest_count", cardManifestCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_observation",
                    label = "Wi-Fi AP metadata and channel graphs",
                    ready = wifiReport.optBoolean("success", false),
                    valueLabel = "$wifiNetworkCount AP(s), $wifiChannelCount channel row(s)",
                    detail = "$wifiDetailCount access-point detail row(s), ${wifiReport.optInt("wifi_access_point_semantic_count", 0)} semantic row(s), ${wifiReport.optInt("wifi_band_coverage_count", 0)} band coverage row(s).",
                    recommendation = "Open Wi-Fi Analyzer cards before advising channel changes, security risk, AP identity, or vendor/OUI context.",
                    fraction = when {
                        wifiNetworkCount > 0 && wifiChannelCount > 0 -> 0.95f
                        wifiReport.optBoolean("success", false) -> 0.7f
                        else -> 0.35f
                    },
                    extra = JSONObject().put("tool_action", "wifi_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_observation",
                    label = "Bluetooth nearby metadata",
                    ready = bluetoothReport.optBoolean("success", false),
                    valueLabel = "$bluetoothDeviceCount device(s)",
                    detail = "$bluetoothMetadataCount metadata row(s), ${bluetoothReport.optInt("bluetooth_service_label_count", 0)} service label(s), ${bluetoothReport.optInt("bluetooth_manufacturer_name_count", 0)} manufacturer name(s), ${bluetoothReport.optInt("bluetooth_signal_history_count", 0)} cached trend row(s).",
                    recommendation = "Open Bluetooth Analyzer cards before explaining proximity, service identity, paired inventory, or BLE manufacturer context.",
                    fraction = when {
                        bluetoothDeviceCount > 0 && bluetoothMetadataCount > 0 -> 0.95f
                        bluetoothReport.optBoolean("success", false) -> 0.7f
                        else -> 0.35f
                    },
                    extra = JSONObject().put("tool_action", "bluetooth_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_observation",
                    label = "Motion and sensor context",
                    ready = sensorReport.optBoolean("sensor_service_available", false),
                    valueLabel = "$sensorCapabilityCount sensor row(s)",
                    detail = "${sensorReport.optInt("motion_sensor_count", 0)} motion sensor(s), ${sensorReport.optInt("ambient_sensor_count", 0)} ambient sensor(s), $motionPoseCount pose estimate row(s).",
                    recommendation = "Use sensor cards for accelerometer, gyroscope, heading, ambient, sampling, power, and workflow trigger context.",
                    fraction = if (sensorCapabilityCount > 0) 0.9f else 0.35f,
                    extra = JSONObject().put("tool_action", "sensor_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "radio_observation",
                    label = "Radio and RF boundaries",
                    ready = radioBandCount > 0,
                    valueLabel = "$radioBandCount band row(s)",
                    detail = "${radioReport.optInt("radio_signal_feature_count", 0)} feature row(s), ${radioReport.optInt("radio_signal_constraint_count", 0)} constraint row(s), external SDR required=${radioReport.optBoolean("requires_external_sdr_for_broad_rf", true)}.",
                    recommendation = "Use radio cards to distinguish public Android Wi-Fi/Bluetooth signal access from AM/FM or broader RF hardware limits.",
                    fraction = if (radioBandCount > 0) 0.85f else 0.35f,
                    extra = JSONObject().put("tool_action", "radio_signal_status"),
                ),
            )
            .put(
                capabilityRow(
                    category = "backend_risk_observation",
                    label = "GPU backend risk triage",
                    ready = backendRiskReport.optBoolean("success", false) && backendRiskCount > 0,
                    valueLabel = "$backendRiskLevel risk, $highBackendRiskCount high row(s)",
                    detail = "$backendRiskCount risk row(s) cover accelerator acceptance, SOC/GPU policy, thermal pressure, memory pressure, power state, model artifact fit, phone validation scope, and fallback routes.",
                    recommendation = "Open GPU Backend Risk before promising MediaTek, Mali, PowerVR, Xclipse, or other non-Adreno acceleration behavior.",
                    fraction = if (backendRiskCount > 0) ((100 - backendRiskScore).coerceIn(5, 100) / 100f) else 0.25f,
                    extra = JSONObject()
                        .put("tool_action", "gpu_backend_risk_report")
                        .put("graph_type", "gpu_backend_risk_matrix")
                        .put("gpu_backend_risk_level", backendRiskLevel)
                        .put("gpu_backend_risk_score", backendRiskScore)
                        .put("gpu_backend_risk_count", backendRiskCount)
                        .put("high_gpu_backend_risk_count", highBackendRiskCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_model_observation",
                    label = "SOC and local model readiness",
                    ready = capabilityCount > 0,
                    valueLabel = "${environmentReport.optInt("ready_capability_count", 0)}/$capabilityCount ready",
                    detail = "SOC/backend policy, preferred local model, UI control, automation, memory, and signal inputs are visible to the agent environment report.",
                    recommendation = "Use SOC, backend-risk, and local-model cards before assuming Snapdragon-only behavior or multimodal local inference availability.",
                    fraction = if (capabilityCount > 0) 0.85f else 0.35f,
                    extra = JSONObject().put("tool_action", "agent_environment_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "kai_observation",
                    label = "Kai operations and interactive routes",
                    ready = kaiOperationsCount > 0,
                    valueLabel = "$kaiOperationsCount operation row(s)",
                    detail = "Provider fallback, native tool bridge, encrypted storage/backup, heartbeat, TTS, image attachment, and shell-boundary rows are visible from the Kai Operations card.",
                    recommendation = "Use these rows before routing a Kai-style workflow through Hermes tools, cards, local model, or generated HTML screens.",
                    fraction = if (kaiOperationsCount > 0) 0.9f else 0.35f,
                    extra = JSONObject().put("tool_action", "agent_environment_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "card_observation",
                    label = "Expandable card coverage",
                    ready = true,
                    valueLabel = "${reportCardTitles(wifiReport).length() + reportCardTitles(bluetoothReport).length() + reportCardTitles(sensorReport).length() + reportCardTitles(signalReport).length() + reportCardTitles(backendRiskReport).length() + reportCardTitles(environmentReport).length()} source cards",
                    detail = "The dashboard points the user to expandable graph cards for Wi-Fi, Bluetooth, sensor, radio, backend-risk, signal-awareness, and agent-environment rows.",
                    recommendation = "Prefer cards for scan-heavy or graph-heavy answers so the user can expand the underlying rows instead of reading a long text dump.",
                    fraction = 0.9f,
                    extra = JSONObject().put("source_surface", "diagnostic_cards"),
                ),
            )
            .put(
                capabilityRow(
                    category = "route_observation",
                    label = "Next-tool route coverage",
                    ready = signalRouteCount > 0,
                    valueLabel = "$signalRouteCount route row(s)",
                    detail = "Signal-awareness and observation-route rows map broad user questions to exact diagnostic actions.",
                    recommendation = "Use route rows to choose between wifi_analyzer_report, bluetooth_analyzer_report, sensor_analyzer_report, radio_signal_status, gpu_backend_risk_report, soc_compatibility_report, or agent_environment_report.",
                    fraction = if (signalRouteCount > 0) 0.9f else 0.45f,
                    extra = JSONObject().put("tool_action", "signal_awareness_report"),
                ),
            )
    }

    private fun agentSignalContextFusionRows(
        wifiReport: JSONObject,
        bluetoothReport: JSONObject,
        sensorReport: JSONObject,
        radioReport: JSONObject,
        signalReport: JSONObject,
        backendRiskReport: JSONObject,
    ): JSONArray {
        val wifiNetworkCount = wifiReport.optInt("total_scan_result_count", wifiReport.optJSONArray("wifi_networks")?.length() ?: 0)
        val wifiBandCoverageCount = wifiReport.optInt("wifi_band_coverage_count", wifiReport.optJSONArray("wifi_band_coverage")?.length() ?: 0)
        val wifiChannelGraphCount = wifiReport.optInt("wifi_channel_graph_count", wifiReport.optJSONArray("wifi_channel_graph")?.length() ?: 0)
        val wifiChannelRatingCount = wifiReport.optJSONArray("wifi_channel_ratings")?.length() ?: 0
        val wifiChannelUtilizationCount = wifiReport.optInt("wifi_channel_utilization_count", wifiReport.optJSONArray("wifi_channel_utilization")?.length() ?: 0)
        val wifiHistoryCount = wifiReport.optJSONArray("wifi_signal_history")?.length() ?: 0
        val bluetoothDeviceCount = bluetoothReport.optInt("bluetooth_device_count", bluetoothReport.optJSONArray("bluetooth_devices")?.length() ?: 0)
        val bluetoothMetadataCount = bluetoothReport.optInt("bluetooth_metadata_count", bluetoothReport.optJSONArray("bluetooth_metadata_summary")?.length() ?: 0)
        val bluetoothHistoryCount = bluetoothReport.optInt("bluetooth_signal_history_count", bluetoothReport.optJSONArray("bluetooth_signal_history")?.length() ?: 0)
        val bluetoothServiceLabelCount = bluetoothReport.optInt("bluetooth_service_label_count", 0)
        val bluetoothManufacturerNameCount = bluetoothReport.optInt("bluetooth_manufacturer_name_count", 0)
        val sensorCapabilityCount = sensorReport.optInt("sensor_capability_count", sensorReport.optJSONArray("sensor_capabilities")?.length() ?: 0)
        val motionPoseCount = sensorReport.optInt("motion_pose_estimate_count", sensorReport.optJSONArray("motion_pose_estimates")?.length() ?: 0)
        val motionHistoryCount = sensorReport.optInt("motion_sensor_history_count", sensorReport.optJSONArray("motion_sensor_history")?.length() ?: 0)
        val radioBandCount = radioReport.optInt("radio_band_plan_count", radioReport.optJSONArray("radio_bands")?.length() ?: 0)
        val radioConstraintCount = radioReport.optInt("radio_signal_constraint_count", radioReport.optJSONArray("radio_signal_constraint_matrix")?.length() ?: 0)
        val signalRouteCount = signalReport.optInt("signal_workflow_route_count", signalReport.optJSONArray("signal_workflow_routes")?.length() ?: 0)
        val backendRiskCount = backendRiskReport.optInt("gpu_backend_risk_count", backendRiskReport.optJSONArray("gpu_backend_risk_matrix")?.length() ?: 0)
        val backendRiskRouteCount = backendRiskReport.optInt("gpu_backend_risk_route_count", backendRiskReport.optJSONArray("gpu_backend_risk_routes")?.length() ?: 0)
        val highBackendRiskCount = backendRiskReport.optInt("high_gpu_backend_risk_count", 0)
        val backendRiskLevel = backendRiskReport.optString("gpu_backend_risk_level").ifBlank { "unknown" }
        val backendRiskScore = backendRiskReport.optInt("gpu_backend_risk_score", 0)
        val wifiReady = wifiReport.optBoolean("success", false) && wifiBandCoverageCount > 0
        val bluetoothReady = bluetoothReport.optBoolean("success", false) &&
            (bluetoothDeviceCount > 0 || bluetoothMetadataCount > 0 || bluetoothHistoryCount > 0)
        val sensorReady = sensorReport.optBoolean("sensor_service_available", false) ||
            sensorCapabilityCount > 0 || motionPoseCount > 0 || motionHistoryCount > 0
        val radioReady = radioBandCount > 0
        val backendRiskReady = backendRiskReport.optBoolean("success", false) && backendRiskCount > 0
        val readyDomainCount = listOf(wifiReady, bluetoothReady, sensorReady, radioReady, backendRiskReady).count { it }

        return JSONArray()
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Gemma signal context contract",
                    ready = readyDomainCount > 0,
                    valueLabel = "$readyDomainCount/5 source domain(s)",
                    detail = "Fuses Wi-Fi channel/band rows, Bluetooth RSSI/history metadata, motion pose context, radio-boundary rows, and backend-risk rows into one card before natural-language reasoning.",
                    recommendation = "Read this matrix before summarizing nearby signals; then open the exact source card for evidence.",
                    fraction = (readyDomainCount / 5f).coerceIn(0.25f, 0.95f),
                    extra = JSONObject()
                        .put("fusion_key", "signal_context_contract")
                        .put("context_domains", JSONArray().put("wifi").put("bluetooth").put("motion_sensors").put("radio_rf_limits").put("backend_risk"))
                        .put("source_actions", JSONArray().put("wifi_analyzer_report").put("bluetooth_analyzer_report").put("sensor_analyzer_report").put("radio_signal_status").put("gpu_backend_risk_report").put("signal_awareness_report")),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Wi-Fi channel and band context",
                    ready = wifiReady,
                    valueLabel = "$wifiNetworkCount AP(s), $wifiChannelGraphCount graph row(s)",
                    detail = "$wifiChannelGraphCount channel envelope row(s), $wifiChannelRatingCount channel rating row(s), $wifiChannelUtilizationCount utilization row(s), $wifiBandCoverageCount band row(s), and $wifiHistoryCount cached history row(s) available for Wi-Fi Analyzer-style graphing.",
                    recommendation = "Use this row to keep channel graph envelopes, rating, band coverage, utilization, RSSI history, vendor, and security metadata together.",
                    fraction = when {
                        wifiNetworkCount > 0 && wifiChannelGraphCount > 0 -> 0.95f
                        wifiNetworkCount > 0 && wifiChannelRatingCount > 0 -> 0.9f
                        wifiBandCoverageCount > 0 -> 0.8f
                        wifiReport.optBoolean("success", false) -> 0.55f
                        else -> 0.25f
                    },
                    extra = JSONObject()
                        .put("fusion_key", "wifi_channel_band_context")
                        .put("source_actions", JSONArray().put("wifi_analyzer_report").put("wifi_channel_graph").put("wifi_channel_rating").put("wifi_channel_utilization"))
                        .put("card_graph_types", JSONArray().put("wifi_band_coverage").put("wifi_channel_graph").put("wifi_channel_rating").put("wifi_channel_utilization").put("wifi_signal_history"))
                        .put("wifi_network_count", wifiNetworkCount)
                        .put("wifi_band_coverage_count", wifiBandCoverageCount)
                        .put("wifi_channel_graph_count", wifiChannelGraphCount)
                        .put("wifi_channel_rating_count", wifiChannelRatingCount)
                        .put("wifi_channel_utilization_count", wifiChannelUtilizationCount)
                        .put("wifi_signal_history_count", wifiHistoryCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Bluetooth RSSI and identity context",
                    ready = bluetoothReady,
                    valueLabel = "$bluetoothDeviceCount device(s), $bluetoothHistoryCount trend row(s)",
                    detail = "$bluetoothMetadataCount metadata row(s), $bluetoothServiceLabelCount service label(s), and $bluetoothManufacturerNameCount manufacturer name(s) available for nearby-device reasoning.",
                    recommendation = "Use this row to keep Bluetooth proximity, service UUID labels, manufacturer IDs, and history trends linked to the same observation.",
                    fraction = when {
                        bluetoothDeviceCount > 0 && bluetoothMetadataCount > 0 -> 0.95f
                        bluetoothHistoryCount > 0 -> 0.75f
                        bluetoothReport.optBoolean("success", false) -> 0.5f
                        else -> 0.25f
                    },
                    extra = JSONObject()
                        .put("fusion_key", "bluetooth_rssi_identity_context")
                        .put("source_actions", JSONArray().put("bluetooth_analyzer_report").put("bluetooth_signal_history").put("bluetooth_scan"))
                        .put("card_graph_types", JSONArray().put("bluetooth_metadata_summary").put("bluetooth_signal_history").put("bluetooth_rssi"))
                        .put("bluetooth_device_count", bluetoothDeviceCount)
                        .put("bluetooth_metadata_count", bluetoothMetadataCount)
                        .put("bluetooth_signal_history_count", bluetoothHistoryCount)
                        .put("bluetooth_service_label_count", bluetoothServiceLabelCount)
                        .put("bluetooth_manufacturer_name_count", bluetoothManufacturerNameCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Motion pose and sensor context",
                    ready = sensorReady,
                    valueLabel = "$sensorCapabilityCount sensor row(s), $motionPoseCount pose row(s)",
                    detail = "$motionHistoryCount cached motion trend row(s) available to bind nearby-signal changes to phone movement, heading, acceleration, and orientation.",
                    recommendation = "Use this row before attributing Wi-Fi or Bluetooth changes to distance, movement, pocket state, heading, or user motion.",
                    fraction = when {
                        motionPoseCount > 0 -> 0.95f
                        sensorCapabilityCount > 0 -> 0.8f
                        motionHistoryCount > 0 -> 0.7f
                        else -> 0.3f
                    },
                    extra = JSONObject()
                        .put("fusion_key", "motion_pose_sensor_context")
                        .put("source_actions", JSONArray().put("sensor_analyzer_report").put("motion_pose").put("motion_sensor_history"))
                        .put("card_graph_types", JSONArray().put("motion_pose_estimate").put("motion_sensor_history").put("sensor_capability"))
                        .put("sensor_capability_count", sensorCapabilityCount)
                        .put("motion_pose_estimate_count", motionPoseCount)
                        .put("motion_sensor_history_count", motionHistoryCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Radio hardware boundary context",
                    ready = radioReady,
                    valueLabel = "$radioBandCount band row(s), $radioConstraintCount constraint row(s)",
                    detail = "AM/FM scan support=${radioReport.optBoolean("am_fm_public_android_scan_supported", false)}, broad RF support=${radioReport.optBoolean("general_radio_spectrum_supported", false)}, external SDR required=${radioReport.optBoolean("requires_external_sdr_for_broad_rf", true)}.",
                    recommendation = "Use this row to keep public Android Wi-Fi/Bluetooth signal access separate from vendor tuner or external SDR requirements.",
                    fraction = if (radioReady) 0.85f else 0.25f,
                    extra = JSONObject()
                        .put("fusion_key", "radio_hardware_boundary_context")
                        .put("source_actions", JSONArray().put("radio_signal_status").put("signal_awareness_report"))
                        .put("card_graph_types", JSONArray().put("radio_frequency_capability").put("radio_signal_constraint_matrix").put("radio_signal_workflow_routes"))
                        .put("radio_band_plan_count", radioBandCount)
                        .put("radio_signal_constraint_count", radioConstraintCount)
                        .put("am_fm_public_android_scan_supported", radioReport.optBoolean("am_fm_public_android_scan_supported", false))
                        .put("requires_external_sdr_for_broad_rf", radioReport.optBoolean("requires_external_sdr_for_broad_rf", true)),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Backend risk and fallback context",
                    ready = backendRiskReady,
                    valueLabel = "$backendRiskLevel risk, $highBackendRiskCount high row(s)",
                    detail = "$backendRiskCount GPU/backend risk row(s) and $backendRiskRouteCount route row(s) bind accelerator acceptance, SOC/GPU family, thermal, memory, power, model artifact fit, and phone-validation scope.",
                    recommendation = "Use this row before claiming local acceleration stability on MediaTek, Mali, PowerVR, Xclipse, or any non-Adreno device.",
                    fraction = if (backendRiskReady) ((100 - backendRiskScore).coerceIn(5, 100) / 100f) else 0.25f,
                    extra = JSONObject()
                        .put("fusion_key", "backend_risk_fallback_context")
                        .put("source_actions", JSONArray().put("gpu_backend_risk_report").put("local_backend_runtime_report").put("soc_compatibility_report").put("device_performance_report"))
                        .put("card_graph_types", JSONArray().put("gpu_backend_risk_matrix").put("gpu_backend_risk_routes").put("runtime_backend_matrix").put("soc_backend_matrix").put("runtime_stability_matrix"))
                        .put("gpu_backend_risk_level", backendRiskLevel)
                        .put("gpu_backend_risk_score", backendRiskScore)
                        .put("gpu_backend_risk_count", backendRiskCount)
                        .put("high_gpu_backend_risk_count", highBackendRiskCount)
                        .put("gpu_backend_risk_route_count", backendRiskRouteCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_signal_context",
                    label = "Source card drill-down",
                    ready = signalRouteCount > 0,
                    valueLabel = "$signalRouteCount signal route row(s)",
                    detail = "Routes link the fused matrix back to Wi-Fi Analyzer, Bluetooth Analyzer, Sensor Analyzer, radio band plan, backend-risk, and signal-awareness cards.",
                    recommendation = "Use the source_actions and card_graph_types fields when the answer needs evidence instead of a text-only summary.",
                    fraction = if (signalRouteCount > 0) 0.9f else 0.45f,
                    extra = JSONObject()
                        .put("fusion_key", "source_card_drill_down")
                        .put("source_actions", JSONArray().put("agent_observation_report").put("wifi_analyzer_report").put("wifi_channel_graph").put("bluetooth_analyzer_report").put("sensor_analyzer_report").put("radio_signal_status").put("gpu_backend_risk_report"))
                        .put("card_graph_types", JSONArray().put("agent_signal_context_matrix").put("wifi_channel_graph").put("wifi_channel_rating").put("bluetooth_signal_history").put("motion_pose_estimate").put("radio_frequency_capability").put("gpu_backend_risk_matrix"))
                        .put("signal_workflow_route_count", signalRouteCount),
                ),
            )
    }

    private fun agentSignalEvidenceRows(
        wifiReport: JSONObject,
        bluetoothReport: JSONObject,
        sensorReport: JSONObject,
        radioReport: JSONObject,
        signalReport: JSONObject,
        backendRiskReport: JSONObject,
        inferenceReport: JSONObject,
    ): JSONArray {
        val wifiNetworkCount = wifiReport.optInt("total_scan_result_count", wifiReport.optJSONArray("wifi_networks")?.length() ?: 0)
        val wifiChannelGraphCount = wifiReport.optInt("wifi_channel_graph_count", wifiReport.optJSONArray("wifi_channel_graph")?.length() ?: 0)
        val wifiChannelRatingCount = wifiReport.optJSONArray("wifi_channel_ratings")?.length() ?: 0
        val wifiBandCoverageCount = wifiReport.optInt("wifi_band_coverage_count", wifiReport.optJSONArray("wifi_band_coverage")?.length() ?: 0)
        val wifiHistoryCount = wifiReport.optJSONArray("wifi_signal_history")?.length() ?: 0
        val bluetoothDeviceCount = bluetoothReport.optInt("bluetooth_device_count", bluetoothReport.optJSONArray("bluetooth_devices")?.length() ?: 0)
        val bluetoothMetadataCount = bluetoothReport.optInt("bluetooth_metadata_count", bluetoothReport.optJSONArray("bluetooth_metadata_summary")?.length() ?: 0)
        val bluetoothHistoryCount = bluetoothReport.optInt("bluetooth_signal_history_count", bluetoothReport.optJSONArray("bluetooth_signal_history")?.length() ?: 0)
        val sensorCapabilityCount = sensorReport.optInt("sensor_capability_count", sensorReport.optJSONArray("sensor_capabilities")?.length() ?: 0)
        val motionHistoryCount = sensorReport.optInt("motion_sensor_history_count", sensorReport.optJSONArray("motion_sensor_history")?.length() ?: 0)
        val motionPoseCount = sensorReport.optInt("motion_pose_estimate_count", sensorReport.optJSONArray("motion_pose_estimates")?.length() ?: 0)
        val radioBandCount = radioReport.optInt("radio_band_plan_count", radioReport.optJSONArray("radio_bands")?.length() ?: 0)
        val radioGraphReady = radioReport.optBoolean("radio_signal_graph_bridge_ready", false) ||
            radioReport.optBoolean("am_fm_public_android_scan_supported", false)
        val signalAwarenessCount = signalReport.optInt("signal_awareness_count", signalReport.optJSONArray("signal_awareness_matrix")?.length() ?: 0)
        val signalConstraintCount = signalReport.optInt("signal_constraint_count", signalReport.optJSONArray("signal_constraint_matrix")?.length() ?: 0)
        val backendRiskCount = backendRiskReport.optInt("gpu_backend_risk_count", backendRiskReport.optJSONArray("gpu_backend_risk_matrix")?.length() ?: 0)
        val highBackendRiskCount = backendRiskReport.optInt("high_gpu_backend_risk_count", 0)
        val backendRiskLevel = backendRiskReport.optString("gpu_backend_risk_level").ifBlank { "unknown" }
        val compatibilityScore = inferenceReport.optInt("local_inference_compatibility_score", 0)
        val compatibilityLevel = inferenceReport.optString("local_inference_compatibility_level").ifBlank { "unknown" }
        val compatibilityCount = inferenceReport.optInt("local_inference_compatibility_count", inferenceReport.optJSONArray("local_inference_compatibility_matrix")?.length() ?: 0)
        val compatibilityReadyCount = inferenceReport.optInt("ready_local_inference_compatibility_count", 0)
        val wifiReady = wifiReport.optBoolean("success", false) && (wifiNetworkCount > 0 || wifiBandCoverageCount > 0 || wifiChannelGraphCount > 0)
        val bluetoothReady = bluetoothReport.optBoolean("success", false) && (bluetoothDeviceCount > 0 || bluetoothMetadataCount > 0 || bluetoothHistoryCount > 0)
        val sensorReady = sensorReport.optBoolean("sensor_service_available", false) || sensorCapabilityCount > 0 || motionHistoryCount > 0 || motionPoseCount > 0
        val radioReady = radioBandCount > 0
        val inferenceReady = inferenceReport.optBoolean("success", false) && compatibilityCount > 0
        val readyDomainCount = listOf(wifiReady, bluetoothReady, sensorReady, radioReady, inferenceReady).count { it }
        val sourceActions = signalEvidenceSourceActions()
        val graphTypes = signalEvidenceGraphTypes()

        return JSONArray()
            .put(
                capabilityRow(
                    category = "signal_evidence",
                    label = "Current signal evidence bundle",
                    ready = readyDomainCount > 0,
                    valueLabel = "$readyDomainCount/5 evidence domain(s)",
                    detail = "Bundles passive Wi-Fi, Bluetooth, motion sensor, AM/FM/RF boundary, and local inference compatibility evidence for Gemma before natural-language reasoning.",
                    recommendation = "Read this bundle first when the user asks what Hermes can currently view or infer from nearby signals and phone context.",
                    fraction = (readyDomainCount / 5f).coerceIn(0.25f, 0.95f),
                    extra = JSONObject()
                        .put("evidence_key", "current_signal_bundle")
                        .put("source_actions", sourceActions)
                        .put("card_graph_types", graphTypes),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_wifi",
                    label = "Wi-Fi AP and channel evidence",
                    ready = wifiReady,
                    valueLabel = "$wifiNetworkCount AP(s), $wifiChannelGraphCount graph row(s)",
                    detail = "$wifiChannelRatingCount channel rating row(s), $wifiBandCoverageCount band coverage row(s), and $wifiHistoryCount cached RSSI history row(s) are available for WiFiAnalyzer-style cards.",
                    recommendation = "Open Wi-Fi Analyzer or channel graph cards before explaining placement, overlap, roaming, security, vendor/OUI, or channel congestion.",
                    fraction = when {
                        wifiNetworkCount > 0 && wifiChannelGraphCount > 0 -> 0.95f
                        wifiBandCoverageCount > 0 || wifiChannelRatingCount > 0 -> 0.8f
                        wifiReport.optBoolean("success", false) -> 0.55f
                        else -> 0.25f
                    },
                    extra = JSONObject()
                        .put("evidence_key", "wifi_ap_channel")
                        .put("tool_action", "wifi_analyzer_report")
                        .put("graph_type", "wifi_channel_graph")
                        .put("source_actions", JSONArray().put("wifi_analyzer_report").put("wifi_channel_graph").put("wifi_channel_rating").put("wifi_channel_utilization"))
                        .put("card_graph_types", JSONArray().put("wifi_channel_graph").put("wifi_channel_rating").put("wifi_channel_utilization").put("wifi_band_coverage").put("wifi_signal_history"))
                        .put("wifi_network_count", wifiNetworkCount)
                        .put("wifi_channel_graph_count", wifiChannelGraphCount)
                        .put("wifi_channel_rating_count", wifiChannelRatingCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_bluetooth",
                    label = "Bluetooth proximity evidence",
                    ready = bluetoothReady,
                    valueLabel = "$bluetoothDeviceCount device(s), $bluetoothHistoryCount trend row(s)",
                    detail = "$bluetoothMetadataCount metadata row(s), ${bluetoothReport.optInt("bluetooth_service_label_count", 0)} service label(s), and ${bluetoothReport.optInt("bluetooth_manufacturer_name_count", 0)} manufacturer label(s) are available for nearby-device cards.",
                    recommendation = "Open Bluetooth Analyzer or signal history cards before explaining proximity, beacons, paired inventory, service identity, or manufacturer context.",
                    fraction = when {
                        bluetoothDeviceCount > 0 && bluetoothMetadataCount > 0 -> 0.95f
                        bluetoothHistoryCount > 0 -> 0.75f
                        bluetoothReport.optBoolean("success", false) -> 0.55f
                        else -> 0.25f
                    },
                    extra = JSONObject()
                        .put("evidence_key", "bluetooth_proximity")
                        .put("tool_action", "bluetooth_analyzer_report")
                        .put("graph_type", "bluetooth_signal_history")
                        .put("source_actions", JSONArray().put("bluetooth_analyzer_report").put("bluetooth_scan").put("bluetooth_signal_history"))
                        .put("card_graph_types", JSONArray().put("bluetooth_metadata_summary").put("bluetooth_signal_history").put("bluetooth_rssi"))
                        .put("bluetooth_device_count", bluetoothDeviceCount)
                        .put("bluetooth_metadata_count", bluetoothMetadataCount)
                        .put("bluetooth_signal_history_count", bluetoothHistoryCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_motion",
                    label = "Motion and sensor evidence",
                    ready = sensorReady,
                    valueLabel = "$sensorCapabilityCount sensor row(s), $motionPoseCount pose row(s)",
                    detail = "$motionHistoryCount cached IMU trend row(s) can bind Wi-Fi or Bluetooth changes to phone movement, heading, acceleration, orientation, and pocket/desk state.",
                    recommendation = "Open Sensor Analyzer, motion history, or motion pose cards before attributing signal changes to distance or movement.",
                    fraction = when {
                        motionPoseCount > 0 -> 0.95f
                        sensorCapabilityCount > 0 -> 0.8f
                        motionHistoryCount > 0 -> 0.7f
                        else -> 0.3f
                    },
                    extra = JSONObject()
                        .put("evidence_key", "motion_sensor_context")
                        .put("tool_action", "sensor_analyzer_report")
                        .put("graph_type", "motion_pose_estimate")
                        .put("source_actions", JSONArray().put("sensor_analyzer_report").put("motion_pose").put("motion_sensor_history").put("sensor_snapshot"))
                        .put("card_graph_types", JSONArray().put("sensor_capability").put("motion_sensor_history").put("motion_pose_estimate"))
                        .put("sensor_capability_count", sensorCapabilityCount)
                        .put("motion_sensor_history_count", motionHistoryCount)
                        .put("motion_pose_estimate_count", motionPoseCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_radio",
                    label = "AM/FM and RF boundary evidence",
                    ready = radioReady,
                    valueLabel = "$radioBandCount band row(s)",
                    detail = "Radio graph bridge ready=$radioGraphReady, external SDR required=${radioReport.optBoolean("requires_external_sdr_for_broad_rf", true)}, signal constraints=$signalConstraintCount.",
                    recommendation = "Open radio cards to distinguish public Wi-Fi/Bluetooth signal access from vendor AM/FM tuner or external SDR requirements.",
                    fraction = if (radioReady) 0.85f else 0.25f,
                    extra = JSONObject()
                        .put("evidence_key", "radio_rf_boundary")
                        .put("tool_action", "radio_signal_status")
                        .put("graph_type", "radio_signal_graph")
                        .put("source_actions", JSONArray().put("radio_signal_status").put("radio_signal_graph").put("signal_capability_status"))
                        .put("card_graph_types", JSONArray().put("radio_signal_graph").put("radio_frequency_capability").put("radio_signal_constraint_matrix"))
                        .put("radio_band_plan_count", radioBandCount)
                        .put("radio_signal_graph_bridge_ready", radioGraphReady)
                        .put("radio_signal_constraint_count", signalConstraintCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_inference",
                    label = "Local inference readiness evidence",
                    ready = inferenceReady,
                    valueLabel = "$compatibilityLevel score $compatibilityScore",
                    detail = "$compatibilityReadyCount/$compatibilityCount compatibility row(s) ready; backend risk is $backendRiskLevel with $highBackendRiskCount high-risk row(s) across $backendRiskCount backend risk row(s).",
                    recommendation = "Open Local Inference Compatibility and GPU Backend Risk before claiming offline Gemma 4 multimodal readiness, especially on MediaTek, Mali, PowerVR, Xclipse, or other non-Adreno phones.",
                    fraction = if (compatibilityCount > 0) (compatibilityScore / 100f).coerceIn(0.15f, 0.95f) else 0.25f,
                    extra = JSONObject()
                        .put("evidence_key", "local_inference_readiness")
                        .put("tool_action", "local_inference_compatibility_report")
                        .put("graph_type", "local_inference_compatibility_matrix")
                        .put("source_actions", JSONArray().put("local_inference_compatibility_report").put("gpu_backend_risk_report").put("soc_compatibility_report").put("local_backend_runtime_report").put("device_performance_report"))
                        .put("card_graph_types", JSONArray().put("local_inference_compatibility_matrix").put("gpu_backend_risk_matrix").put("runtime_backend_matrix").put("soc_backend_matrix").put("runtime_stability_matrix"))
                        .put("local_inference_compatibility_score", compatibilityScore)
                        .put("local_inference_compatibility_level", compatibilityLevel)
                        .put("gpu_backend_risk_level", backendRiskLevel),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_policy",
                    label = "Permission and refresh evidence",
                    ready = signalAwarenessCount > 0,
                    valueLabel = "$signalAwarenessCount awareness row(s)",
                    detail = "Signal-awareness rows expose permission gates, cached histories, scan throttling, external-radio constraints, and passive-vs-live refresh policy.",
                    recommendation = "Use this row before requesting active Wi-Fi, Bluetooth, or sensor refreshes; use cached evidence when Android permissions or throttling block live scans.",
                    fraction = if (signalAwarenessCount > 0) 0.85f else 0.35f,
                    extra = JSONObject()
                        .put("evidence_key", "permission_refresh_policy")
                        .put("tool_action", "signal_awareness_report")
                        .put("graph_type", "signal_constraint_matrix")
                        .put("source_actions", JSONArray().put("signal_awareness_report").put("wifi_analyzer_report").put("bluetooth_analyzer_report").put("sensor_analyzer_report"))
                        .put("card_graph_types", JSONArray().put("signal_awareness_matrix").put("signal_workflow_routes").put("signal_constraint_matrix"))
                        .put("signal_awareness_count", signalAwarenessCount)
                        .put("signal_constraint_count", signalConstraintCount),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Evidence drill-down route",
                    ready = true,
                    valueLabel = "source cards",
                    detail = "Routes expose the exact analyzer, graph, scorecard, and manifest actions behind this evidence bundle.",
                    recommendation = "Use signal_evidence_routes when the answer needs an expandable card instead of a text-only summary.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("evidence_key", "source_card_drill_down")
                        .put("tool_action", "agent_signal_evidence_report")
                        .put("graph_type", "signal_evidence_routes")
                        .put("source_actions", sourceActions)
                        .put("card_graph_types", graphTypes),
                ),
            )
    }

    private fun agentSignalEvidenceRouteRows(): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open signal evidence bundle",
                    ready = true,
                    valueLabel = "agent_signal_evidence_report",
                    detail = "Use for the compact current-evidence view across Wi-Fi, Bluetooth, sensors, radio boundaries, and local inference compatibility.",
                    recommendation = "Run first for user questions about what Hermes or Gemma can currently view from nearby signals.",
                    fraction = 0.95f,
                    extra = JSONObject().put("tool_action", "agent_signal_evidence_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open Wi-Fi graph evidence",
                    ready = true,
                    valueLabel = "wifi_channel_graph",
                    detail = "Use for AP channel envelopes, RSSI, overlap pressure, channel widths, bands, and WiFiAnalyzer-style graph rows.",
                    recommendation = "Use refresh=false first, refresh=true only when the user needs current live scan evidence.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "wifi_channel_graph"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open Bluetooth proximity evidence",
                    ready = true,
                    valueLabel = "bluetooth_analyzer_report",
                    detail = "Use for paired/nearby devices, RSSI history, service UUID labels, manufacturer metadata, and proximity cards.",
                    recommendation = "Use bluetooth_signal_history after scans when movement or trend evidence matters.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "bluetooth_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open motion pose evidence",
                    ready = true,
                    valueLabel = "motion_pose",
                    detail = "Use for heading, tilt, angular velocity, acceleration state, and motion-aware signal interpretation.",
                    recommendation = "Prefer bounded sampling with only needed sensor_types when current pose matters.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "motion_pose"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open radio boundary evidence",
                    ready = true,
                    valueLabel = "radio_signal_graph",
                    detail = "Use for AM/FM band plan, signal graph schemas, receiver profiles, and external SDR/vendor bridge constraints.",
                    recommendation = "Keep AM/FM or broad RF claims behind public API, vendor, or external hardware evidence.",
                    fraction = 0.85f,
                    extra = JSONObject().put("tool_action", "radio_signal_graph"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open local inference evidence",
                    ready = true,
                    valueLabel = "local_inference_compatibility_report",
                    detail = "Use for Gemma 4/LiteRT/GGUF readiness, MediaTek/non-Adreno compatibility, backend risk, runtime health, and thermal/memory guardrails.",
                    recommendation = "Run before promising offline multimodal or local GPU behavior on the phone.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "local_inference_compatibility_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_evidence_route",
                    label = "Open card manifest evidence",
                    ready = true,
                    valueLabel = "agent_card_manifest_report",
                    detail = "Use for graph_type, source_action, refresh_policy, and permission_gate fields for each expandable top card.",
                    recommendation = "Open when Gemma needs to choose the exact card behind a signal-evidence explanation.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "agent_card_manifest_report"),
                ),
            )
    }

    private fun signalEvidenceSourceActions(): JSONArray = JSONArray()
        .put("wifi_analyzer_report")
        .put("wifi_channel_graph")
        .put("bluetooth_analyzer_report")
        .put("bluetooth_signal_history")
        .put("sensor_analyzer_report")
        .put("motion_pose")
        .put("radio_signal_status")
        .put("radio_signal_graph")
        .put("signal_awareness_report")
        .put("gpu_backend_risk_report")
        .put("local_inference_compatibility_report")
        .put("agent_card_manifest_report")

    private fun signalEvidenceGraphTypes(): JSONArray = JSONArray()
        .put("signal_evidence_matrix")
        .put("signal_evidence_routes")
        .put("wifi_channel_graph")
        .put("wifi_channel_rating")
        .put("wifi_channel_utilization")
        .put("bluetooth_signal_history")
        .put("motion_pose_estimate")
        .put("motion_sensor_history")
        .put("radio_signal_graph")
        .put("radio_frequency_capability")
        .put("signal_awareness_matrix")
        .put("signal_constraint_matrix")
        .put("gpu_backend_risk_matrix")
        .put("local_inference_compatibility_matrix")

    private fun agentObservationRouteRows(): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open the full observation dashboard",
                    ready = true,
                    valueLabel = "agent_observation_report",
                    detail = "Use for one compact Gemma-visible view across Wi-Fi, Bluetooth, sensors, radio, backend risk, SOC, local model, Kai operations, and cards.",
                    recommendation = "Run first for broad nearby-signal or phone-capability questions.",
                    fraction = 0.95f,
                    extra = JSONObject().put("tool_action", "agent_observation_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open Wi-Fi analyzer cards",
                    ready = true,
                    valueLabel = "wifi_analyzer_report",
                    detail = "Use for AP metadata, channel graph envelopes, channel ratings, utilization, band coverage, semantic labels, vendor/OUI, filters, exports, and signal history.",
                    recommendation = "Use refresh=false for planning and refresh=true only when current scan data is needed.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "wifi_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open Bluetooth analyzer cards",
                    ready = true,
                    valueLabel = "bluetooth_analyzer_report",
                    detail = "Use for nearby/paired inventory, RSSI trend, proximity, service labels, manufacturer metadata, and scan-policy boundaries.",
                    recommendation = "Use refresh=false for passive rows and refresh=true only when the user needs a live nearby scan.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "bluetooth_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open signal context fusion card",
                    ready = true,
                    valueLabel = "agent_observation_report",
                    detail = "Use for the combined Wi-Fi channel/band, Bluetooth RSSI/history, motion pose, and radio-boundary matrix before writing a nearby-signal explanation.",
                    recommendation = "Open after the observation dashboard when the user asks what the agent can see across nearby signals.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "agent_observation_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open sensor analyzer cards",
                    ready = true,
                    valueLabel = "sensor_analyzer_report",
                    detail = "Use for accelerometer, gyroscope, magnetic, ambient, hardware metadata, sampling policy, motion trends, and pose estimates.",
                    recommendation = "Keep include_snapshot=false until a live reading is actually needed.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "sensor_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open radio and RF limit cards",
                    ready = true,
                    valueLabel = "radio_signal_status",
                    detail = "Use for AM/FM band-plan rows, vendor radio hints, Wi-Fi/Bluetooth radio routes, SDR constraints, and public Android RF limits.",
                    recommendation = "Use before promising broad AM/FM or microwave scanning on normal phones.",
                    fraction = 0.85f,
                    extra = JSONObject().put("tool_action", "radio_signal_status"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open GPU backend risk cards",
                    ready = true,
                    valueLabel = "gpu_backend_risk_report",
                    detail = "Use for accelerator acceptance, SOC/GPU policy, thermal, memory, power, model artifact fit, phone validation, and CPU fallback rows.",
                    recommendation = "Run before local inference policy changes or non-Adreno device compatibility claims.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "gpu_backend_risk_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "agent_observation_route",
                    label = "Open SOC and Kai environment cards",
                    ready = true,
                    valueLabel = "agent_environment_report",
                    detail = "Use for MediaTek/Mali/PowerVR backend policy, local model readiness, tool bridge, memory, heartbeat, TTS, image, and Kai operation parity.",
                    recommendation = "Use when the question mixes device compatibility with autonomous-agent readiness.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "agent_environment_report"),
                ),
            )
    }

    private data class CardManifestSource(
        val action: String,
        val report: JSONObject,
        val refreshPolicy: String,
        val permissionGate: String,
    )

    private fun agentCardManifestSources(
        wifiReport: JSONObject,
        bluetoothReport: JSONObject,
        sensorReport: JSONObject,
        radioReport: JSONObject,
        backendRiskReport: JSONObject,
        signalReport: JSONObject,
        environmentReport: JSONObject,
    ): List<CardManifestSource> = listOf(
        CardManifestSource("wifi_analyzer_report", wifiReport, "passive_by_default_refresh_when_needed", "nearby_wifi_or_location_permission"),
        CardManifestSource("bluetooth_analyzer_report", bluetoothReport, "passive_by_default_refresh_when_needed", "bluetooth_scan_or_connect_permission"),
        CardManifestSource("sensor_analyzer_report", sensorReport, "passive_metadata_live_snapshot_optional", "sensor_hardware_availability"),
        CardManifestSource("radio_signal_status", radioReport, "passive_capability_boundary", "vendor_radio_bridge_or_external_sdr_for_am_fm"),
        CardManifestSource("gpu_backend_risk_report", backendRiskReport, "passive_backend_triage", "phone_validation_required_for_acceleration_claims"),
        CardManifestSource("signal_awareness_report", signalReport, "passive_fused_context", "source_report_permissions"),
        CardManifestSource("agent_environment_report", environmentReport, "passive_agent_readiness", "settings_and_local_state"),
    )

    private fun agentCardManifestSourceActions(sources: List<CardManifestSource>): JSONArray {
        return JSONArray().also { actions ->
            sources.forEach { source -> actions.put(source.action) }
        }
    }

    private fun agentCardManifestRows(sources: List<CardManifestSource>): JSONArray {
        val rows = JSONArray()
        sources.forEach { source ->
            val cards = source.report.optJSONArray("cards") ?: JSONArray()
            for (index in 0 until cards.length()) {
                val card = cards.optJSONObject(index) ?: continue
                val title = card.optString("title").ifBlank { "Diagnostic card ${index + 1}" }
                val graphType = card.optString("graph_type").ifBlank { "diagnostic_card" }
                val rowCount = card.optInt("row_count", card.optJSONArray("rows")?.length() ?: 0)
                rows.put(
                    capabilityRow(
                        category = "agent_card_manifest",
                        label = title,
                        ready = source.report.optBoolean("success", false) && rowCount > 0,
                        valueLabel = "$graphType via ${source.action}",
                        detail = "Card exposes $rowCount row(s) from ${source.action}; refresh_policy=${source.refreshPolicy}; permission_gate=${source.permissionGate}.",
                        recommendation = "Open this expandable card for evidence, or call ${source.action} only when its refresh policy says current data is needed.",
                        fraction = when {
                            source.report.optBoolean("success", false) && rowCount > 0 -> 0.95f
                            source.report.optBoolean("success", false) -> 0.7f
                            else -> 0.35f
                        },
                        extra = JSONObject()
                            .put("source_action", source.action)
                            .put("tool_action", source.action)
                            .put("graph_type", graphType)
                            .put("card_title", title)
                            .put("card_index", index)
                            .put("row_count", rowCount)
                            .put("refresh_policy", source.refreshPolicy)
                            .put("permission_gate", source.permissionGate)
                            .put("source_report_success", source.report.optBoolean("success", false))
                            .put("source_report_scope", source.report.optString("report_scope")),
                    ),
                )
            }
        }
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "Wi-Fi channel graph route",
            sourceAction = "wifi_channel_graph",
            graphType = "wifi_channel_graph",
            refreshPolicy = "active_refresh_on_request",
            permissionGate = "nearby_wifi_or_location_permission",
            detail = "WiFiAnalyzer-style AP channel envelopes can be requested even when the passive report has no current AP rows.",
        )
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "Wi-Fi channel rating route",
            sourceAction = "wifi_channel_rating",
            graphType = "wifi_channel_rating",
            refreshPolicy = "active_refresh_on_request",
            permissionGate = "nearby_wifi_or_location_permission",
            detail = "Wi-Fi channel scoring is available as a target card for choosing quiet 2.4 GHz, 5 GHz, and 6 GHz candidates.",
        )
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "Bluetooth nearby RSSI route",
            sourceAction = "bluetooth_scan",
            graphType = "bluetooth_rssi",
            refreshPolicy = "active_scan_on_request",
            permissionGate = "bluetooth_scan_or_connect_permission",
            detail = "Nearby Bluetooth and BLE RSSI cards can be requested when the user needs fresh proximity rows.",
        )
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "Bluetooth signal history route",
            sourceAction = "bluetooth_signal_history",
            graphType = "bluetooth_signal_history",
            refreshPolicy = "passive_after_scan",
            permissionGate = "bluetooth_scan_or_connect_permission",
            detail = "Bluetooth trend cards expose cached RSSI movement after paired or nearby scans.",
        )
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "Motion pose route",
            sourceAction = "motion_pose",
            graphType = "motion_pose_estimate",
            refreshPolicy = "bounded_live_sample_on_request",
            permissionGate = "motion_sensor_hardware",
            detail = "Motion pose cards fuse accelerometer, gyroscope, rotation, magnetic, gravity, and linear acceleration rows.",
        )
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "AM/FM signal graph route",
            sourceAction = "radio_signal_graph",
            graphType = "radio_signal_graph",
            refreshPolicy = "bridge_samples_required",
            permissionGate = "vendor_radio_bridge_or_external_sdr",
            detail = "AM/FM signal graph rows require vendor tuner or SDR sample input before Hermes can show station-like data.",
        )
        putCanonicalCardManifestRoute(
            rows = rows,
            label = "GPU backend risk route",
            sourceAction = "gpu_backend_risk_report",
            graphType = "gpu_backend_risk_matrix",
            refreshPolicy = "passive_backend_triage",
            permissionGate = "phone_validation_required_for_acceleration_claims",
            detail = "Backend risk cards keep MediaTek, Mali, PowerVR, Xclipse, non-Adreno, thermal, memory, and fallback evidence discoverable.",
        )
        return rows
    }

    private fun putCanonicalCardManifestRoute(
        rows: JSONArray,
        label: String,
        sourceAction: String,
        graphType: String,
        refreshPolicy: String,
        permissionGate: String,
        detail: String,
    ) {
        if (manifestContainsGraphType(rows, graphType)) return
        rows.put(
            capabilityRow(
                category = "agent_card_manifest",
                label = label,
                ready = true,
                valueLabel = "$graphType via $sourceAction",
                detail = "$detail refresh_policy=$refreshPolicy; permission_gate=$permissionGate.",
                recommendation = "Use $sourceAction when this graph type is the best evidence surface for the user's question.",
                fraction = 0.82f,
                extra = JSONObject()
                    .put("source_action", sourceAction)
                    .put("tool_action", sourceAction)
                    .put("graph_type", graphType)
                    .put("card_title", label)
                    .put("row_count", 0)
                    .put("refresh_policy", refreshPolicy)
                    .put("permission_gate", permissionGate)
                    .put("source_report_success", true)
                    .put("source_report_scope", "canonical_card_route"),
            ),
        )
    }

    private fun manifestContainsGraphType(rows: JSONArray, graphType: String): Boolean {
        for (index in 0 until rows.length()) {
            if (rows.optJSONObject(index)?.optString("graph_type") == graphType) return true
        }
        return false
    }

    private fun cardGraphTypeList(rows: JSONArray): JSONArray {
        val graphTypes = linkedSetOf<String>()
        for (index in 0 until rows.length()) {
            rows.optJSONObject(index)
                ?.optString("graph_type")
                ?.takeIf { it.isNotBlank() }
                ?.let { graphTypes.add(it) }
        }
        return JSONArray().also { array ->
            graphTypes.forEach { array.put(it) }
        }
    }

    private fun observationSummaryJson(report: JSONObject, action: String): JSONObject {
        return JSONObject()
            .put("action", action)
            .put("success", report.optBoolean("success", false))
            .put("report_scope", report.optString("report_scope"))
            .put("card_titles", reportCardTitles(report))
            .put("card_count", report.optJSONArray("cards")?.length() ?: 0)
    }

    private fun reportCardTitles(report: JSONObject, limit: Int = 6): JSONArray {
        val titles = JSONArray()
        val cards = report.optJSONArray("cards") ?: return titles
        for (index in 0 until minOf(cards.length(), limit)) {
            val title = cards.optJSONObject(index)?.optString("title").orEmpty()
            if (title.isNotBlank()) titles.put(title)
        }
        return titles
    }

    private fun workflowReadinessRows(
        diagnostics: JSONObject,
        signalStatus: JSONObject,
        preferredModel: JSONObject,
        automationStatus: JSONObject,
        modelRouting: JSONObject,
        availableSensors: JSONArray,
    ): JSONArray {
        val wifiPermission = diagnostics.optJSONObject("wifi_scan_permission_status") ?: JSONObject()
        val bluetoothPermission = diagnostics.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject()
        val sensorNames = jsonStringList(availableSensors)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "wireless_workflow",
                    label = "Analyze nearby Wi-Fi",
                    ready = wifiPermission.optBoolean("can_read_scan_results", false),
                    valueLabel = if (wifiPermission.optBoolean("can_read_scan_results", false)) "call wifi_ap_details" else "grant location/Wi-Fi scan access",
                    detail = "Best next tool: android_device_diagnostics_tool action=wifi_ap_details or wifi_export.",
                    recommendation = "Use before choosing channels, interpreting nearby signals, or explaining AP security/vendor metadata.",
                    fraction = if (wifiPermission.optBoolean("can_read_scan_results", false)) 1f else 0.45f,
                ),
            )
            .put(
                capabilityRow(
                    category = "wireless_workflow",
                    label = "Inspect nearby Bluetooth",
                    ready = bluetoothPermission.optBoolean("can_scan_nearby_devices", false),
                    valueLabel = if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) "call bluetooth_scan" else "grant Bluetooth scan access",
                    detail = "Best next tool: android_device_diagnostics_tool action=bluetooth_scan.",
                    recommendation = "Use for nearby BLE beacons, paired devices, service UUID labels, manufacturer names, and RSSI proximity.",
                    fraction = if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) 1f else 0.45f,
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_workflow",
                    label = "Sample motion/environment",
                    ready = sensorNames.isNotEmpty(),
                    valueLabel = "${sensorNames.size} sensor type(s)",
                    detail = "Best next tool: android_device_diagnostics_tool action=sensor_snapshot sensor_types=accelerometer,gyroscope,magnetic_field,light,proximity.",
                    recommendation = "Use for orientation, movement, ambient, and trigger-aware mobile workflows.",
                    fraction = (sensorNames.size / 8f).coerceIn(0.15f, 1f),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_agent_workflow",
                    label = "Run local multimodal agent",
                    ready = preferredModel.optBoolean("ready") && modelRouting.optBoolean("vision_capable", false),
                    valueLabel = if (preferredModel.optBoolean("ready")) modelRouting.optString("active_model").ifBlank { "preferred model ready" } else "model import needed",
                    detail = "Best next tools: local chat with image attachment, android_ui_tool visual_snapshot, and diagnostics/tool calls.",
                    recommendation = "Use the SOC backend policy and model routing status before assuming image or GPU availability.",
                    fraction = when {
                        preferredModel.optBoolean("ready") && modelRouting.optBoolean("vision_capable", false) -> 1f
                        preferredModel.optBoolean("ready") -> 0.7f
                        else -> 0.35f
                    },
                ),
            )
            .put(
                capabilityRow(
                    category = "automation_workflow",
                    label = "Route Kai-style tool orchestration",
                    ready = true,
                    valueLabel = "call agent_environment_report",
                    detail = "Best next tools: android_device_diagnostics_tool action=agent_environment_report or tool_catalog, then route provider/model/tool work through the Kai Operations card.",
                    recommendation = "Use before combining provider fallback, native tools, memory, backup, TTS, image, or shell work in one local-agent plan.",
                    fraction = 0.85f,
                ),
            )
            .put(
                capabilityRow(
                    category = "automation_workflow",
                    label = "Autonomous status and dispatch",
                    ready = automationStatus.optBoolean("standby_heartbeat_supported", false),
                    valueLabel = if (automationStatus.optBoolean("standby_heartbeat_supported", false)) "operator status ready" else "automation status only",
                    detail = "Best next tools: android_automation_tool operator_standby_status, operator_model_routing, operator_heartbeat.",
                    recommendation = "Use for heartbeat/self-check style progress visibility and remote task dispatch.",
                    fraction = if (automationStatus.optBoolean("standby_heartbeat_supported", false)) 0.95f else 0.45f,
                ),
            )
            .put(
                capabilityRow(
                    category = "rf_workflow",
                    label = "Explain radio signal limits",
                    ready = signalStatus.optBoolean("wifi_signal_analysis_supported", false) || signalStatus.optBoolean("bluetooth_signal_access_supported", false),
                    valueLabel = if (signalStatus.optBoolean("requires_external_sdr_for_broad_rf", true)) "external SDR for broad RF" else "built-in RF ready",
                    detail = "Best next tool: android_device_diagnostics_tool action=signal_capability_status or radio_signal_status.",
                    recommendation = "Use this row to be explicit about public Android AM/FM/microwave limitations.",
                    fraction = 0.5f,
                ),
            )
    }

    private fun signalAwarenessRows(
        diagnostics: JSONObject,
        signalStatus: JSONObject,
        radioStatus: JSONObject,
        preferredModel: JSONObject,
        socProfile: JSONObject,
        availableSensors: JSONArray,
        cachedWifiHistory: JSONArray,
        cachedBluetoothHistory: JSONArray,
        cachedMotionHistory: JSONArray,
        cachedMotionPoseEstimates: JSONArray,
        backendRiskReport: JSONObject,
    ): JSONArray {
        val wifiPermission = diagnostics.optJSONObject("wifi_scan_permission_status") ?: JSONObject()
        val bluetoothPermission = diagnostics.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject()
        val sensorNames = jsonStringList(availableSensors)
        val motionSensors = sensorNames.filter { it in MOTION_SENSOR_TYPES }
        val ambientSensors = sensorNames.filter { it in AMBIENT_SENSOR_TYPES }
        val wifiReady = diagnostics.optBoolean("wifi_supported", false) && wifiPermission.optBoolean("can_read_scan_results", false)
        val bluetoothReady = diagnostics.optBoolean("bluetooth_supported", false) &&
            (bluetoothPermission.optBoolean("can_scan_nearby_devices", false) || bluetoothPermission.optBoolean("can_read_paired_devices", false))
        val backendRiskCount = backendRiskReport.optInt("gpu_backend_risk_count", backendRiskReport.optJSONArray("gpu_backend_risk_matrix")?.length() ?: 0)
        val highBackendRiskCount = backendRiskReport.optInt("high_gpu_backend_risk_count", 0)
        val backendRiskLevel = backendRiskReport.optString("gpu_backend_risk_level").ifBlank { "unknown" }
        val backendRiskScore = backendRiskReport.optInt("gpu_backend_risk_score", 0)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Wi-Fi scan surface",
                    ready = wifiReady,
                    valueLabel = if (wifiReady) "live scan ready" else if (diagnostics.optBoolean("wifi_supported", false)) "permission gated" else "no Wi-Fi feature",
                    detail = "${cachedWifiHistory.length()} cached trend row(s); AP detail, channel rating, vendor/OUI, security, width, standard, and export rows are available when scan access is ready.",
                    recommendation = "Use wifi_ap_details or wifi_export before making channel, roaming, or nearby-network decisions.",
                    fraction = if (wifiReady) 1f else if (diagnostics.optBoolean("wifi_supported", false)) 0.55f else 0.1f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_ap_details")
                        .put("permission_gate", "nearby Wi-Fi/location scan access"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Cached Wi-Fi trend memory",
                    ready = cachedWifiHistory.length() > 0,
                    valueLabel = "${cachedWifiHistory.length()} tracked AP(s)",
                    detail = "Hermes keeps bounded Wi-Fi RSSI history so Gemma can compare current, average, min/max, trend, and last-seen metadata after scans.",
                    recommendation = "Run wifi_scan periodically when diagnosing changing signal strength or room-to-room network quality.",
                    fraction = if (cachedWifiHistory.length() > 0) (cachedWifiHistory.length() / 8f).coerceIn(0.35f, 1f) else 0.25f,
                    extra = JSONObject().put("tool_action", "wifi_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Bluetooth proximity metadata",
                    ready = bluetoothReady,
                    valueLabel = if (bluetoothReady) "scan or paired metadata ready" else "permission gated",
                    detail = "${cachedBluetoothHistory.length()} cached trend row(s); Bluetooth rows can expose paired devices, BLE RSSI, proximity, class/category, service UUID labels, manufacturer names, and connectable status.",
                    recommendation = "Use bluetooth_scan before reasoning about nearby peripherals, beacons, wearables, audio devices, or service advertisements.",
                    fraction = if (bluetoothReady) 1f else if (diagnostics.optBoolean("bluetooth_supported", false)) 0.55f else 0.1f,
                    extra = JSONObject()
                        .put("tool_action", "bluetooth_scan")
                        .put("permission_gate", "Bluetooth connect/scan access"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Cached Bluetooth trend memory",
                    ready = cachedBluetoothHistory.length() > 0,
                    valueLabel = "${cachedBluetoothHistory.length()} tracked device(s)",
                    detail = "Hermes keeps bounded Bluetooth RSSI history so Gemma can compare current, average, min/max, trend, proximity, and last-seen metadata after BLE scans.",
                    recommendation = "Run bluetooth_signal_history after scans when diagnosing moving beacons, wearables, controllers, or audio devices.",
                    fraction = if (cachedBluetoothHistory.length() > 0) (cachedBluetoothHistory.length() / 8f).coerceIn(0.35f, 1f) else 0.25f,
                    extra = JSONObject().put("tool_action", "bluetooth_signal_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Motion/orientation sensors",
                    ready = motionSensors.isNotEmpty(),
                    valueLabel = "${motionSensors.size} motion type(s)",
                    detail = "${cachedMotionHistory.length()} cached trend row(s); motion sensors present: ${motionSensors.joinToString(", ").ifBlank { "none reported" }}.",
                    recommendation = "Use motion_sensor_history for recent accelerometer/gyroscope trends or sensor_snapshot for a one-shot current reading before motion-aware workflows.",
                    fraction = (motionSensors.size / MOTION_SENSOR_TYPES.size.toFloat()).coerceIn(0.1f, 1f),
                    extra = JSONObject().put("tool_action", "motion_sensor_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Cached motion trend memory",
                    ready = cachedMotionHistory.length() > 0,
                    valueLabel = "${cachedMotionHistory.length()} tracked sensor(s)",
                    detail = "Hermes keeps bounded IMU magnitude history so Gemma can compare current, average, range, trend, stability, current vector, and recent series after sensor samples.",
                    recommendation = "Run motion_sensor_history when diagnosing movement changes, orientation shifts, device handling, or sensor stability.",
                    fraction = if (cachedMotionHistory.length() > 0) (cachedMotionHistory.length() / MOTION_HISTORY_SENSOR_TYPES.size.toFloat()).coerceIn(0.35f, 1f) else 0.25f,
                    extra = JSONObject().put("tool_action", "motion_sensor_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Motion pose fusion",
                    ready = cachedMotionPoseEstimates.length() > 0 || motionSensors.isNotEmpty(),
                    valueLabel = if (cachedMotionPoseEstimates.length() > 0) "${cachedMotionPoseEstimates.length()} pose row(s)" else "sample needed",
                    detail = "Hermes fuses accelerometer/gravity, magnetic-field, rotation-vector, gyroscope, and linear-acceleration rows into pose, heading, angular-motion, and acceleration-state context for Gemma.",
                    recommendation = "Use motion_pose or sensor_snapshot with accelerometer, magnetic_field, rotation_vector, gyroscope, and linear_acceleration before orientation-aware automations.",
                    fraction = if (cachedMotionPoseEstimates.length() > 0) 0.9f else if (motionSensors.isNotEmpty()) 0.55f else 0.2f,
                    extra = JSONObject().put("tool_action", "motion_pose"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Ambient/environment sensors",
                    ready = ambientSensors.isNotEmpty(),
                    valueLabel = "${ambientSensors.size} ambient type(s)",
                    detail = "Ambient sensors present: ${ambientSensors.joinToString(", ").ifBlank { "none reported" }}.",
                    recommendation = "Use sensor_snapshot for light, proximity, pressure, temperature, humidity, or magnetic-field context when environment can affect the workflow.",
                    fraction = (ambientSensors.size / AMBIENT_SENSOR_TYPES.size.toFloat()).coerceIn(0.1f, 1f),
                    extra = JSONObject().put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Radio/RF limits",
                    ready = radioStatus.optBoolean("am_fm_public_android_scan_supported", false),
                    valueLabel = if (signalStatus.optBoolean("requires_external_sdr_for_broad_rf", true)) "external SDR needed" else "built-in radio feed",
                    detail = "AM/FM, broad RF, and microwave scans are not exposed through normal public Android APIs on this device path.",
                    recommendation = "Use radio_signal_status and external SDR/vendor bridges for broad RF, AM, FM, or microwave work.",
                    fraction = if (radioStatus.optBoolean("am_fm_public_android_scan_supported", false)) 0.8f else 0.25f,
                    extra = JSONObject()
                        .put("tool_action", "radio_signal_status")
                        .put("hardware_gate", "external SDR or vendor radio API"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "SOC backend compatibility",
                    ready = true,
                    valueLabel = socProfile.optString("litert_lm_acceleration_label").ifBlank { "SOC-neutral policy" },
                    detail = "${socProfile.optString("soc_family_label").ifBlank { "unknown SOC" }} | ${socProfile.optString("gpu_family_label").ifBlank { "unknown GPU" }} | ${socProfile.optString("primary_abi").ifBlank { "unknown ABI" }}",
                    recommendation = "Keep MediaTek/Mali/PowerVR, Tensor/Mali, Exynos/Xclipse, Snapdragon/Adreno, Unisoc, and CPU fallback paths visible before local model work.",
                    fraction = 0.95f,
                    extra = JSONObject().put("source_surface", "soc_profile"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "GPU backend risk triage",
                    ready = backendRiskReport.optBoolean("success", false) && backendRiskCount > 0,
                    valueLabel = "$backendRiskLevel risk, $highBackendRiskCount high row(s)",
                    detail = "$backendRiskCount risk row(s) cover accelerator acceptance, SOC/GPU policy, thermal, memory, power, model artifact fit, validation scope, and fallback routing.",
                    recommendation = "Use gpu_backend_risk_report before answering whether this phone can safely use local GPU acceleration.",
                    fraction = if (backendRiskCount > 0) ((100 - backendRiskScore).coerceIn(5, 100) / 100f) else 0.25f,
                    extra = JSONObject()
                        .put("tool_action", "gpu_backend_risk_report")
                        .put("graph_type", "gpu_backend_risk_matrix")
                        .put("gpu_backend_risk_level", backendRiskLevel)
                        .put("gpu_backend_risk_score", backendRiskScore),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_awareness",
                    label = "Local multimodal reasoning",
                    ready = preferredModel.optBoolean("ready"),
                    valueLabel = preferredModel.optString("title").ifBlank { "model import needed" },
                    detail = "Signal reports are structured so Gemma can read wireless metadata, sensor metadata, radio limits, and SOC policy as compact top cards.",
                    recommendation = "Import and prefer a local Gemma LiteRT-LM or Qwen GGUF model before treating signal-aware workflows as offline-ready.",
                    fraction = if (preferredModel.optBoolean("ready")) 1f else 0.35f,
                    extra = JSONObject().put("source_surface", "preferred_local_model"),
                ),
            )
    }

    private fun wifiAnalyzerFeatureRows(
        wifiAvailable: Boolean,
        permissionStatus: JSONObject,
        networkCount: Int,
        returnedNetworkCount: Int,
        accessPointDetailCount: Int,
        channelRatingCount: Int,
        channelGraphCount: Int,
        vendorCount: Int,
        filterCount: Int,
        historyCount: Int,
        semanticCount: Int,
        bandCoverageCount: Int,
        observedBandCount: Int,
        channelUtilizationCount: Int,
        widthCount: Int,
        standardCount: Int,
        exportReady: Boolean,
    ): JSONArray {
        val scanReady = wifiAvailable && permissionStatus.optBoolean("can_read_scan_results", false)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Identify nearby access points",
                    ready = scanReady && networkCount > 0,
                    valueLabel = if (scanReady) "$networkCount AP(s)" else "permission gated",
                    detail = "$returnedNetworkCount AP row(s) are ready for top cards; complete AP details include SSID/BSSID, RSSI, channel, band, security, width, standard, OUI/vendor, and distance metadata.",
                    recommendation = "Use wifi_scan for a compact graph or wifi_ap_details for the complete AP table before network-placement decisions.",
                    fraction = if (scanReady && networkCount > 0) 1f else if (scanReady) 0.7f else 0.4f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_scan")
                        .put("feature_source", "WiFiAnalyzer nearby access points"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Channel signal graph",
                    ready = channelGraphCount > 0,
                    valueLabel = "$channelGraphCount AP envelope row(s)",
                    detail = "Hermes maps each nearby AP onto a WiFiAnalyzer-style channel graph envelope with dBm, channel width, channel span, frequency span, and overlap pressure.",
                    recommendation = "Use wifi_channel_graph when the user asks what the channel graph looks like or which visible APs overlap by width.",
                    fraction = if (channelGraphCount > 0) 1f else if (scanReady) 0.55f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_channel_graph")
                        .put("feature_source", "WiFiAnalyzer channel graph"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Channel rating",
                    ready = channelRatingCount > 0,
                    valueLabel = "$channelRatingCount channel score row(s)",
                    detail = "Hermes scores candidate 2.4GHz, 5GHz, and 6GHz channels from crowding, overlap, signal strength, width metadata, and 6GHz preferred candidate channels.",
                    recommendation = "Use wifi_channel_rating to pick a candidate channel instead of judging only by strongest RSSI.",
                    fraction = if (channelRatingCount > 0) 1f else if (scanReady) 0.55f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_channel_rating")
                        .put("feature_source", "WiFiAnalyzer channel rating"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Channel utilization occupancy",
                    ready = channelUtilizationCount > 0,
                    valueLabel = "$channelUtilizationCount utilization row(s)",
                    detail = "Hermes infers channel occupancy from visible AP counts, overlap, RSSI pressure, channel width, security modes, and SSID samples. It does not claim airtime counters that Android scan APIs do not expose.",
                    recommendation = "Use wifi_channel_utilization when the user asks which Wi-Fi channels look busy, noisy, crowded, or interference-heavy.",
                    fraction = if (channelUtilizationCount > 0) 0.95f else if (scanReady) 0.6f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_channel_utilization")
                        .put("feature_source", "WiFiAnalyzer channel utilization parity"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Access-point signal history",
                    ready = historyCount > 0,
                    valueLabel = "$historyCount tracked AP(s)",
                    detail = "Cached AP history keeps current, average, min/max, trend, sample count, and last-seen metadata for Gemma-readable signal-over-time cards.",
                    recommendation = "Run wifi_scan periodically when diagnosing roaming, room-to-room signal changes, or unstable RSSI.",
                    fraction = if (historyCount > 0) 0.95f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_scan")
                        .put("feature_source", "WiFiAnalyzer signal strength over time"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Pause/resume scan control",
                    ready = true,
                    valueLabel = "scan_mode ready",
                    detail = "Hermes accepts scan_mode=paused to reuse cached AP rows/history without active startScan, and scan_mode=resumed to request a fresh Android scan on direct Wi-Fi actions.",
                    recommendation = "Pause repeated scans during passive review; resume only when the user asks for a fresh nearby Wi-Fi reading.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_scan")
                        .put("feature_source", "WiFiAnalyzer pause/resume scanning"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Band coverage and 2.4/5/6GHz visibility",
                    ready = observedBandCount > 0 || scanReady,
                    valueLabel = "$observedBandCount observed band(s)",
                    detail = "Hermes summarizes observed 2.4GHz, 5GHz, and 6GHz AP counts with channels, widths, standards, security attention, hidden SSIDs, and best rated channel hints.",
                    recommendation = "Use wifi_scan or wifi_analyzer_report before advising channel plans so Gemma can see which bands are actually visible on this device.",
                    fraction = if (observedBandCount > 0) 0.95f else if (scanReady) 0.6f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_analyzer_report")
                        .put("feature_source", "WiFiAnalyzer 2.4/5/6GHz band switching"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Band, security, signal, and SSID filters",
                    ready = filterCount > 0,
                    valueLabel = "$filterCount filter group(s)",
                    detail = "Filter facets cover Wi-Fi band, signal quality, security mode, and visible SSID groups for compact Gemma context.",
                    recommendation = "Use wifi_ap_details when a user asks to narrow APs by band, security, signal quality, or SSID.",
                    fraction = if (filterCount > 0) 1f else if (scanReady) 0.6f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_ap_details")
                        .put("feature_source", "WiFiAnalyzer filters"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Agent AP semantic and risk labels",
                    ready = semanticCount > 0,
                    valueLabel = "$semanticCount AP semantic row(s)",
                    detail = "Hermes labels likely private routers, public/guest hotspots, hidden SSIDs, passpoint/venue APs, mesh/repeater candidates, IoT/device APs, and open/WEP/WPS attention rows for Gemma-readable reasoning.",
                    recommendation = "Use wifi_ap_details or wifi_analyzer_report when the user asks what nearby networks are or which ones deserve security attention.",
                    fraction = if (semanticCount > 0) 0.95f else if (scanReady) 0.6f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_ap_details")
                        .put("feature_source", "WiFiAnalyzer AP details, hidden SSID, and security interpretation"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Vendor/OUI lookup",
                    ready = vendorCount > 0,
                    valueLabel = "$vendorCount vendor group(s)",
                    detail = "Hermes uses local OUI prefix hints from scan metadata and does not perform network lookups for vendor labels.",
                    recommendation = "Use vendor rows as hints, and preserve BSSID/OUI fields for Gemma when explaining nearby infrastructure.",
                    fraction = if (vendorCount > 0) 0.9f else 0.45f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_ap_details")
                        .put("feature_source", "WiFiAnalyzer vendor/OUI database lookup"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "HT/VHT/HE/EHT width and standard metadata",
                    ready = widthCount > 0 || standardCount > 0,
                    valueLabel = "$widthCount width group(s), $standardCount standard group(s)",
                    detail = "Android scan metadata can expose 40/80/160/320 MHz width and 802.11n/ac/ax/be standard hints when hardware and OS support them.",
                    recommendation = "Use width and standard rows to avoid assuming throughput from band or RSSI alone.",
                    fraction = if (widthCount > 0 || standardCount > 0) 0.9f else 0.45f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_ap_details")
                        .put("feature_source", "WiFiAnalyzer HT/VHT detection"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Export access point details",
                    ready = exportReady,
                    valueLabel = if (exportReady) "$accessPointDetailCount export row(s)" else "scan needed",
                    detail = "Wi-Fi AP details can be exported as JSON, CSV, or both for follow-up analysis or bug reports.",
                    recommendation = "Use wifi_export when the user needs a portable AP table instead of only an in-chat graph.",
                    fraction = if (exportReady) 1f else if (scanReady) 0.6f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "wifi_export")
                        .put("feature_source", "WiFiAnalyzer export access points details"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_parity",
                    label = "Wi-Fi safety boundary",
                    ready = true,
                    valueLabel = "analysis only",
                    detail = "Hermes reports signal, channel, metadata, and export rows; it is not a password cracking, phishing, or network intrusion tool.",
                    recommendation = "Keep analysis limited to observable Android scan metadata and user-authorized workflows.",
                    fraction = 1f,
                    extra = JSONObject().put("feature_source", "WiFiAnalyzer privacy boundary"),
                ),
            )
    }

    private fun wifiAnalyzerWorkflowRows(
        permissionStatus: JSONObject,
        scanStatus: JSONObject,
        historyCount: Int,
        channelGraphCount: Int,
        channelRatingCount: Int,
        accessPointDetailCount: Int,
    ): JSONArray {
        val scanReady = permissionStatus.optBoolean("can_read_scan_results", false)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route compact signal graph",
                    ready = scanReady,
                    valueLabel = "wifi_scan",
                    detail = "Use when the user asks what nearby Wi-Fi looks like or wants a compact RSSI/channel graph.",
                    recommendation = "Request location/nearby Wi-Fi permission first when scan rows are gated.",
                    fraction = if (scanReady) 0.95f else 0.45f,
                    extra = JSONObject().put("tool_action", "wifi_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route best-channel analysis",
                    ready = channelRatingCount > 0 || scanReady,
                    valueLabel = "wifi_channel_rating",
                    detail = "Use for channel crowding, overlap, width-aware score, and recommended channel rows.",
                    recommendation = "Prefer this route for router-placement or channel-selection questions.",
                    fraction = if (channelRatingCount > 0) 1f else if (scanReady) 0.75f else 0.45f,
                    extra = JSONObject().put("tool_action", "wifi_channel_rating"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route channel graph envelopes",
                    ready = channelGraphCount > 0 || scanReady,
                    valueLabel = "wifi_channel_graph",
                    detail = "Use for WiFiAnalyzer-style AP channel graph rows with dBm, channel width, channel span, frequency span, and overlap pressure.",
                    recommendation = "Prefer this route when the user asks what visible access points overlap on the channel graph.",
                    fraction = if (channelGraphCount > 0) 1f else if (scanReady) 0.75f else 0.45f,
                    extra = JSONObject().put("tool_action", "wifi_channel_graph"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route full AP metadata",
                    ready = accessPointDetailCount > 0 || scanReady,
                    valueLabel = "wifi_ap_details",
                    detail = "Use for SSID/BSSID, security, vendor/OUI, standard, width, estimated distance, filter facets, and AP table cards.",
                    recommendation = "Choose this when Gemma needs inspectable metadata instead of only a graph.",
                    fraction = if (accessPointDetailCount > 0) 1f else if (scanReady) 0.75f else 0.45f,
                    extra = JSONObject().put("tool_action", "wifi_ap_details"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route AP export",
                    ready = accessPointDetailCount > 0 || scanReady,
                    valueLabel = "wifi_export",
                    detail = "Use for JSON/CSV access-point details when the user needs a portable table.",
                    recommendation = "Include export_format=json, csv, or both depending on the requested artifact.",
                    fraction = if (accessPointDetailCount > 0) 1f else if (scanReady) 0.7f else 0.4f,
                    extra = JSONObject().put("tool_action", "wifi_export"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route history-first explanation",
                    ready = historyCount > 0,
                    valueLabel = "wifi_signal_history",
                    detail = "Use cached RSSI trend rows when Android scan throttling or permissions make live refresh unreliable.",
                    recommendation = "Explain scan age and trend metadata when using cached rows.",
                    fraction = if (historyCount > 0) 0.9f else 0.35f,
                    extra = JSONObject().put("tool_action", "wifi_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route pause or resume scan mode",
                    ready = true,
                    valueLabel = "scan_mode",
                    detail = "Use scan_mode=paused with wifi_scan, wifi_channel_graph, wifi_ap_details, wifi_export, wifi_channel_rating, or wifi_channel_utilization to keep analysis cached; use scan_mode=resumed to request a fresh direct scan.",
                    recommendation = "Prefer paused mode for repeated card review and resumed mode for explicit fresh-scan requests.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "wifi_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_analyzer_route",
                    label = "Route scan policy explanation",
                    ready = true,
                    valueLabel = "wifi_analyzer_report",
                    detail = scanStatus.optString("android_scan_throttle_note").ifBlank { "Android scan policy metadata is available in the report." },
                    recommendation = "Use this report before repeatedly refreshing scans so the agent can avoid noisy or throttled polling.",
                    fraction = 0.8f,
                    extra = JSONObject().put("tool_action", "wifi_analyzer_report"),
                ),
            )
    }

    private fun wifiScanPolicyRows(
        wifiAvailable: Boolean,
        permissionStatus: JSONObject,
        scanStatus: JSONObject,
        scanSucceeded: Boolean,
    ): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "wifi_scan_policy",
                    label = "Wi-Fi service availability",
                    ready = wifiAvailable,
                    valueLabel = if (wifiAvailable) "service present" else "no Wi-Fi service",
                    detail = "Android must expose WifiManager before Hermes can read scan result metadata.",
                    recommendation = "Report lack of Wi-Fi hardware or service honestly instead of fabricating nearby AP rows.",
                    fraction = if (wifiAvailable) 1f else 0.1f,
                    extra = JSONObject().put("constraint_type", "hardware_service"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_scan_policy",
                    label = "Location and nearby Wi-Fi permissions",
                    ready = permissionStatus.optBoolean("can_read_scan_results", false),
                    valueLabel = if (permissionStatus.optBoolean("can_read_scan_results", false)) "scan readable" else "permission gated",
                    detail = "Fine location, nearby Wi-Fi on Android 13+, and enabled location services can gate scan reads.",
                    recommendation = "Use open_location_settings or app settings when the user wants live Wi-Fi analysis and permissions are missing.",
                    fraction = if (permissionStatus.optBoolean("can_read_scan_results", false)) 0.95f else 0.45f,
                    extra = JSONObject()
                        .put("constraint_type", "permission")
                        .put("permission_gate", "fine location, nearby Wi-Fi, location services"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_scan_policy",
                    label = "Android scan throttling",
                    ready = true,
                    valueLabel = if (scanStatus.optBoolean("refresh_accepted", false)) "refresh accepted" else "passive or cached",
                    detail = scanStatus.optString("android_scan_throttle_note").ifBlank { "Android may throttle active Wi-Fi scans." },
                    recommendation = "Prefer passive scan reads, scan age, and cached history over tight refresh loops.",
                    fraction = 0.8f,
                    extra = JSONObject().put("constraint_type", "android_policy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_scan_policy",
                    label = "Pause/resume scan mode",
                    ready = true,
                    valueLabel = scanStatus.optJSONObject("wifi_scan_control")?.optString("scan_mode").orEmpty().ifBlank { WIFI_SCAN_MODE_AUTO },
                    detail = "scan_mode=paused suppresses active refresh even when refresh=true, while scan_mode=resumed requests a fresh Android scan on direct Wi-Fi actions.",
                    recommendation = "Use paused mode for cached signal dashboards and resumed mode for explicit fresh nearby-signal scans.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("constraint_type", "agent_scan_control")
                        .put("tool_action", "wifi_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_scan_policy",
                    label = "Passive report default",
                    ready = true,
                    valueLabel = if (scanSucceeded) "scan context reused" else "no active refresh",
                    detail = "wifi_analyzer_report does not force a scan refresh; it reuses currently available scan data or returns readiness/policy rows.",
                    recommendation = "Use refresh=true only on the narrower wifi_scan, wifi_channel_graph, wifi_ap_details, wifi_export, or wifi_channel_rating actions when the user needs a fresh scan.",
                    fraction = 0.85f,
                    extra = JSONObject().put("constraint_type", "scan_cadence"),
                ),
            )
            .put(
                capabilityRow(
                    category = "wifi_scan_policy",
                    label = "Analysis and privacy boundary",
                    ready = true,
                    valueLabel = "metadata only",
                    detail = "Hermes Wi-Fi analysis uses Android scan metadata and local OUI hints; it does not crack passwords, phish, or probe networks.",
                    recommendation = "Keep user-facing answers scoped to signal, channel, metadata, and user-authorized troubleshooting.",
                    fraction = 1f,
                    extra = JSONObject().put("constraint_type", "privacy_safety"),
                ),
            )
    }

    private fun bluetoothAnalyzerFeatureRows(
        bluetoothAvailable: Boolean,
        bluetoothLeSupported: Boolean,
        bluetoothEnabled: Boolean,
        permissionStatus: JSONObject,
        deviceCount: Int,
        metadataCount: Int,
        serviceUuidCount: Int,
        manufacturerIdCount: Int,
        rssiDeviceCount: Int,
        historyCount: Int,
        categoryCount: Int,
    ): JSONArray {
        val canReadPaired = permissionStatus.optBoolean("can_read_paired_devices", false)
        val canScanNearby = permissionStatus.optBoolean("can_scan_nearby_devices", false)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Identify paired devices",
                    ready = bluetoothAvailable && canReadPaired,
                    valueLabel = if (deviceCount > 0) "$deviceCount device row(s)" else if (canReadPaired) "inventory ready" else "permission gated",
                    detail = "Hermes can read bonded-device identity, type, class, and pairing metadata when Android grants Bluetooth connect access.",
                    recommendation = "Use bluetooth_scan for paired-device inventory before explaining remembered headsets, wearables, controllers, or beacons.",
                    fraction = if (bluetoothAvailable && canReadPaired) 0.9f else 0.4f,
                    extra = JSONObject()
                        .put("feature_source", "Android bonded devices")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Scan nearby BLE devices",
                    ready = bluetoothLeSupported && bluetoothEnabled && canScanNearby,
                    valueLabel = when {
                        bluetoothLeSupported && bluetoothEnabled && canScanNearby -> "BLE scan ready"
                        !bluetoothLeSupported -> "no BLE feature"
                        !bluetoothEnabled -> "Bluetooth disabled"
                        else -> "permission gated"
                    },
                    detail = "Nearby BLE rows come from Android BluetoothLeScanner and can include advertisements, RSSI, service UUID labels, manufacturer names, and connectability hints.",
                    recommendation = "Use refresh=true on bluetooth_scan only when the user needs a fresh nearby-device sample.",
                    fraction = if (bluetoothLeSupported && bluetoothEnabled && canScanNearby) 0.95f else if (bluetoothLeSupported) 0.45f else 0.2f,
                    extra = JSONObject()
                        .put("feature_source", "Android BluetoothLeScanner")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Pause/resume BLE scan control",
                    ready = true,
                    valueLabel = "scan_mode ready",
                    detail = "Hermes accepts scan_mode=paused to reuse paired/passive rows and cached RSSI history without starting BluetoothLeScanner, and scan_mode=resumed to request a fresh BLE sample on direct Bluetooth actions.",
                    recommendation = "Pause repeated BLE scans during passive review; resume only when the user asks for a fresh nearby Bluetooth reading.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("feature_source", "Bluetooth active scan cadence control")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "RSSI proximity graph",
                    ready = rssiDeviceCount > 0 || (bluetoothLeSupported && canScanNearby),
                    valueLabel = if (rssiDeviceCount > 0) "$rssiDeviceCount RSSI row(s)" else "scan route ready",
                    detail = "Bluetooth graph cards convert RSSI and optional TX power into proximity labels and distance estimates when Android exposes them.",
                    recommendation = "Use bluetooth_scan for proximity or beacon-strength questions, and explain when paired rows do not expose RSSI.",
                    fraction = if (rssiDeviceCount > 0) 1f else if (bluetoothLeSupported && canScanNearby) 0.75f else 0.35f,
                    extra = JSONObject()
                        .put("feature_source", "BLE RSSI metadata")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "RSSI trend history graph",
                    ready = historyCount > 0 || (bluetoothLeSupported && canScanNearby),
                    valueLabel = if (historyCount > 0) "$historyCount tracked device(s)" else "scan route ready",
                    detail = "Hermes keeps bounded Bluetooth RSSI history from BLE scan observations so Gemma can compare current, average, min/max, trend, and last-seen metadata.",
                    recommendation = "Use bluetooth_signal_history after scans when the user asks whether nearby Bluetooth devices are approaching, fading, or stable.",
                    fraction = if (historyCount > 0) (historyCount / 8f).coerceIn(0.45f, 1f) else if (bluetoothLeSupported && canScanNearby) 0.7f else 0.3f,
                    extra = JSONObject()
                        .put("feature_source", "bounded BLE RSSI history")
                        .put("tool_action", "bluetooth_signal_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Service UUID labels",
                    ready = serviceUuidCount > 0 || (bluetoothLeSupported && canScanNearby),
                    valueLabel = if (serviceUuidCount > 0) "$serviceUuidCount UUID group(s)" else "advertisement gated",
                    detail = "Service UUIDs are mapped to assigned labels where known so the agent can infer device role, such as heart-rate, battery, HID, audio, beacon, or vendor service surfaces.",
                    recommendation = "Prefer service label rows before claiming a device capability; tell the user when Android did not expose or recognize them.",
                    fraction = if (serviceUuidCount > 0) 1f else if (bluetoothLeSupported && canScanNearby) 0.7f else 0.35f,
                    extra = JSONObject()
                        .put("feature_source", "BLE advertisement metadata")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Manufacturer names",
                    ready = manufacturerIdCount > 0 || (bluetoothLeSupported && canScanNearby),
                    valueLabel = if (manufacturerIdCount > 0) "$manufacturerIdCount manufacturer group(s)" else "advertisement gated",
                    detail = "Manufacturer IDs are mapped to company names where known, with payload byte counts for beacon/vendor-specific devices without connecting to them.",
                    recommendation = "Use manufacturer name rows for beacon and vendor analysis, while avoiding overconfident identity claims from partial advertisements.",
                    fraction = if (manufacturerIdCount > 0) 1f else if (bluetoothLeSupported && canScanNearby) 0.7f else 0.35f,
                    extra = JSONObject()
                        .put("feature_source", "BLE manufacturer data")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Device category hints",
                    ready = categoryCount > 0 || metadataCount > 0 || canReadPaired || canScanNearby,
                    valueLabel = if (categoryCount > 0) "$categoryCount category group(s)" else "class inference ready",
                    detail = "Hermes groups Bluetooth class, device type, and metadata summary rows so Gemma can reason about audio, wearable, HID, beacon, and unknown devices.",
                    recommendation = "Use category rows as hints, not final identity, because Android metadata can be sparse or vendor-specific.",
                    fraction = if (categoryCount > 0 || metadataCount > 0) 0.9f else if (canReadPaired || canScanNearby) 0.7f else 0.35f,
                    extra = JSONObject()
                        .put("feature_source", "Android Bluetooth class and summary rows")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Gemma-readable Bluetooth cards",
                    ready = true,
                    valueLabel = "${deviceCount + metadataCount} card row(s)",
                    detail = "Bluetooth analyzer reports create top-level expandable cards for readiness, routes, scan policy, nearby RSSI rows, and metadata summaries.",
                    recommendation = "Show Bluetooth cards before long explanations when the user asks what nearby devices or metadata the agent can see.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("feature_source", "Hermes diagnostic cards")
                        .put("tool_action", "bluetooth_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_parity",
                    label = "Bluetooth safety boundary",
                    ready = true,
                    valueLabel = "metadata only",
                    detail = "Hermes reports Android-exposed Bluetooth metadata and does not pair, connect, track people, exploit devices, or bypass OS permissions.",
                    recommendation = "Keep analysis scoped to user-authorized nearby/paired metadata and explain missing permission or hardware gates directly.",
                    fraction = 1f,
                    extra = JSONObject().put("feature_source", "Bluetooth privacy boundary"),
                ),
            )
    }

    private fun bluetoothAnalyzerWorkflowRows(
        permissionStatus: JSONObject,
        deviceCount: Int,
        serviceUuidCount: Int,
        manufacturerIdCount: Int,
        rssiDeviceCount: Int,
        historyCount: Int,
    ): JSONArray {
        val canReadPaired = permissionStatus.optBoolean("can_read_paired_devices", false)
        val canScanNearby = permissionStatus.optBoolean("can_scan_nearby_devices", false)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route nearby Bluetooth scan",
                    ready = canScanNearby,
                    valueLabel = "bluetooth_scan",
                    detail = "Use when the user asks what Bluetooth or BLE devices are nearby and a fresh RSSI/advertisement sample is useful.",
                    recommendation = "Set refresh=true only for an explicit fresh nearby scan; otherwise reuse passive report rows.",
                    fraction = if (canScanNearby) 0.95f else 0.45f,
                    extra = JSONObject().put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route paired-device inventory",
                    ready = canReadPaired,
                    valueLabel = "bluetooth_scan",
                    detail = "Use for bonded headsets, watches, controllers, keyboards, cars, and other devices Android already knows.",
                    recommendation = "Ask for Bluetooth connect access before promising paired-device names on Android 12+.",
                    fraction = if (canReadPaired) 0.9f else 0.4f,
                    extra = JSONObject().put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route proximity explanation",
                    ready = rssiDeviceCount > 0 || canScanNearby,
                    valueLabel = "bluetooth_scan",
                    detail = "Use when the answer depends on RSSI, proximity labels, TX power distance estimates, or strongest-nearby ordering.",
                    recommendation = "Explain RSSI as a rough proximity indicator, not a precise location fix.",
                    fraction = if (rssiDeviceCount > 0) 1f else if (canScanNearby) 0.75f else 0.35f,
                    extra = JSONObject().put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route Bluetooth signal history",
                    ready = historyCount > 0 || canScanNearby,
                    valueLabel = "bluetooth_signal_history",
                    detail = "Use for RSSI trend, average, min/max, last-seen, and approaching/fading/stable comparisons across recent BLE scan observations.",
                    recommendation = "Run bluetooth_scan refresh=true first when the history is empty and the user explicitly needs a fresh sample.",
                    fraction = if (historyCount > 0) 0.9f else if (canScanNearby) 0.7f else 0.35f,
                    extra = JSONObject().put("tool_action", "bluetooth_signal_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route pause or resume BLE scan mode",
                    ready = true,
                    valueLabel = "scan_mode",
                    detail = "Use scan_mode=paused with bluetooth_scan or bluetooth_signal_history to keep analysis cached; use scan_mode=resumed to request a fresh direct BLE sample.",
                    recommendation = "Prefer paused mode for repeated card review and resumed mode for explicit fresh-scan requests.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route service/manufacturer semantics",
                    ready = serviceUuidCount > 0 || manufacturerIdCount > 0 || canScanNearby || deviceCount > 0,
                    valueLabel = "bluetooth_scan",
                    detail = "Use for service UUID, service data UUID, manufacturer ID, payload byte count, class, category, and connectability rows.",
                    recommendation = "Prefer metadata rows before naming a device role; note sparse advertisements when service labels or manufacturer names are absent.",
                    fraction = if (serviceUuidCount > 0 || manufacturerIdCount > 0) 1f else if (canScanNearby || deviceCount > 0) 0.7f else 0.35f,
                    extra = JSONObject().put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_analyzer_route",
                    label = "Route scan policy explanation",
                    ready = true,
                    valueLabel = "bluetooth_analyzer_report",
                    detail = "Use before repeated scans so the agent can explain Bluetooth enablement, permissions, legacy location gates, and scan cadence.",
                    recommendation = "Prefer this report for permission or policy questions instead of firing a fresh BLE scan.",
                    fraction = 0.85f,
                    extra = JSONObject().put("tool_action", "bluetooth_analyzer_report"),
                ),
            )
    }

    private fun bluetoothScanPolicyRows(
        bluetoothAvailable: Boolean,
        bluetoothLeSupported: Boolean,
        bluetoothEnabled: Boolean,
        permissionStatus: JSONObject,
        scanStatus: JSONObject,
        scanSucceeded: Boolean,
    ): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Bluetooth service availability",
                    ready = bluetoothAvailable,
                    valueLabel = if (bluetoothAvailable) "service present" else "no Bluetooth service",
                    detail = "Android must expose a Bluetooth adapter or service before Hermes can read paired or nearby Bluetooth metadata.",
                    recommendation = "Report missing Bluetooth hardware or service honestly instead of inventing nearby device rows.",
                    fraction = if (bluetoothAvailable) 1f else 0.1f,
                    extra = JSONObject().put("constraint_type", "hardware_service"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Bluetooth enabled state",
                    ready = bluetoothEnabled,
                    valueLabel = if (bluetoothEnabled) "enabled" else "disabled or unreadable",
                    detail = "Nearby BLE scans require Bluetooth to be enabled; paired metadata can also be limited when Android blocks adapter state reads.",
                    recommendation = "Ask the user to enable Bluetooth before promising a fresh nearby scan.",
                    fraction = if (bluetoothEnabled) 0.95f else 0.4f,
                    extra = JSONObject().put("constraint_type", "adapter_state"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Connect and scan permissions",
                    ready = permissionStatus.optBoolean("can_read_paired_devices", false) || permissionStatus.optBoolean("can_scan_nearby_devices", false),
                    valueLabel = when {
                        permissionStatus.optBoolean("can_read_paired_devices", false) && permissionStatus.optBoolean("can_scan_nearby_devices", false) -> "connect and scan granted"
                        permissionStatus.optBoolean("can_read_paired_devices", false) -> "paired only"
                        permissionStatus.optBoolean("can_scan_nearby_devices", false) -> "scan only"
                        else -> "permission gated"
                    },
                    detail = "Android 12+ separates BLUETOOTH_CONNECT for paired devices from BLUETOOTH_SCAN for nearby BLE scans.",
                    recommendation = "Use app settings when the user wants Bluetooth analysis and connect/scan permissions are missing.",
                    fraction = if (permissionStatus.optBoolean("can_read_paired_devices", false) && permissionStatus.optBoolean("can_scan_nearby_devices", false)) 0.95f else if (permissionStatus.optBoolean("can_read_paired_devices", false) || permissionStatus.optBoolean("can_scan_nearby_devices", false)) 0.7f else 0.35f,
                    extra = JSONObject()
                        .put("constraint_type", "permission")
                        .put("permission_gate", "Bluetooth connect and scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Legacy location gate",
                    ready = !permissionStatus.optBoolean("requires_location_for_legacy_scan", false),
                    valueLabel = if (permissionStatus.optBoolean("requires_location_for_legacy_scan", false)) "location needed" else "not blocking",
                    detail = "Android versions before 12 can require fine location and enabled location services before BLE scan results are exposed.",
                    recommendation = "Explain legacy location requirements when scan rows are missing on older devices.",
                    fraction = if (permissionStatus.optBoolean("requires_location_for_legacy_scan", false)) 0.45f else 0.85f,
                    extra = JSONObject()
                        .put("constraint_type", "permission")
                        .put("permission_gate", "fine location and location services"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Active scan cadence",
                    ready = true,
                    valueLabel = if (scanSucceeded) "passive context reused" else "no active scan",
                    detail = "bluetooth_analyzer_report does not force a BLE refresh; it reuses paired/passive rows and names scan availability before any active sampling.",
                    recommendation = "Use timeout_ms and refresh=true only on bluetooth_scan when an active nearby sample is warranted.",
                    fraction = 0.85f,
                    extra = JSONObject()
                        .put("constraint_type", "scan_cadence")
                        .put("scan_error", scanStatus.opt("scan_error") ?: JSONObject.NULL),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Pause/resume BLE scan mode",
                    ready = true,
                    valueLabel = scanStatus.optJSONObject("bluetooth_scan_control")?.optString("scan_mode").orEmpty().ifBlank { BLUETOOTH_SCAN_MODE_AUTO },
                    detail = "scan_mode=paused suppresses active BluetoothLeScanner work even when refresh=true, while scan_mode=resumed requests a fresh BLE sample on direct Bluetooth actions.",
                    recommendation = "Use paused mode for cached Bluetooth dashboards and resumed mode for explicit fresh nearby-device scans.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("constraint_type", "agent_scan_control")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "bluetooth_scan_policy",
                    label = "Analysis and privacy boundary",
                    ready = true,
                    valueLabel = if (bluetoothLeSupported) "metadata only" else "no BLE metadata",
                    detail = "Hermes analyzes Android-exposed paired and advertisement metadata; it does not connect, pair, track identities, or bypass the OS permission model.",
                    recommendation = "Keep Bluetooth answers scoped to observable metadata and user-authorized troubleshooting.",
                    fraction = 1f,
                    extra = JSONObject().put("constraint_type", "privacy_safety"),
                ),
            )
    }

    private fun sensorAnalyzerFeatureRows(
        sensorServiceAvailable: Boolean,
        availableSensors: List<String>,
        capabilityCount: Int,
        motionSensorCount: Int,
        ambientSensorCount: Int,
        wakeUpSensorCount: Int,
        directChannelSensorCount: Int,
        sampledSensorCount: Int,
        motionHistoryCount: Int,
        motionPoseEstimateCount: Int,
    ): JSONArray {
        val hasAccelerometer = "accelerometer" in availableSensors
        val hasGyroscope = "gyroscope" in availableSensors
        val hasRotationContext = "rotation_vector" in availableSensors ||
            (hasGyroscope && "magnetic_field" in availableSensors)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Motion and orientation sensors",
                    ready = sensorServiceAvailable && motionSensorCount > 0,
                    valueLabel = if (motionSensorCount > 0) "$motionSensorCount motion type(s)" else "no motion rows",
                    detail = "Hermes can expose accelerometer, gyroscope, gravity, linear-acceleration, and rotation-vector metadata when Android reports those sensors.",
                    recommendation = "Use sensor_snapshot for the exact motion types needed by the workflow instead of sampling every sensor.",
                    fraction = if (motionSensorCount > 0) (motionSensorCount / MOTION_SENSOR_TYPES.size.toFloat()).coerceIn(0.35f, 1f) else 0.2f,
                    extra = JSONObject()
                        .put("feature_source", "Android SensorManager")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Accelerometer access",
                    ready = hasAccelerometer,
                    valueLabel = if (hasAccelerometer) "accelerometer ready" else "not reported",
                    detail = "Accelerometer rows support movement, shake, tilt, posture-change, and device-handling context for local workflows.",
                    recommendation = "Ask for sensor_snapshot sensor_types=accelerometer when the user needs current acceleration values.",
                    fraction = if (hasAccelerometer) 0.95f else 0.25f,
                    extra = JSONObject()
                        .put("feature_source", "Sensor.TYPE_ACCELEROMETER")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Gyroscope access",
                    ready = hasGyroscope,
                    valueLabel = if (hasGyroscope) "gyroscope ready" else "not reported",
                    detail = "Gyroscope rows expose angular velocity for rotation-aware automation, stabilization, and device-orientation reasoning.",
                    recommendation = "Use gyroscope together with accelerometer or rotation-vector rows before making orientation claims.",
                    fraction = if (hasGyroscope) 0.95f else 0.25f,
                    extra = JSONObject()
                        .put("feature_source", "Sensor.TYPE_GYROSCOPE")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Rotation and heading context",
                    ready = hasRotationContext,
                    valueLabel = when {
                        "rotation_vector" in availableSensors -> "rotation vector ready"
                        hasRotationContext -> "gyro plus magnetic"
                        else -> "orientation gated"
                    },
                    detail = "Rotation-vector, gyroscope, and magnetic-field rows help Gemma reason about orientation and heading without camera-only assumptions.",
                    recommendation = "Prefer rotation_vector when available; otherwise combine gyroscope and magnetic_field with clear uncertainty.",
                    fraction = if (hasRotationContext) 0.9f else 0.3f,
                    extra = JSONObject()
                        .put("feature_source", "Android orientation sensors")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Motion trend history graph",
                    ready = motionHistoryCount > 0,
                    valueLabel = "$motionHistoryCount trend row(s)",
                    detail = "Motion history rows preserve current, average, range, trend, stability, current vector, and recent magnitude series for IMU-style context across bounded samples.",
                    recommendation = "Use motion_sensor_history when comparing movement or orientation changes across recent samples instead of relying on one point in time.",
                    fraction = if (motionHistoryCount > 0) (motionHistoryCount / MOTION_HISTORY_SENSOR_TYPES.size.toFloat()).coerceIn(0.35f, 1f) else 0.3f,
                    extra = JSONObject()
                        .put("feature_source", "Hermes motion sensor history")
                        .put("tool_action", "motion_sensor_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Motion pose fusion rows",
                    ready = motionPoseEstimateCount > 0 || hasAccelerometer || "rotation_vector" in availableSensors,
                    valueLabel = if (motionPoseEstimateCount > 0) "$motionPoseEstimateCount pose row(s)" else "sample needed",
                    detail = "Pose rows fuse accelerometer/gravity, magnetic-field, rotation-vector, gyroscope, and linear-acceleration values into roll, pitch, tilt, heading, angular-motion, and movement-state context.",
                    recommendation = "Use motion_pose or sensor_snapshot when the workflow needs orientation claims rather than raw vector magnitudes.",
                    fraction = if (motionPoseEstimateCount > 0) 0.95f else if (hasRotationContext || hasAccelerometer) 0.65f else 0.25f,
                    extra = JSONObject()
                        .put("feature_source", "Hermes IMU fusion")
                        .put("tool_action", "motion_pose"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Environmental and proximity context",
                    ready = ambientSensorCount > 0,
                    valueLabel = if (ambientSensorCount > 0) "$ambientSensorCount ambient type(s)" else "none reported",
                    detail = "Light, proximity, pressure, temperature, humidity, and magnetic rows can add local context to user workflows when hardware exists.",
                    recommendation = "Use ambient rows as context signals and tell the user when the device does not expose those sensors.",
                    fraction = if (ambientSensorCount > 0) (ambientSensorCount / AMBIENT_SENSOR_TYPES.size.toFloat()).coerceIn(0.3f, 1f) else 0.2f,
                    extra = JSONObject()
                        .put("feature_source", "Android ambient sensors")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Sensor hardware metadata",
                    ready = capabilityCount > 0,
                    valueLabel = "$capabilityCount capability row(s)",
                    detail = "Sensor Analyzer reports preserve vendor, range, resolution, power, min/max delay, FIFO, wake-up, direct-channel, and reporting-mode metadata.",
                    recommendation = "Use hardware metadata before choosing polling cadence, battery strategy, or workflow thresholds.",
                    fraction = if (capabilityCount > 0) 0.9f else 0.25f,
                    extra = JSONObject()
                        .put("feature_source", "Sensor metadata")
                        .put("tool_action", "sensor_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Wake-up and direct-channel hints",
                    ready = wakeUpSensorCount > 0 || directChannelSensorCount > 0,
                    valueLabel = "$wakeUpSensorCount wake-up, $directChannelSensorCount direct",
                    detail = "Wake-up and direct-channel fields help the agent explain power and low-latency limits without relying on a specific SOC vendor.",
                    recommendation = "Treat these as hardware hints; normal Hermes sampling still stays in the Android app permission model.",
                    fraction = if (wakeUpSensorCount > 0 || directChannelSensorCount > 0) 0.85f else 0.45f,
                    extra = JSONObject()
                        .put("feature_source", "Sensor hardware flags")
                        .put("tool_action", "sensor_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Sensor watcher automation route",
                    ready = availableSensors.isNotEmpty(),
                    valueLabel = if (availableSensors.isNotEmpty()) "watcher route ready" else "no sensor types",
                    detail = "Hermes automation can route explicit sensor events through saved watcher records for motion-aware workflows.",
                    recommendation = "Use start_sensor_watcher only for intentional workflows and keep one-shot sensor_snapshot as the default diagnostic path.",
                    fraction = if (availableSensors.isNotEmpty()) 0.8f else 0.25f,
                    extra = JSONObject()
                        .put("feature_source", "Hermes automation watcher")
                        .put("tool_action", "start_sensor_watcher"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Gemma-readable sensor cards",
                    ready = true,
                    valueLabel = "${capabilityCount + sampledSensorCount} card row(s)",
                    detail = "Sensor analyzer reports create top-level expandable cards for readiness, routes, sampling policy, and hardware rows Gemma can compactly inspect.",
                    recommendation = "Show sensor cards before long explanations when the user asks what motion or ambient context the agent can see.",
                    fraction = 0.9f,
                    extra = JSONObject()
                        .put("feature_source", "Hermes diagnostic cards")
                        .put("tool_action", "sensor_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_parity",
                    label = "Sensor privacy and power boundary",
                    ready = true,
                    valueLabel = "bounded sampling",
                    detail = "Hermes reports Android-exposed sensor metadata and bounded one-shot samples; it does not infer hidden location or run endless background polling by default.",
                    recommendation = "Keep sensor answers scoped to user-authorized local context, sensor availability, and explicit automation records.",
                    fraction = 1f,
                    extra = JSONObject().put("feature_source", "Sensor safety boundary"),
                ),
            )
    }

    private fun sensorAnalyzerWorkflowRows(
        availableSensors: List<String>,
        motionSensorCount: Int,
        ambientSensorCount: Int,
        capabilityCount: Int,
        motionHistoryCount: Int,
        motionPoseEstimateCount: Int,
    ): JSONArray {
        val hasAccelerometer = "accelerometer" in availableSensors
        val hasGyroscope = "gyroscope" in availableSensors
        val hasRotationContext = "rotation_vector" in availableSensors ||
            (hasGyroscope && "magnetic_field" in availableSensors)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route one-shot motion sample",
                    ready = motionSensorCount > 0,
                    valueLabel = "sensor_snapshot",
                    detail = "Use for current accelerometer, gyroscope, gravity, linear acceleration, or rotation-vector values.",
                    recommendation = "Pass only the needed sensor_types and a bounded timeout_ms.",
                    fraction = if (motionSensorCount > 0) 0.95f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "sensor_snapshot")
                        .put("sensor_types", "accelerometer,gyroscope,linear_acceleration,rotation_vector"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route motion trend history",
                    ready = motionSensorCount > 0 || motionHistoryCount > 0,
                    valueLabel = "motion_sensor_history",
                    detail = "Use for cached accelerometer, gyroscope, linear-acceleration, rotation-vector, magnetic-field magnitude, trend, stability, and vector rows.",
                    recommendation = "Set sample=true only when the user needs a fresh bounded reading; otherwise reuse cached motion history.",
                    fraction = if (motionHistoryCount > 0) 0.9f else if (motionSensorCount > 0) 0.65f else 0.3f,
                    extra = JSONObject()
                        .put("tool_action", "motion_sensor_history")
                        .put("sensor_types", "accelerometer,gyroscope,linear_acceleration,rotation_vector,magnetic_field"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route motion pose fusion",
                    ready = hasAccelerometer || hasRotationContext || motionPoseEstimateCount > 0,
                    valueLabel = "motion_pose",
                    detail = "Use for fused roll, pitch, tilt, heading, angular velocity, and acceleration-state rows that top cards and Gemma can read directly.",
                    recommendation = "Prefer rotation_vector for high-confidence pose; otherwise combine accelerometer/gravity with magnetic_field and expose confidence.",
                    fraction = if (motionPoseEstimateCount > 0) 0.95f else if (hasRotationContext) 0.85f else if (hasAccelerometer) 0.65f else 0.3f,
                    extra = JSONObject()
                        .put("tool_action", "motion_pose")
                        .put("sensor_types", "accelerometer,gravity,magnetic_field,rotation_vector,gyroscope,linear_acceleration"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route accelerometer workflow",
                    ready = hasAccelerometer,
                    valueLabel = "accelerometer",
                    detail = "Use when the task needs movement, shake, phone posture, pocket handling, or coarse activity context.",
                    recommendation = "Sample accelerometer before building motion-triggered automation thresholds.",
                    fraction = if (hasAccelerometer) 0.9f else 0.3f,
                    extra = JSONObject()
                        .put("tool_action", "sensor_snapshot")
                        .put("sensor_types", "accelerometer"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route orientation workflow",
                    ready = hasGyroscope || hasRotationContext,
                    valueLabel = if (hasRotationContext) "rotation context" else "gyroscope",
                    detail = "Use when device orientation, angular velocity, heading, or stabilization context matters.",
                    recommendation = "Prefer rotation_vector, then gyroscope plus magnetic_field when available.",
                    fraction = if (hasRotationContext) 0.9f else if (hasGyroscope) 0.75f else 0.3f,
                    extra = JSONObject()
                        .put("tool_action", "sensor_snapshot")
                        .put("sensor_types", "gyroscope,rotation_vector,magnetic_field"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route ambient context workflow",
                    ready = ambientSensorCount > 0,
                    valueLabel = if (ambientSensorCount > 0) "ambient sensors" else "not reported",
                    detail = "Use for light, proximity, magnetic, pressure, temperature, or humidity context before environment-sensitive workflows.",
                    recommendation = "Explain missing hardware directly because many phones only expose a subset of ambient sensors.",
                    fraction = if (ambientSensorCount > 0) 0.85f else 0.3f,
                    extra = JSONObject()
                        .put("tool_action", "sensor_snapshot")
                        .put("sensor_types", "light,proximity,pressure,ambient_temperature,relative_humidity,magnetic_field"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route sensor hardware metadata",
                    ready = capabilityCount > 0,
                    valueLabel = "sensor_analyzer_report",
                    detail = "Use for vendor, range, resolution, power, FIFO, wake-up, direct-channel, and sampling-rate metadata without forcing a live sample.",
                    recommendation = "Prefer this passive report for planning and policy answers.",
                    fraction = if (capabilityCount > 0) 0.85f else 0.3f,
                    extra = JSONObject().put("tool_action", "sensor_analyzer_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route sensor watcher automation",
                    ready = availableSensors.isNotEmpty(),
                    valueLabel = "start_sensor_watcher",
                    detail = "Use for explicit, saved Hermes automations that react to sensor events instead of ad hoc chat diagnostics.",
                    recommendation = "Create watcher records deliberately and expose thresholds clearly to the user.",
                    fraction = if (availableSensors.isNotEmpty()) 0.75f else 0.25f,
                    extra = JSONObject().put("tool_action", "start_sensor_watcher"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_analyzer_route",
                    label = "Route sampling policy explanation",
                    ready = true,
                    valueLabel = "sensor_analyzer_report",
                    detail = "Use before repeated sampling so the agent can explain timeout, power, hardware, privacy, and background constraints.",
                    recommendation = "Keep sensor_analyzer_report passive and call sensor_snapshot only for a needed current reading.",
                    fraction = 0.85f,
                    extra = JSONObject().put("tool_action", "sensor_analyzer_report"),
                ),
            )
    }

    private fun sensorSamplingPolicyRows(
        sensorServiceAvailable: Boolean,
        requestedSensorCount: Int,
        requestedAvailableCount: Int,
        wakeUpSensorCount: Int,
        directChannelSensorCount: Int,
        activeSampleRequested: Boolean,
        timeoutMs: Long,
    ): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "sensor_sampling_policy",
                    label = "Sensor service availability",
                    ready = sensorServiceAvailable,
                    valueLabel = if (sensorServiceAvailable) "service present" else "no SensorManager",
                    detail = "Android must expose SensorManager before Hermes can read motion, orientation, ambient, or hardware metadata rows.",
                    recommendation = "Report missing sensor service honestly instead of inventing motion or ambient rows.",
                    fraction = if (sensorServiceAvailable) 1f else 0.1f,
                    extra = JSONObject().put("constraint_type", "hardware_service"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_sampling_policy",
                    label = "Requested sensor availability",
                    ready = requestedAvailableCount > 0,
                    valueLabel = "$requestedAvailableCount/$requestedSensorCount requested",
                    detail = "Sensor Analyzer compares requested accelerometer, gyroscope, orientation, and ambient types with the Android sensor catalog.",
                    recommendation = "Ask for a smaller sensor_types list when only one workflow signal is needed.",
                    fraction = if (requestedSensorCount > 0) (requestedAvailableCount / requestedSensorCount.toFloat()).coerceIn(0.1f, 1f) else 0.1f,
                    extra = JSONObject().put("constraint_type", "hardware_inventory"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_sampling_policy",
                    label = "Passive report default",
                    ready = true,
                    valueLabel = if (activeSampleRequested) "sample included" else "no live sample",
                    detail = "sensor_analyzer_report stays passive unless include_snapshot or sample is explicitly requested.",
                    recommendation = "Use sensor_snapshot for a current reading and sensor_analyzer_report for planning, readiness, and policy cards.",
                    fraction = if (activeSampleRequested) 0.8f else 0.9f,
                    extra = JSONObject().put("constraint_type", "sampling_cadence"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_sampling_policy",
                    label = "Bounded one-shot timeout",
                    ready = true,
                    valueLabel = "${timeoutMs}ms timeout",
                    detail = "One-shot sensor sampling clamps timeout_ms between 150ms and ${MAX_SENSOR_TIMEOUT_MS}ms to avoid hanging chat turns.",
                    recommendation = "Use short timeouts for chat diagnostics and saved watcher automation for ongoing sensor workflows.",
                    fraction = 0.85f,
                    extra = JSONObject().put("constraint_type", "latency_power"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_sampling_policy",
                    label = "Power and low-latency hints",
                    ready = true,
                    valueLabel = "$wakeUpSensorCount wake-up, $directChannelSensorCount direct",
                    detail = "Power, FIFO, wake-up, min/max delay, reporting mode, and direct-channel fields are metadata hints, not permission to bypass Android policy.",
                    recommendation = "Use these fields when choosing thresholds or explaining battery impact.",
                    fraction = if (wakeUpSensorCount > 0 || directChannelSensorCount > 0) 0.85f else 0.65f,
                    extra = JSONObject().put("constraint_type", "power_policy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "sensor_sampling_policy",
                    label = "Analysis and privacy boundary",
                    ready = true,
                    valueLabel = "local metadata",
                    detail = "Hermes uses local Android sensor metadata and bounded readings; it does not infer hidden location, identify people, or poll forever by default.",
                    recommendation = "Keep sensor explanations scoped to observable local context and explicit user workflows.",
                    fraction = 1f,
                    extra = JSONObject().put("constraint_type", "privacy_safety"),
                ),
            )
    }

    private fun sensorSamplingStatusJson(
        sensorServiceAvailable: Boolean,
        requestedSensorCount: Int,
        availableSensorCount: Int,
        requestedAvailableCount: Int,
        sampledSensorCount: Int,
        activeSampleRequested: Boolean,
        timeoutMs: Long,
    ): JSONObject {
        return JSONObject()
            .put("sensor_service_available", sensorServiceAvailable)
            .put("requested_sensor_count", requestedSensorCount)
            .put("available_sensor_count", availableSensorCount)
            .put("requested_available_sensor_count", requestedAvailableCount)
            .put("active_sample_requested", activeSampleRequested)
            .put("sampled_sensor_count", sampledSensorCount)
            .put("sample_timeout_ms", timeoutMs)
            .put("default_timeout_ms", DEFAULT_SENSOR_TIMEOUT_MS)
            .put("max_timeout_ms", MAX_SENSOR_TIMEOUT_MS)
            .put("passive_report_default", true)
            .put("android_sensor_policy_note", "Normal Android sensors can be read through SensorManager, but Hermes keeps analyzer reports passive and bounds one-shot sampling to reduce latency and power use.")
    }

    private fun signalWorkflowRouteRows(
        diagnostics: JSONObject,
        signalStatus: JSONObject,
        preferredModel: JSONObject,
        availableSensors: JSONArray,
        cachedWifiHistory: JSONArray,
        cachedBluetoothHistory: JSONArray,
        cachedMotionHistory: JSONArray,
        cachedMotionPoseEstimates: JSONArray,
        backendRiskReport: JSONObject,
    ): JSONArray {
        val wifiPermission = diagnostics.optJSONObject("wifi_scan_permission_status") ?: JSONObject()
        val bluetoothPermission = diagnostics.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject()
        val sensorNames = jsonStringList(availableSensors)
        val backendRiskLevel = backendRiskReport.optString("gpu_backend_risk_level").ifBlank { "unknown" }
        val backendRiskScore = backendRiskReport.optInt("gpu_backend_risk_score", 0)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route Wi-Fi analyzer work",
                    ready = wifiPermission.optBoolean("can_read_scan_results", false),
                    valueLabel = "wifi_ap_details",
                    detail = "Use for live AP details, security/standard/width summaries, vendor/OUI lookup, channel rating, distance estimate, and export rows.",
                    recommendation = "Ask for location/nearby Wi-Fi permission when this row is not ready.",
                    fraction = if (wifiPermission.optBoolean("can_read_scan_results", false)) 1f else 0.45f,
                    extra = JSONObject().put("tool_action", "wifi_ap_details"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route Wi-Fi trend work",
                    ready = cachedWifiHistory.length() > 0,
                    valueLabel = "wifi_scan",
                    detail = "Use cached signal history rows when comparing changing RSSI, trend, last seen, and AP stability across scans.",
                    recommendation = "Refresh scans sparingly because Android may throttle active Wi-Fi scanning.",
                    fraction = if (cachedWifiHistory.length() > 0) 0.85f else 0.35f,
                    extra = JSONObject().put("tool_action", "wifi_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route Bluetooth proximity work",
                    ready = bluetoothPermission.optBoolean("can_scan_nearby_devices", false) || bluetoothPermission.optBoolean("can_read_paired_devices", false),
                    valueLabel = "bluetooth_scan",
                    detail = "Use for BLE/paired-device rows, RSSI proximity, service UUID, manufacturer ID, and category metadata.",
                    recommendation = "Ask for Bluetooth scan/connect permissions when nearby scan rows are not available.",
                    fraction = if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) 1f else if (bluetoothPermission.optBoolean("can_read_paired_devices", false)) 0.75f else 0.45f,
                    extra = JSONObject().put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route Bluetooth trend work",
                    ready = cachedBluetoothHistory.length() > 0 || bluetoothPermission.optBoolean("can_scan_nearby_devices", false),
                    valueLabel = "bluetooth_signal_history",
                    detail = "Use cached Bluetooth signal history rows when comparing changing RSSI, proximity, trend, last seen, and nearby-device stability across scans.",
                    recommendation = "Refresh BLE scans only when the user needs a fresh sample because scan permission and battery policy can gate repeated reads.",
                    fraction = if (cachedBluetoothHistory.length() > 0) 0.85f else if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) 0.7f else 0.35f,
                    extra = JSONObject().put("tool_action", "bluetooth_signal_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route motion trend work",
                    ready = sensorNames.any { it in MOTION_HISTORY_SENSOR_TYPES } || cachedMotionHistory.length() > 0,
                    valueLabel = "motion_sensor_history",
                    detail = "Use for cached accelerometer, gyroscope, linear-acceleration, rotation, magnetic magnitude, stability, and trend rows across recent samples.",
                    recommendation = "Use sample=true only when a fresh bounded IMU reading is needed for the current workflow.",
                    fraction = if (cachedMotionHistory.length() > 0) 0.85f else if (sensorNames.any { it in MOTION_HISTORY_SENSOR_TYPES }) 0.65f else 0.3f,
                    extra = JSONObject().put("tool_action", "motion_sensor_history"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route motion pose work",
                    ready = sensorNames.any { it in setOf("accelerometer", "gravity", "magnetic_field", "rotation_vector", "gyroscope", "linear_acceleration") } || cachedMotionPoseEstimates.length() > 0,
                    valueLabel = "motion_pose",
                    detail = "Use fused pose rows when Gemma needs roll, pitch, tilt, heading, angular velocity, or movement state rather than raw IMU vectors alone.",
                    recommendation = "Prefer rotation_vector when available; otherwise combine accelerometer/gravity with magnetic_field and expose confidence.",
                    fraction = if (cachedMotionPoseEstimates.length() > 0) 0.9f else if (sensorNames.any { it in MOTION_HISTORY_SENSOR_TYPES }) 0.65f else 0.3f,
                    extra = JSONObject().put("tool_action", "motion_pose"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route sensor context work",
                    ready = sensorNames.isNotEmpty(),
                    valueLabel = "sensor_snapshot",
                    detail = "Use for accelerometer, gyroscope, magnetic, ambient, proximity, pressure, humidity, temperature, and sensor hardware metadata.",
                    recommendation = "Sample only the sensor types that matter to the workflow to reduce latency and power use.",
                    fraction = (sensorNames.size / SENSOR_TYPE_LABELS.size.toFloat()).coerceIn(0.15f, 1f),
                    extra = JSONObject().put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route broad RF explanation",
                    ready = true,
                    valueLabel = "signal_capability_status",
                    detail = "Use for AM/FM, broad RF, microwave, Wi-Fi, Bluetooth, audio, camera, and sensor capability boundaries.",
                    recommendation = "Tell the user when public Android APIs cannot scan a requested RF band and name the needed external hardware/API.",
                    fraction = 0.75f,
                    extra = JSONObject().put("tool_action", "signal_capability_status"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route backend risk triage",
                    ready = backendRiskReport.optBoolean("success", false),
                    valueLabel = "gpu_backend_risk_report",
                    detail = "Use for MediaTek, Mali, PowerVR, Xclipse, non-Adreno, thermal, memory, power, model artifact, and CPU fallback risk context.",
                    recommendation = "Run before local inference policy changes or before promising GPU acceleration on this device; current passive level=$backendRiskLevel score=$backendRiskScore.",
                    fraction = if (backendRiskReport.optBoolean("success", false)) ((100 - backendRiskScore).coerceIn(5, 100) / 100f) else 0.35f,
                    extra = JSONObject().put("tool_action", "gpu_backend_risk_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_route",
                    label = "Route local reasoning",
                    ready = preferredModel.optBoolean("ready"),
                    valueLabel = if (preferredModel.optBoolean("ready")) "local model" else "import model",
                    detail = "Use this report before wireless or sensor-heavy reasoning so the local agent has SOC, permission, and signal limits in context.",
                    recommendation = "Run signal_awareness_report first for broad situational context, then call the narrow scanner.",
                    fraction = if (preferredModel.optBoolean("ready")) 1f else 0.4f,
                    extra = JSONObject().put("tool_action", "signal_awareness_report"),
                ),
            )
    }

    private fun signalConstraintRows(diagnostics: JSONObject, signalStatus: JSONObject, radioStatus: JSONObject): JSONArray {
        val wifiPermission = diagnostics.optJSONObject("wifi_scan_permission_status") ?: JSONObject()
        val bluetoothPermission = diagnostics.optJSONObject("bluetooth_scan_permission_status") ?: JSONObject()
        return JSONArray()
            .put(
                capabilityRow(
                    category = "signal_constraint",
                    label = "Wi-Fi scan permission and throttling",
                    ready = wifiPermission.optBoolean("can_read_scan_results", false),
                    valueLabel = if (wifiPermission.optBoolean("can_read_scan_results", false)) "readable" else "permission needed",
                    detail = "Nearby Wi-Fi/location permissions and enabled location services can gate scan reads; Android may throttle active refreshes.",
                    recommendation = "Prefer cached scan age/history when Android denies or throttles an active refresh.",
                    fraction = if (wifiPermission.optBoolean("can_read_scan_results", false)) 0.9f else 0.45f,
                    extra = JSONObject().put("constraint_type", "permission"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_constraint",
                    label = "Bluetooth scan permission",
                    ready = bluetoothPermission.optBoolean("can_scan_nearby_devices", false),
                    valueLabel = if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) "scan readable" else "permission needed",
                    detail = "Android 12+ can require BLUETOOTH_SCAN/CONNECT and may expose paired metadata separately from live BLE scan results.",
                    recommendation = "Use paired-device metadata when scan permission is missing; request scan access for proximity and advertisement context.",
                    fraction = if (bluetoothPermission.optBoolean("can_scan_nearby_devices", false)) 0.9f else 0.45f,
                    extra = JSONObject().put("constraint_type", "permission"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_constraint",
                    label = "AM/FM tuner public API",
                    ready = radioStatus.optBoolean("am_fm_public_android_scan_supported", false),
                    valueLabel = if (radioStatus.optBoolean("am_fm_public_android_scan_supported", false)) "vendor exposed" else "not public",
                    detail = "Normal Android apps cannot read AM/FM tuner scan results through public APIs even when a vendor declares radio features.",
                    recommendation = "Use external SDR or vendor-specific bridges for actual AM/FM station scans.",
                    fraction = if (radioStatus.optBoolean("am_fm_public_android_scan_supported", false)) 0.8f else 0.25f,
                    extra = JSONObject().put("constraint_type", "hardware_api"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_constraint",
                    label = "Broad RF and microwave hardware",
                    ready = !signalStatus.optBoolean("requires_external_sdr_for_broad_rf", true),
                    valueLabel = if (signalStatus.optBoolean("requires_external_sdr_for_broad_rf", true)) "external hardware" else "built-in",
                    detail = "Public Android phone APIs do not expose a general-purpose RF or microwave spectrum analyzer.",
                    recommendation = "Attach an SDR or specialized vendor radio bridge before claiming broad RF or microwave scan support.",
                    fraction = if (signalStatus.optBoolean("requires_external_sdr_for_broad_rf", true)) 0.25f else 0.8f,
                    extra = JSONObject().put("constraint_type", "hardware"),
                ),
            )
            .put(
                capabilityRow(
                    category = "signal_constraint",
                    label = "Sensor sampling latency",
                    ready = true,
                    valueLabel = "bounded sample",
                    detail = "One-shot sensor snapshots use a bounded timeout; missing samples can mean no default sensor, sensor delay, or hardware reporting limits.",
                    recommendation = "Set timeout_ms and sensor_types narrowly when the workflow needs faster response.",
                    fraction = 0.75f,
                    extra = JSONObject().put("constraint_type", "runtime"),
                ),
            )
    }

    private fun radioBandPlanRows(
        vendorBroadcastRadioDeclared: Boolean,
        wifiSupported: Boolean,
        bluetoothSupported: Boolean,
    ): JSONArray = JSONArray()
        .put(
            radioBandRow(
                band = "AM broadcast",
                sourceType = "am_broadcast",
                frequencyMinKhz = 530,
                frequencyMaxKhz = 1700,
                channelStepLabel = "9/10 kHz regional channel spacing",
                hardwareHintSupported = vendorBroadcastRadioDeclared,
                accessPath = "OEM Broadcast Radio HAL or external SDR",
                reason = if (vendorBroadcastRadioDeclared) {
                    "A vendor broadcast-radio feature is declared, but Android public app APIs still do not expose AM tuner scan rows."
                } else {
                    "Android public app APIs do not expose AM tuner scan rows on normal phones."
                },
                agentUsage = "Use as an AM band boundary card and route actual station scans to vendor radio or SDR hardware.",
            ),
        )
        .put(
            radioBandRow(
                band = "FM broadcast",
                sourceType = "fm_broadcast",
                frequencyMinMhz = 87.5,
                frequencyMaxMhz = 108.0,
                channelStepLabel = "50/100/200 kHz regional spacing",
                hardwareHintSupported = vendorBroadcastRadioDeclared,
                accessPath = "OEM Broadcast Radio HAL, vendor FM app, or external SDR",
                reason = if (vendorBroadcastRadioDeclared) {
                    "A vendor broadcast-radio feature is declared, but Android public app APIs still do not expose FM tuner/RDS scan rows."
                } else {
                    "Android public app APIs do not expose FM tuner or RDS scan rows on normal phones."
                },
                agentUsage = "Use as an FM band boundary card and explain when external tuner hardware is needed.",
                metadataFields = listOf("frequency", "station", "rssi_if_vendor_exposed", "rds_if_vendor_exposed"),
            ),
        )
        .put(
            radioBandRow(
                band = "Wi-Fi 2.4 GHz",
                sourceType = "wifi_public_scan",
                frequencyMinMhz = 2401.0,
                frequencyMaxMhz = 2484.0,
                channelStepLabel = "20/40 MHz channel-width metadata",
                publicAndroidScanSupported = wifiSupported,
                builtInAndroidSource = wifiSupported,
                accessPath = "wifi_scan, wifi_channel_graph, wifi_channel_rating, wifi_channel_utilization",
                reason = if (wifiSupported) {
                    "Android exposes nearby Wi-Fi RSSI, channel, frequency, width, security, and vendor/OUI metadata when permissions and scan throttling allow it."
                } else {
                    "This device does not declare Wi-Fi support."
                },
                agentUsage = "Route to Wi-Fi Analyzer cards for channel graph, time/history graph, utilization, filters, vendor, and export rows.",
                metadataFields = listOf("ssid", "bssid", "rssi_dbm", "frequency_mhz", "channel", "channel_width", "security", "vendor_oui"),
            ),
        )
        .put(
            radioBandRow(
                band = "Wi-Fi 5/6 GHz",
                sourceType = "wifi_public_scan",
                frequencyMinMhz = 5150.0,
                frequencyMaxMhz = 7125.0,
                channelStepLabel = "20/40/80/160/320 MHz channel-width metadata",
                publicAndroidScanSupported = wifiSupported,
                builtInAndroidSource = wifiSupported,
                accessPath = "wifi_scan, wifi_channel_graph, wifi_channel_rating, wifi_channel_utilization",
                reason = if (wifiSupported) {
                    "Android exposes 5/6 GHz access-point frequency and channel metadata through Wi-Fi scans when available on the device."
                } else {
                    "This device does not declare Wi-Fi support."
                },
                agentUsage = "Route to Wi-Fi channel graph, occupancy, and channel-rating cards before making placement/interference recommendations.",
                metadataFields = listOf("ssid", "bssid", "rssi_dbm", "frequency_mhz", "channel", "channel_width", "standard"),
            ),
        )
        .put(
            radioBandRow(
                band = "Bluetooth 2.4 GHz",
                sourceType = "bluetooth_public_scan",
                frequencyMinMhz = 2402.0,
                frequencyMaxMhz = 2480.0,
                channelStepLabel = "BLE advertising/service metadata, not raw channel sweep",
                publicAndroidScanSupported = bluetoothSupported,
                builtInAndroidSource = bluetoothSupported,
                accessPath = "bluetooth_scan, bluetooth_signal_history",
                reason = if (bluetoothSupported) {
                    "Android exposes nearby Bluetooth/BLE identity, service UUID, manufacturer, RSSI, proximity metadata, and analyzer filters with the right permissions."
                } else {
                    "This device does not declare Bluetooth support."
                },
                agentUsage = "Route to Bluetooth Analyzer cards for filtered device proximity, service UUID labels, manufacturer names, and RSSI trend rows.",
                metadataFields = listOf("name", "address", "rssi_dbm", "service_uuids", "manufacturer_ids", "proximity_label", "applied_bluetooth_filters"),
            ),
        )
        .put(
            radioBandRow(
                band = "External SDR / broad RF",
                sourceType = "external_sdr",
                frequencyMinMhz = 0.5,
                frequencyMaxMhz = 6000.0,
                channelStepLabel = "receiver-dependent",
                requiresExternalHardware = true,
                accessPath = "USB/Bluetooth/Wi-Fi SDR or vendor radio bridge",
                reason = "Attach an SDR or vendor radio bridge for broad RF, AM, FM, airband, weather, or microwave spectrum data.",
                agentUsage = "Do not claim broad RF scan rows until a receiver bridge provides samples; use this row to route setup.",
                metadataFields = listOf("center_frequency", "span", "sample_rate", "power_db", "modulation", "waterfall_if_bridge_exposes_it"),
            ),
        )

    private fun radioBandRow(
        band: String,
        sourceType: String,
        frequencyMinKhz: Int? = null,
        frequencyMaxKhz: Int? = null,
        frequencyMinMhz: Double? = null,
        frequencyMaxMhz: Double? = null,
        channelStepLabel: String,
        publicAndroidScanSupported: Boolean = false,
        builtInAndroidSource: Boolean = false,
        hardwareHintSupported: Boolean = false,
        requiresExternalHardware: Boolean = false,
        accessPath: String,
        reason: String,
        agentUsage: String,
        metadataFields: List<String> = emptyList(),
    ): JSONObject {
        val scanState = when {
            publicAndroidScanSupported -> "public_android_metadata_route"
            hardwareHintSupported -> "vendor_feature_declared_no_public_scan"
            requiresExternalHardware -> "external_receiver_required"
            else -> "not_public_android_api"
        }
        val row = JSONObject()
            .put("band", band)
            .put("source_type", sourceType)
            .put("supported", publicAndroidScanSupported)
            .put("sampled", false)
            .put("public_android_scan_supported", publicAndroidScanSupported)
            .put("built_in_android_source", builtInAndroidSource)
            .put("hardware_hint_supported", hardwareHintSupported)
            .put("requires_external_hardware", requiresExternalHardware)
            .put("data_available", false)
            .put("scan_state", scanState)
            .put("channel_step", channelStepLabel)
            .put("access_path", accessPath)
            .put("reason", reason)
            .put("agent_usage", agentUsage)
            .put("metadata_fields", JSONArray(metadataFields))
        frequencyMinKhz?.let { row.put("frequency_min_khz", it) }
        frequencyMaxKhz?.let { row.put("frequency_max_khz", it) }
        frequencyMinMhz?.let { row.put("frequency_min_mhz", it) }
        frequencyMaxMhz?.let { row.put("frequency_max_mhz", it) }
        return row
    }

    private fun radioReceiverProfileRows(
        vendorBroadcastRadioDeclared: Boolean,
        wifiSupported: Boolean,
        bluetoothSupported: Boolean,
    ): JSONArray = JSONArray()
        .put(
            radioReceiverProfileRow(
                receiverId = "am_vendor_or_sdr",
                label = "AM station receiver profile",
                sourceType = "am_broadcast",
                frequencyMinKhz = 530,
                frequencyMaxKhz = 1700,
                channelStepLabel = "9/10 kHz regional channel spacing",
                publicAndroidScanSupported = false,
                vendorBridgePossible = vendorBroadcastRadioDeclared,
                requiresVendorBridge = true,
                routeAction = "radio_signal_graph",
                accessPath = "OEM Broadcast Radio HAL bridge, vendor app bridge, or external SDR",
                scanState = if (vendorBroadcastRadioDeclared) "vendor_bridge_required" else "external_or_vendor_receiver_required",
                reason = if (vendorBroadcastRadioDeclared) {
                    "Device declares a broadcast-radio feature, but Hermes still needs an OEM/vendor bridge to receive AM station rows."
                } else {
                    "No public Android AM tuner scan API is available; AM station rows require a vendor bridge or SDR receiver."
                },
                recommendation = "Use this profile as the required AM scan schema and avoid returning empty station rows until a bridge reports samples.",
                stationFields = listOf("frequency_khz", "station_label", "signal_dbuv_or_rssi_dbm", "snr_db", "modulation", "scan_timestamp_ms"),
                sampleFields = listOf("frequency_khz", "power_db", "bandwidth_hz", "sample_rate_hz", "receiver_id"),
            ),
        )
        .put(
            radioReceiverProfileRow(
                receiverId = "fm_vendor_or_sdr",
                label = "FM station receiver profile",
                sourceType = "fm_broadcast",
                frequencyMinMhz = 87.5,
                frequencyMaxMhz = 108.0,
                channelStepLabel = "50/100/200 kHz regional spacing",
                publicAndroidScanSupported = false,
                vendorBridgePossible = vendorBroadcastRadioDeclared,
                requiresVendorBridge = true,
                routeAction = "radio_signal_graph",
                accessPath = "OEM Broadcast Radio HAL bridge, vendor FM app bridge, or external SDR",
                scanState = if (vendorBroadcastRadioDeclared) "vendor_bridge_required" else "external_or_vendor_receiver_required",
                reason = if (vendorBroadcastRadioDeclared) {
                    "Device declares a broadcast-radio feature, but Hermes still needs a vendor bridge to read FM tuner, station, or RDS rows."
                } else {
                    "No public Android FM tuner/RDS scan API is available; FM station rows require a vendor bridge or SDR receiver."
                },
                recommendation = "Use this profile as the required FM scan schema and include RDS fields only when the receiver bridge reports them.",
                stationFields = listOf("frequency_mhz", "station_label", "rds_program_service", "rds_radio_text", "signal_dbuv_or_rssi_dbm", "snr_db", "stereo", "scan_timestamp_ms"),
                sampleFields = listOf("frequency_mhz", "power_db", "bandwidth_hz", "sample_rate_hz", "receiver_id"),
            ),
        )
        .put(
            radioReceiverProfileRow(
                receiverId = "wifi_public_metadata",
                label = "Wi-Fi public metadata receiver",
                sourceType = "wifi_public_scan",
                frequencyMinMhz = 2401.0,
                frequencyMaxMhz = 7125.0,
                channelStepLabel = "20/40/80/160/320 MHz channel-width metadata",
                publicAndroidScanSupported = wifiSupported,
                builtInAndroidSource = wifiSupported,
                routeAction = "wifi_analyzer_report",
                accessPath = "wifi_scan, wifi_channel_graph, wifi_channel_rating, wifi_channel_utilization, wifi_signal_history",
                scanState = if (wifiSupported) "public_android_metadata_route" else "no_wifi_feature",
                reason = if (wifiSupported) {
                    "Android exposes access-point frequency, channel, RSSI, width, security, and history metadata through Wi-Fi APIs when permissions allow it."
                } else {
                    "This device does not declare Wi-Fi support."
                },
                recommendation = "Use Wi-Fi Analyzer cards for graphable RF metadata instead of broad raw spectrum claims.",
                stationFields = listOf("ssid", "bssid", "frequency_mhz", "channel", "rssi_dbm", "security_mode", "bssid_vendor", "scan_timestamp_ms"),
                sampleFields = listOf("frequency_mhz", "channel", "rssi_dbm", "channel_width", "wifi_standard", "estimated_distance_meters"),
            ),
        )
        .put(
            radioReceiverProfileRow(
                receiverId = "bluetooth_public_metadata",
                label = "Bluetooth proximity receiver",
                sourceType = "bluetooth_public_scan",
                frequencyMinMhz = 2402.0,
                frequencyMaxMhz = 2480.0,
                channelStepLabel = "BLE advertising/service metadata",
                publicAndroidScanSupported = bluetoothSupported,
                builtInAndroidSource = bluetoothSupported,
                routeAction = "bluetooth_analyzer_report",
                accessPath = "bluetooth_scan, bluetooth_signal_history",
                scanState = if (bluetoothSupported) "public_android_metadata_route" else "no_bluetooth_feature",
                reason = if (bluetoothSupported) {
                    "Android exposes nearby Bluetooth/BLE identity, service UUID, manufacturer, RSSI, history metadata, and filter facets with permissions."
                } else {
                    "This device does not declare Bluetooth support."
                },
                recommendation = "Use Bluetooth Analyzer cards for 2.4 GHz proximity and identity context rather than raw channel sweeps.",
                stationFields = listOf("device_name", "address", "rssi_dbm", "service_uuids", "manufacturer_ids", "proximity_label", "applied_bluetooth_filters", "scan_timestamp_ms"),
                sampleFields = listOf("rssi_dbm", "average_rssi_dbm", "trend_db", "service_labels", "manufacturer_names"),
            ),
        )
        .put(
            radioReceiverProfileRow(
                receiverId = "external_sdr_bridge",
                label = "External SDR spectrum receiver",
                sourceType = "external_sdr",
                frequencyMinMhz = 0.5,
                frequencyMaxMhz = 6000.0,
                channelStepLabel = "receiver-dependent center/span/sample-rate",
                publicAndroidScanSupported = false,
                requiresExternalHardware = true,
                routeAction = "radio_signal_graph",
                accessPath = "USB/Bluetooth/Wi-Fi SDR bridge or vendor radio bridge",
                scanState = "external_receiver_required",
                reason = "Broad RF, airband, weather, arbitrary AM/FM scans, and microwave-like spectrum work need receiver hardware and a bridge that reports sample metadata.",
                recommendation = "Require receiver_id, center_frequency, span, sample_rate, and power rows before showing spectrum or waterfall data.",
                stationFields = listOf("frequency_hz", "label", "modulation", "power_db", "snr_db", "receiver_id", "scan_timestamp_ms"),
            sampleFields = listOf("center_frequency_hz", "span_hz", "sample_rate_hz", "bin_width_hz", "power_db", "waterfall_row", "receiver_id"),
            ),
        )

    private fun radioSignalGraphRows(arguments: JSONObject, vendorBroadcastRadioDeclared: Boolean): JSONArray {
        val limit = arguments.optInt("sample_limit", arguments.optInt("limit", 32)).coerceIn(1, 64)
        val sampleSource = arguments.optString("sample_source").ifBlank {
            arguments.optString("receiver_source").ifBlank { "vendor_or_sdr_bridge" }
        }
        val rows = JSONArray()
        val samples = radioSampleRowsFromArguments(arguments, limit)
        for (index in 0 until samples.length()) {
            val sample = samples.optJSONObject(index) ?: continue
            rows.put(radioSampleGraphRow(sample, index, sampleSource))
        }
        rows
            .put(
                radioGraphBoundaryRow(
                    label = "AM broadcast band",
                    band = "AM broadcast",
                    sourceType = "am_broadcast",
                    receiverId = "am_vendor_or_sdr",
                    frequencyMinKhz = 530,
                    frequencyMaxKhz = 1700,
                    valueLabel = if (vendorBroadcastRadioDeclared) "vendor bridge required" else "external receiver required",
                    scanState = if (vendorBroadcastRadioDeclared) "vendor_bridge_required" else "external_or_vendor_receiver_required",
                    recommendation = "Route station samples to radio_signal_graph only after a vendor radio bridge or SDR reports AM rows.",
                    fraction = if (vendorBroadcastRadioDeclared) 0.55 else 0.35,
                ),
            )
            .put(
                radioGraphBoundaryRow(
                    label = "FM broadcast band",
                    band = "FM broadcast",
                    sourceType = "fm_broadcast",
                    receiverId = "fm_vendor_or_sdr",
                    frequencyMinMhz = 87.5,
                    frequencyMaxMhz = 108.0,
                    valueLabel = if (vendorBroadcastRadioDeclared) "vendor bridge required" else "external receiver required",
                    scanState = if (vendorBroadcastRadioDeclared) "vendor_bridge_required" else "external_or_vendor_receiver_required",
                    recommendation = "Use radio_signal_graph for FM station/RDS samples only when a vendor bridge or SDR provides them.",
                    fraction = if (vendorBroadcastRadioDeclared) 0.55 else 0.35,
                ),
            )
        return rows
    }

    private fun radioSampleRowsFromArguments(arguments: JSONObject, limit: Int): JSONArray {
        val rows = JSONArray()
        val arrayKeys = listOf(
            "radio_signal_graph_sample_rows",
            "radio_signal_graph_rows",
            "radio_scan_rows",
            "radio_samples",
            "samples",
            "stations",
        )
        for (key in arrayKeys) {
            val samples = arguments.optJSONArray(key) ?: continue
            for (index in 0 until samples.length()) {
                if (rows.length() >= limit) return rows
                val sample = samples.optJSONObject(index) ?: continue
                if (radioSampleLooksGraphable(sample)) rows.put(sample)
            }
        }
        val objectKeys = listOf("radio_sample", "sample", "station")
        for (key in objectKeys) {
            if (rows.length() >= limit) return rows
            val sample = arguments.optJSONObject(key) ?: continue
            if (radioSampleLooksGraphable(sample)) rows.put(sample)
        }
        return rows
    }

    private fun radioSampleLooksGraphable(sample: JSONObject): Boolean {
        if (sample.optBoolean("sampled", false)) return true
        if (sample.optString("station_label").isNotBlank() || sample.optString("label").isNotBlank()) return true
        return listOf(
            "frequency_khz",
            "frequency_mhz",
            "frequency_hz",
            "center_frequency_hz",
            "rssi_dbm",
            "level_dbm",
            "signal_dbuv_or_rssi_dbm",
            "power_db",
        ).any { key -> radioOptionalDouble(sample, key) != null }
    }

    private fun radioSampleGraphRow(sample: JSONObject, index: Int, sampleSource: String): JSONObject {
        val frequencyKhz = radioOptionalDouble(sample, "frequency_khz")
        val frequencyMhz = radioOptionalDouble(sample, "frequency_mhz")
            ?: frequencyKhz?.div(1000.0)
            ?: radioOptionalDouble(sample, "frequency_hz")?.div(1_000_000.0)
            ?: radioOptionalDouble(sample, "center_frequency_hz")?.div(1_000_000.0)
        val band = sample.optString("band").ifBlank {
            inferRadioSampleBand(frequencyMhz, frequencyKhz, sample.optString("modulation"))
        }
        val receiverId = sample.optString("receiver_id").ifBlank { receiverIdForRadioBand(band) }
        val label = sample.optString("station_label").ifBlank {
            sample.optString("label").ifBlank {
                radioFrequencyLabel(frequencyMhz, frequencyKhz)?.let { "$band $it" } ?: "Radio sample ${index + 1}"
            }
        }
        val rssiDbm = listOf("rssi_dbm", "level_dbm", "signal_dbuv_or_rssi_dbm")
            .firstNotNullOfOrNull { key -> radioOptionalDouble(sample, key) }
        val powerDb = listOf("power_db", "power_dbm")
            .firstNotNullOfOrNull { key -> radioOptionalDouble(sample, key) }
        val signalValue = rssiDbm ?: powerDb
        val valueLabel = sample.optString("value_label").ifBlank {
            when {
                rssiDbm != null -> "${formatDecimal(rssiDbm, if (rssiDbm % 1.0 == 0.0) 0 else 1)} dBm"
                powerDb != null -> "${formatDecimal(powerDb, if (powerDb % 1.0 == 0.0) 0 else 1)} dB"
                else -> "sample available"
            }
        }
        val row = JSONObject()
            .put("category", "radio_signal_graph")
            .put("graph_row_role", "sample")
            .put("label", label)
            .put("band", band)
            .put("source_type", sample.optString("source_type").ifBlank { "vendor_or_sdr_sample" })
            .put("receiver_id", receiverId)
            .put("sampled", true)
            .put("data_available", true)
            .put("sample_source", sampleSource)
            .put("public_android_scan_supported", false)
            .put("requires_vendor_or_external_receiver", true)
            .put("scan_state", sample.optString("scan_state").ifBlank { "bridge_sample_reported" })
            .put("value_label", valueLabel)
            .put("modulation", sample.optString("modulation").ifBlank { modulationForRadioBand(band) })
            .put("recommendation", sample.optString("recommendation").ifBlank { "Treat as receiver-provided radio metadata; do not infer broader spectrum coverage from one sample." })
            .put("fraction", signalValue?.let { radioSignalFraction(it) } ?: 0.75)
        frequencyKhz?.let { row.put("frequency_khz", it) }
        frequencyMhz?.let { row.put("frequency_mhz", it) }
        radioFrequencyLabel(frequencyMhz, frequencyKhz)?.let { row.put("frequency_label", it) }
        rssiDbm?.let {
            row.put("rssi_dbm", it)
            row.put("signal_dbuv_or_rssi_dbm", it)
        }
        powerDb?.let { row.put("power_db", it) }
        radioOptionalDouble(sample, "snr_db")?.let { row.put("snr_db", it) }
        radioOptionalLong(sample, "scan_timestamp_ms")?.let { row.put("scan_timestamp_ms", it) }
        sample.optString("rds_program_service").takeIf { it.isNotBlank() }?.let { row.put("rds_program_service", it) }
        sample.optString("rds_radio_text").takeIf { it.isNotBlank() }?.let { row.put("rds_radio_text", it) }
        return row
    }

    private fun radioGraphBoundaryRow(
        label: String,
        band: String,
        sourceType: String,
        receiverId: String,
        frequencyMinKhz: Int? = null,
        frequencyMaxKhz: Int? = null,
        frequencyMinMhz: Double? = null,
        frequencyMaxMhz: Double? = null,
        valueLabel: String,
        scanState: String,
        recommendation: String,
        fraction: Double,
    ): JSONObject {
        val row = JSONObject()
            .put("category", "radio_signal_graph")
            .put("graph_row_role", "band_boundary")
            .put("label", label)
            .put("band", band)
            .put("source_type", sourceType)
            .put("receiver_id", receiverId)
            .put("sampled", false)
            .put("data_available", false)
            .put("public_android_scan_supported", false)
            .put("requires_vendor_or_external_receiver", true)
            .put("scan_state", scanState)
            .put("value_label", valueLabel)
            .put("recommendation", recommendation)
            .put("fraction", fraction)
        frequencyMinKhz?.let { row.put("frequency_min_khz", it) }
        frequencyMaxKhz?.let { row.put("frequency_max_khz", it) }
        frequencyMinMhz?.let { row.put("frequency_min_mhz", it) }
        frequencyMaxMhz?.let { row.put("frequency_max_mhz", it) }
        return row
    }

    private fun sampledRadioSignalGraphRows(rows: JSONArray): JSONArray {
        val sampledRows = JSONArray()
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            if (row.optBoolean("sampled", false)) sampledRows.put(row)
        }
        return sampledRows
    }

    private fun radioOptionalDouble(row: JSONObject, key: String): Double? {
        return when (val value = row.opt(key)) {
            is Number -> value.toDouble()
            is String -> value.trim().toDoubleOrNull()
            else -> null
        }
    }

    private fun radioOptionalLong(row: JSONObject, key: String): Long? {
        return when (val value = row.opt(key)) {
            is Number -> value.toLong()
            is String -> value.trim().toLongOrNull()
            else -> null
        }
    }

    private fun inferRadioSampleBand(frequencyMhz: Double?, frequencyKhz: Double?, modulation: String): String {
        val normalizedModulation = modulation.lowercase(Locale.US)
        return when {
            normalizedModulation == "am" -> "AM broadcast"
            normalizedModulation == "fm" -> "FM broadcast"
            frequencyKhz != null && frequencyKhz in 530.0..1700.0 -> "AM broadcast"
            frequencyMhz != null && frequencyMhz in 0.53..1.7 -> "AM broadcast"
            frequencyMhz != null && frequencyMhz in 87.5..108.0 -> "FM broadcast"
            else -> "External SDR / broad RF"
        }
    }

    private fun receiverIdForRadioBand(band: String): String = when (band) {
        "AM broadcast" -> "am_vendor_or_sdr"
        "FM broadcast" -> "fm_vendor_or_sdr"
        else -> "external_sdr_bridge"
    }

    private fun modulationForRadioBand(band: String): String = when (band) {
        "AM broadcast" -> "am"
        "FM broadcast" -> "fm"
        else -> "unknown"
    }

    private fun radioFrequencyLabel(frequencyMhz: Double?, frequencyKhz: Double?): String? {
        return when {
            frequencyKhz != null -> "${formatDecimal(frequencyKhz, if (frequencyKhz % 1.0 == 0.0) 0 else 1)} kHz"
            frequencyMhz != null && frequencyMhz < 2.0 -> "${formatDecimal(frequencyMhz * 1000.0, if ((frequencyMhz * 1000.0) % 1.0 == 0.0) 0 else 1)} kHz"
            frequencyMhz != null -> "${formatDecimal(frequencyMhz, if (frequencyMhz % 1.0 == 0.0) 0 else 1)} MHz"
            else -> null
        }
    }

    private fun radioSignalFraction(valueDb: Double): Double = ((valueDb + 110.0) / 80.0).coerceIn(0.08, 1.0)

    private fun radioReceiverProfileRow(
        receiverId: String,
        label: String,
        sourceType: String,
        frequencyMinKhz: Int? = null,
        frequencyMaxKhz: Int? = null,
        frequencyMinMhz: Double? = null,
        frequencyMaxMhz: Double? = null,
        channelStepLabel: String,
        publicAndroidScanSupported: Boolean,
        builtInAndroidSource: Boolean = false,
        vendorBridgePossible: Boolean = false,
        requiresVendorBridge: Boolean = false,
        requiresExternalHardware: Boolean = false,
        routeAction: String,
        accessPath: String,
        scanState: String,
        reason: String,
        recommendation: String,
        stationFields: List<String>,
        sampleFields: List<String>,
    ): JSONObject {
        val graphableMetadataSupported = publicAndroidScanSupported || vendorBridgePossible || requiresExternalHardware
        val row = JSONObject()
            .put("category", "radio_receiver_profile")
            .put("receiver_id", receiverId)
            .put("label", label)
            .put("source_type", sourceType)
            .put("ready", true)
            .put("value_label", scanState)
            .put("public_android_scan_supported", publicAndroidScanSupported)
            .put("built_in_android_source", builtInAndroidSource)
            .put("vendor_bridge_possible", vendorBridgePossible)
            .put("requires_vendor_bridge", requiresVendorBridge)
            .put("requires_external_hardware", requiresExternalHardware)
            .put("sample_rows_available", false)
            .put("graphable_metadata_supported", graphableMetadataSupported)
            .put("scan_state", scanState)
            .put("channel_step", channelStepLabel)
            .put("route_action", routeAction)
            .put("access_path", accessPath)
            .put("reason", reason)
            .put("recommendation", recommendation)
            .put("station_metadata_fields", JSONArray(stationFields))
            .put("sample_fields", JSONArray(sampleFields))
            .put("graph_row_schema", JSONArray((stationFields + sampleFields).distinct()))
            .put(
                "detail",
                "$accessPath | $channelStepLabel | ${if (publicAndroidScanSupported) "public Android metadata route" else if (vendorBridgePossible) "vendor bridge required" else if (requiresExternalHardware) "external receiver required" else "not public Android API"}",
            )
            .put(
                "fraction",
                when {
                    publicAndroidScanSupported -> 0.9f
                    vendorBridgePossible -> 0.65f
                    requiresExternalHardware -> 0.45f
                    else -> 0.3f
                },
            )
        frequencyMinKhz?.let { row.put("frequency_min_khz", it) }
        frequencyMaxKhz?.let { row.put("frequency_max_khz", it) }
        frequencyMinMhz?.let { row.put("frequency_min_mhz", it) }
        frequencyMaxMhz?.let { row.put("frequency_max_mhz", it) }
        return row
    }

    private fun radioSignalFeatureRows(
        vendorBroadcastRadioDeclared: Boolean,
        wifiSupported: Boolean,
        bluetoothSupported: Boolean,
    ): JSONArray = JSONArray()
        .put(
            capabilityRow(
                category = "radio_signal_feature",
                label = "AM/FM band plan cards",
                ready = true,
                valueLabel = "mapped",
                detail = "AM and FM frequency boundaries are represented as graph rows even when public Android scan samples are unavailable.",
                recommendation = "Show the band plan card for user context, then route real station scans to vendor radio or external SDR hardware.",
                fraction = 0.8f,
                extra = JSONObject().put("tool_action", "radio_signal_status"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_feature",
                label = "AM/FM public scan API",
                ready = false,
                valueLabel = "not public",
                detail = "Normal Android SDK APIs do not expose tuner sweep, station list, RDS, S-meter, or demodulated AM/FM rows to third-party apps.",
                recommendation = "Avoid promising built-in AM/FM scanning; use vendor radio bridge or external SDR integration when the user has compatible hardware.",
                fraction = 0.2f,
                extra = JSONObject().put("constraint_type", "android_public_api"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_feature",
                label = "Vendor broadcast radio hint",
                ready = vendorBroadcastRadioDeclared,
                valueLabel = if (vendorBroadcastRadioDeclared) "feature declared" else "no feature",
                detail = "Checks vendor feature names such as android.hardware.broadcastradio, android.hardware.radio, android.hardware.fm, and android.hardware.fmradio.",
                recommendation = "Treat this as a hardware hint only; it is not proof that Hermes can tune or read station rows through public APIs.",
                fraction = if (vendorBroadcastRadioDeclared) 0.65f else 0.35f,
                extra = JSONObject().put("feature_names", JSONArray(BROADCAST_RADIO_FEATURE_NAMES)),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_feature",
                label = "Wi-Fi radio metadata route",
                ready = wifiSupported,
                valueLabel = if (wifiSupported) "wifi_scan" else "no Wi-Fi",
                detail = "Wi-Fi scans provide the radio-like channel/frequency/RSSI/time-history surface that Android exposes reliably to apps.",
                recommendation = "Use Wi-Fi Analyzer cards for channel graph, channel rating, utilization, filters, vendor/OUI, estimated distance, and export rows.",
                fraction = if (wifiSupported) 0.9f else 0.25f,
                extra = JSONObject().put("tool_action", "wifi_channel_utilization"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_feature",
                label = "Bluetooth 2.4 GHz metadata route",
                ready = bluetoothSupported,
                valueLabel = if (bluetoothSupported) "bluetooth_scan" else "no Bluetooth",
                detail = "Bluetooth/BLE diagnostics provide nearby device RSSI, service UUID, manufacturer, category, and trend metadata rather than raw spectrum sweeps.",
                recommendation = "Use Bluetooth Analyzer and Bluetooth history cards for proximity reasoning.",
                fraction = if (bluetoothSupported) 0.85f else 0.25f,
                extra = JSONObject().put("tool_action", "bluetooth_signal_history"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_feature",
                label = "External SDR bridge path",
                ready = false,
                valueLabel = "hardware needed",
                detail = "Broad RF, airband, weather radio, arbitrary AM/FM scans, and microwave-like spectrum work need an external receiver and bridge.",
                recommendation = "Ask the user to attach/configure SDR or vendor receiver hardware before expecting spectrum samples.",
                fraction = 0.3f,
                extra = JSONObject().put("constraint_type", "external_receiver"),
            ),
        )

    private fun radioSignalWorkflowRows(
        vendorBroadcastRadioDeclared: Boolean,
        wifiSupported: Boolean,
        bluetoothSupported: Boolean,
    ): JSONArray = JSONArray()
        .put(
            capabilityRow(
                category = "radio_signal_route",
                label = "Route AM/FM requests",
                ready = true,
                valueLabel = "radio_signal_status",
                detail = "Use the radio band plan and scan-boundary rows to explain AM/FM availability without pretending public tuner access.",
                recommendation = if (vendorBroadcastRadioDeclared) {
                    "Mention the vendor radio hint, then request the OEM/vendor bridge path for station rows."
                } else {
                    "Name external SDR or vendor radio hardware as the required source for actual AM/FM scans."
                },
                fraction = 0.8f,
                extra = JSONObject().put("tool_action", "radio_signal_status"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_route",
                label = "Route Wi-Fi spectrum work",
                ready = wifiSupported,
                valueLabel = "wifi_channel_utilization",
                detail = "Use Wi-Fi channel utilization, rating, and signal-history rows for the graphable radio-frequency data Android actually exposes.",
                recommendation = "Run wifi_channel_utilization or wifi_analyzer_report before making interference/channel recommendations.",
                fraction = if (wifiSupported) 0.9f else 0.25f,
                extra = JSONObject().put("tool_action", "wifi_channel_utilization"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_route",
                label = "Route Bluetooth proximity work",
                ready = bluetoothSupported,
                valueLabel = "bluetooth_signal_history",
                detail = "Use nearby Bluetooth and trend cards for 2.4 GHz device proximity context, service UUID labels, manufacturer names, and RSSI changes.",
                recommendation = "Run bluetooth_signal_history when the user asks what nearby Bluetooth devices are doing over time.",
                fraction = if (bluetoothSupported) 0.85f else 0.25f,
                extra = JSONObject().put("tool_action", "bluetooth_signal_history"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_route",
                label = "Route external SDR setup",
                ready = false,
                valueLabel = "external bridge",
                detail = "No receiver bridge is bundled for arbitrary RF samples, so Hermes should surface setup requirements instead of empty scan rows.",
                recommendation = "Use a future SDR bridge/tool only after it reports sample rate, center frequency, span, and power rows.",
                fraction = 0.3f,
                extra = JSONObject().put("tool_action", "tool_catalog"),
            ),
        )

    private fun radioSignalConstraintRows(vendorBroadcastRadioDeclared: Boolean): JSONArray = JSONArray()
        .put(
            capabilityRow(
                category = "radio_signal_constraint",
                label = "Public Android AM/FM scan access",
                ready = false,
                valueLabel = "unavailable",
                detail = "AM/FM tuner scans, RDS, station lists, and raw receiver levels are not exposed through normal Android app APIs.",
                recommendation = "Return explicit unavailable rows and preserve the band plan for context.",
                fraction = 0.2f,
                extra = JSONObject().put("constraint_type", "android_public_api"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_constraint",
                label = "Vendor/OEM radio feature",
                ready = vendorBroadcastRadioDeclared,
                valueLabel = if (vendorBroadcastRadioDeclared) "hint only" else "not declared",
                detail = "PackageManager feature names can hint at OEM radio hardware, but they do not grant public tuning or scan access.",
                recommendation = "Use an OEM service, vendor app bridge, Shizuku-backed integration, or external hardware only with explicit user setup.",
                fraction = if (vendorBroadcastRadioDeclared) 0.55f else 0.25f,
                extra = JSONObject().put("constraint_type", "vendor_hal"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_constraint",
                label = "External receiver requirement",
                ready = false,
                valueLabel = "required for broad RF",
                detail = "Broad RF and microwave-like spectrum analysis needs receiver hardware that can provide samples to Hermes.",
                recommendation = "Do not invent spectrum rows; ask for an SDR/vendor bridge and verify sample metadata first.",
                fraction = 0.25f,
                extra = JSONObject().put("constraint_type", "external_receiver"),
            ),
        )
        .put(
            capabilityRow(
                category = "radio_signal_constraint",
                label = "Permission-gated built-in radios",
                ready = true,
                valueLabel = "Wi-Fi/Bluetooth",
                detail = "Android-exposed radio metadata still depends on location, nearby-device, Bluetooth, and scan-throttling policies.",
                recommendation = "Prefer cached history rows when active scans are throttled or permissions are missing.",
                fraction = 0.75f,
                extra = JSONObject().put("constraint_type", "permission"),
            ),
        )

    private fun localBackendStatusJson(status: LocalBackendStatus): JSONObject {
        return JSONObject()
            .put("backend_kind", status.backendKind.persistedValue)
            .put("started", status.started)
            .put("base_url", status.baseUrl.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
            .put("model_name", status.modelName.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
            .put("source_model_path", status.sourceModelPath.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
            .put("status_message", status.statusMessage.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
            .put("health_url", status.baseUrl.takeIf { it.isNotBlank() }?.removeSuffix("/v1")?.plus("/health") ?: JSONObject.NULL)
    }

    private fun liteRtRuntimeHealthJson(): JSONObject {
        return LiteRtLmOpenAiProxy.currentHealthJson()
            ?: JSONObject()
                .put("available", false)
                .put("status", "not_running")
                .put("backend", "litert-lm")
                .put("accelerator", JSONObject.NULL)
                .put("gpu_policy", "LiteRT-LM proxy is not currently running; start a local LiteRT-LM/AICore backend to expose live /health accelerator fields.")
    }

    private fun devicePerformanceProfileJson(context: Context): JSONObject {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val powerManager = context.getSystemService(PowerManager::class.java)
        val memorySummary = memorySummaryJson(context)
        val battery = batteryStateJson(context)
        val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && powerManager != null) {
            powerManager.currentThermalStatus
        } else {
            THERMAL_STATUS_UNSUPPORTED
        }
        val mediaPerformanceClass = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.VERSION.MEDIA_PERFORMANCE_CLASS
        } else {
            0
        }
        return JSONObject()
            .put("thermal_api_supported", Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && powerManager != null)
            .put("thermal_status", thermalStatus)
            .put("thermal_status_label", thermalStatusLabel(thermalStatus))
            .put("thermal_status_severity", thermalStatusSeverity(thermalStatus))
            .put("thermal_throttling_risk", thermalThrottlingRiskLabel(thermalStatus))
            .put("power_save_mode", powerManager?.isPowerSaveMode ?: false)
            .put("interactive", powerManager?.isInteractive ?: true)
            .put("low_ram_device", activityManager?.isLowRamDevice ?: false)
            .put("memory_class_mb", activityManager?.memoryClass ?: 0)
            .put("large_memory_class_mb", activityManager?.largeMemoryClass ?: 0)
            .put("available_memory_bytes", memorySummary.optLong("available_bytes", 0L))
            .put("available_memory_label", memorySummary.optString("available_label").ifBlank { "unknown" })
            .put("total_memory_bytes", memorySummary.optLong("total_bytes", 0L))
            .put("total_memory_label", memorySummary.optString("total_label").ifBlank { "unknown" })
            .put("memory_pressure_low", memorySummary.optBoolean("low_memory", false))
            .put("memory_threshold_bytes", memorySummary.optLong("threshold_bytes", 0L))
            .put("memory_threshold_label", memorySummary.optString("threshold_label").ifBlank { "unknown" })
            .put("app_data_free_bytes", memorySummary.optLong("app_data_free_bytes", 0L))
            .put("app_data_free_label", memorySummary.optString("app_data_free_label").ifBlank { "unknown" })
            .put("media_performance_class", mediaPerformanceClass)
            .put("media_performance_class_label", mediaPerformanceClassLabel(mediaPerformanceClass))
            .put("battery_status", battery.optInt("battery_status", BatteryManager.BATTERY_STATUS_UNKNOWN))
            .put("battery_status_label", battery.optString("battery_status_label"))
            .put("battery_plugged", battery.optInt("battery_plugged", 0))
            .put("battery_plugged_label", battery.optString("battery_plugged_label"))
            .put("battery_level_percent", battery.opt("battery_level_percent") ?: JSONObject.NULL)
            .put("battery_temperature_celsius", battery.opt("battery_temperature_celsius") ?: JSONObject.NULL)
    }

    private fun batteryStateJson(context: Context): JSONObject {
        val batteryIntent = runCatching {
            context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        }.getOrNull()
        val status = batteryIntent?.getIntExtra(
            BatteryManager.EXTRA_STATUS,
            BatteryManager.BATTERY_STATUS_UNKNOWN,
        ) ?: BatteryManager.BATTERY_STATUS_UNKNOWN
        val plugged = batteryIntent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val temperatureTenths = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?: Int.MIN_VALUE
        val percent = if (level >= 0 && scale > 0) (level.toDouble() * 100.0 / scale.toDouble()) else null
        val temperatureCelsius = if (temperatureTenths != Int.MIN_VALUE) temperatureTenths / 10.0 else null
        return JSONObject()
            .put("battery_status", status)
            .put("battery_status_label", batteryStatusLabel(status))
            .put("battery_plugged", plugged)
            .put("battery_plugged_label", batteryPluggedLabel(plugged))
            .put("battery_level_percent", percent ?: JSONObject.NULL)
            .put("battery_temperature_celsius", temperatureCelsius ?: JSONObject.NULL)
    }

    private fun gpuBackendRiskMatrixRows(
        socProfile: JSONObject,
        performanceProfile: JSONObject,
        runtimeHealth: JSONObject,
        currentBackend: LocalBackendStatus,
        preferredModel: JSONObject,
        selectedBackend: BackendKind,
        offlineAirplaneMode: Boolean,
        deviceIdentity: JSONObject,
    ): JSONArray {
        val healthAvailable = runtimeHealth.optString("status") == "ok" || runtimeHealth.optBoolean("available", false)
        val accelerator = runtimeHealth.optString("accelerator").takeIf { it.isNotBlank() && it != "null" } ?: "not running"
        val fallbackToCpu = runtimeHealth.optBoolean("gpu_fallback_to_cpu", false)
        val selectedLocalBackend = selectedBackend != BackendKind.NONE
        val supportsArm = socProfile.optBoolean("supports_arm64", false) || socProfile.optBoolean("supports_arm", false)
        val supportsX86 = socProfile.optBoolean("supports_x86_64", false) || socProfile.optBoolean("supports_x86", false)
        val thermalStatus = performanceProfile.optInt("thermal_status", THERMAL_STATUS_UNSUPPORTED)
        val lowRam = performanceProfile.optBoolean("low_ram_device", false)
        val memoryPressureLow = performanceProfile.optBoolean("memory_pressure_low", false)
        val memoryClass = performanceProfile.optInt("memory_class_mb", 0)
        val powerSaveMode = performanceProfile.optBoolean("power_save_mode", false)
        val batteryTemperature = performanceProfile.optDouble("battery_temperature_celsius", Double.NaN)
        val batteryHot = !batteryTemperature.isNaN() && batteryTemperature >= 45.0
        val preferredModelReady = preferredModel.optBoolean("ready", false)
        val likelyEmulator = deviceIdentity.optBoolean("likely_emulator", false)
        val liveAcceleratorRisk = when {
            healthAvailable && accelerator == "gpu" && !fallbackToCpu -> 10
            healthAvailable && fallbackToCpu -> 45
            healthAvailable && accelerator == "cpu" -> 40
            currentBackend.started && !healthAvailable -> 50
            selectedLocalBackend -> 35
            else -> 20
        }
        val socPolicyRisk = when {
            supportsArm -> 15
            supportsX86 -> 45
            else -> 35
        }
        val thermalRisk = when {
            thermalStatus == THERMAL_STATUS_UNSUPPORTED -> 25
            thermalStatus >= PowerManager.THERMAL_STATUS_CRITICAL -> 90
            thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> 70
            thermalStatus == PowerManager.THERMAL_STATUS_MODERATE -> 45
            else -> 15
        }
        val memoryRisk = when {
            memoryPressureLow -> 80
            lowRam -> 65
            memoryClass in 1 until 128 -> 55
            memoryClass in 128 until 192 -> 35
            else -> 20
        }
        val powerRisk = when {
            batteryHot && powerSaveMode -> 80
            batteryHot -> 65
            powerSaveMode -> 45
            else -> 20
        }
        val modelRisk = when {
            selectedLocalBackend && !preferredModelReady -> 55
            !preferredModelReady -> 35
            else -> 15
        }
        val validationRisk = if (likelyEmulator) 55 else 15
        return JSONArray()
            .put(
                gpuRiskRow(
                    label = "Live accelerator acceptance",
                    ready = liveAcceleratorRisk < 60,
                    valueLabel = if (healthAvailable) accelerator else if (currentBackend.started) "health unavailable" else "runtime not started",
                    detail = listOf(
                        "selected=${selectedBackend.persistedValue}",
                        "started=${currentBackend.started}",
                        "gpu_policy=${runtimeHealth.optString("gpu_policy").ifBlank { "unavailable" }}",
                        "opencl=${runtimeHealth.opt("opencl_available") ?: JSONObject.NULL}",
                        "fallback_to_cpu=$fallbackToCpu",
                    ).joinToString(" | "),
                    recommendation = "Read this row before promising GPU acceleration; CPU fallback is a valid mitigation when the delegate is rejected.",
                    riskScore = liveAcceleratorRisk,
                    extra = JSONObject()
                        .put("source_surface", "/health")
                        .put("runtime_signal", "accelerator_acceptance")
                        .put("tool_action", "local_backend_runtime_report"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "SOC/GPU policy coverage",
                    ready = true,
                    valueLabel = "${socProfile.optString("soc_family_label").ifBlank { "Android SOC" }} / ${socProfile.optString("gpu_family_label").ifBlank { "unknown GPU" }}",
                    detail = listOf(
                        "soc=${socProfile.optString("soc_family").ifBlank { "unknown" }}",
                        "gpu=${socProfile.optString("gpu_family_hint").ifBlank { "unknown" }}",
                        "arm=$supportsArm",
                        "x86=$supportsX86",
                    ).joinToString(" | "),
                    recommendation = "Treat MediaTek/Mali/Immortalis/PowerVR/Xclipse as covered policy paths that still require live accelerator probing.",
                    riskScore = socPolicyRisk,
                    extra = JSONObject()
                        .put("source_surface", "soc_profile")
                        .put("runtime_signal", "soc_gpu_policy")
                        .put("tool_action", "soc_compatibility_report"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "Thermal throttle risk",
                    ready = thermalRisk < 60,
                    valueLabel = performanceProfile.optString("thermal_throttling_risk").ifBlank { "unknown" },
                    detail = "thermal=${performanceProfile.optString("thermal_status_label").ifBlank { "unknown" }} | api=${performanceProfile.optBoolean("thermal_api_supported", false)}",
                    recommendation = "Reduce model size, response length, scan cadence, or GPU-heavy bursts when thermal risk rises.",
                    riskScore = thermalRisk,
                    extra = JSONObject()
                        .put("source_surface", "PowerManager.currentThermalStatus")
                        .put("runtime_signal", "thermal"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "Memory pressure risk",
                    ready = memoryRisk < 60,
                    valueLabel = "${memoryClass}MB app class",
                    detail = "low_ram=$lowRam | pressure_low=$memoryPressureLow | available=${performanceProfile.optString("available_memory_label").ifBlank { "unknown" }} | threshold=${performanceProfile.optString("memory_threshold_label").ifBlank { "unknown" }}",
                    recommendation = "Use smaller local models, shorter context, or CPU fallback when low-RAM or pressure flags are active.",
                    riskScore = memoryRisk,
                    extra = JSONObject()
                        .put("source_surface", "ActivityManager.MemoryInfo")
                        .put("runtime_signal", "memory_pressure"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "Power saver and battery heat",
                    ready = powerRisk < 60,
                    valueLabel = if (powerSaveMode) "power saver" else "normal power",
                    detail = "plugged=${performanceProfile.optString("battery_plugged_label").ifBlank { "unknown" }} | temp_c=${performanceProfile.opt("battery_temperature_celsius") ?: JSONObject.NULL} | level=${performanceProfile.opt("battery_level_percent") ?: JSONObject.NULL}",
                    recommendation = "Avoid repeated active scans or long GPU runs when battery heat or power saver is active.",
                    riskScore = powerRisk,
                    extra = JSONObject()
                        .put("source_surface", "PowerManager and battery state")
                        .put("runtime_signal", "power_battery"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "Model artifact fit",
                    ready = modelRisk < 60,
                    valueLabel = preferredModel.optString("runtime_flavor").ifBlank { "model import needed" },
                    detail = listOf(
                        preferredModel.optString("title").ifBlank { "no preferred model" },
                        preferredModel.optString("destination_file_name").ifBlank { "no file" },
                        preferredModel.optString("record_status").ifBlank { "no record" },
                        "file_exists=${preferredModel.optBoolean("file_exists", false)}",
                    ).joinToString(" | "),
                    recommendation = "Confirm the preferred model matches the selected backend before treating GPU/SOC state as the blocker.",
                    riskScore = modelRisk,
                    extra = JSONObject()
                        .put("source_surface", "preferred_local_model")
                        .put("runtime_signal", "model_artifact")
                        .put("tool_action", "soc_compatibility_report"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "Phone validation scope",
                    ready = validationRisk < 60,
                    valueLabel = if (likelyEmulator) "emulator evidence" else "phone evidence",
                    detail = "likely_emulator=$likelyEmulator | primary_abi=${socProfile.optString("primary_abi").ifBlank { "unknown" }} | offline_airplane_mode=$offlineAirplaneMode",
                    recommendation = "Treat x86/emulator success as UI and CPU fallback coverage; validate ARM phone behavior before claiming GPU compatibility is fully green.",
                    riskScore = validationRisk,
                    extra = JSONObject()
                        .put("constraint_type", "validation_scope")
                        .put("runtime_signal", "phone_validation"),
                ),
            )
            .put(
                gpuRiskRow(
                    label = "Fallback action route",
                    ready = true,
                    valueLabel = "triage available",
                    detail = "Use runtime, SOC, stability, and preflight reports together before changing backend or model policy.",
                    recommendation = "Route follow-up through gpu_backend_risk_report, local_backend_runtime_report, soc_compatibility_report, and device_performance_report.",
                    riskScore = 15,
                    extra = JSONObject()
                        .put("tool_action", "gpu_backend_risk_report")
                        .put("runtime_signal", "triage_route"),
                ),
            )
    }

    private fun gpuBackendRiskRouteRows(): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "gpu_backend_risk_route",
                    label = "Route GPU backend risk triage",
                    ready = true,
                    valueLabel = "gpu_backend_risk_report",
                    detail = "Use when a user asks if local inference will be stable on MediaTek, Mali, PowerVR, or another non-Adreno phone.",
                    recommendation = "Start with this matrix, then drill into runtime health, SOC policy, or phone preflight rows.",
                    fraction = 0.95f,
                    extra = JSONObject().put("tool_action", "gpu_backend_risk_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "gpu_backend_risk_route",
                    label = "Route live runtime health",
                    ready = true,
                    valueLabel = "local_backend_runtime_report",
                    detail = "Use for selected backend, current local runtime state, /health accelerator, GPU policy, and modality support.",
                    recommendation = "Run when the risk row says the runtime is started or GPU fallback status is ambiguous.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "local_backend_runtime_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "gpu_backend_risk_route",
                    label = "Route SOC and artifact policy",
                    ready = true,
                    valueLabel = "soc_compatibility_report",
                    detail = "Use for SOC/GPU identity, ABI selection, LiteRT-LM artifact matching, and Adreno-only assumption checks.",
                    recommendation = "Run when the question is about device family compatibility rather than current runtime pressure.",
                    fraction = 0.88f,
                    extra = JSONObject().put("tool_action", "soc_compatibility_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "gpu_backend_risk_route",
                    label = "Route stability guardrails",
                    ready = true,
                    valueLabel = "device_performance_report",
                    detail = "Use for thermal, low-RAM, power saver, battery heat, and media performance class rows.",
                    recommendation = "Run before repeated scans, long local model calls, or multimodal local inference.",
                    fraction = 0.86f,
                    extra = JSONObject().put("tool_action", "device_performance_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "gpu_backend_risk_route",
                    label = "Route phone workflow preflight",
                    ready = true,
                    valueLabel = "social_gmail_goal_preflight",
                    detail = "Use when backend risk must be combined with installed apps, accessibility state, permissions, and preferred model readiness.",
                    recommendation = "Use this route before end-to-end TikTok, Instagram, or Gmail phone workflows.",
                    fraction = 0.82f,
                    extra = JSONObject().put("tool_action", "social_gmail_goal_preflight"),
                ),
            )
    }

    private fun localInferenceCompatibilityRows(
        socProfile: JSONObject,
        performanceProfile: JSONObject,
        runtimeHealth: JSONObject,
        currentBackend: LocalBackendStatus,
        preferredModel: JSONObject,
        selectedBackend: BackendKind,
        offlineAirplaneMode: Boolean,
        deviceIdentity: JSONObject,
        riskRows: JSONArray,
    ): JSONArray {
        val supportsArm = socProfile.optBoolean("supports_arm64", false) || socProfile.optBoolean("supports_arm", false)
        val supportsX86 = socProfile.optBoolean("supports_x86_64", false) || socProfile.optBoolean("supports_x86", false)
        val likelyNonAdreno = !socProfile.optBoolean("likely_adreno_gpu", false)
        val healthAvailable = runtimeHealth.optString("status") == "ok" || runtimeHealth.optBoolean("available", false)
        val accelerator = runtimeHealth.optString("accelerator").takeIf { it.isNotBlank() && it != "null" } ?: "not running"
        val fallbackToCpu = runtimeHealth.optBoolean("gpu_fallback_to_cpu", false)
        val selectedLocalBackend = selectedBackend != BackendKind.NONE
        val thermalStatus = performanceProfile.optInt("thermal_status", THERMAL_STATUS_UNSUPPORTED)
        val thermalBlocked = thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE
        val lowRam = performanceProfile.optBoolean("low_ram_device", false)
        val memoryPressureLow = performanceProfile.optBoolean("memory_pressure_low", false)
        val powerSaveMode = performanceProfile.optBoolean("power_save_mode", false)
        val likelyEmulator = deviceIdentity.optBoolean("likely_emulator", false)
        val maxRiskScore = maxRiskScore(riskRows)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "SOC and GPU family coverage",
                    ready = supportsArm || supportsX86,
                    valueLabel = socProfile.optString("litert_lm_acceleration_label").ifBlank { "Android SOC" },
                    detail = listOf(
                        socProfile.optString("soc_family_label").ifBlank { "unknown SOC" },
                        socProfile.optString("gpu_family_label").ifBlank { "unknown GPU" },
                        socProfile.optString("primary_abi").ifBlank { "unknown ABI" },
                        "arm=$supportsArm",
                        "x86=$supportsX86",
                    ).joinToString(" | "),
                    recommendation = "Use SOC-neutral GPU probing with CPU fallback; do not treat Snapdragon/Adreno as the only supported acceleration path.",
                    fraction = when {
                        supportsArm -> 0.95f
                        supportsX86 -> 0.6f
                        else -> 0.35f
                    },
                    extra = JSONObject()
                        .put("tool_action", "soc_compatibility_report")
                        .put("graph_type", "soc_backend_matrix"),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "Live accelerator acceptance",
                    ready = !selectedLocalBackend || (healthAvailable && (!fallbackToCpu || accelerator == "cpu")),
                    valueLabel = if (healthAvailable) accelerator else if (currentBackend.started) "health unavailable" else "runtime not started",
                    detail = "selected=${selectedBackend.persistedValue} | started=${currentBackend.started} | gpu_policy=${runtimeHealth.optString("gpu_policy").ifBlank { "unavailable" }} | fallback_to_cpu=$fallbackToCpu",
                    recommendation = "Start or inspect the local backend before claiming GPU readiness; CPU fallback is acceptable but should be named to the user.",
                    fraction = when {
                        healthAvailable && accelerator == "gpu" && !fallbackToCpu -> 1f
                        healthAvailable && (accelerator == "cpu" || fallbackToCpu) -> 0.72f
                        currentBackend.started -> 0.45f
                        selectedLocalBackend -> 0.55f
                        else -> 0.62f
                    },
                    extra = JSONObject()
                        .put("tool_action", "local_backend_runtime_report")
                        .put("graph_type", "runtime_backend_matrix"),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "Model artifact fit",
                    ready = preferredModel.optBoolean("ready", false),
                    valueLabel = preferredModel.optString("runtime_flavor").ifBlank { "model import needed" },
                    detail = listOf(
                        preferredModel.optString("title").ifBlank { "no preferred model" },
                        preferredModel.optString("destination_file_name").ifBlank { "no file" },
                        preferredModel.optString("record_status").ifBlank { "no record" },
                        "file_exists=${preferredModel.optBoolean("file_exists", false)}",
                    ).joinToString(" | "),
                    recommendation = "Prefer a generic or matching MediaTek/Mali/PowerVR LiteRT-LM artifact before vendor-specific Qualcomm/Adreno bundles on non-Adreno phones.",
                    fraction = if (preferredModel.optBoolean("ready", false)) 0.95f else 0.35f,
                    extra = JSONObject()
                        .put("tool_action", "soc_compatibility_report")
                        .put("source_surface", "preferred_local_model"),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "Thermal memory and power runway",
                    ready = !thermalBlocked && !memoryPressureLow,
                    valueLabel = performanceProfile.optString("thermal_throttling_risk").ifBlank { "unknown thermal" },
                    detail = "thermal=${performanceProfile.optString("thermal_status_label").ifBlank { "unknown" }} | memory=${performanceProfile.optString("available_memory_label").ifBlank { "unknown" }} | low_ram=$lowRam | power_saver=$powerSaveMode",
                    recommendation = "Reduce scan cadence, model size, or response length when thermal, low-RAM, memory-pressure, or power-saver signals are weak.",
                    fraction = when {
                        thermalBlocked || memoryPressureLow -> 0.3f
                        lowRam || powerSaveMode -> 0.62f
                        else -> 0.88f
                    },
                    extra = JSONObject()
                        .put("tool_action", "device_performance_report")
                        .put("graph_type", "runtime_stability_matrix"),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "MediaTek and non-Adreno fallback policy",
                    ready = true,
                    valueLabel = if (likelyNonAdreno) "non-Adreno path visible" else "Adreno path visible",
                    detail = "soc=${socProfile.optString("soc_family").ifBlank { "unknown" }} | gpu=${socProfile.optString("gpu_family_hint").ifBlank { "unknown" }} | risk=${riskLevelForScore(maxRiskScore)} score=$maxRiskScore",
                    recommendation = "Keep MediaTek/Mali/Immortalis/PowerVR/Xclipse as first-class policy rows and disclose CPU fallback until live GPU acceptance is proven.",
                    fraction = if (likelyNonAdreno) 0.92f else 0.86f,
                    extra = JSONObject()
                        .put("tool_action", "gpu_backend_risk_report")
                        .put("graph_type", "gpu_backend_risk_matrix")
                        .put("gpu_backend_risk_level", riskLevelForScore(maxRiskScore))
                        .put("gpu_backend_risk_score", maxRiskScore),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "Phone validation scope",
                    ready = !likelyEmulator,
                    valueLabel = if (likelyEmulator) "emulator evidence" else "phone evidence",
                    detail = "likely_emulator=$likelyEmulator | primary_abi=${socProfile.optString("primary_abi").ifBlank { "unknown" }} | offline_airplane_mode=$offlineAirplaneMode",
                    recommendation = "Treat emulator success as UI and CPU-path coverage; validate on an ARM phone before saying MediaTek or non-Adreno acceleration is fully green.",
                    fraction = if (likelyEmulator) 0.42f else 0.9f,
                    extra = JSONObject()
                        .put("validation_scope", if (likelyEmulator) "emulator" else "physical_or_host_device")
                        .put("tool_action", "social_gmail_goal_preflight"),
                ),
            )
            .put(
                capabilityRow(
                    category = "local_inference_compatibility",
                    label = "Agent drill-down route",
                    ready = true,
                    valueLabel = "SOC -> risk -> runtime -> stability",
                    detail = "Scorecard rows route to soc_compatibility_report, gpu_backend_risk_report, local_backend_runtime_report, and device_performance_report.",
                    recommendation = "Use this card as Gemma's first local-inference summary, then drill into the weakest row's source action.",
                    fraction = 0.95f,
                    extra = JSONObject()
                        .put("tool_action", "local_inference_compatibility_report")
                        .put("source_actions", JSONArray().put("soc_compatibility_report").put("gpu_backend_risk_report").put("local_backend_runtime_report").put("device_performance_report")),
                ),
            )
    }

    private fun gpuRiskRow(
        label: String,
        ready: Boolean,
        valueLabel: String,
        detail: String,
        recommendation: String,
        riskScore: Int,
        extra: JSONObject = JSONObject(),
    ): JSONObject {
        val normalizedScore = riskScore.coerceIn(0, 100)
        return capabilityRow(
            category = "gpu_backend_risk",
            label = label,
            ready = ready,
            valueLabel = valueLabel,
            detail = detail,
            recommendation = recommendation,
            fraction = ((100 - normalizedScore) / 100f).coerceIn(0.05f, 1f),
            extra = extra
                .put("risk_score", normalizedScore)
                .put("risk_level", riskLevelForScore(normalizedScore))
                .put("mitigation", recommendation),
        )
    }

    private fun riskLevelForScore(score: Int): String = when {
        score >= 85 -> "critical"
        score >= 60 -> "high"
        score >= 35 -> "moderate"
        else -> "low"
    }

    private fun maxRiskScore(rows: JSONArray): Int {
        var maxScore = 0
        for (index in 0 until rows.length()) {
            maxScore = maxOf(maxScore, rows.optJSONObject(index)?.optInt("risk_score", 0) ?: 0)
        }
        return maxScore
    }

    private fun averageCapabilityValue(rows: JSONArray): Int {
        if (rows.length() == 0) return 0
        var total = 0
        for (index in 0 until rows.length()) {
            total += rows.optJSONObject(index)?.optInt("value", 0) ?: 0
        }
        return (total.toDouble() / rows.length().toDouble()).roundToInt().coerceIn(0, 100)
    }

    private fun compatibilityLevelForScore(score: Int): String = when {
        score >= 85 -> "ready"
        score >= 65 -> "watch"
        score >= 45 -> "limited"
        else -> "blocked"
    }

    private fun countHighRiskRows(rows: JSONArray): Int {
        var count = 0
        for (index in 0 until rows.length()) {
            val level = rows.optJSONObject(index)?.optString("risk_level").orEmpty()
            if (level == "high" || level == "critical") count += 1
        }
        return count
    }

    private fun devicePerformanceMatrixRows(profile: JSONObject, socProfile: JSONObject): JSONArray {
        val thermalStatus = profile.optInt("thermal_status", THERMAL_STATUS_UNSUPPORTED)
        val powerSaveMode = profile.optBoolean("power_save_mode", false)
        val lowRam = profile.optBoolean("low_ram_device", false)
        val memoryClass = profile.optInt("memory_class_mb", 0)
        val largeMemoryClass = profile.optInt("large_memory_class_mb", 0)
        val mediaPerformanceClass = profile.optInt("media_performance_class", 0)
        val socLabel = socProfile.optString("soc_family_label").ifBlank { "Android SOC" }
        val gpuLabel = socProfile.optString("gpu_family_label").ifBlank { "GPU unknown" }
        return JSONArray()
            .put(
                capabilityRow(
                    category = "runtime_stability",
                    label = "Thermal throttling status",
                    ready = thermalStatus < PowerManager.THERMAL_STATUS_SEVERE,
                    valueLabel = profile.optString("thermal_status_label").ifBlank { "unknown" },
                    detail = "thermal_api_supported=${profile.optBoolean("thermal_api_supported", false)} | risk=${profile.optString("thermal_throttling_risk").ifBlank { "unknown" }}",
                    recommendation = "If thermal status reaches severe or higher, prefer CPU fallback, smaller local models, shorter responses, or wait before GPU-heavy LiteRT-LM runs.",
                    fraction = thermalStatusFraction(thermalStatus),
                    extra = JSONObject().put("source_surface", "PowerManager.currentThermalStatus"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_stability",
                    label = "Low-RAM and memory class",
                    ready = !lowRam && memoryClass >= 128,
                    valueLabel = "${memoryClass}MB app / ${largeMemoryClass}MB large",
                    detail = "low_ram_device=$lowRam | available=${profile.optString("available_memory_label").ifBlank { "unknown" }} | total=${profile.optString("total_memory_label").ifBlank { "unknown" }} | threshold=${profile.optString("memory_threshold_label").ifBlank { "unknown" }}",
                    recommendation = "Use memory class and low-RAM status before enabling multimodal local inference, speculative decoding, or large model artifacts on lower-end MediaTek phones.",
                    fraction = when {
                        lowRam -> 0.35f
                        memoryClass >= 384 -> 0.95f
                        memoryClass >= 192 -> 0.8f
                        memoryClass >= 128 -> 0.65f
                        else -> 0.4f
                    },
                    extra = JSONObject().put("source_surface", "ActivityManager memory class"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_stability",
                    label = "Battery and power saver state",
                    ready = !powerSaveMode,
                    valueLabel = if (powerSaveMode) "power saver enabled" else "normal power policy",
                    detail = "battery=${profile.optString("battery_status_label").ifBlank { "unknown" }} | plugged=${profile.optString("battery_plugged_label").ifBlank { "unknown" }} | level=${profile.opt("battery_level_percent") ?: JSONObject.NULL} | temp_c=${profile.opt("battery_temperature_celsius") ?: JSONObject.NULL}",
                    recommendation = "When power saver is enabled or battery temperature is high, reduce scan refreshes and prefer short local inference bursts.",
                    fraction = if (powerSaveMode) 0.55f else 0.85f,
                    extra = JSONObject().put("source_surface", "PowerManager and ACTION_BATTERY_CHANGED"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_stability",
                    label = "Android media performance class",
                    ready = true,
                    valueLabel = profile.optString("media_performance_class_label").ifBlank { "not declared" },
                    detail = "media_performance_class=$mediaPerformanceClass; not all phones declare this even when local inference can still run.",
                    recommendation = "Use this as a guardrail for video/image/audio expectations, not as the only gate for Gemma local inference.",
                    fraction = if (mediaPerformanceClass > 0) 0.85f else 0.55f,
                    extra = JSONObject().put("source_surface", "Build.VERSION.MEDIA_PERFORMANCE_CLASS"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_stability",
                    label = "MediaTek/non-Adreno stability guardrail",
                    ready = true,
                    valueLabel = "$socLabel / $gpuLabel",
                    detail = "SOC family=${socProfile.optString("soc_family").ifBlank { "unknown" }} | GPU family=${socProfile.optString("gpu_family_hint").ifBlank { "unknown" }} | ABI=${socProfile.optString("primary_abi").ifBlank { "unknown" }}",
                    recommendation = "Combine SOC/GPU detection with thermal, low-RAM, power, and live /health accelerator fields before treating non-Adreno phones as unsupported.",
                    fraction = 0.95f,
                    extra = JSONObject().put("feature_source", "MediaTek and non-Adreno runtime stability policy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_stability",
                    label = "Local inference cadence policy",
                    ready = true,
                    valueLabel = "bounded bursts",
                    detail = "Use cached Wi-Fi/Bluetooth rows, one-shot sensor samples, and passive stability state before repeated refreshes or long model/tool loops.",
                    recommendation = "Route long local Gemma work through runtime_backend_report plus this stability matrix so scans and inference adapt to current phone pressure.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "device_performance_report"),
                ),
            )
    }

    private fun thermalStatusLabel(status: Int): String = when (status) {
        THERMAL_STATUS_UNSUPPORTED -> "unsupported"
        PowerManager.THERMAL_STATUS_NONE -> "none"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
        else -> "unknown($status)"
    }

    private fun thermalStatusSeverity(status: Int): String = when {
        status == THERMAL_STATUS_UNSUPPORTED -> "unsupported"
        status <= PowerManager.THERMAL_STATUS_LIGHT -> "low"
        status == PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        status == PowerManager.THERMAL_STATUS_SEVERE -> "high"
        else -> "critical"
    }

    private fun thermalThrottlingRiskLabel(status: Int): String = when {
        status == THERMAL_STATUS_UNSUPPORTED -> "api unavailable"
        status <= PowerManager.THERMAL_STATUS_LIGHT -> "low"
        status == PowerManager.THERMAL_STATUS_MODERATE -> "medium"
        status == PowerManager.THERMAL_STATUS_SEVERE -> "high"
        else -> "critical"
    }

    private fun thermalStatusFraction(status: Int): Float = when {
        status == THERMAL_STATUS_UNSUPPORTED -> 0.55f
        status <= PowerManager.THERMAL_STATUS_LIGHT -> 0.95f
        status == PowerManager.THERMAL_STATUS_MODERATE -> 0.75f
        status == PowerManager.THERMAL_STATUS_SEVERE -> 0.45f
        else -> 0.2f
    }

    private fun mediaPerformanceClassLabel(mediaPerformanceClass: Int): String {
        return if (mediaPerformanceClass > 0) {
            "Android $mediaPerformanceClass media performance class"
        } else {
            "not declared"
        }
    }

    private fun batteryStatusLabel(status: Int): String = when (status) {
        BatteryManager.BATTERY_STATUS_CHARGING -> "charging"
        BatteryManager.BATTERY_STATUS_DISCHARGING -> "discharging"
        BatteryManager.BATTERY_STATUS_FULL -> "full"
        BatteryManager.BATTERY_STATUS_NOT_CHARGING -> "not charging"
        else -> "unknown"
    }

    private fun batteryPluggedLabel(plugged: Int): String {
        val labels = buildList {
            if (plugged and BatteryManager.BATTERY_PLUGGED_AC != 0) add("ac")
            if (plugged and BatteryManager.BATTERY_PLUGGED_USB != 0) add("usb")
            if (plugged and BatteryManager.BATTERY_PLUGGED_WIRELESS != 0) add("wireless")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && plugged and BatteryManager.BATTERY_PLUGGED_DOCK != 0) {
                add("dock")
            }
        }
        return labels.joinToString("+").ifBlank { "unplugged" }
    }

    private fun runtimeBackendMatrixRows(
        selectedBackend: BackendKind,
        currentBackend: LocalBackendStatus,
        runtimeHealth: JSONObject,
        socProfile: JSONObject,
        preferredModel: JSONObject,
        offlineAirplaneMode: Boolean,
    ): JSONArray {
        val healthAvailable = runtimeHealth.optString("status") == "ok"
        val healthBackendOrder = jsonStringList(runtimeHealth.optJSONArray("litert_backend_order"))
        val selectedLabel = when (selectedBackend) {
            BackendKind.NONE -> "remote provider"
            BackendKind.LLAMA_CPP -> "llama.cpp"
            BackendKind.LITERT_LM -> "LiteRT-LM"
            BackendKind.AICORE -> "AICore"
        }
        val currentLabel = when {
            currentBackend.started -> "${currentBackend.backendKind.persistedValue} running"
            currentBackend.backendKind == BackendKind.NONE -> "no local runtime"
            else -> "${currentBackend.backendKind.persistedValue} stopped"
        }
        return JSONArray()
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "Selected on-device backend",
                    ready = selectedBackend != BackendKind.NONE,
                    valueLabel = selectedLabel,
                    detail = "offline_airplane_mode=$offlineAirplaneMode | selected=${selectedBackend.persistedValue}",
                    recommendation = "Use LiteRT-LM or AICore for Gemma Android local inference; use llama.cpp for GGUF models when selected.",
                    fraction = if (selectedBackend != BackendKind.NONE) 0.9f else 0.35f,
                    extra = JSONObject().put("source_surface", "AppSettingsStore.onDeviceBackend"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "Current local backend state",
                    ready = currentBackend.started,
                    valueLabel = currentLabel,
                    detail = listOf(
                        currentBackend.statusMessage.ifBlank { "no status message" },
                        currentBackend.modelName.ifBlank { "no active model" },
                        currentBackend.baseUrl.ifBlank { "no local base URL" },
                    ).joinToString(" | "),
                    recommendation = "Do not infer runtime health from SOC labels alone; inspect the current backend state and /health accelerator row.",
                    fraction = if (currentBackend.started) 1f else 0.35f,
                    extra = JSONObject()
                        .put("source_surface", "OnDeviceBackendManager.currentStatus")
                        .put("health_url", currentBackend.baseUrl.takeIf { it.isNotBlank() }?.removeSuffix("/v1")?.plus("/health") ?: JSONObject.NULL),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "LiteRT-LM /health accelerator",
                    ready = healthAvailable,
                    valueLabel = runtimeHealth.optString("accelerator").ifBlank { "not running" },
                    detail = listOf(
                        "gpu_policy=${runtimeHealth.optString("gpu_policy").ifBlank { "unavailable" }}",
                        "opencl=${runtimeHealth.opt("opencl_available") ?: JSONObject.NULL}",
                        "fallback_to_cpu=${runtimeHealth.opt("gpu_fallback_to_cpu") ?: JSONObject.NULL}",
                        "soc=${runtimeHealth.optString("soc_family").ifBlank { socProfile.optString("soc_family").ifBlank { "unknown" } }}",
                        "gpu=${runtimeHealth.optString("gpu_family").ifBlank { socProfile.optString("gpu_family_hint").ifBlank { "unknown" } }}",
                        "order=${healthBackendOrder.joinToString(">").ifBlank { "not reported" }}",
                    ).joinToString(" | "),
                    recommendation = "Use this row to distinguish working GPU acceleration, GPU-to-CPU fallback, and stopped runtime states on MediaTek/Mali/PowerVR phones.",
                    fraction = if (healthAvailable && runtimeHealth.optString("accelerator") == "gpu") 1f else if (healthAvailable) 0.8f else 0.25f,
                    extra = JSONObject().put("source_surface", "/health"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "Model artifact compatibility",
                    ready = preferredModel.optBoolean("ready", false),
                    valueLabel = preferredModel.optString("runtime_flavor").ifBlank { "model import needed" },
                    detail = listOf(
                        preferredModel.optString("title").ifBlank { "no preferred model" },
                        preferredModel.optString("destination_file_name").ifBlank { "no file" },
                        preferredModel.optString("record_status").ifBlank { "no record" },
                        if (preferredModel.optBoolean("file_exists", false)) "${preferredModel.optLong("file_bytes", 0L)} bytes" else "file missing",
                    ).joinToString(" | "),
                    recommendation = "Keep web .task, .litertlm, and GGUF artifact compatibility visible before blaming SOC or GPU compatibility.",
                    fraction = if (preferredModel.optBoolean("ready", false)) 1f else 0.35f,
                    extra = JSONObject().put("source_surface", "preferred_local_model"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "Multimodal runtime policy",
                    ready = healthAvailable,
                    valueLabel = if (healthAvailable) {
                        "image=${runtimeHealth.optBoolean("image_input_supported", false)} audio=${runtimeHealth.optBoolean("audio_input_supported", false)}"
                    } else {
                        "runtime not started"
                    },
                    detail = listOf(
                        runtimeHealth.optString("modality_policy").ifBlank { "Start LiteRT-LM/AICore to expose modality policy." },
                        "vision=${runtimeHealth.optString("vision_accelerator").ifBlank { "unknown" }}",
                        "audio=${runtimeHealth.optString("audio_accelerator").ifBlank { "unknown" }}",
                    ).joinToString(" | "),
                    recommendation = "Use the live modality policy before promising Gemma image/audio input on a specific phone.",
                    fraction = if (healthAvailable && !runtimeHealth.optBoolean("multimodal_fallback", false)) 0.95f else if (healthAvailable) 0.65f else 0.3f,
                    extra = JSONObject().put("source_surface", "LiteRT-LM modality policy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "MediaTek/non-Snapdragon fallback policy",
                    ready = true,
                    valueLabel = socProfile.optString("litert_lm_acceleration_label").ifBlank { "GPU probe + CPU fallback" },
                    detail = socProfile.optString("litert_lm_backend_strategy").ifBlank { "GPU-first on ARM devices when LiteRT-LM accepts the accelerator, then CPU fallback; CPU-only on x86 emulator/device builds." },
                    recommendation = "Treat GPU initialization failure as a fallback path; do not mark MediaTek, Mali, Immortalis, PowerVR/IMG, Xclipse, Tensor, Exynos, or Unisoc unsupported solely because they are not Adreno.",
                    fraction = 0.95f,
                    extra = JSONObject().put("source_surface", "soc_profile"),
                ),
            )
            .put(
                capabilityRow(
                    category = "runtime_backend",
                    label = "Runtime health check route",
                    ready = currentBackend.started && currentBackend.baseUrl.isNotBlank(),
                    valueLabel = currentBackend.baseUrl.takeIf { it.isNotBlank() }?.removeSuffix("/v1")?.plus("/health") ?: "no endpoint",
                    detail = "The runtime report stays passive; it reads current in-process state and exposes the health endpoint path when a local backend is already running.",
                    recommendation = "Start the selected backend through normal app settings or chat startup before expecting live accelerator fields.",
                    fraction = if (currentBackend.started && currentBackend.baseUrl.isNotBlank()) 0.9f else 0.3f,
                    extra = JSONObject().put("tool_action", "local_backend_runtime_report"),
                ),
            )
    }

    private fun socBackendMatrixRows(socProfile: JSONObject, preferredModel: JSONObject): JSONArray {
        val nativeAbiCandidates = jsonStringList(socProfile.optJSONArray("native_abi_candidates"))
        val supportedAbis = jsonStringList(socProfile.optJSONArray("supported_abis"))
        val primaryAbi = socProfile.optString("primary_abi").ifBlank { supportedAbis.firstOrNull().orEmpty() }
        val socFamily = socProfile.optString("soc_family")
        val gpuFamily = socProfile.optString("gpu_family_hint")
        val supportsArm = socProfile.optBoolean("supports_arm64", false) || socProfile.optBoolean("supports_arm", false)
        val supportsX86 = socProfile.optBoolean("supports_x86_64", false) || socProfile.optBoolean("supports_x86", false)
        val artifactPolicy = socProfile.optJSONObject("litert_lm_artifact_selection_policy") ?: JSONObject()
        val detectedBackend = when {
            socProfile.optBoolean("likely_mediatek", false) -> "MediaTek covered"
            socProfile.optBoolean("likely_mali_gpu", false) -> "Mali covered"
            socProfile.optBoolean("likely_powervr_img_gpu", false) -> "PowerVR covered"
            socProfile.optBoolean("likely_adreno_gpu", false) -> "Adreno covered"
            supportsX86 && !supportsArm -> "x86 CPU fallback"
            else -> "generic ARM covered"
        }
        return JSONArray()
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "Detected SOC family",
                    ready = socFamily.isNotBlank() && socFamily != "unknown",
                    valueLabel = socProfile.optString("soc_family_label").ifBlank { "unknown SOC" },
                    detail = listOf(
                        socProfile.optString("soc_manufacturer").ifBlank { "manufacturer unknown" },
                        socProfile.optString("soc_model").ifBlank { "model unknown" },
                        socProfile.optString("hardware").ifBlank { "hardware unknown" },
                        socProfile.optString("board").ifBlank { "board unknown" },
                    ).joinToString(" | "),
                    recommendation = "Keep the report SOC-neutral; do not assume Snapdragon just because local inference is enabled.",
                    fraction = if (socFamily.isNotBlank() && socFamily != "unknown") 0.95f else 0.6f,
                    extra = JSONObject().put("source_surface", "soc_profile"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "GPU family hint",
                    ready = gpuFamily.isNotBlank() && gpuFamily != "unknown",
                    valueLabel = socProfile.optString("gpu_family_label").ifBlank { "unknown GPU" },
                    detail = listOf(
                        "adreno=${socProfile.optBoolean("likely_adreno_gpu", false)}",
                        "mali=${socProfile.optBoolean("likely_mali_gpu", false)}",
                        "powervr_img=${socProfile.optBoolean("likely_powervr_img_gpu", false)}",
                        "xclipse=${socProfile.optBoolean("likely_xclipse_gpu", false)}",
                    ).joinToString(" | "),
                    recommendation = "Probe LiteRT-LM accelerators from GPU hints, then fall back to CPU when the accelerator is not accepted.",
                    fraction = if (gpuFamily.isNotBlank() && gpuFamily != "unknown") 0.9f else 0.55f,
                    extra = JSONObject().put("source_surface", "gpu_family_hint"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "Native ABI selection",
                    ready = nativeAbiCandidates.isNotEmpty() || primaryAbi.isNotBlank(),
                    valueLabel = primaryAbi.ifBlank { "ABI unknown" },
                    detail = listOf(
                        "candidates=${nativeAbiCandidates.joinToString(", ").ifBlank { "none" }}",
                        "supported=${supportedAbis.joinToString(", ").ifBlank { "none" }}",
                        socProfile.optString("native_abi_strategy").ifBlank { "Native ABI strategy unavailable." },
                    ).joinToString(" | "),
                    recommendation = "Package and choose native artifacts by ABI, not by SOC marketing name.",
                    fraction = if (nativeAbiCandidates.isNotEmpty() || primaryAbi.isNotBlank()) 0.95f else 0.45f,
                    extra = JSONObject().put("source_surface", "native_abi_candidates"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "LiteRT-LM accelerator policy",
                    ready = true,
                    valueLabel = socProfile.optString("litert_lm_acceleration_label").ifBlank { "GPU probe + CPU fallback" },
                    detail = socProfile.optString("litert_lm_backend_strategy").ifBlank { "GPU-first on ARM, CPU fallback, CPU-only on x86 emulator/device builds." },
                    recommendation = "Use this row before promising GPU acceleration or rejecting non-Snapdragon phones.",
                    fraction = 0.95f,
                    extra = JSONObject().put("source_surface", "litert_backend_strategy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "SOC-specific LiteRT artifact selection",
                    ready = artifactPolicy.optBoolean("soc_aware_selection_enabled", false),
                    valueLabel = artifactPolicy.optString("preferred_device_family_label").ifBlank { "generic first" },
                    detail = artifactPolicy.optString("selection_order").ifBlank {
                        "Generic .litertlm artifacts are preferred; matching SOC/GPU-specific bundles are selected before mismatched SOC-specific bundles when generic artifacts are absent."
                    },
                    recommendation = artifactPolicy.optString("recommendation").ifBlank {
                        "Avoid selecting Qualcomm-specific LiteRT-LM artifacts on MediaTek phones unless the user explicitly pins that file."
                    },
                    fraction = if (artifactPolicy.optBoolean("soc_aware_selection_enabled", false)) 0.9f else 0.65f,
                    extra = JSONObject(artifactPolicy.toString()).put("source_surface", "model_download_artifact_selection"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "MediaTek/Mali/PowerVR coverage",
                    ready = true,
                    valueLabel = detectedBackend,
                    detail = "Dimensity/Helio, Mali/Immortalis, PowerVR/IMG, Tensor/Mali, Exynos/Xclipse, Snapdragon/Adreno, Unisoc, and generic ARM devices all use GPU probing with CPU fallback.",
                    recommendation = "Keep MediaTek and PowerVR as first-class compatibility cases instead of treating them as unsupported outliers.",
                    fraction = 0.95f,
                    extra = JSONObject().put("feature_source", "MediaTek Mali PowerVR compatibility policy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "Preferred local model",
                    ready = preferredModel.optBoolean("ready", false),
                    valueLabel = preferredModel.optString("title").ifBlank { "model import needed" },
                    detail = listOf(
                        preferredModel.optString("runtime_flavor").ifBlank { "unknown runtime" },
                        preferredModel.optString("record_status").ifBlank { "no preferred record" },
                        if (preferredModel.optBoolean("file_exists", false)) "${preferredModel.optLong("file_bytes", 0L)} bytes" else "file missing",
                    ).joinToString(" | "),
                    recommendation = "Import and prefer a Gemma LiteRT-LM or Qwen GGUF model before treating SOC readiness as runnable local inference.",
                    fraction = if (preferredModel.optBoolean("ready", false)) 1f else 0.35f,
                    extra = JSONObject().put("source_surface", "preferred_local_model"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_parity",
                    label = "x86 emulator separation",
                    ready = true,
                    valueLabel = if (socProfile.optBoolean("supports_x86", false) || socProfile.optBoolean("supports_x86_64", false)) "CPU-only emulator/device" else "phone ABI path",
                    detail = "x86/x86_64 runs prove CPU fallback and UI logic, but do not prove ARM phone GPU acceleration on Mali, PowerVR, Adreno, or Xclipse.",
                    recommendation = "Validate ARM phone behavior separately from emulator-only runs before calling GPU compatibility done.",
                    fraction = 0.85f,
                    extra = JSONObject().put("constraint_type", "emulator_separation"),
                ),
            )
    }

    private fun socBackendRouteRows(socProfile: JSONObject, preferredModel: JSONObject): JSONArray {
        return JSONArray()
            .put(
                capabilityRow(
                    category = "soc_backend_route",
                    label = "Route SOC compatibility report",
                    ready = true,
                    valueLabel = "soc_compatibility_report",
                    detail = "Use for SOC family, GPU hint, ABI candidate, MediaTek/Mali/PowerVR coverage, and LiteRT-LM backend policy cards.",
                    recommendation = "Run this report when the user asks whether a non-Snapdragon phone can use Hermes local inference.",
                    fraction = 0.95f,
                    extra = JSONObject().put("tool_action", "soc_compatibility_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_route",
                    label = "Route full agent environment",
                    ready = true,
                    valueLabel = "agent_environment_report",
                    detail = "Use when SOC policy needs to be interpreted alongside Kai parity, local model readiness, automation, memory, UI control, and wireless inputs.",
                    recommendation = "Choose this route when backend compatibility is one part of a broader autonomous-agent readiness question.",
                    fraction = 0.9f,
                    extra = JSONObject().put("tool_action", "agent_environment_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_route",
                    label = "Route cross-signal context",
                    ready = true,
                    valueLabel = "signal_awareness_report",
                    detail = "Use when Wi-Fi, Bluetooth, sensors, radio limits, and SOC backend policy all need to be visible to Gemma together.",
                    recommendation = "Run this before wireless or sensor-heavy local reasoning so backend limits are not separated from signal inputs.",
                    fraction = 0.85f,
                    extra = JSONObject().put("tool_action", "signal_awareness_report"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_route",
                    label = "Route phone preflight",
                    ready = preferredModel.optBoolean("ready", false),
                    valueLabel = "social_gmail_goal_preflight",
                    detail = "Use when SOC/backend readiness must be combined with package, accessibility, and preferred-model checks before a full phone workflow.",
                    recommendation = "Treat missing preferred model or phone permissions as blockers even when the SOC policy itself is compatible.",
                    fraction = if (preferredModel.optBoolean("ready", false)) 0.9f else 0.45f,
                    extra = JSONObject().put("tool_action", "social_gmail_goal_preflight"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_route",
                    label = "Route current signal limits",
                    ready = true,
                    valueLabel = "signal_capability_status",
                    detail = "${socProfile.optString("soc_family_label").ifBlank { "Android SOC" }} backend policy can be paired with public Android radio/sensor limits.",
                    recommendation = "Use this route when a hardware question includes AM/FM, broad RF, microwave, or sensor availability.",
                    fraction = 0.8f,
                    extra = JSONObject().put("tool_action", "signal_capability_status"),
                ),
            )
    }

    private fun socBackendConstraintRows(socProfile: JSONObject): JSONArray {
        val supportedAbis = jsonStringList(socProfile.optJSONArray("supported_abis"))
        val nativeAbiCandidates = jsonStringList(socProfile.optJSONArray("native_abi_candidates"))
        val supportsArm = socProfile.optBoolean("supports_arm64", false) || socProfile.optBoolean("supports_arm", false)
        val supportsX86 = socProfile.optBoolean("supports_x86_64", false) || socProfile.optBoolean("supports_x86", false)
        return JSONArray()
            .put(
                capabilityRow(
                    category = "soc_backend_constraint",
                    label = "Avoid Adreno-only assumptions",
                    ready = true,
                    valueLabel = "SOC-neutral",
                    detail = "The policy recognizes MediaTek/Mali/Immortalis, PowerVR/IMG, Tensor/Mali, Exynos/Xclipse, Unisoc, Snapdragon/Adreno, and generic ARM paths.",
                    recommendation = "Do not gate local inference support only on Qualcomm or Adreno labels.",
                    fraction = 1f,
                    extra = JSONObject().put("constraint_type", "soc_policy"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_constraint",
                    label = "ARM native artifact coverage",
                    ready = supportsArm || nativeAbiCandidates.isNotEmpty(),
                    valueLabel = if (supportsArm) "ARM path present" else if (supportsX86) "x86 CPU fallback" else "ABI unknown",
                    detail = "Supported ABI(s): ${supportedAbis.joinToString(", ").ifBlank { "none reported" }}. Candidate native ABI(s): ${nativeAbiCandidates.joinToString(", ").ifBlank { "none selected" }}.",
                    recommendation = "Build and package ARM native assets for phone validation; use x86 only as emulator coverage.",
                    fraction = if (supportsArm) 0.95f else if (supportsX86) 0.65f else 0.35f,
                    extra = JSONObject().put("constraint_type", "native_abi"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_constraint",
                    label = "GPU probe then CPU fallback",
                    ready = true,
                    valueLabel = "fallback required",
                    detail = socProfile.optString("litert_lm_backend_strategy").ifBlank { "GPU-first on ARM devices when LiteRT-LM accepts the accelerator, then CPU fallback." },
                    recommendation = "Treat failed GPU accelerator initialization as a fallback path, not as device incompatibility.",
                    fraction = 0.95f,
                    extra = JSONObject().put("constraint_type", "backend_fallback"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_constraint",
                    label = "x86 emulator is not phone GPU proof",
                    ready = true,
                    valueLabel = if (supportsX86) "x86 detected" else "phone validation needed",
                    detail = "Emulator CPU-only success does not validate ARM phone GPU delegates, SOC governors, or vendor OpenCL/Vulkan behavior.",
                    recommendation = "Run physical ARM phone smoke tests before claiming MediaTek/Mali/PowerVR compatibility is fully green.",
                    fraction = 0.85f,
                    extra = JSONObject().put("constraint_type", "validation_scope"),
                ),
            )
            .put(
                capabilityRow(
                    category = "soc_backend_constraint",
                    label = "Public Android capability probes",
                    ready = true,
                    valueLabel = "feature API first",
                    detail = "Hermes uses Android SDK feature, permission, sensor, Wi-Fi, Bluetooth, camera, storage, ABI, and Build fields before making backend decisions.",
                    recommendation = "Prefer public API probes and explicit rows over hard-coded SOC brand assumptions.",
                    fraction = 0.9f,
                    extra = JSONObject().put("constraint_type", "android_api"),
                ),
            )
    }

    private fun capabilityRow(
        category: String,
        label: String,
        ready: Boolean,
        valueLabel: String,
        detail: String,
        recommendation: String,
        fraction: Float,
        extra: JSONObject = JSONObject(),
    ): JSONObject {
        val row = JSONObject()
            .put("category", category)
            .put("label", label)
            .put("ready", ready)
            .put("value_label", valueLabel)
            .put("detail", detail)
            .put("recommendation", recommendation)
            .put("fraction", fraction.coerceIn(0.05f, 1f))
            .put("value", (fraction.coerceIn(0.05f, 1f) * 100).roundToInt())
        extra.keys().forEach { key -> row.put(key, extra.opt(key)) }
        return row
    }

    private fun countReadyRows(rows: JSONArray): Int {
        var count = 0
        for (index in 0 until rows.length()) {
            if (rows.optJSONObject(index)?.optBoolean("ready", false) == true) count += 1
        }
        return count
    }

    private fun compactModelRoutingJson(status: JSONObject): JSONObject {
        return JSONObject()
            .put("ready", status.optBoolean("ready", false))
            .put("active_provider", status.optString("active_provider"))
            .put("active_provider_label", status.optString("active_provider_label"))
            .put("active_model", status.optString("active_model"))
            .put("vision_capable", status.optBoolean("vision_capable", false))
            .put("model_routing_supported", status.optBoolean("model_routing_supported", false))
            .put("roles", status.optJSONArray("roles") ?: JSONArray())
    }

    private fun agentPersonaStatusJson(settings: AppSettings): JSONObject {
        val customPrompt = AppSettings.normalizeCustomSystemPrompt(settings.customSystemPrompt)
        return JSONObject()
            .put("ready", true)
            .put("custom_system_prompt_enabled", customPrompt.isNotBlank())
            .put("custom_system_prompt_chars", customPrompt.length)
            .put("max_custom_system_prompt_chars", AppSettings.MAX_CUSTOM_SYSTEM_PROMPT_CHARS)
            .put("settings_key", "custom_system_prompt")
            .put("settings_bundle_kind", AppSettings.EXPORT_KIND)
            .put("secrets_included", false)
            .put("prompt_injection_point", "native_chat_system_prompt")
            .put("kai_parity_source", "customizable soul")
    }

    private fun compactHindsightStatusJson(status: JSONObject): JSONObject {
        return JSONObject()
            .put("memory_count", status.optInt("memory_count", 0))
            .put("reinforced_memory_count", status.optInt("reinforced_memory_count", 0))
            .put("promoted_memory_count", status.optInt("promoted_memory_count", 0))
            .put("promotion_hit_threshold", status.optInt("promotion_hit_threshold", 0))
    }

    private fun compactAutomationStandbyStatusJson(status: JSONObject): JSONObject {
        return JSONObject()
            .put("ready", status.optBoolean("ready", false))
            .put("standby_heartbeat_supported", status.optBoolean("standby_heartbeat_supported", false))
            .put("batch_heartbeat_supported", status.optBoolean("batch_heartbeat_supported", false))
            .put("model_routing_supported", status.optBoolean("model_routing_supported", false))
            .put("heartbeat_interval_seconds", status.optInt("heartbeat_interval_seconds", 0))
            .put("enabled_automation_count", status.optInt("enabled_automation_count", 0))
            .put("recent_run_count", status.optInt("recent_run_count", 0))
            .put("supported_dispatch_channels", status.optJSONArray("supported_dispatch_channels") ?: JSONArray())
    }

    private fun compactSignalCapabilityJson(status: JSONObject): JSONObject {
        return JSONObject()
            .put("audio_frequency_analysis_supported", status.optBoolean("audio_frequency_analysis_supported", false))
            .put("wifi_signal_analysis_supported", status.optBoolean("wifi_signal_analysis_supported", false))
            .put("bluetooth_signal_access_supported", status.optBoolean("bluetooth_signal_access_supported", false))
            .put("magnetic_field_sensor_supported", status.optBoolean("magnetic_field_sensor_supported", false))
            .put("motion_sensor_analysis_supported", status.optJSONArray("motion_sensor_analysis_supported") ?: JSONArray())
            .put("am_fm_public_android_scan_supported", status.optBoolean("am_fm_public_android_scan_supported", false))
            .put("vendor_broadcast_radio_feature_declared", status.optBoolean("vendor_broadcast_radio_feature_declared", false))
            .put("wifi_radio_metadata_supported", status.optBoolean("wifi_radio_metadata_supported", false))
            .put("bluetooth_radio_metadata_supported", status.optBoolean("bluetooth_radio_metadata_supported", false))
            .put("requires_external_sdr_for_broad_rf", status.optBoolean("requires_external_sdr_for_broad_rf", true))
            .put("radio_band_plan_count", status.optInt("radio_band_plan_count", 0))
            .put("radio_receiver_profile_count", status.optInt("radio_receiver_profile_count", 0))
            .put("ready_radio_receiver_profile_count", status.optInt("ready_radio_receiver_profile_count", 0))
            .put("radio_signal_feature_count", status.optInt("radio_signal_feature_count", 0))
            .put("ready_radio_signal_feature_count", status.optInt("ready_radio_signal_feature_count", 0))
            .put("radio_signal_workflow_route_count", status.optInt("radio_signal_workflow_route_count", 0))
            .put("radio_signal_constraint_count", status.optInt("radio_signal_constraint_count", 0))
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
                    .put(toolJson("schedule_task", "Kai-compatible scheduled reminder alias backed by Hermes native Android automation notification records, not background AI prompt execution.", "task, title, task_id, time, at, interval_minutes, days_of_week, enabled"))
                    .put(toolJson("list_tasks", "Kai-compatible alias for listing saved Hermes Android automation task records.", "limit"))
                    .put(toolJson("cancel_task", "Kai-compatible alias for deleting a saved Hermes Android automation by task_id.", "task_id"))
                    .put(toolJson("android_automation_tool", "Run/open/create saved automations, watcher tasks, overlays, notifications, widgets, Tasker-style triggers, Kai-compatible scheduled task aliases, and secret-free app settings export/import.", "action, trigger, task_id, data_uri, bundle_json, settings_json"))
                    .put(toolJson("android_device_diagnostics_tool", "Inspect resource-heavy apps, Wi-Fi signals/channel graph envelopes/channel ratings/AP detail and export rows/vendor OUI/filter facets plus active Wi-Fi band/security/signal/SSID/RSSI filters, Bluetooth nearby devices/service UUID labels/manufacturer names/proximity/history/filter facets, camera, sensors, SOC compatibility, overlay, Gemma-visible signal evidence bundles and agent observation dashboards, radio/RF capability limits, Kai-style agent environment parity, and the social/Gmail end-to-end phone preflight.", "action, limit, detail_limit, export_format, scan_mode, refresh, filter_band, filter_security, filter_signal, filter_ssid, min_rssi_dbm, max_rssi_dbm, filter_device_name, filter_bluetooth_service, filter_bluetooth_manufacturer, filter_bluetooth_category, filter_bluetooth_proximity, sensor_types, timeout_ms"))
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

    private fun unavailableSensorRows(requested: List<String>): JSONArray {
        val rows = JSONArray()
        requested.forEach { key -> rows.put(unavailableSensorJson(key)) }
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
        return appendWifiSemanticFields(json)
    }

    private fun bluetoothDeviceJson(device: BluetoothDevice): JSONObject {
        val bluetoothClass = runCatching { device.bluetoothClass }.getOrNull()
        val serviceUuids = runCatching {
            device.uuids
                ?.mapNotNull { it?.uuid?.toString() }
                ?.distinct()
                .orEmpty()
        }.getOrDefault(emptyList())
        val serviceLabels = bluetoothServiceLabels(serviceUuids)
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
            .put("service_labels", JSONArray(serviceLabels.take(MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE)))
            .put("service_uuid_count", serviceUuids.size)
            .put("semantic_context", bluetoothSemanticContext(serviceLabels = serviceLabels))
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
            .put("service_labels", bluetoothServiceLabelsJson(scanRecord))
            .put("service_uuid_count", scanRecord?.serviceUuids?.size ?: 0)
            .put("service_data_uuids", bluetoothServiceDataUuidsJson(scanRecord))
            .put("service_data_labels", bluetoothServiceDataLabelsJson(scanRecord))
            .put("manufacturer_ids", bluetoothManufacturerIdsJson(scanRecord))
            .put("manufacturer_names", bluetoothManufacturerNamesJson(scanRecord))
            .put("manufacturer_data_count", scanRecord?.manufacturerSpecificData?.size() ?: 0)
            .put("manufacturer_data_bytes", bluetoothManufacturerDataBytes(scanRecord))
            .put(
                "semantic_context",
                bluetoothSemanticContext(
                    serviceLabels = jsonStringList(bluetoothServiceLabelsJson(scanRecord)),
                    manufacturerNames = jsonStringList(bluetoothManufacturerNamesJson(scanRecord)),
                ),
            )
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

    private fun bluetoothServiceLabelsJson(scanRecord: ScanRecord?): JSONArray {
        val values = scanRecord?.serviceUuids
            ?.mapNotNull { it?.uuid?.toString() }
            ?.distinct()
            .orEmpty()
        return JSONArray(bluetoothServiceLabels(values).take(MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE))
    }

    private fun bluetoothServiceDataLabelsJson(scanRecord: ScanRecord?): JSONArray {
        val values = scanRecord?.serviceData
            ?.keys
            ?.mapNotNull { it?.uuid?.toString() }
            ?.distinct()
            .orEmpty()
        return JSONArray(bluetoothServiceLabels(values).take(MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE))
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

    private fun bluetoothManufacturerNamesJson(scanRecord: ScanRecord?): JSONArray {
        val ids = jsonStringList(bluetoothManufacturerIdsJson(scanRecord))
        return JSONArray(
            ids.mapNotNull { bluetoothManufacturerIdLabel(it).takeIf(String::isNotBlank) }
                .distinct()
                .take(MAX_BLUETOOTH_MANUFACTURER_IDS_PER_DEVICE),
        )
    }

    private fun bluetoothManufacturerDataBytes(scanRecord: ScanRecord?): Int {
        val data = scanRecord?.manufacturerSpecificData ?: return 0
        var bytes = 0
        for (index in 0 until data.size()) {
            bytes += data.valueAt(index)?.size ?: 0
        }
        return bytes
    }

    internal fun bluetoothServiceUuidLabel(uuid: String): String {
        val assignedNumber = bluetoothAssignedServiceNumber(uuid) ?: return ""
        return BLUETOOTH_SERVICE_UUID_LABELS[assignedNumber].orEmpty()
    }

    internal fun bluetoothManufacturerIdLabel(companyId: String): String {
        val normalized = normalizedBluetoothCompanyId(companyId) ?: return ""
        return BLUETOOTH_COMPANY_ID_LABELS[normalized].orEmpty()
    }

    private fun bluetoothServiceLabels(values: List<String>): List<String> {
        return values.mapNotNull { bluetoothServiceUuidLabel(it).takeIf(String::isNotBlank) }.distinct()
    }

    private fun bluetoothSemanticContext(
        serviceLabels: List<String> = emptyList(),
        manufacturerNames: List<String> = emptyList(),
    ): String {
        return listOfNotNull(
            serviceLabels.take(3).joinToString(", ").takeIf { it.isNotBlank() }?.let { "services=$it" },
            manufacturerNames.take(3).joinToString(", ").takeIf { it.isNotBlank() }?.let { "manufacturers=$it" },
        ).joinToString(" | ")
    }

    private fun bluetoothAssignedServiceNumber(uuid: String): String? {
        val normalized = uuid.trim().lowercase(Locale.US)
        val shortHex = when {
            Regex("""^0x[0-9a-f]{4}$""").matches(normalized) -> normalized.drop(2)
            Regex("""^[0-9a-f]{4}$""").matches(normalized) -> normalized
            else -> Regex("""^0000([0-9a-f]{4})-0000-1000-8000-00805f9b34fb$""")
                .matchEntire(normalized)
                ?.groupValues
                ?.getOrNull(1)
        } ?: return null
        return "0x${shortHex.uppercase(Locale.US)}"
    }

    private fun normalizedBluetoothCompanyId(companyId: String): String? {
        val trimmed = companyId.trim()
        val value = when {
            trimmed.startsWith("0x", ignoreCase = true) -> trimmed.drop(2).toIntOrNull(16)
            Regex("""[0-9a-fA-F]{1,4}""").matches(trimmed) -> trimmed.toIntOrNull(16)
            trimmed.all { it.isDigit() } -> trimmed.toIntOrNull()
            else -> null
        } ?: return null
        if (value !in 0..0xFFFF) return null
        return "0x${value.toString(16).uppercase(Locale.US).padStart(4, '0')}"
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
            jsonStringList(row, "service_data_uuids").forEach { uuid ->
                add("service_data_uuid", uuid, row)
            }
            jsonStringList(row, "manufacturer_ids").forEach { manufacturerId ->
                add("manufacturer_id", manufacturerId, row)
            }
        }

        val rows = summaries.map { (key, accumulator) ->
            val parts = key.split('|', limit = 2)
            val summaryType = parts.firstOrNull().orEmpty()
            val label = parts.getOrNull(1).orEmpty()
            val semanticLabel = bluetoothSemanticMetadataLabel(summaryType, label)
            JSONObject()
                .put("summary_type", summaryType)
                .put("label", label)
                .put("semantic_label", semanticLabel ?: JSONObject.NULL)
                .put("display_label", semanticLabel ?: label)
                .put("count", accumulator.count)
                .put("paired_count", accumulator.pairedCount)
                .put("connectable_count", accumulator.connectableCount)
                .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                .put("sample_devices", JSONArray(accumulator.sampleDevices.take(MAX_BLUETOOTH_SUMMARY_SAMPLES)))
                .put(
                    "recommendation",
                    when (summaryType) {
                        "service_uuid" -> "BLE service UUID advertised nearby${semanticLabel?.let { " ($it)" }.orEmpty()}; use it to infer device capability before connecting."
                        "service_data_uuid" -> "BLE service-data UUID advertised nearby${semanticLabel?.let { " ($it)" }.orEmpty()}; use it to interpret advertisement payload context before connecting."
                        "manufacturer_id" -> "Manufacturer data advertised nearby${semanticLabel?.let { " ($it)" }.orEmpty()}; useful for beacon or vendor-specific device identification."
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

    private fun bluetoothSemanticMetadataLabel(summaryType: String, label: String): String? = when (summaryType) {
        "service_uuid", "service_data_uuid" -> bluetoothServiceUuidLabel(label).takeIf { it.isNotBlank() }
        "manufacturer_id" -> bluetoothManufacturerIdLabel(label).takeIf { it.isNotBlank() }
        else -> null
    }

    internal fun mergeBluetoothSignalHistory(existing: JSONObject, devices: JSONArray, observedAtMs: Long): JSONObject {
        val records = linkedMapOf<String, JSONObject>()
        val existingRecords = existing.optJSONArray("devices") ?: JSONArray()
        for (index in 0 until existingRecords.length()) {
            val record = existingRecords.optJSONObject(index) ?: continue
            val key = record.optString("key").ifBlank { bluetoothHistoryKey(record) }
            if (key.isNotBlank()) {
                records[key] = record.put("key", key)
            }
        }
        for (index in 0 until minOf(devices.length(), MAX_BLUETOOTH_HISTORY_DEVICES_PER_SCAN)) {
            val device = devices.optJSONObject(index) ?: continue
            val rssi = jsonIntOrNull(device, "rssi_dbm") ?: continue
            val key = bluetoothHistoryKey(device)
            if (key.isBlank()) continue
            val record = records.getOrPut(key) { JSONObject().put("key", key) }
            record
                .put("device_name", device.optString("device_name").ifBlank { device.optString("advertised_name") }.ifBlank { "<unnamed>" })
                .put("advertised_name", device.optString("advertised_name"))
                .put("address", device.optString("address"))
                .put("device_type", device.optString("device_type").ifBlank { "unknown" })
                .put("device_category", device.optString("device_category").ifBlank { "unknown" })
                .put("bond_state", device.optString("bond_state").ifBlank { "unknown" })
                .put("connectable", device.opt("connectable") ?: JSONObject.NULL)
                .put("service_uuids", device.optJSONArray("service_uuids") ?: JSONArray())
                .put("service_labels", device.optJSONArray("service_labels") ?: JSONArray())
                .put("service_data_uuids", device.optJSONArray("service_data_uuids") ?: JSONArray())
                .put("service_data_labels", device.optJSONArray("service_data_labels") ?: JSONArray())
                .put("manufacturer_ids", device.optJSONArray("manufacturer_ids") ?: JSONArray())
                .put("manufacturer_names", device.optJSONArray("manufacturer_names") ?: JSONArray())
                .put("semantic_context", device.optString("semantic_context"))
            val observations = record.optJSONArray("observations") ?: JSONArray()
            observations.put(JSONObject().put("observed_at_ms", observedAtMs).put("rssi_dbm", rssi))
            record.put("observations", trimBluetoothObservations(observations))
        }
        val ordered = records.values
            .filter { record -> (record.optJSONArray("observations")?.length() ?: 0) > 0 }
            .sortedWith(
                compareByDescending<JSONObject> { lastBluetoothObservationTime(it) }
                    .thenByDescending { currentBluetoothRssi(it) ?: Int.MIN_VALUE }
                    .thenBy { it.optString("device_name") },
            )
            .take(MAX_BLUETOOTH_HISTORY_DEVICES)
        return JSONObject()
            .put("updated_at_ms", observedAtMs)
            .put("devices", JSONArray().also { array -> ordered.forEach(array::put) })
    }

    internal fun bluetoothSignalHistoryRowsFromStore(store: JSONObject, nowMs: Long = System.currentTimeMillis()): JSONArray {
        val records = store.optJSONArray("devices") ?: JSONArray()
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
                val lastSeenMs = (nowMs - lastBluetoothObservationTime(record)).coerceAtLeast(0L)
                add(
                    JSONObject()
                        .put("device_name", record.optString("device_name").ifBlank { "<unnamed>" })
                        .put("advertised_name", record.optString("advertised_name"))
                        .put("address", record.optString("address"))
                        .put("device_type", record.optString("device_type").ifBlank { "unknown" })
                        .put("device_category", record.optString("device_category").ifBlank { "unknown" })
                        .put("bond_state", record.optString("bond_state").ifBlank { "unknown" })
                        .put("connectable", record.opt("connectable") ?: JSONObject.NULL)
                        .put("service_uuids", record.optJSONArray("service_uuids") ?: JSONArray())
                        .put("service_labels", record.optJSONArray("service_labels") ?: JSONArray())
                        .put("service_data_uuids", record.optJSONArray("service_data_uuids") ?: JSONArray())
                        .put("service_data_labels", record.optJSONArray("service_data_labels") ?: JSONArray())
                        .put("manufacturer_ids", record.optJSONArray("manufacturer_ids") ?: JSONArray())
                        .put("manufacturer_names", record.optJSONArray("manufacturer_names") ?: JSONArray())
                        .put("semantic_context", record.optString("semantic_context"))
                        .put("sample_count", rssiValues.size)
                        .put("current_rssi_dbm", currentRssi)
                        .put("average_rssi_dbm", averageRssi)
                        .put("min_rssi_dbm", rssiValues.minOrNull() ?: currentRssi)
                        .put("max_rssi_dbm", rssiValues.maxOrNull() ?: currentRssi)
                        .put("trend_db", trendDb)
                        .put("trend_label", bluetoothSignalTrendLabel(trendDb))
                        .put("proximity_label", bluetoothProximityLabel(currentRssi))
                        .put("last_seen_ms", lastSeenMs)
                        .put("rssi_series", bluetoothObservationSeries(observations)),
                )
            }
        }
            .sortedWith(
                compareByDescending<JSONObject> { it.optInt("current_rssi_dbm", Int.MIN_VALUE) }
                    .thenByDescending { it.optInt("sample_count", 0) }
                    .thenBy { it.optString("device_name") },
            )
            .take(MAX_BLUETOOTH_HISTORY_ROWS)
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun bluetoothFilteredDeviceRows(devices: JSONArray, arguments: JSONObject): JSONArray {
        return bluetoothFilteredDeviceRowsForSpec(devices, bluetoothScanFilterSpec(arguments))
    }

    private fun bluetoothScanFilterSpec(arguments: JSONObject): BluetoothScanFilterSpec {
        return BluetoothScanFilterSpec(
            deviceNameQuery = jsonStringArgument(
                arguments,
                "filter_device_name",
                "filter_bluetooth_name",
                "device_name_filter",
                "bluetooth_name",
                "name",
            ).orEmpty(),
            addressQuery = jsonStringArgument(
                arguments,
                "filter_bluetooth_address",
                "filter_address",
                "address_filter",
                "bluetooth_address",
                "address",
            ).orEmpty(),
            serviceQuery = jsonStringArgument(
                arguments,
                "filter_bluetooth_service",
                "filter_service",
                "service_filter",
                "service_uuid",
                "service_label",
            ).orEmpty(),
            manufacturerQuery = jsonStringArgument(
                arguments,
                "filter_bluetooth_manufacturer",
                "filter_manufacturer",
                "manufacturer_filter",
                "manufacturer_id",
                "manufacturer_name",
            ).orEmpty(),
            categoryQuery = jsonStringArgument(
                arguments,
                "filter_bluetooth_category",
                "filter_category",
                "category_filter",
                "device_category",
            ).orEmpty(),
            proximityLabels = jsonStringListArgument(
                arguments,
                "filter_bluetooth_proximity",
                "filter_proximity",
                "proximity_filter",
                "proximity_label",
            ).mapNotNull(::normalizedBluetoothProximityFilter).toCollection(linkedSetOf()),
            minRssiDbm = intArgument(arguments, "min_rssi_dbm", "rssi_min_dbm", "minimum_rssi_dbm", "min_signal_dbm"),
            maxRssiDbm = intArgument(arguments, "max_rssi_dbm", "rssi_max_dbm", "maximum_rssi_dbm", "max_signal_dbm"),
        )
    }

    private fun bluetoothFilteredDeviceRowsForSpec(devices: JSONArray, filterSpec: BluetoothScanFilterSpec): JSONArray {
        if (!filterSpec.active) return devices
        return JSONArray().also { filtered ->
            for (index in 0 until devices.length()) {
                val row = devices.optJSONObject(index) ?: continue
                if (bluetoothDeviceMatchesFilterSpec(row, filterSpec)) filtered.put(row)
            }
        }
    }

    private fun bluetoothDeviceMatchesFilterSpec(row: JSONObject, filterSpec: BluetoothScanFilterSpec): Boolean {
        if (!bluetoothFilterTextMatches(
                filterSpec.deviceNameQuery,
                row.optString("device_name"),
                row.optString("advertised_name"),
            )
        ) {
            return false
        }
        if (!bluetoothFilterTextMatches(filterSpec.addressQuery, row.optString("address"))) {
            return false
        }
        if (!bluetoothFilterTextMatches(
                filterSpec.serviceQuery,
                row.optString("semantic_context"),
                *jsonStringList(row, "service_uuids").toTypedArray(),
                *jsonStringList(row, "service_labels").toTypedArray(),
                *jsonStringList(row, "service_data_uuids").toTypedArray(),
                *jsonStringList(row, "service_data_labels").toTypedArray(),
            )
        ) {
            return false
        }
        if (!bluetoothFilterTextMatches(
                filterSpec.manufacturerQuery,
                row.optString("semantic_context"),
                *jsonStringList(row, "manufacturer_ids").toTypedArray(),
                *jsonStringList(row, "manufacturer_names").toTypedArray(),
            )
        ) {
            return false
        }
        if (!bluetoothFilterTextMatches(
                filterSpec.categoryQuery,
                row.optString("device_category"),
                row.optString("device_type"),
                row.optString("device_class"),
                row.optString("major_device_class"),
            )
        ) {
            return false
        }

        val rssiDbm = jsonIntOrNull(row, "rssi_dbm")
        val proximityLabel = row.optString("proximity_label").ifBlank { rssiDbm?.let(::bluetoothProximityLabel).orEmpty() }
        if (filterSpec.proximityLabels.isNotEmpty() && proximityLabel !in filterSpec.proximityLabels) return false
        filterSpec.minRssiDbm?.let { minRssi -> if (rssiDbm == null || rssiDbm < minRssi) return false }
        filterSpec.maxRssiDbm?.let { maxRssi -> if (rssiDbm == null || rssiDbm > maxRssi) return false }

        return true
    }

    private fun bluetoothFilterSummaryJson(filterSpec: BluetoothScanFilterSpec, totalDeviceCount: Int, matchedDeviceCount: Int): JSONObject {
        return JSONObject()
            .put("active", filterSpec.active)
            .put("active_filter_count", filterSpec.activeFilterCount)
            .put("total_device_count", totalDeviceCount)
            .put("matched_device_count", matchedDeviceCount)
            .put("match_fraction", if (totalDeviceCount > 0) matchedDeviceCount.toDouble() / totalDeviceCount else 0.0)
            .put("requested_filters", bluetoothRequestedFilterJson(filterSpec))
            .put(
                "agent_usage",
                "Use these filters before answering Bluetooth questions that name a device, address, service UUID/label, manufacturer, category, proximity bucket, or RSSI threshold.",
            )
    }

    private fun bluetoothFilterApplicationRows(filterSpec: BluetoothScanFilterSpec, totalDeviceCount: Int, matchedDeviceCount: Int): JSONArray {
        if (!filterSpec.active) return JSONArray()
        val rows = JSONArray()
        fun addRow(key: String, label: String, valueLabel: String, detail: String) {
            rows.put(
                JSONObject()
                    .put("category", "bluetooth_filter")
                    .put("filter_key", key)
                    .put("label", label)
                    .put("ready", matchedDeviceCount > 0)
                    .put("value_label", valueLabel)
                    .put("detail", "$detail Matched $matchedDeviceCount of $totalDeviceCount Bluetooth row(s).")
                    .put(
                        "recommendation",
                        if (matchedDeviceCount > 0) {
                            "Use the filtered Bluetooth cards for the user's narrowed question, while keeping total device count visible for context."
                        } else {
                            "No Bluetooth rows matched this filter; relax the filter or request a resumed scan before concluding the device is absent."
                        },
                    )
                    .put("fraction", if (totalDeviceCount > 0) (matchedDeviceCount.toFloat() / totalDeviceCount).coerceIn(0.05f, 1f) else 0.05f),
            )
        }
        if (filterSpec.deviceNameQuery.isNotBlank()) addRow("device_name", "Device name contains", filterSpec.deviceNameQuery, "Included rows whose device or advertised name contains this text.")
        if (filterSpec.addressQuery.isNotBlank()) addRow("address", "Address contains", filterSpec.addressQuery, "Included rows whose Bluetooth address contains this text.")
        if (filterSpec.serviceQuery.isNotBlank()) addRow("service", "Service contains", filterSpec.serviceQuery, "Included rows whose service UUID, service label, service-data UUID, or semantic context contains this text.")
        if (filterSpec.manufacturerQuery.isNotBlank()) addRow("manufacturer", "Manufacturer contains", filterSpec.manufacturerQuery, "Included rows whose manufacturer ID, manufacturer name, or semantic context contains this text.")
        if (filterSpec.categoryQuery.isNotBlank()) addRow("category", "Category contains", filterSpec.categoryQuery, "Included rows whose Android class, device type, or category contains this text.")
        if (filterSpec.proximityLabels.isNotEmpty()) addRow("proximity", "Proximity filter", filterSpec.proximityLabels.joinToString(", "), "Included rows in the selected proximity bucket(s).")
        filterSpec.minRssiDbm?.let { addRow("min_rssi_dbm", "Minimum RSSI", "$it dBm", "Included Bluetooth rows at or above $it dBm.") }
        filterSpec.maxRssiDbm?.let { addRow("max_rssi_dbm", "Maximum RSSI", "$it dBm", "Included Bluetooth rows at or below $it dBm.") }
        return rows
    }

    private fun bluetoothRequestedFilterJson(filterSpec: BluetoothScanFilterSpec): JSONObject {
        return JSONObject()
            .put("device_name_query", if (filterSpec.deviceNameQuery.isBlank()) JSONObject.NULL else filterSpec.deviceNameQuery)
            .put("address_query", if (filterSpec.addressQuery.isBlank()) JSONObject.NULL else filterSpec.addressQuery)
            .put("service_query", if (filterSpec.serviceQuery.isBlank()) JSONObject.NULL else filterSpec.serviceQuery)
            .put("manufacturer_query", if (filterSpec.manufacturerQuery.isBlank()) JSONObject.NULL else filterSpec.manufacturerQuery)
            .put("category_query", if (filterSpec.categoryQuery.isBlank()) JSONObject.NULL else filterSpec.categoryQuery)
            .put("proximity_labels", JSONArray(filterSpec.proximityLabels.toList()))
            .put("min_rssi_dbm", filterSpec.minRssiDbm ?: JSONObject.NULL)
            .put("max_rssi_dbm", filterSpec.maxRssiDbm ?: JSONObject.NULL)
    }

    private fun bluetoothAnalyzerFilterSummaryJson(devices: JSONArray): JSONArray {
        val facets = linkedMapOf<String, BluetoothFilterFacetAccumulator>()
        fun addFacet(filterKey: String, label: String, row: JSONObject) {
            if (label.isBlank() || label == "unknown") return
            val key = "$filterKey|$label"
            val accumulator = facets.getOrPut(key) { BluetoothFilterFacetAccumulator(filterKey = filterKey, label = label) }
            accumulator.count += 1
            jsonIntOrNull(row, "rssi_dbm")?.let { rssi ->
                accumulator.strongestRssiDbm = maxOf(accumulator.strongestRssiDbm ?: rssi, rssi)
            }
        }
        for (index in 0 until devices.length()) {
            val row = devices.optJSONObject(index) ?: continue
            addFacet("category", row.optString("device_category"), row)
            addFacet("proximity", row.optString("proximity_label"), row)
            jsonStringList(row, "service_labels").forEach { addFacet("service", it, row) }
            jsonStringList(row, "manufacturer_names").forEach { addFacet("manufacturer", it, row) }
        }
        val rows = facets.values
            .sortedWith(
                compareBy<BluetoothFilterFacetAccumulator> { bluetoothFilterFacetSortKey(it.filterKey) }
                    .thenByDescending { it.count }
                    .thenBy { it.label },
            )
            .take(MAX_BLUETOOTH_METADATA_SUMMARY_ROWS)
            .map { accumulator ->
                JSONObject()
                    .put("category", "bluetooth_filter_facet")
                    .put("filter_key", accumulator.filterKey)
                    .put("label", accumulator.label)
                    .put("count", accumulator.count)
                    .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                    .put(
                        "recommendation",
                        "Use filter_${if (accumulator.filterKey == "category") "bluetooth_category" else "bluetooth_${accumulator.filterKey}"}=${accumulator.label} to narrow Bluetooth cards before explaining nearby-device evidence.",
                    )
            }
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    private fun normalizedBluetoothProximityFilter(value: String): String? {
        val normalized = value.trim().lowercase(Locale.US).replace("_", " ")
        if (normalized.isBlank() || normalized in setOf("all", "any", "*")) return null
        return when {
            normalized in setOf("immediate", "very near", "very close") -> "immediate"
            normalized in setOf("near", "close", "strong") -> "near"
            normalized in setOf("room", "medium", "moderate") -> "room"
            normalized in setOf("far", "distant", "weak") -> "far"
            else -> normalized
        }
    }

    private fun bluetoothFilterTextMatches(query: String, vararg values: String): Boolean {
        if (query.isBlank()) return true
        return values.any { value -> value.contains(query, ignoreCase = true) }
    }

    private fun bluetoothFilterFacetSortKey(filterKey: String): Int = when (filterKey) {
        "proximity" -> 0
        "category" -> 1
        "service" -> 2
        "manufacturer" -> 3
        else -> 4
    }

    internal fun mergeMotionSensorHistory(existing: JSONObject, samples: JSONArray, observedAtMs: Long): JSONObject {
        val records = linkedMapOf<String, JSONObject>()
        val existingRecords = existing.optJSONArray("sensors") ?: JSONArray()
        for (index in 0 until existingRecords.length()) {
            val record = existingRecords.optJSONObject(index) ?: continue
            val key = record.optString("key").ifBlank { motionSensorHistoryKey(record) }
            if (key.isNotBlank()) {
                records[key] = record.put("key", key)
            }
        }
        for (index in 0 until minOf(samples.length(), MAX_MOTION_HISTORY_SENSORS_PER_SAMPLE)) {
            val sample = samples.optJSONObject(index) ?: continue
            val sensorType = canonicalSensorType(sample.optString("sensor_type"))
            if (sensorType !in MOTION_HISTORY_SENSOR_TYPES) continue
            if (!sample.optBoolean("sampled", false) || !sample.optBoolean("available", true)) continue
            val values = sample.optJSONArray("values") ?: continue
            val magnitude = motionVectorMagnitude(values) ?: continue
            val key = motionSensorHistoryKey(sample)
            if (key.isBlank()) continue
            val record = records.getOrPut(key) { JSONObject().put("key", key) }
            record
                .put("sensor_type", sensorType)
                .put("sensor_label", sample.optString("sensor_label").ifBlank { SENSOR_TYPE_LABELS[sensorType] ?: sensorType })
                .put("sensor_name", sample.optString("sensor_name"))
                .put("vendor", sample.optString("vendor"))
                .put("unit", sample.optString("unit").ifBlank { unitForSensorType(sensorType) })
                .put("maximum_range", jsonValueOrNull(sample, "maximum_range"))
                .put("resolution", jsonValueOrNull(sample, "resolution"))
                .put("power_ma", jsonValueOrNull(sample, "power_ma"))
                .put("reporting_mode", sample.optString("reporting_mode"))
            val observations = record.optJSONArray("observations") ?: JSONArray()
            observations.put(
                JSONObject()
                    .put("observed_at_ms", observedAtMs)
                    .put("timestamp_nanos", jsonValueOrNull(sample, "timestamp_nanos"))
                    .put("magnitude", magnitude)
                    .put("values", copyJsonArray(values))
                    .put("accuracy", jsonValueOrNull(sample, "accuracy"))
                    .put("accuracy_label", sample.optString("accuracy_label").ifBlank { "unknown" }),
            )
            record.put("observations", trimMotionSensorObservations(observations))
        }
        val ordered = records.values
            .filter { record -> (record.optJSONArray("observations")?.length() ?: 0) > 0 }
            .sortedWith(
                compareByDescending<JSONObject> { lastMotionSensorObservationTime(it) }
                    .thenBy { motionSensorSortKey(it.optString("sensor_type")) }
                    .thenBy { it.optString("sensor_label") },
            )
            .take(MAX_MOTION_HISTORY_SENSORS)
        return JSONObject()
            .put("updated_at_ms", observedAtMs)
            .put("sensors", JSONArray().also { array -> ordered.forEach(array::put) })
    }

    internal fun motionSensorHistoryRowsFromStore(store: JSONObject, nowMs: Long = System.currentTimeMillis()): JSONArray {
        val records = store.optJSONArray("sensors") ?: JSONArray()
        val rows = buildList {
            for (index in 0 until records.length()) {
                val record = records.optJSONObject(index) ?: continue
                val observations = record.optJSONArray("observations") ?: continue
                val magnitudes = buildList {
                    for (sampleIndex in 0 until observations.length()) {
                        jsonDoubleOrNull(observations.optJSONObject(sampleIndex) ?: continue, "magnitude")?.let(::add)
                    }
                }
                if (magnitudes.isEmpty()) continue
                val firstMagnitude = magnitudes.first()
                val currentMagnitude = magnitudes.last()
                val averageMagnitude = magnitudes.average()
                val minMagnitude = magnitudes.minOrNull() ?: currentMagnitude
                val maxMagnitude = magnitudes.maxOrNull() ?: currentMagnitude
                val trendMagnitude = currentMagnitude - firstMagnitude
                val rangeMagnitude = maxMagnitude - minMagnitude
                val sensorType = record.optString("sensor_type").ifBlank { motionSensorHistoryKey(record) }
                val lastObservation = lastMotionSensorObservation(record)
                val lastSeenMs = (nowMs - lastMotionSensorObservationTime(record)).coerceAtLeast(0L)
                add(
                    JSONObject()
                        .put("sensor_type", sensorType)
                        .put("sensor_label", record.optString("sensor_label").ifBlank { SENSOR_TYPE_LABELS[sensorType] ?: sensorType })
                        .put("sensor_name", record.optString("sensor_name"))
                        .put("vendor", record.optString("vendor"))
                        .put("unit", record.optString("unit").ifBlank { unitForSensorType(sensorType) })
                        .put("magnitude_unit", record.optString("unit").ifBlank { unitForSensorType(sensorType) })
                        .put("sample_count", magnitudes.size)
                        .put("current_magnitude", currentMagnitude)
                        .put("average_magnitude", averageMagnitude)
                        .put("min_magnitude", minMagnitude)
                        .put("max_magnitude", maxMagnitude)
                        .put("trend_magnitude", trendMagnitude)
                        .put("trend_label", motionSensorTrendLabel(sensorType, trendMagnitude))
                        .put("stability_delta", rangeMagnitude)
                        .put("stability_label", motionSensorStabilityLabel(sensorType, rangeMagnitude, averageMagnitude))
                        .put("last_seen_ms", lastSeenMs)
                        .put("current_values", lastObservation?.optJSONArray("values") ?: JSONArray())
                        .put("accuracy_label", lastObservation?.optString("accuracy_label").orEmpty().ifBlank { "unknown" })
                        .put("magnitude_series", motionSensorObservationSeries(observations)),
                )
            }
        }
            .sortedWith(
                compareBy<JSONObject> { motionSensorSortKey(it.optString("sensor_type")) }
                    .thenBy { it.optLong("last_seen_ms", Long.MAX_VALUE) }
                    .thenByDescending { it.optInt("sample_count", 0) }
                    .thenBy { it.optString("sensor_label") },
            )
            .take(MAX_MOTION_HISTORY_ROWS)
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun motionPoseEstimateRows(samples: JSONArray, motionHistoryRows: JSONArray = JSONArray()): JSONArray {
        val vectors = latestMotionVectors(samples, motionHistoryRows)
        val rows = JSONArray()
        val rotationVector = vectors["rotation_vector"]?.takeIf { it.size >= 3 }
        val gravitySource = if (vectors.containsKey("gravity")) "gravity" else "accelerometer"
        val gravityVector = vectors["gravity"] ?: vectors["accelerometer"]
        val magneticVector = vectors["magnetic_field"]?.takeIf { it.size >= 3 }
        val pose = when {
            rotationVector != null -> poseFromRotationVector(rotationVector)
            gravityVector != null -> poseFromGravityMagnetic(gravitySource, gravityVector, magneticVector)
            else -> null
        }
        if (pose != null) {
            rows.put(
                JSONObject()
                    .put("pose_type", "device_pose")
                    .put("label", "Device pose estimate")
                    .put("value_label", pose.valueLabel)
                    .put("pose_label", pose.faceOrientationLabel)
                    .put("pose_source", pose.source)
                    .put("source_sensors", jsonStringArray(pose.sourceSensors))
                    .put("roll_degrees", roundedDegrees(pose.rollDegrees))
                    .put("pitch_degrees", roundedDegrees(pose.pitchDegrees))
                    .put("tilt_degrees", roundedDegrees(pose.tiltDegrees))
                    .put("azimuth_degrees", pose.azimuthDegrees?.let(::roundedDegrees) ?: JSONObject.NULL)
                    .put("heading_label", pose.headingLabel ?: JSONObject.NULL)
                    .put("face_orientation_label", pose.faceOrientationLabel)
                    .put("confidence_label", pose.confidenceLabel)
                    .put("workflow_hint", pose.workflowHint)
                    .put("fraction", pose.fraction),
            )
        }
        vectors["gyroscope"]?.takeIf { it.isNotEmpty() }?.let { gyro ->
            val angularVelocity = vectorMagnitude(gyro) ?: 0.0
            val motionState = angularMotionStateLabel(angularVelocity)
            rows.put(
                JSONObject()
                    .put("pose_type", "angular_motion")
                    .put("label", "Angular motion state")
                    .put("value_label", "${formatDecimal(angularVelocity, 2)} rad/s $motionState")
                    .put("pose_source", "gyroscope")
                    .put("source_sensors", jsonStringArray(listOf("gyroscope")))
                    .put("angular_velocity_rad_s", angularVelocity)
                    .put("motion_state_label", motionState)
                    .put("confidence_label", "medium")
                    .put("workflow_hint", "Use this row to decide whether orientation-sensitive work should wait for the phone to stop rotating.")
                    .put("fraction", (angularVelocity / 2.0).toFloat().coerceIn(0.08f, 1f)),
            )
        }
        val accelerationSource = if (vectors.containsKey("linear_acceleration")) "linear_acceleration" else "accelerometer"
        vectors[accelerationSource]?.takeIf { it.isNotEmpty() }?.let { acceleration ->
            val magnitude = vectorMagnitude(acceleration) ?: 0.0
            val deltaFromGravity = if (accelerationSource == "accelerometer") abs(magnitude - STANDARD_GRAVITY) else magnitude
            val motionState = accelerationMotionStateLabel(accelerationSource, deltaFromGravity)
            rows.put(
                JSONObject()
                    .put("pose_type", "acceleration_state")
                    .put("label", "Acceleration state")
                    .put("value_label", "${formatDecimal(deltaFromGravity, 2)} m/s^2 $motionState")
                    .put("pose_source", accelerationSource)
                    .put("source_sensors", jsonStringArray(listOf(accelerationSource)))
                    .put("acceleration_magnitude", magnitude)
                    .put("acceleration_delta_from_gravity", deltaFromGravity)
                    .put("motion_state_label", motionState)
                    .put("confidence_label", if (accelerationSource == "linear_acceleration") "high" else "medium")
                    .put("workflow_hint", "Use this row to distinguish steady handling from active movement before triggering motion-aware workflows.")
                    .put("fraction", (deltaFromGravity / 4.0).toFloat().coerceIn(0.08f, 1f)),
            )
        }
        return rows
    }

    private fun bluetoothDistinctStringCount(devices: JSONArray, key: String): Int {
        val values = linkedSetOf<String>()
        for (index in 0 until devices.length()) {
            val row = devices.optJSONObject(index) ?: continue
            values.addAll(jsonStringList(row, key))
        }
        return values.size
    }

    private fun bluetoothRssiDeviceCount(devices: JSONArray): Int {
        var count = 0
        for (index in 0 until devices.length()) {
            val row = devices.optJSONObject(index) ?: continue
            if (jsonIntOrNull(row, "rssi_dbm") != null) count += 1
        }
        return count
    }

    private fun bluetoothDistinctCategoryCount(devices: JSONArray): Int {
        val values = linkedSetOf<String>()
        for (index in 0 until devices.length()) {
            val row = devices.optJSONObject(index) ?: continue
            row.optString("device_category")
                .ifBlank { row.optString("major_device_class") }
                .ifBlank { row.optString("device_type") }
                .takeIf { it.isNotBlank() && it != "unknown" }
                ?.let(values::add)
        }
        return values.size
    }

    private fun normalizedBluetoothScanMode(arguments: JSONObject): String {
        val requested = arguments.optString("scan_mode").ifBlank {
            arguments.optString("bluetooth_scan_mode").ifBlank {
                arguments.optString("scan_control")
            }
        }
        return when (requested.trim().lowercase(Locale.US)) {
            "pause", "paused", "passive", "cached", "hold", "stop" -> BLUETOOTH_SCAN_MODE_PAUSED
            "resume", "resumed", "active", "fresh", "refresh", "live" -> BLUETOOTH_SCAN_MODE_RESUMED
            else -> BLUETOOTH_SCAN_MODE_AUTO
        }
    }

    private fun effectiveBluetoothRefreshRequested(refreshRequested: Boolean, scanMode: String): Boolean {
        return when (scanMode) {
            BLUETOOTH_SCAN_MODE_PAUSED -> false
            BLUETOOTH_SCAN_MODE_RESUMED -> true
            else -> refreshRequested
        }
    }

    private fun bluetoothScanControlJson(
        scanMode: String,
        userRefreshRequested: Boolean,
        effectiveRefreshRequested: Boolean,
        refreshAccepted: Boolean,
    ): JSONObject {
        return JSONObject()
            .put("scan_mode", scanMode)
            .put("pause_resume_supported", true)
            .put("user_refresh_requested", userRefreshRequested)
            .put("effective_refresh_requested", effectiveRefreshRequested)
            .put("refresh_suppressed_by_pause", userRefreshRequested && !effectiveRefreshRequested && scanMode == BLUETOOTH_SCAN_MODE_PAUSED)
            .put("refresh_accepted", refreshAccepted)
            .put("paused_uses_cached_scan_results", scanMode == BLUETOOTH_SCAN_MODE_PAUSED)
            .put("resumed_requests_active_scan", scanMode == BLUETOOTH_SCAN_MODE_RESUMED)
            .put("android_scope", "Per diagnostic request; Android exposes paired devices and cached Hermes history, while active BLE scans require permission and may be sparse.")
            .put("agent_instruction", bluetoothScanModeInstruction(scanMode, userRefreshRequested, effectiveRefreshRequested, refreshAccepted))
    }

    private fun bluetoothScanModeInstruction(
        scanMode: String,
        userRefreshRequested: Boolean,
        effectiveRefreshRequested: Boolean,
        refreshAccepted: Boolean,
    ): String {
        return when {
            scanMode == BLUETOOTH_SCAN_MODE_PAUSED && userRefreshRequested ->
                "Active Bluetooth scan was paused; explain paired rows and cached RSSI history instead of scanning again."
            scanMode == BLUETOOTH_SCAN_MODE_PAUSED ->
                "Use paired Bluetooth rows and cached RSSI history while scan mode is paused."
            scanMode == BLUETOOTH_SCAN_MODE_RESUMED && refreshAccepted ->
                "Android accepted the resumed BLE scan request; read RSSI, service, and manufacturer rows before advising."
            scanMode == BLUETOOTH_SCAN_MODE_RESUMED && effectiveRefreshRequested ->
                "A resumed BLE scan was requested; Android may still return only paired or sparse advertisement rows."
            else ->
                "Auto mode follows the refresh argument and reports Bluetooth permission, enablement, and sparse-advertisement limits honestly."
        }
    }

    private fun bluetoothScanStatusJson(
        bluetoothAvailable: Boolean,
        bluetoothLeSupported: Boolean,
        bluetoothEnabled: Boolean,
        permissionStatus: JSONObject,
        scanResult: JSONObject?,
        returnedDeviceCount: Int,
        metadataCount: Int,
        serviceUuidCount: Int,
        manufacturerIdCount: Int,
        rssiDeviceCount: Int,
    ): JSONObject {
        val scanError = when {
            scanResult == null -> JSONObject.NULL
            scanResult.has("scan_error") -> scanResult.opt("scan_error") ?: JSONObject.NULL
            scanResult.has("error") -> scanResult.optString("error")
            else -> JSONObject.NULL
        }
        val refreshRequested = scanResult?.optBoolean("refresh_requested", false) ?: false
        val effectiveRefreshRequested = scanResult?.optBoolean("effective_refresh_requested", refreshRequested) ?: false
        val scanMode = scanResult?.optString("bluetooth_scan_mode").orEmpty().ifBlank { BLUETOOTH_SCAN_MODE_AUTO }
        return JSONObject()
            .put("bluetooth_available", bluetoothAvailable)
            .put("bluetooth_le_supported", bluetoothLeSupported)
            .put("bluetooth_enabled", bluetoothEnabled)
            .put("refresh_requested", refreshRequested)
            .put("effective_refresh_requested", effectiveRefreshRequested)
            .put("refresh_accepted", scanResult?.optBoolean("refresh_accepted", false) ?: false)
            .put("bluetooth_scan_mode", scanMode)
            .put(
                "bluetooth_scan_control",
                scanResult?.optJSONObject("bluetooth_scan_control")
                    ?: bluetoothScanControlJson(BLUETOOTH_SCAN_MODE_AUTO, false, false, false),
            )
            .put("scan_error", scanError)
            .put("can_read_paired_devices", permissionStatus.optBoolean("can_read_paired_devices", false))
            .put("can_scan_nearby_devices", permissionStatus.optBoolean("can_scan_nearby_devices", false))
            .put("returned_device_count", returnedDeviceCount)
            .put("metadata_summary_count", metadataCount)
            .put("rssi_device_count", rssiDeviceCount)
            .put("service_uuid_count", serviceUuidCount)
            .put("manufacturer_id_count", manufacturerIdCount)
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
        if (isRobolectricRuntime()) return false
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

    internal fun isRobolectricRuntime(
        fingerprint: String = Build.FINGERPRINT.orEmpty(),
        model: String = Build.MODEL.orEmpty(),
        manufacturer: String = Build.MANUFACTURER.orEmpty(),
        brand: String = Build.BRAND.orEmpty(),
        device: String = Build.DEVICE.orEmpty(),
        product: String = Build.PRODUCT.orEmpty(),
    ): Boolean {
        val buildValues = listOf(fingerprint, model, manufacturer, brand, device, product)
            .map { it.lowercase(Locale.US) }
        val hasRobolectricClass = runCatching {
            Class.forName(
                "org.robolectric.RuntimeEnvironment",
                false,
                HermesDeviceDiagnosticsBridge::class.java.classLoader,
            )
        }.isSuccess
        return hasRobolectricClass || buildValues.any { it.contains("robolectric") }
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

    private fun motionPoseDefaultArguments(arguments: JSONObject): JSONObject {
        val copy = JSONObject(arguments.toString())
        if (!copy.has("sensor_types") && !copy.has("sensors") && !copy.has("sensor_type")) {
            copy.put("sensor_types", DEFAULT_MOTION_POSE_SENSOR_TYPES.joinToString(","))
        }
        return copy
    }

    private fun requestedMotionHistorySensorTypes(arguments: JSONObject): List<String> {
        return if (arguments.has("sensor_types") || arguments.has("sensors") || arguments.has("sensor_type")) {
            requestedSensorTypes(arguments)
                .filter { it in MOTION_HISTORY_SENSOR_TYPES }
                .ifEmpty { DEFAULT_MOTION_HISTORY_SENSOR_TYPES }
                .take(MAX_MOTION_HISTORY_SENSORS_PER_SAMPLE)
        } else {
            DEFAULT_MOTION_HISTORY_SENSOR_TYPES
        }
    }

    private fun sensorAnalyzerRequestedSensorTypes(arguments: JSONObject): List<String> {
        return if (arguments.has("sensor_types") || arguments.has("sensors") || arguments.has("sensor_type")) {
            requestedSensorTypes(arguments)
        } else {
            SENSOR_TYPE_LABELS.keys.toList()
        }
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

    internal fun wifiChannelUtilizationRowsForNetworks(networks: JSONArray): JSONArray {
        data class ChannelSample(
            val ssid: String,
            val channel: Int,
            val band: String,
            val rssiDbm: Int,
            val widthLabel: String,
            val securityMode: String,
            val frequencyMhz: Int,
        )

        val samples = buildList {
            for (index in 0 until networks.length()) {
                val row = networks.optJSONObject(index) ?: continue
                val frequencyMhz = jsonIntOrNull(row, "frequency_mhz") ?: 0
                val channel = jsonIntOrNull(row, "channel")
                    ?: channelForFrequencyMhz(frequencyMhz)
                    ?: continue
                val rssiDbm = jsonIntOrNull(row, "rssi_dbm") ?: continue
                val band = canonicalWifiBandLabel(row.optString("band"), frequencyMhz)
                if (band == "unknown") continue
                add(
                    ChannelSample(
                        ssid = row.optString("display_ssid").ifBlank { row.optString("ssid").ifBlank { "<hidden>" } },
                        channel = channel,
                        band = band,
                        rssiDbm = rssiDbm,
                        widthLabel = row.optString("channel_width").ifBlank { "20MHz" },
                        securityMode = row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) },
                        frequencyMhz = frequencyMhz,
                    ),
                )
            }
        }
        if (samples.isEmpty()) return JSONArray()

        val rows = samples
            .map { it.band to it.channel }
            .distinct()
            .map { (band, channel) ->
                val overlapping = samples.filter { sample ->
                    sample.band == band && wifiChannelOverlapWeight(band, channel, sample.channel) > 0.0
                }
                val sameChannel = overlapping.filter { it.channel == channel }
                val strongestRssi = overlapping.maxOfOrNull { it.rssiDbm }
                val averageRssi = if (overlapping.isNotEmpty()) {
                    (overlapping.sumOf { it.rssiDbm }.toDouble() / overlapping.size).roundToInt()
                } else {
                    null
                }
                val pressureScore = overlapping.sumOf { sample ->
                    val signalWeight = ((sample.rssiDbm + 100).coerceIn(0, 70)) / 70.0
                    val overlapWeight = wifiChannelOverlapWeight(band, channel, sample.channel)
                    signalWeight * overlapWeight * wifiWidthWeight(sample.widthLabel) * 45.0
                }.roundToInt().coerceIn(0, 100)
                val maxWidthMhz = overlapping.mapNotNull { channelWidthMhz(it.widthLabel) }.maxOrNull()
                val wideChannelCount = overlapping.count { (channelWidthMhz(it.widthLabel) ?: 20) >= 80 }
                val securityModes = overlapping.map { it.securityMode.ifBlank { "unknown" } }.distinct().take(MAX_WIFI_SUMMARY_SAMPLES)
                val sampleSsids = sameChannel.map { it.ssid }.distinct().take(MAX_WIFI_SUMMARY_SAMPLES)
                val utilizationLabel = wifiChannelUtilizationLabel(pressureScore)
                JSONObject()
                    .put("band", band)
                    .put("channel", channel)
                    .put("frequency_hint_mhz", frequencyHintMhzForWifiChannel(band, channel) ?: sameChannel.firstOrNull()?.frequencyMhz ?: JSONObject.NULL)
                    .put("network_count", sameChannel.size)
                    .put("overlap_count", overlapping.size)
                    .put("strongest_rssi_dbm", strongestRssi ?: JSONObject.NULL)
                    .put("average_rssi_dbm", averageRssi ?: JSONObject.NULL)
                    .put("channel_pressure_score", pressureScore)
                    .put("utilization_label", utilizationLabel)
                    .put("max_channel_width_mhz", maxWidthMhz ?: JSONObject.NULL)
                    .put("wide_channel_count", wideChannelCount)
                    .put("security_modes", JSONArray(securityModes))
                    .put("sample_ssids", JSONArray(sampleSsids))
                    .put(
                        "recommendation",
                        wifiChannelUtilizationRecommendation(
                            score = pressureScore,
                            sameChannelCount = sameChannel.size,
                            overlapCount = overlapping.size,
                            wideChannelCount = wideChannelCount,
                            strongestRssi = strongestRssi,
                        ),
                    )
            }
            .sortedWith(
                compareBy<JSONObject> { wifiBandSortKey(it.optString("band")) }
                    .thenByDescending { it.optInt("channel_pressure_score") }
                    .thenBy { it.optInt("channel") },
            )
            .take(MAX_WIFI_CHANNEL_UTILIZATION_ROWS)

        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun wifiChannelGraphRows(networks: JSONArray, detailLimit: Int = MAX_WIFI_RESULTS): JSONArray {
        data class ChannelGraphSample(
            val rank: Int,
            val ssid: String,
            val bssid: String,
            val band: String,
            val channel: Int,
            val frequencyMhz: Int?,
            val rssiDbm: Int,
            val channelWidth: String,
            val channelWidthMhz: Int,
            val spanStart: Int,
            val spanEnd: Int,
            val securityMode: String,
            val vendor: String,
            val distanceMeters: Double?,
        )

        val details = wifiAccessPointDetailRows(networks, MAX_WIFI_RESULTS)
        val samples = buildList {
            for (index in 0 until details.length()) {
                val row = details.optJSONObject(index) ?: continue
                val frequencyMhz = jsonIntOrNull(row, "frequency_mhz")
                val channel = jsonIntOrNull(row, "channel")
                    ?: frequencyMhz?.let(::channelForFrequencyMhz)
                    ?: continue
                val rssiDbm = jsonIntOrNull(row, "rssi_dbm") ?: continue
                val band = canonicalWifiBandLabel(row.optString("band"), frequencyMhz ?: 0)
                if (band == "unknown") continue
                val widthLabel = row.optString("channel_width").ifBlank { "20MHz" }
                val widthMhz = (jsonIntOrNull(row, "channel_width_mhz") ?: channelWidthMhz(widthLabel) ?: 20)
                    .coerceAtLeast(20)
                val halfSpan = wifiChannelGraphHalfSpan(widthMhz)
                add(
                    ChannelGraphSample(
                        rank = row.optInt("rank", index + 1),
                        ssid = row.optString("display_ssid").ifBlank { row.optString("ssid").ifBlank { "<hidden>" } },
                        bssid = row.optString("bssid"),
                        band = band,
                        channel = channel,
                        frequencyMhz = frequencyMhz,
                        rssiDbm = rssiDbm,
                        channelWidth = widthLabel,
                        channelWidthMhz = widthMhz,
                        spanStart = channel - halfSpan,
                        spanEnd = channel + halfSpan,
                        securityMode = row.optString("security_mode").ifBlank { "unknown" },
                        vendor = row.optString("bssid_vendor").ifBlank { "Unknown vendor" },
                        distanceMeters = jsonDoubleOrNull(row, "estimated_distance_m")
                            ?: jsonDoubleOrNull(row, "estimated_distance_meters"),
                    ),
                )
            }
        }
        val rows = samples
            .sortedWith(
                compareBy<ChannelGraphSample> { wifiBandSortKey(it.band) }
                    .thenBy { it.channel }
                    .thenByDescending { it.rssiDbm },
            )
            .take(detailLimit.coerceIn(1, MAX_WIFI_CHANNEL_GRAPH_ROWS))
            .mapIndexed { outputIndex, sample ->
                val overlapping = samples.filter { other ->
                    !(other.rank == sample.rank && other.bssid == sample.bssid) &&
                        other.band == sample.band &&
                        wifiChannelGraphSpanOverlap(sample.spanStart, sample.spanEnd, other.spanStart, other.spanEnd) > 0
                }
                val sameChannelCount = overlapping.count { it.channel == sample.channel }
                val sampleWidth = (sample.spanEnd - sample.spanStart + 1).coerceAtLeast(1)
                val overlapPressure = overlapping.sumOf { other ->
                    val overlapWidth = wifiChannelGraphSpanOverlap(sample.spanStart, sample.spanEnd, other.spanStart, other.spanEnd)
                    val overlapFraction = overlapWidth.toDouble() / sampleWidth.toDouble()
                    val signalWeight = ((other.rssiDbm + 100).coerceIn(0, 70)) / 70.0
                    overlapFraction * signalWeight * wifiWidthWeight(other.channelWidth) * 70.0
                }.roundToInt().coerceIn(0, 100)
                val frequencyHalfWidthMhz = sample.channelWidthMhz / 2
                JSONObject()
                    .put("rank", outputIndex + 1)
                    .put("source_rank", sample.rank)
                    .put("display_ssid", sample.ssid)
                    .put("ssid", sample.ssid)
                    .put("bssid", sample.bssid)
                    .put("band", sample.band)
                    .put("channel", sample.channel)
                    .put("graph_x_channel", sample.channel)
                    .put("rssi_dbm", sample.rssiDbm)
                    .put("graph_y_dbm", sample.rssiDbm)
                    .put("signal_quality", wifiSignalQualityLabel(sample.rssiDbm))
                    .put("frequency_mhz", sample.frequencyMhz ?: JSONObject.NULL)
                    .put("frequency_span_start_mhz", sample.frequencyMhz?.minus(frequencyHalfWidthMhz) ?: JSONObject.NULL)
                    .put("frequency_span_end_mhz", sample.frequencyMhz?.plus(frequencyHalfWidthMhz) ?: JSONObject.NULL)
                    .put("channel_width", sample.channelWidth)
                    .put("channel_width_mhz", sample.channelWidthMhz)
                    .put("graph_width_channels", (sample.spanEnd - sample.spanStart + 1).coerceAtLeast(1))
                    .put("channel_span_start", sample.spanStart)
                    .put("channel_span_end", sample.spanEnd)
                    .put("overlap_network_count", overlapping.size)
                    .put("same_channel_network_count", sameChannelCount)
                    .put("overlap_pressure_score", overlapPressure)
                    .put("overlap_sample_ssids", JSONArray(overlapping.sortedByDescending { it.rssiDbm }.map { it.ssid }.distinct().take(MAX_WIFI_SUMMARY_SAMPLES)))
                    .put("security_mode", sample.securityMode)
                    .put("bssid_vendor", sample.vendor)
                    .put("estimated_distance_m", sample.distanceMeters ?: JSONObject.NULL)
                    .put("graph_shape", "channel_width_envelope")
                    .put("recommendation", wifiChannelGraphRecommendation(overlapping.size, overlapPressure, sample.channelWidthMhz, sample.rssiDbm))
            }
        return JSONArray().also { array -> rows.forEach(array::put) }
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

    internal fun wifiFilteredNetworkRows(networks: JSONArray, arguments: JSONObject): JSONArray {
        return wifiFilteredNetworkRows(networks, wifiScanFilterSpec(arguments))
    }

    private fun wifiScanFilterSpec(arguments: JSONObject): WifiScanFilterSpec {
        val includeHidden = booleanArgument(arguments, "include_hidden", "include_hidden_ssid", "show_hidden_ssid")
        return WifiScanFilterSpec(
            bands = jsonStringListArgument(
                arguments,
                "filter_band",
                "band_filter",
                "wifi_band_filter",
                "wifi_band",
                "band",
                "bands",
            ).mapNotNull(::normalizedWifiBandFilter).toCollection(linkedSetOf()),
            securityModes = jsonStringListArgument(
                arguments,
                "filter_security",
                "security_filter",
                "wifi_security_filter",
                "security_mode",
                "security",
            ).mapNotNull(::normalizedWifiSecurityFilter).toCollection(linkedSetOf()),
            signalQualities = jsonStringListArgument(
                arguments,
                "filter_signal",
                "filter_signal_strength",
                "signal_filter",
                "signal_strength",
                "signal_quality",
            ).mapNotNull(::normalizedWifiSignalFilter).toCollection(linkedSetOf()),
            ssidQuery = jsonStringArgument(arguments, "filter_ssid", "ssid_filter", "ssid_query", "ssid_contains", "ssid").orEmpty(),
            bssidQuery = jsonStringArgument(arguments, "filter_bssid", "bssid_filter", "bssid_query", "bssid_contains", "bssid").orEmpty(),
            vendorQuery = jsonStringArgument(arguments, "filter_vendor", "vendor_filter", "vendor_query", "vendor_contains", "bssid_vendor").orEmpty(),
            minRssiDbm = intArgument(arguments, "min_rssi_dbm", "rssi_min_dbm", "minimum_rssi_dbm", "min_signal_dbm"),
            maxRssiDbm = intArgument(arguments, "max_rssi_dbm", "rssi_max_dbm", "maximum_rssi_dbm", "max_signal_dbm"),
            excludeHidden = includeHidden == false || booleanArgument(arguments, "exclude_hidden", "exclude_hidden_ssid") == true,
            hiddenOnly = booleanArgument(arguments, "hidden_only", "only_hidden", "filter_hidden_only") == true,
        )
    }

    private fun wifiFilteredNetworkRows(networks: JSONArray, filterSpec: WifiScanFilterSpec): JSONArray {
        if (!filterSpec.active) return networks
        return JSONArray().also { filtered ->
            for (index in 0 until networks.length()) {
                val row = networks.optJSONObject(index) ?: continue
                if (wifiNetworkMatchesFilterSpec(row, filterSpec)) filtered.put(row)
            }
        }
    }

    private fun wifiNetworkMatchesFilterSpec(row: JSONObject, filterSpec: WifiScanFilterSpec): Boolean {
        val frequencyMhz = jsonIntOrNull(row, "frequency_mhz") ?: 0
        val band = canonicalWifiBandLabel(row.optString("band"), frequencyMhz)
        if (filterSpec.bands.isNotEmpty() && band !in filterSpec.bands) return false

        val securityMode = row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) }
        if (filterSpec.securityModes.isNotEmpty() && filterSpec.securityModes.none { it.equals(securityMode, ignoreCase = true) }) {
            return false
        }

        val rssiDbm = jsonIntOrNull(row, "rssi_dbm")
        val signalQuality = rssiDbm?.let(::wifiSignalQualityLabel) ?: row.optString("signal_quality").ifBlank { "unknown" }
        if (filterSpec.signalQualities.isNotEmpty() && filterSpec.signalQualities.none { it.equals(signalQuality, ignoreCase = true) }) {
            return false
        }
        filterSpec.minRssiDbm?.let { minRssi -> if (rssiDbm == null || rssiDbm < minRssi) return false }
        filterSpec.maxRssiDbm?.let { maxRssi -> if (rssiDbm == null || rssiDbm > maxRssi) return false }

        val displaySsid = row.optString("display_ssid").ifBlank { row.optString("ssid") }
        val hidden = row.optBoolean("hidden_ssid", displaySsid.isBlank() || displaySsid == "<hidden>")
        if (filterSpec.hiddenOnly && !hidden) return false
        if (filterSpec.excludeHidden && hidden) return false

        if (!wifiFilterTextMatches(displaySsid, filterSpec.ssidQuery)) return false
        if (!wifiFilterTextMatches(row.optString("bssid"), filterSpec.bssidQuery)) return false
        val vendor = row.optString("bssid_vendor").ifBlank {
            wifiOuiVendorLabel(row.optString("bssid_oui").ifBlank { wifiBssidOui(row.optString("bssid")) })
        }
        if (!wifiFilterTextMatches(vendor, filterSpec.vendorQuery)) return false

        return true
    }

    private fun wifiFilterSummaryJson(filterSpec: WifiScanFilterSpec, totalNetworkCount: Int, matchedNetworkCount: Int): JSONObject {
        return JSONObject()
            .put("active", filterSpec.active)
            .put("active_filter_count", filterSpec.activeFilterCount)
            .put("total_network_count", totalNetworkCount)
            .put("matched_network_count", matchedNetworkCount)
            .put("match_fraction", if (totalNetworkCount > 0) matchedNetworkCount.toDouble() / totalNetworkCount else 0.0)
            .put("requested_filters", wifiRequestedFilterJson(filterSpec))
            .put(
                "agent_usage",
                "Use these filters before answering Wi-Fi questions that name a band, SSID, security mode, vendor, hidden-SSID handling, or RSSI threshold.",
            )
    }

    private fun wifiFilterApplicationRows(filterSpec: WifiScanFilterSpec, totalNetworkCount: Int, matchedNetworkCount: Int): JSONArray {
        if (!filterSpec.active) return JSONArray()
        val rows = JSONArray()
        fun addRow(key: String, label: String, valueLabel: String, detail: String) {
            rows.put(
                JSONObject()
                    .put("category", "wifi_filter")
                    .put("filter_key", key)
                    .put("label", label)
                    .put("ready", matchedNetworkCount > 0)
                    .put("value_label", valueLabel)
                    .put("detail", "$detail Matched $matchedNetworkCount of $totalNetworkCount visible AP row(s).")
                    .put(
                        "recommendation",
                        if (matchedNetworkCount > 0) {
                            "Use the filtered Wi-Fi cards for the user's narrowed question, but keep total_scan_result_count visible for context."
                        } else {
                            "No AP rows matched this filter; relax the filter or request a fresh scan before concluding the network is absent."
                        },
                    )
                    .put("fraction", if (totalNetworkCount > 0) (matchedNetworkCount.toFloat() / totalNetworkCount).coerceIn(0.05f, 1f) else 0.05f),
            )
        }
        if (filterSpec.bands.isNotEmpty()) {
            addRow("band", "Band filter", filterSpec.bands.joinToString(", "), "Included Wi-Fi bands: ${filterSpec.bands.joinToString(", ")}.")
        }
        if (filterSpec.securityModes.isNotEmpty()) {
            addRow("security", "Security filter", filterSpec.securityModes.joinToString(", "), "Included security modes: ${filterSpec.securityModes.joinToString(", ")}.")
        }
        if (filterSpec.signalQualities.isNotEmpty()) {
            addRow("signal_strength", "Signal filter", filterSpec.signalQualities.joinToString(", "), "Included signal quality buckets: ${filterSpec.signalQualities.joinToString(", ")}.")
        }
        filterSpec.minRssiDbm?.let { addRow("min_rssi_dbm", "Minimum RSSI", "$it dBm", "Included APs at or above $it dBm.") }
        filterSpec.maxRssiDbm?.let { addRow("max_rssi_dbm", "Maximum RSSI", "$it dBm", "Included APs at or below $it dBm.") }
        if (filterSpec.ssidQuery.isNotBlank()) {
            addRow("ssid", "SSID contains", filterSpec.ssidQuery, "Included APs whose SSID/display SSID contains this text.")
        }
        if (filterSpec.bssidQuery.isNotBlank()) {
            addRow("bssid", "BSSID contains", filterSpec.bssidQuery, "Included APs whose BSSID contains this text.")
        }
        if (filterSpec.vendorQuery.isNotBlank()) {
            addRow("vendor", "Vendor contains", filterSpec.vendorQuery, "Included APs whose local OUI/vendor label contains this text.")
        }
        if (filterSpec.hiddenOnly) {
            addRow("hidden_only", "Hidden SSIDs only", "hidden only", "Included only rows marked hidden by Android scan metadata.")
        } else if (filterSpec.excludeHidden) {
            addRow("include_hidden", "Hidden SSIDs excluded", "exclude hidden", "Excluded rows marked hidden by Android scan metadata.")
        }
        return rows
    }

    private fun wifiRequestedFilterJson(filterSpec: WifiScanFilterSpec): JSONObject {
        return JSONObject()
            .put("bands", JSONArray(filterSpec.bands.toList()))
            .put("security_modes", JSONArray(filterSpec.securityModes.toList()))
            .put("signal_qualities", JSONArray(filterSpec.signalQualities.toList()))
            .put("ssid_query", if (filterSpec.ssidQuery.isBlank()) JSONObject.NULL else filterSpec.ssidQuery)
            .put("bssid_query", if (filterSpec.bssidQuery.isBlank()) JSONObject.NULL else filterSpec.bssidQuery)
            .put("vendor_query", if (filterSpec.vendorQuery.isBlank()) JSONObject.NULL else filterSpec.vendorQuery)
            .put("min_rssi_dbm", filterSpec.minRssiDbm ?: JSONObject.NULL)
            .put("max_rssi_dbm", filterSpec.maxRssiDbm ?: JSONObject.NULL)
            .put("exclude_hidden", filterSpec.excludeHidden)
            .put("hidden_only", filterSpec.hiddenOnly)
    }

    private fun normalizedWifiBandFilter(value: String): String? {
        val normalized = value.trim().lowercase(Locale.US).replace(" ", "")
        if (normalized.isBlank() || normalized in setOf("all", "any", "*")) return null
        return when {
            normalized.contains("2.4") || normalized == "24ghz" || normalized == "2g" || normalized == "2ghz" -> "2.4GHz"
            normalized.startsWith("5") -> "5GHz"
            normalized.startsWith("6") || normalized == "6e" -> "6GHz"
            normalized.startsWith("60") -> "60GHz"
            else -> canonicalWifiBandLabel(value).takeIf { it != "unknown" }
        }
    }

    private fun normalizedWifiSecurityFilter(value: String): String? {
        val normalized = value.trim().lowercase(Locale.US).replace("_", " ")
        if (normalized.isBlank() || normalized in setOf("all", "any", "*")) return null
        return when {
            "wpa3" in normalized || "sae" in normalized -> "WPA3"
            "wpa2" in normalized || "rsn" in normalized -> "WPA2"
            normalized == "wpa" || normalized.startsWith("wpa ") -> "WPA"
            "enhanced" in normalized || "owe" in normalized -> "Enhanced Open"
            "wep" in normalized -> "WEP"
            "open" in normalized -> "Open"
            else -> value.trim()
        }
    }

    private fun normalizedWifiSignalFilter(value: String): String? {
        val normalized = value.trim().lowercase(Locale.US).replace("_", " ")
        if (normalized.isBlank() || normalized in setOf("all", "any", "*")) return null
        return when {
            normalized in setOf("excellent", "strong", "very strong") -> "excellent"
            normalized == "good" -> "good"
            normalized in setOf("fair", "medium", "moderate") -> "fair"
            normalized in setOf("weak", "poor", "low") -> "weak"
            else -> normalized
        }
    }

    private fun wifiFilterTextMatches(value: String, query: String): Boolean {
        return query.isBlank() || value.contains(query, ignoreCase = true)
    }

    internal fun wifiAccessPointSemanticRows(details: JSONArray, limit: Int = MAX_WIFI_RESULTS): JSONArray {
        val rows = buildList {
            for (index in 0 until details.length()) {
                val row = details.optJSONObject(index) ?: continue
                val semanticRow = appendWifiSemanticFields(JSONObject(row.toString()))
                add(
                    JSONObject()
                        .put("rank", semanticRow.optInt("rank", index + 1))
                        .put("display_ssid", semanticRow.optString("display_ssid").ifBlank { semanticRow.optString("ssid").ifBlank { "<hidden>" } })
                        .put("ssid", semanticRow.optString("ssid").ifBlank { semanticRow.optString("display_ssid").ifBlank { "<hidden>" } })
                        .put("hidden_ssid", semanticRow.optBoolean("hidden_ssid", false))
                        .put("bssid", semanticRow.optString("bssid"))
                        .put("bssid_vendor", semanticRow.optString("bssid_vendor").ifBlank { "Unknown vendor" })
                        .put("rssi_dbm", jsonValueOrNull(semanticRow, "rssi_dbm"))
                        .put("signal_quality", semanticRow.optString("signal_quality").ifBlank { "unknown" })
                        .put("frequency_mhz", jsonValueOrNull(semanticRow, "frequency_mhz"))
                        .put("channel", jsonValueOrNull(semanticRow, "channel"))
                        .put("band", semanticRow.optString("band").ifBlank { "unknown" })
                        .put("channel_width", semanticRow.optString("channel_width").ifBlank { "unknown" })
                        .put("wifi_standard", semanticRow.optString("wifi_standard").ifBlank { "unknown" })
                        .put("security_mode", semanticRow.optString("security_mode").ifBlank { "unknown" })
                        .put("security_risk_label", semanticRow.optString("security_risk_label").ifBlank { "unknown_security" })
                        .put("semantic_label", semanticRow.optString("semantic_label").ifBlank { "nearby AP" })
                        .put("semantic_tags", semanticRow.optJSONArray("semantic_tags") ?: JSONArray())
                        .put("estimated_distance_m", jsonValueOrNull(semanticRow, "estimated_distance_m"))
                        .put("passpoint_network", jsonValueOrNull(semanticRow, "passpoint_network"))
                        .put("80211mc_responder", jsonValueOrNull(semanticRow, "80211mc_responder"))
                        .put("recommendation", semanticRow.optString("semantic_recommendation").ifBlank { wifiAccessPointSemanticRecommendation(semanticRow) })
                        .put("agent_usage", "Use this row when Gemma needs to explain what a nearby AP likely is, whether it deserves security attention, and which Wi-Fi diagnostic action should follow."),
                )
            }
        }
            .sortedWith(
                compareByDescending<JSONObject> { wifiSemanticAttentionScore(it) }
                    .thenByDescending { jsonIntOrNull(it, "rssi_dbm") ?: Int.MIN_VALUE }
                    .thenBy { it.optString("display_ssid") },
            )
            .take(limit.coerceIn(1, MAX_WIFI_RESULTS))
        return JSONArray().also { array -> rows.forEach(array::put) }
    }

    internal fun wifiBandCoverageRows(networks: JSONArray, bandSummary: JSONArray = JSONArray(), ratings: JSONArray = JSONArray()): JSONArray {
        val accumulators = linkedMapOf<String, WifiBandCoverageAccumulator>()
        WIFI_BAND_COVERAGE_BASE_BANDS.forEach { band -> accumulators[band] = WifiBandCoverageAccumulator() }
        for (index in 0 until networks.length()) {
            val row = networks.optJSONObject(index) ?: continue
            val frequencyMhz = jsonIntOrNull(row, "frequency_mhz") ?: 0
            val band = canonicalWifiBandLabel(row.optString("band"), frequencyMhz)
            val accumulator = accumulators.getOrPut(band.takeIf { it != "unknown" } ?: "unknown") { WifiBandCoverageAccumulator() }
            accumulator.networkCount += 1
            jsonIntOrNull(row, "rssi_dbm")?.let { rssi ->
                accumulator.strongestRssiDbm = maxOf(accumulator.strongestRssiDbm ?: rssi, rssi)
            }
            (jsonIntOrNull(row, "channel") ?: channelForFrequencyMhz(frequencyMhz))?.let { accumulator.channels.add(it.toString()) }
            row.optString("channel_width").takeIf { it.isNotBlank() && it != "unknown" }?.let(accumulator.widths::add)
            row.optString("wifi_standard").takeIf { it.isNotBlank() && it != "unknown" }?.let(accumulator.standards::add)
            if (row.optBoolean("hidden_ssid", false)) accumulator.hiddenSsidCount += 1
            val security = row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) }
            if (security in WIFI_ATTENTION_SECURITY_MODES || "WPS" in row.optString("capabilities").uppercase(Locale.US)) {
                accumulator.securityAttentionCount += 1
            }
        }
        val summaryByBand = mutableMapOf<String, JSONObject>()
        for (index in 0 until bandSummary.length()) {
            val row = bandSummary.optJSONObject(index) ?: continue
            summaryByBand[row.optString("band")] = row
        }
        val bestRatingByBand = mutableMapOf<String, JSONObject>()
        for (index in 0 until ratings.length()) {
            val row = ratings.optJSONObject(index) ?: continue
            val band = row.optString("band")
            val previous = bestRatingByBand[band]
            if (previous == null || row.optInt("score") > previous.optInt("score")) bestRatingByBand[band] = row
        }
        val bands = (accumulators.keys + summaryByBand.keys + bestRatingByBand.keys)
            .distinct()
            .sortedBy(::wifiBandSortKey)
        return JSONArray().also { array ->
            bands.forEach { band ->
                val accumulator = accumulators[band] ?: WifiBandCoverageAccumulator()
                val summary = summaryByBand[band]
                val best = bestRatingByBand[band]
                val recommendedChannel = jsonIntOrNull(summary ?: JSONObject(), "recommended_channel")
                    ?: best?.optInt("channel")
                val recommendedScore = jsonIntOrNull(summary ?: JSONObject(), "recommended_score")
                    ?: best?.optInt("score")
                array.put(
                    JSONObject()
                        .put("band", band)
                        .put("network_count", accumulator.networkCount)
                        .put("observed", accumulator.networkCount > 0)
                        .put("coverage_label", if (accumulator.networkCount > 0) "observed" else "not observed in latest scan")
                        .put("rated_channel_count", summary?.optInt("rated_channel_count") ?: ratingsForBandCount(ratings, band))
                        .put("recommended_channel", recommendedChannel ?: JSONObject.NULL)
                        .put("recommended_score", recommendedScore ?: JSONObject.NULL)
                        .put("strongest_rssi_dbm", accumulator.strongestRssiDbm ?: JSONObject.NULL)
                        .put("channel_count", accumulator.channels.size)
                        .put("visible_channels", JSONArray(accumulator.channels.take(MAX_WIFI_SUMMARY_SAMPLES)))
                        .put("observed_widths", JSONArray(accumulator.widths.take(MAX_WIFI_SUMMARY_SAMPLES)))
                        .put("observed_standards", JSONArray(accumulator.standards.take(MAX_WIFI_SUMMARY_SAMPLES)))
                        .put("hidden_ssid_count", accumulator.hiddenSsidCount)
                        .put("security_attention_count", accumulator.securityAttentionCount)
                        .put("recommendation", wifiBandCoverageRecommendation(band, accumulator, recommendedChannel, recommendedScore)),
                )
            }
        }
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
                val detail = JSONObject()
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
                        .put("scan_age_ms", timestampMicros?.let(::wifiScanAgeMs) ?: JSONObject.NULL)
                add(appendWifiSemanticFields(detail))
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
        val backendOrder = when {
            supportedAbis.any { HermesAndroidHardwareProfile.isX86Abi(it) } -> listOf("cpu")
            supportedAbis.any { HermesAndroidHardwareProfile.isArmAbi(it) } -> listOf("gpu", "cpu")
            else -> listOf("cpu")
        }
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
            .put("litert_lm_backend_order", JSONArray(backendOrder))
            .put("litert_lm_backend_strategy", "GPU-first on ARM devices when LiteRT-LM accepts the accelerator, then CPU fallback; CPU-only on x86 emulator/device builds.")
            .put("litert_lm_artifact_selection_policy", liteRtLmArtifactSelectionPolicyJson(hardwareProfile))
            .put("compatibility_strategy", "Use Android SDK feature, permission, sensor, Wi-Fi, Bluetooth, camera, and storage APIs; avoid Adreno-only or Snapdragon-only assumptions.")
            .put("native_abi_strategy", HermesAndroidHardwareProfile.nativeAbiStrategy(supportedAbis))
    }

    private fun liteRtLmArtifactSelectionPolicyJson(hardwareProfile: HermesAndroidHardwareProfile.Profile): JSONObject {
        return JSONObject()
            .put("soc_aware_selection_enabled", true)
            .put("generic_litertlm_preferred", true)
            .put("matching_soc_specific_rank", 10)
            .put("matching_gpu_specific_rank", 12)
            .put("unmatched_soc_specific_rank", 80)
            .put("preferred_soc_family", hardwareProfile.socFamily)
            .put("preferred_gpu_family", hardwareProfile.gpuFamily)
            .put("preferred_device_family_label", listOf(hardwareProfile.socLabel, hardwareProfile.gpuLabel).filter { it != "unknown" }.joinToString("/").ifBlank { "generic Android" })
            .put("matching_soc_patterns", JSONArray(liteRtLmArtifactSocPatterns(hardwareProfile.socFamily)))
            .put("matching_gpu_patterns", JSONArray(liteRtLmArtifactGpuPatterns(hardwareProfile.gpuFamily)))
            .put("selection_order", "Prefer generic .litertlm first; if only SOC/GPU-specific LiteRT-LM bundles are available, prefer matching ${hardwareProfile.socLabel}/${hardwareProfile.gpuLabel} tags before mismatched vendor bundles.")
            .put("recommendation", "On MediaTek/MT/Dimensity/Helio devices, prefer generic or MediaTek-tagged LiteRT-LM artifacts before Qualcomm/SM/Adreno-specific artifacts.")
    }

    private fun liteRtLmArtifactSocPatterns(socFamily: String): List<String> = when (socFamily) {
        "mediatek" -> listOf("mediatek", "mtk", "mt[0-9]+", "dimensity", "helio")
        "qualcomm_snapdragon" -> listOf("qualcomm", "snapdragon", "qcom", "sm[0-9]+", "sdm[0-9]+", "msm[0-9]+")
        "google_tensor" -> listOf("tensor", "gs[0-9]+")
        "samsung_exynos" -> listOf("exynos", "s5e[0-9]+")
        "unisoc" -> listOf("unisoc", "spreadtrum", "ums[0-9]+")
        else -> emptyList()
    }

    private fun liteRtLmArtifactGpuPatterns(gpuFamily: String): List<String> = when (gpuFamily) {
        "mali_immortalis" -> listOf("immortalis", "mali")
        "mali" -> listOf("mali", "valhall", "bifrost")
        "powervr_img" -> listOf("powervr", "imgtec", "rogue")
        "adreno" -> listOf("adreno")
        "xclipse" -> listOf("xclipse", "amd-rdna", "amd_rdna")
        else -> emptyList()
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
        val observed = observedChannels.filter { it > 0 }.distinct().sorted()
        val base = when (band) {
            "2.4GHz" -> listOf(1, 6, 11)
            "5GHz" -> listOf(36, 40, 44, 48, 149, 153, 157, 161)
            "6GHz" -> listOf(5, 21, 37, 53, 69, 85, 101, 117, 133, 149, 165, 181, 197, 213, 229)
            else -> emptyList()
        }
        return (observed + base)
            .distinct()
            .take(MAX_WIFI_CANDIDATE_CHANNELS_PER_BAND)
            .sorted()
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

    private fun wifiChannelGraphHalfSpan(widthMhz: Int): Int {
        return (widthMhz.coerceAtLeast(20) / 10.0).roundToInt().coerceAtLeast(2)
    }

    private fun wifiChannelGraphSpanOverlap(firstStart: Int, firstEnd: Int, secondStart: Int, secondEnd: Int): Int {
        return (minOf(firstEnd, secondEnd) - maxOf(firstStart, secondStart) + 1).coerceAtLeast(0)
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

    private fun wifiChannelGraphRecommendation(overlapCount: Int, pressureScore: Int, widthMhz: Int, rssiDbm: Int): String {
        return when {
            overlapCount == 0 -> "Clear visible channel envelope for this AP; keep this row as a baseline when comparing nearby movement or router placement."
            pressureScore >= 75 -> "Heavy visible channel overlap around this AP. Compare wifi_channel_rating rows before selecting a router channel."
            pressureScore >= 45 -> "Moderate channel overlap. Wide ${widthMhz}MHz use may trade throughput for more interference exposure."
            rssiDbm < -75 -> "Weak AP signal with some overlap; move closer or compare the Wi-Fi history card before changing router settings."
            else -> "Some nearby AP overlap is visible; use this graph row with channel rating and utilization cards for final guidance."
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

    private fun wifiChannelUtilizationLabel(score: Int): String = when {
        score >= 75 -> "heavily_used"
        score >= 50 -> "crowded"
        score >= 25 -> "moderate"
        else -> "quiet"
    }

    private fun wifiChannelUtilizationRecommendation(
        score: Int,
        sameChannelCount: Int,
        overlapCount: Int,
        wideChannelCount: Int,
        strongestRssi: Int?,
    ): String {
        return when {
            overlapCount == 0 || score < 25 -> "Quiet observed channel: little visible AP pressure in the latest Android scan cache."
            score >= 75 && wideChannelCount > 0 -> "High utilization with wide-channel APs nearby: compare width and channel-rating rows before choosing this channel."
            score >= 75 || sameChannelCount >= 4 || (strongestRssi ?: -100) > -50 -> "Heavily used channel: strong or repeated AP pressure is visible in the latest scan."
            score >= 50 -> "Crowded channel: visible overlap exists; prefer a lower-utilization or higher-rated channel when possible."
            else -> "Moderate channel use: acceptable if its channel rating and security rows also look healthy."
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

    private fun readWifiSignalHistory(context: Context): JSONObject {
        val prefs = context.getSharedPreferences(WIFI_SIGNAL_HISTORY_PREFS, Context.MODE_PRIVATE)
        return runCatching {
            JSONObject(prefs.getString(WIFI_SIGNAL_HISTORY_KEY, "{}").orEmpty().ifBlank { "{}" })
        }.getOrDefault(JSONObject())
    }

    private fun updateBluetoothSignalHistory(context: Context, devices: JSONArray, observedAtMs: Long): JSONObject {
        val prefs = context.getSharedPreferences(BLUETOOTH_SIGNAL_HISTORY_PREFS, Context.MODE_PRIVATE)
        val existing = runCatching {
            JSONObject(prefs.getString(BLUETOOTH_SIGNAL_HISTORY_KEY, "{}").orEmpty().ifBlank { "{}" })
        }.getOrDefault(JSONObject())
        val updated = mergeBluetoothSignalHistory(existing, devices, observedAtMs)
        prefs.edit().putString(BLUETOOTH_SIGNAL_HISTORY_KEY, updated.toString()).apply()
        return updated
    }

    private fun readBluetoothSignalHistory(context: Context): JSONObject {
        val prefs = context.getSharedPreferences(BLUETOOTH_SIGNAL_HISTORY_PREFS, Context.MODE_PRIVATE)
        return runCatching {
            JSONObject(prefs.getString(BLUETOOTH_SIGNAL_HISTORY_KEY, "{}").orEmpty().ifBlank { "{}" })
        }.getOrDefault(JSONObject())
    }

    private fun updateMotionSensorHistory(context: Context, samples: JSONArray, observedAtMs: Long): JSONObject {
        val prefs = context.getSharedPreferences(MOTION_SENSOR_HISTORY_PREFS, Context.MODE_PRIVATE)
        val existing = runCatching {
            JSONObject(prefs.getString(MOTION_SENSOR_HISTORY_KEY, "{}").orEmpty().ifBlank { "{}" })
        }.getOrDefault(JSONObject())
        val updated = mergeMotionSensorHistory(existing, samples, observedAtMs)
        prefs.edit().putString(MOTION_SENSOR_HISTORY_KEY, updated.toString()).apply()
        return updated
    }

    private fun readMotionSensorHistory(context: Context): JSONObject {
        val prefs = context.getSharedPreferences(MOTION_SENSOR_HISTORY_PREFS, Context.MODE_PRIVATE)
        return runCatching {
            JSONObject(prefs.getString(MOTION_SENSOR_HISTORY_KEY, "{}").orEmpty().ifBlank { "{}" })
        }.getOrDefault(JSONObject())
    }

    private fun wifiHistoryKey(row: JSONObject): String {
        val bssid = row.optString("bssid").trim().lowercase(Locale.US)
        if (bssid.isNotBlank()) return bssid
        val ssid = row.optString("ssid").trim().lowercase(Locale.US)
        val frequency = row.optInt("frequency_mhz", 0)
        val channel = row.opt("channel")?.toString().orEmpty()
        return listOf(ssid, frequency.toString(), channel).joinToString("|").takeIf { ssid.isNotBlank() }.orEmpty()
    }

    private fun bluetoothHistoryKey(row: JSONObject): String {
        val address = row.optString("address").trim().lowercase(Locale.US)
        if (address.isNotBlank()) return address
        val advertisedName = row.optString("advertised_name").trim().lowercase(Locale.US)
        val deviceName = row.optString("device_name").trim().lowercase(Locale.US)
        val serviceFingerprint = jsonStringList(row, "service_uuids").take(2).joinToString(",")
        val manufacturerFingerprint = jsonStringList(row, "manufacturer_ids").take(2).joinToString(",")
        val name = advertisedName.ifBlank { deviceName }
        return listOf(name, serviceFingerprint, manufacturerFingerprint)
            .joinToString("|")
            .takeIf { name.isNotBlank() }
            .orEmpty()
    }

    private fun motionSensorHistoryKey(row: JSONObject): String {
        val sensorType = canonicalSensorType(row.optString("sensor_type"))
        if (sensorType in MOTION_HISTORY_SENSOR_TYPES) return sensorType
        return row.optString("sensor_label").trim().lowercase(Locale.US)
            .replace(' ', '_')
            .takeIf { it.isNotBlank() }
            .orEmpty()
    }

    private fun trimWifiObservations(observations: JSONArray): JSONArray {
        val start = (observations.length() - MAX_WIFI_HISTORY_SAMPLES_PER_NETWORK).coerceAtLeast(0)
        val trimmed = JSONArray()
        for (index in start until observations.length()) {
            observations.optJSONObject(index)?.let(trimmed::put)
        }
        return trimmed
    }

    private fun trimBluetoothObservations(observations: JSONArray): JSONArray {
        val start = (observations.length() - MAX_BLUETOOTH_HISTORY_SAMPLES_PER_DEVICE).coerceAtLeast(0)
        val trimmed = JSONArray()
        for (index in start until observations.length()) {
            observations.optJSONObject(index)?.let(trimmed::put)
        }
        return trimmed
    }

    private fun trimMotionSensorObservations(observations: JSONArray): JSONArray {
        val start = (observations.length() - MAX_MOTION_HISTORY_SAMPLES_PER_SENSOR).coerceAtLeast(0)
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

    private fun lastBluetoothObservationTime(record: JSONObject): Long {
        val observations = record.optJSONArray("observations") ?: return 0L
        for (index in observations.length() - 1 downTo 0) {
            val time = observations.optJSONObject(index)?.optLong("observed_at_ms", 0L) ?: 0L
            if (time > 0L) return time
        }
        return 0L
    }

    private fun lastMotionSensorObservationTime(record: JSONObject): Long {
        val observations = record.optJSONArray("observations") ?: return 0L
        for (index in observations.length() - 1 downTo 0) {
            val time = observations.optJSONObject(index)?.optLong("observed_at_ms", 0L) ?: 0L
            if (time > 0L) return time
        }
        return 0L
    }

    private fun lastMotionSensorObservation(record: JSONObject): JSONObject? {
        val observations = record.optJSONArray("observations") ?: return null
        for (index in observations.length() - 1 downTo 0) {
            observations.optJSONObject(index)?.let { return it }
        }
        return null
    }

    private fun currentWifiRssi(record: JSONObject): Int? {
        val observations = record.optJSONArray("observations") ?: return null
        for (index in observations.length() - 1 downTo 0) {
            jsonIntOrNull(observations.optJSONObject(index) ?: continue, "rssi_dbm")?.let { return it }
        }
        return null
    }

    private fun currentBluetoothRssi(record: JSONObject): Int? {
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

    private fun bluetoothSignalTrendLabel(trendDb: Int): String = when {
        trendDb >= 5 -> "approaching"
        trendDb <= -5 -> "fading"
        else -> "stable"
    }

    private fun motionSensorTrendLabel(sensorType: String, trendMagnitude: Double): String {
        val threshold = motionSensorTrendThreshold(sensorType)
        return when {
            trendMagnitude >= threshold -> "increasing"
            trendMagnitude <= -threshold -> "decreasing"
            else -> "stable"
        }
    }

    private fun motionSensorStabilityLabel(sensorType: String, rangeMagnitude: Double, averageMagnitude: Double): String {
        val threshold = maxOf(motionSensorTrendThreshold(sensorType), abs(averageMagnitude) * 0.08)
        return when {
            rangeMagnitude <= threshold -> "steady"
            rangeMagnitude <= threshold * 3 -> "drifting"
            else -> "changing"
        }
    }

    private fun motionSensorTrendThreshold(sensorType: String): Double = when (sensorType) {
        "gyroscope" -> 0.15
        "rotation_vector" -> 0.05
        "magnetic_field" -> 2.0
        else -> 0.5
    }

    private data class MotionPoseEstimate(
        val source: String,
        val sourceSensors: List<String>,
        val rollDegrees: Double,
        val pitchDegrees: Double,
        val tiltDegrees: Double,
        val azimuthDegrees: Double?,
        val headingLabel: String?,
        val faceOrientationLabel: String,
        val confidenceLabel: String,
        val valueLabel: String,
        val workflowHint: String,
        val fraction: Float,
    )

    private fun poseFromRotationVector(values: List<Double>): MotionPoseEstimate? {
        val x = values.getOrNull(0) ?: return null
        val y = values.getOrNull(1) ?: return null
        val z = values.getOrNull(2) ?: return null
        val w = values.getOrNull(3) ?: sqrt((1.0 - x * x - y * y - z * z).coerceAtLeast(0.0))
        val roll = radiansToDegrees(atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        val pitch = radiansToDegrees(asin((2.0 * (w * y - z * x)).coerceIn(-1.0, 1.0)))
        val azimuth = normalizeDegrees(
            radiansToDegrees(atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))),
        )
        val tilt = poseTiltDegrees(roll, pitch)
        val heading = headingLabel(azimuth)
        val face = faceOrientationLabel(roll, pitch, tilt, null)
        return MotionPoseEstimate(
            source = "rotation_vector",
            sourceSensors = listOf("rotation_vector"),
            rollDegrees = roll,
            pitchDegrees = pitch,
            tiltDegrees = tilt,
            azimuthDegrees = azimuth,
            headingLabel = heading,
            faceOrientationLabel = face,
            confidenceLabel = "high",
            valueLabel = listOf(face.replace('_', ' '), "heading $heading").joinToString(" | "),
            workflowHint = "Use rotation-vector pose as the preferred orientation signal when it is available because Android has already fused IMU inputs.",
            fraction = 0.95f,
        )
    }

    private fun poseFromGravityMagnetic(
        gravitySource: String,
        gravityValues: List<Double>,
        magneticValues: List<Double>?,
    ): MotionPoseEstimate? {
        val x = gravityValues.getOrNull(0) ?: return null
        val y = gravityValues.getOrNull(1) ?: return null
        val z = gravityValues.getOrNull(2) ?: return null
        val magnitude = vectorMagnitude(gravityValues)?.takeIf { it > 0.01 } ?: return null
        val roll = radiansToDegrees(atan2(y, z))
        val pitch = radiansToDegrees(atan2(-x, sqrt(y * y + z * z)))
        val tilt = radiansToDegrees(atan2(sqrt(x * x + y * y), abs(z)))
        val heading = magneticValues
            ?.takeIf { (vectorMagnitude(it) ?: 0.0) > 0.01 }
            ?.let { magnetic -> tiltCompensatedHeadingDegrees(roll, pitch, magnetic) }
        val face = faceOrientationLabel(roll, pitch, tilt, z)
        val headingLabel = heading?.let(::headingLabel)
        val sourceSensors = if (heading != null) listOf(gravitySource, "magnetic_field") else listOf(gravitySource)
        return MotionPoseEstimate(
            source = sourceSensors.joinToString("+"),
            sourceSensors = sourceSensors,
            rollDegrees = roll,
            pitchDegrees = pitch,
            tiltDegrees = tilt,
            azimuthDegrees = heading,
            headingLabel = headingLabel,
            faceOrientationLabel = face,
            confidenceLabel = if (heading != null) "high" else "medium",
            valueLabel = listOfNotNull(face.replace('_', ' '), headingLabel?.let { "heading $it" }).joinToString(" | "),
            workflowHint = if (heading != null) {
                "Use this fused gravity and magnetic-field pose for heading-aware workflows, while keeping compass accuracy and nearby-metal interference in mind."
            } else {
                "Use this as tilt and face-orientation context only; heading needs magnetic_field or rotation_vector data."
            },
            fraction = if (heading != null && magnitude > 0.01) 0.9f else 0.65f,
        )
    }

    private fun latestMotionVectors(samples: JSONArray, motionHistoryRows: JSONArray): Map<String, List<Double>> {
        val vectors = linkedMapOf<String, List<Double>>()
        fun addVector(row: JSONObject, key: String) {
            val sensorType = canonicalSensorType(row.optString("sensor_type"))
            if (sensorType !in MOTION_HISTORY_SENSOR_TYPES) return
            val values = jsonDoubleList(row.optJSONArray(key))
            if (values.isNotEmpty()) vectors[sensorType] = values
        }
        for (index in 0 until motionHistoryRows.length()) {
            val row = motionHistoryRows.optJSONObject(index) ?: continue
            addVector(row, "current_values")
        }
        for (index in 0 until samples.length()) {
            val row = samples.optJSONObject(index) ?: continue
            if (!row.optBoolean("available", true)) continue
            if (row.has("sampled") && !row.optBoolean("sampled", false)) continue
            addVector(row, "values")
        }
        return vectors
    }

    private fun vectorMagnitude(values: List<Double>): Double? {
        if (values.isEmpty()) return null
        var sum = 0.0
        values.forEach { value -> sum += value * value }
        return sqrt(sum)
    }

    private fun tiltCompensatedHeadingDegrees(rollDegrees: Double, pitchDegrees: Double, magneticValues: List<Double>): Double {
        val roll = degreesToRadians(rollDegrees)
        val pitch = degreesToRadians(pitchDegrees)
        val mx = magneticValues.getOrElse(0) { 0.0 }
        val my = magneticValues.getOrElse(1) { 0.0 }
        val mz = magneticValues.getOrElse(2) { 0.0 }
        val horizontalX = mx * cos(pitch) + mz * sin(pitch)
        val horizontalY = mx * sin(roll) * sin(pitch) + my * cos(roll) - mz * sin(roll) * cos(pitch)
        return normalizeDegrees(radiansToDegrees(atan2(horizontalY, horizontalX)))
    }

    private fun faceOrientationLabel(rollDegrees: Double, pitchDegrees: Double, tiltDegrees: Double, zAxis: Double?): String {
        return when {
            tiltDegrees <= 30.0 && zAxis != null && zAxis < 0.0 -> "face_down"
            tiltDegrees <= 30.0 && zAxis != null -> "face_up"
            tiltDegrees <= 30.0 -> "level"
            abs(pitchDegrees) >= 60.0 -> if (pitchDegrees < 0.0) "portrait_upright" else "portrait_inverted"
            abs(rollDegrees) >= 60.0 -> if (rollDegrees > 0.0) "landscape_right" else "landscape_left"
            else -> "tilted"
        }
    }

    private fun angularMotionStateLabel(angularVelocityRadS: Double): String = when {
        angularVelocityRadS < 0.08 -> "steady"
        angularVelocityRadS < 0.5 -> "minor_rotation"
        angularVelocityRadS < 1.5 -> "rotating"
        else -> "fast_rotation"
    }

    private fun accelerationMotionStateLabel(source: String, deltaFromGravity: Double): String = when {
        source == "linear_acceleration" && deltaFromGravity < 0.2 -> "steady"
        source == "linear_acceleration" && deltaFromGravity < 1.0 -> "light_movement"
        source == "accelerometer" && deltaFromGravity < 0.4 -> "steady_with_gravity"
        source == "accelerometer" && deltaFromGravity < 1.5 -> "moving"
        deltaFromGravity < 4.0 -> "active_movement"
        else -> "impact_or_fast_motion"
    }

    private fun poseTiltDegrees(rollDegrees: Double, pitchDegrees: Double): Double {
        return sqrt(rollDegrees * rollDegrees + pitchDegrees * pitchDegrees).coerceIn(0.0, 180.0)
    }

    private fun degreesToRadians(value: Double): Double = value * PI / 180.0

    private fun radiansToDegrees(value: Double): Double = value * 180.0 / PI

    private fun normalizeDegrees(value: Double): Double {
        val normalized = value % 360.0
        return if (normalized < 0.0) normalized + 360.0 else normalized
    }

    private fun headingLabel(azimuthDegrees: Double): String {
        val labels = listOf("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        val index = Math.floorMod((normalizeDegrees(azimuthDegrees) / 45.0).roundToInt(), labels.size)
        return labels[index]
    }

    private fun roundedDegrees(value: Double): Double = (value * 10.0).roundToInt() / 10.0

    private fun formatDecimal(value: Double, places: Int): String = String.format(Locale.US, "%.${places}f", value)

    private fun motionSensorSortKey(sensorType: String): Int = when (canonicalSensorType(sensorType)) {
        "accelerometer" -> 0
        "gyroscope" -> 1
        "linear_acceleration" -> 2
        "gravity" -> 3
        "rotation_vector" -> 4
        "magnetic_field" -> 5
        else -> 6
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

    private fun bluetoothObservationSeries(observations: JSONArray): JSONArray {
        val series = JSONArray()
        val start = (observations.length() - MAX_BLUETOOTH_HISTORY_SERIES_POINTS).coerceAtLeast(0)
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

    private fun motionSensorObservationSeries(observations: JSONArray): JSONArray {
        val series = JSONArray()
        val start = (observations.length() - MAX_MOTION_HISTORY_SERIES_POINTS).coerceAtLeast(0)
        for (index in start until observations.length()) {
            val observation = observations.optJSONObject(index) ?: continue
            series.put(
                JSONObject()
                    .put("observed_at_ms", observation.optLong("observed_at_ms", 0L))
                    .put("magnitude", jsonDoubleOrNull(observation, "magnitude") ?: JSONObject.NULL),
            )
        }
        return series
    }

    private fun motionVectorMagnitude(values: JSONArray): Double? {
        var sum = 0.0
        var count = 0
        for (index in 0 until values.length()) {
            val value = when (val raw = values.opt(index)) {
                is Number -> raw.toDouble()
                is String -> raw.toDoubleOrNull()
                else -> null
            } ?: continue
            sum += value * value
            count += 1
        }
        return if (count > 0) sqrt(sum) else null
    }

    private fun copyJsonArray(values: JSONArray): JSONArray {
        val copy = JSONArray()
        for (index in 0 until values.length()) {
            copy.put(values.opt(index) ?: JSONObject.NULL)
        }
        return copy
    }

    private fun jsonStringArray(values: List<String>): JSONArray {
        val array = JSONArray()
        values.forEach(array::put)
        return array
    }

    private fun jsonDoubleList(values: JSONArray?): List<Double> {
        if (values == null) return emptyList()
        return buildList {
            for (index in 0 until values.length()) {
                when (val value = values.opt(index)) {
                    is Number -> value.toDouble()
                    is String -> value.toDoubleOrNull()
                    else -> null
                }?.let(::add)
            }
        }
    }

    private fun countSampledSensors(samples: JSONArray): Int {
        var count = 0
        for (index in 0 until samples.length()) {
            val row = samples.optJSONObject(index) ?: continue
            if (row.optBoolean("sampled", false)) count += 1
        }
        return count
    }

    private fun jsonIntOrNull(row: JSONObject, key: String): Int? {
        return when (val value = row.opt(key)) {
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }
    }

    private fun intArgument(arguments: JSONObject, vararg keys: String): Int? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is Number -> value.toInt()
            is String -> value.trim().toIntOrNull()
            else -> null
        }
    }

    private fun booleanArgument(arguments: JSONObject, vararg keys: String): Boolean? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is Boolean -> value
            is Number -> value.toInt() != 0
            is String -> when (value.trim().lowercase(Locale.US)) {
                "1", "true", "yes", "on", "enabled" -> true
                "0", "false", "no", "off", "disabled" -> false
                else -> null
            }
            else -> null
        }
    }

    private fun jsonStringArgument(arguments: JSONObject, vararg keys: String): String? {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return null
        return when (val value = arguments.opt(key)) {
            is JSONArray -> buildList {
                for (index in 0 until value.length()) value.optString(index).trim().takeIf { it.isNotBlank() }?.let(::add)
            }.joinToString(" ").takeIf { it.isNotBlank() }
            else -> value?.toString()?.trim()?.takeIf { it.isNotBlank() }
        }
    }

    private fun jsonStringListArgument(arguments: JSONObject, vararg keys: String): List<String> {
        val key = keys.firstOrNull { candidate -> arguments.has(candidate) && !arguments.isNull(candidate) } ?: return emptyList()
        return jsonStringListValue(arguments.opt(key))
    }

    private fun jsonStringListValue(value: Any?): List<String> {
        return when (value) {
            is JSONArray -> buildList {
                for (index in 0 until value.length()) {
                    addAll(jsonStringListValue(value.opt(index)))
                }
            }
            is String -> value
                .split(',', ';', '|')
                .map { it.trim() }
                .filter { it.isNotBlank() }
            null, JSONObject.NULL -> emptyList()
            else -> listOf(value.toString().trim()).filter { it.isNotBlank() }
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

    private fun normalizedWifiScanMode(arguments: JSONObject): String {
        val requested = arguments.optString("scan_mode").ifBlank {
            arguments.optString("wifi_scan_mode").ifBlank {
                arguments.optString("scan_control")
            }
        }
        return when (requested.trim().lowercase(Locale.US)) {
            "pause", "paused", "passive", "cached", "hold", "stop" -> WIFI_SCAN_MODE_PAUSED
            "resume", "resumed", "active", "fresh", "refresh", "live" -> WIFI_SCAN_MODE_RESUMED
            else -> WIFI_SCAN_MODE_AUTO
        }
    }

    private fun effectiveWifiRefreshRequested(refreshRequested: Boolean, scanMode: String): Boolean {
        return when (scanMode) {
            WIFI_SCAN_MODE_PAUSED -> false
            WIFI_SCAN_MODE_RESUMED -> true
            else -> refreshRequested
        }
    }

    private fun wifiScanControlJson(
        scanMode: String,
        userRefreshRequested: Boolean,
        effectiveRefreshRequested: Boolean,
        refreshAccepted: Boolean,
    ): JSONObject {
        return JSONObject()
            .put("scan_mode", scanMode)
            .put("pause_resume_supported", true)
            .put("user_refresh_requested", userRefreshRequested)
            .put("effective_refresh_requested", effectiveRefreshRequested)
            .put("refresh_suppressed_by_pause", userRefreshRequested && !effectiveRefreshRequested && scanMode == WIFI_SCAN_MODE_PAUSED)
            .put("refresh_accepted", refreshAccepted)
            .put("paused_uses_cached_scan_results", scanMode == WIFI_SCAN_MODE_PAUSED)
            .put("resumed_requests_active_scan", scanMode == WIFI_SCAN_MODE_RESUMED)
            .put("android_scope", "Per diagnostic request; Android exposes cached scan results and may throttle active startScan calls.")
            .put("agent_instruction", wifiScanModeInstruction(scanMode, userRefreshRequested, effectiveRefreshRequested, refreshAccepted))
    }

    private fun wifiScanModeInstruction(
        scanMode: String,
        userRefreshRequested: Boolean,
        effectiveRefreshRequested: Boolean,
        refreshAccepted: Boolean,
    ): String {
        return when {
            scanMode == WIFI_SCAN_MODE_PAUSED && userRefreshRequested ->
                "Active Wi-Fi refresh was paused; explain cached scan age/history instead of polling again."
            scanMode == WIFI_SCAN_MODE_PAUSED ->
                "Use cached Wi-Fi scan rows/history and avoid active refresh while scan mode is paused."
            scanMode == WIFI_SCAN_MODE_RESUMED && refreshAccepted ->
                "Android accepted the resumed active scan request; read scan age and rows before advising."
            scanMode == WIFI_SCAN_MODE_RESUMED && effectiveRefreshRequested ->
                "A resumed active scan was requested; Android may still return cached rows if throttled."
            else ->
                "Auto mode follows the refresh argument and reports Android throttling or cached scan age honestly."
        }
    }

    private fun wifiScanStatusJson(
        refreshRequested: Boolean,
        refreshAccepted: Boolean,
        wifiEnabled: Boolean,
        permissionStatus: JSONObject,
        totalScanResultCount: Int,
        returnedNetworkCount: Int,
        latestScanAgeMs: Long?,
        scanMode: String = WIFI_SCAN_MODE_AUTO,
        userRefreshRequested: Boolean = refreshRequested,
    ): JSONObject {
        return JSONObject()
            .put("refresh_requested", refreshRequested)
            .put("refresh_accepted", refreshAccepted)
            .put("user_refresh_requested", userRefreshRequested)
            .put("wifi_scan_mode", scanMode)
            .put("wifi_scan_control", wifiScanControlJson(scanMode, userRefreshRequested, refreshRequested, refreshAccepted))
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
        return jsonStringList(row.optJSONArray(key))
    }

    private fun jsonStringList(array: JSONArray?): List<String> {
        if (array == null) return emptyList()
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

    private fun appendWifiSemanticFields(row: JSONObject): JSONObject {
        val semanticLabel = wifiAccessPointSemanticLabel(row)
        val riskLabel = wifiAccessPointSecurityRiskLabel(row)
        val tags = wifiAccessPointSemanticTags(row, semanticLabel, riskLabel)
        return row
            .put("semantic_label", semanticLabel)
            .put("security_risk_label", riskLabel)
            .put("semantic_tags", JSONArray(tags))
            .put("semantic_recommendation", wifiAccessPointSemanticRecommendation(row, semanticLabel, riskLabel))
    }

    private fun wifiAccessPointSemanticLabel(row: JSONObject): String {
        val ssid = row.optString("display_ssid").ifBlank { row.optString("ssid") }.lowercase(Locale.US)
        val security = row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) }
        val capabilities = row.optString("capabilities").uppercase(Locale.US)
        return when {
            row.optBoolean("passpoint_network", false) -> "enterprise/passpoint AP"
            row.optBoolean("hidden_ssid", false) -> "hidden SSID AP"
            ssid.contains("mesh") || ssid.contains("node") || ssid.contains("repeater") || ssid.contains("extender") -> "mesh/repeater candidate"
            ssid.contains("guest") || ssid.contains("hotspot") || ssid.contains("public") || ssid.contains("free wifi") -> "guest/public hotspot"
            ssid.contains("iot") || ssid.contains("camera") || ssid.contains("printer") || ssid.contains("thermostat") || ssid.contains("smart") -> "IoT/device AP"
            security == "Open" || security == "WEP" || "WPS" in capabilities -> "open/legacy attention AP"
            security == "WPA3" || security == "WPA2" -> "private router/AP"
            else -> "nearby AP"
        }
    }

    private fun wifiAccessPointSecurityRiskLabel(row: JSONObject): String {
        val security = row.optString("security_mode").ifBlank { wifiSecurityLabel(row.optString("capabilities")) }
        val capabilities = row.optString("capabilities").uppercase(Locale.US)
        return when {
            "WPS" in capabilities -> "wps_attention"
            security == "WEP" -> "legacy_weak_security"
            security == "Open" -> "open_network"
            security == "Enhanced Open" -> "encrypted_open"
            security == "WPA3" -> "strong_security"
            security == "WPA2" || security == "WPA" -> "standard_security"
            else -> "unknown_security"
        }
    }

    private fun wifiAccessPointSemanticTags(row: JSONObject, semanticLabel: String, riskLabel: String): List<String> {
        val tags = linkedSetOf<String>()
        tags += semanticLabel.replace("/", "_").replace(" ", "_").lowercase(Locale.US)
        tags += riskLabel
        if (row.optBoolean("hidden_ssid", false)) tags += "hidden_ssid"
        if (row.optBoolean("passpoint_network", false)) tags += "passpoint"
        if (row.optBoolean("80211mc_responder", false)) tags += "80211mc_rtt"
        if (row.optString("bssid_vendor") == "Locally administered / randomized") tags += "randomized_bssid"
        jsonIntOrNull(row, "rssi_dbm")?.let { rssi ->
            tags += when {
                rssi >= -50 -> "excellent_signal"
                rssi >= -60 -> "good_signal"
                rssi >= -70 -> "fair_signal"
                else -> "weak_signal"
            }
        }
        val widthMhz = jsonIntOrNull(row, "channel_width_mhz") ?: channelWidthMhz(row.optString("channel_width"))
        if ((widthMhz ?: 0) >= 80) tags += "wide_channel"
        when (row.optString("wifi_standard")) {
            "802.11be" -> tags += "wifi7"
            "802.11ax" -> tags += "wifi6_or_6e"
            "802.11ac" -> tags += "wifi5"
        }
        return tags.toList()
    }

    private fun wifiAccessPointSemanticRecommendation(row: JSONObject): String {
        return wifiAccessPointSemanticRecommendation(
            row = row,
            semanticLabel = row.optString("semantic_label").ifBlank { wifiAccessPointSemanticLabel(row) },
            riskLabel = row.optString("security_risk_label").ifBlank { wifiAccessPointSecurityRiskLabel(row) },
        )
    }

    private fun wifiAccessPointSemanticRecommendation(row: JSONObject, semanticLabel: String, riskLabel: String): String {
        val channel = row.opt("channel").takeUnless { it == null || it == JSONObject.NULL }?.toString()
        val band = row.optString("band").ifBlank { "unknown band" }
        return when (riskLabel) {
            "open_network" -> "Open network: treat as public or captive-portal only until the user confirms trust; compare $band channel ${channel ?: "unknown"} congestion before use."
            "legacy_weak_security" -> "Legacy WEP network: flag as weak security even if RSSI is strong."
            "wps_attention" -> "WPS is advertised: preserve BSSID/vendor/channel details and avoid treating this AP as fully hardened."
            "strong_security" -> "WPA3 network: prefer only when channel pressure, RSSI, and user trust are also acceptable."
            "standard_security" -> "WPA/WPA2 network: inspect channel pressure, vendor/OUI, and SSID context before recommending it."
            else -> when (semanticLabel) {
                "enterprise/passpoint AP" -> "Passpoint or venue network: explain it as infrastructure-managed and still check channel pressure/security metadata."
                "hidden SSID AP" -> "Hidden SSID: preserve BSSID, OUI/vendor, channel, and RSSI because the display name is intentionally absent."
                "mesh/repeater candidate" -> "Mesh/repeater candidate: use signal history and channel rows to reason about roaming or extender placement."
                "guest/public hotspot" -> "Guest/public hotspot: keep answers scoped to connectivity quality and trust boundaries."
                "IoT/device AP" -> "IoT/device AP: avoid assuming normal router throughput; inspect security, band, and proximity."
                else -> "Nearby AP: use band, channel, security, RSSI, vendor/OUI, and history rows for the next recommendation."
            }
        }
    }

    private fun wifiSemanticAttentionScore(row: JSONObject): Int {
        return when (row.optString("security_risk_label")) {
            "legacy_weak_security", "open_network", "wps_attention" -> 4
            "unknown_security" -> 3
            "encrypted_open" -> 2
            else -> if (row.optBoolean("hidden_ssid", false)) 2 else 1
        }
    }

    private fun ratingsForBandCount(ratings: JSONArray, band: String): Int {
        var count = 0
        for (index in 0 until ratings.length()) {
            if (ratings.optJSONObject(index)?.optString("band") == band) count += 1
        }
        return count
    }

    private fun countWifiObservedBands(rows: JSONArray): Int {
        var count = 0
        for (index in 0 until rows.length()) {
            val row = rows.optJSONObject(index) ?: continue
            if (row.optInt("network_count", 0) > 0) count += 1
        }
        return count
    }

    private fun wifiBandCoverageRecommendation(
        band: String,
        accumulator: WifiBandCoverageAccumulator,
        recommendedChannel: Int?,
        recommendedScore: Int?,
    ): String {
        if (accumulator.networkCount == 0) {
            return "$band was not visible in the latest scan; avoid claiming hardware support or absence without a fresh scan and permission context."
        }
        val bestChannel = recommendedChannel?.let { " Best rated visible channel hint: ch $it${recommendedScore?.let { score -> " ($score/100)" } ?: ""}." }.orEmpty()
        val securityNote = if (accumulator.securityAttentionCount > 0) {
            " ${accumulator.securityAttentionCount} AP(s) on this band advertise open, WEP, or WPS attention metadata."
        } else {
            ""
        }
        val hiddenNote = if (accumulator.hiddenSsidCount > 0) {
            " ${accumulator.hiddenSsidCount} hidden SSID AP(s) need BSSID/channel-based reasoning."
        } else {
            ""
        }
        return when (band) {
            "2.4GHz" -> "2.4GHz is range-friendly but overlap-prone; compare channels 1/6/11 and pressure rows before router guidance.$bestChannel$securityNote$hiddenNote"
            "5GHz" -> "5GHz usually has more channel room and shorter range; compare wide-channel contention and DFS-like gaps before placement advice.$bestChannel$securityNote$hiddenNote"
            "6GHz" -> "6GHz visibility indicates Wi-Fi 6E/7-class surroundings; inspect width, RSSI, and client support before assuming best throughput.$bestChannel$securityNote$hiddenNote"
            else -> "$band is visible; preserve observed channels, widths, standards, security attention, and RSSI before making network recommendations.$bestChannel$securityNote$hiddenNote"
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

    private data class WifiScanFilterSpec(
        val bands: Set<String> = emptySet(),
        val securityModes: Set<String> = emptySet(),
        val signalQualities: Set<String> = emptySet(),
        val ssidQuery: String = "",
        val bssidQuery: String = "",
        val vendorQuery: String = "",
        val minRssiDbm: Int? = null,
        val maxRssiDbm: Int? = null,
        val excludeHidden: Boolean = false,
        val hiddenOnly: Boolean = false,
    ) {
        val activeFilterCount: Int
            get() = listOf(
                bands.isNotEmpty(),
                securityModes.isNotEmpty(),
                signalQualities.isNotEmpty(),
                ssidQuery.isNotBlank(),
                bssidQuery.isNotBlank(),
                vendorQuery.isNotBlank(),
                minRssiDbm != null,
                maxRssiDbm != null,
                excludeHidden,
                hiddenOnly,
            ).count { it }
        val active: Boolean
            get() = activeFilterCount > 0
    }

    private data class BluetoothScanFilterSpec(
        val deviceNameQuery: String = "",
        val addressQuery: String = "",
        val serviceQuery: String = "",
        val manufacturerQuery: String = "",
        val categoryQuery: String = "",
        val proximityLabels: Set<String> = emptySet(),
        val minRssiDbm: Int? = null,
        val maxRssiDbm: Int? = null,
    ) {
        val activeFilterCount: Int
            get() = listOf(
                deviceNameQuery.isNotBlank(),
                addressQuery.isNotBlank(),
                serviceQuery.isNotBlank(),
                manufacturerQuery.isNotBlank(),
                categoryQuery.isNotBlank(),
                proximityLabels.isNotEmpty(),
                minRssiDbm != null,
                maxRssiDbm != null,
            ).count { it }
        val active: Boolean
            get() = activeFilterCount > 0
    }

    private data class BluetoothFilterFacetAccumulator(
        val filterKey: String,
        val label: String,
        var count: Int = 0,
        var strongestRssiDbm: Int? = null,
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

    private data class WifiBandCoverageAccumulator(
        var networkCount: Int = 0,
        var strongestRssiDbm: Int? = null,
        var hiddenSsidCount: Int = 0,
        var securityAttentionCount: Int = 0,
        val channels: LinkedHashSet<String> = linkedSetOf(),
        val widths: LinkedHashSet<String> = linkedSetOf(),
        val standards: LinkedHashSet<String> = linkedSetOf(),
    )

    private val ACTIONS = listOf(
        "status",
        "top_apps",
        "wifi_scan",
        "wifi_filtered_scan",
        "wifi_analyzer_report",
        "wifi_channel_graph",
        "wifi_channel_rating",
        "wifi_channel_utilization",
        "wifi_ap_details",
        "wifi_export",
        "bluetooth_analyzer_report",
        "bluetooth_scan",
        "bluetooth_signal_history",
        "sensor_analyzer_report",
        "motion_sensor_history",
        "motion_pose",
        "sensor_snapshot",
        "camera_status",
        "radio_signal_status",
        "radio_signal_graph",
        "radio_analyzer_report",
        "signal_capability_status",
        "local_backend_runtime_report",
        "soc_compatibility_report",
        "gpu_backend_risk_report",
        "local_inference_compatibility_report",
        "device_performance_report",
        "signal_awareness_report",
        "agent_signal_evidence_report",
        "agent_observation_report",
        "agent_card_manifest_report",
        "agent_environment_report",
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
    private val AMBIENT_SENSOR_TYPES = setOf("magnetic_field", "light", "proximity", "pressure", "ambient_temperature", "relative_humidity")
    private val MOTION_HISTORY_SENSOR_TYPES = setOf("accelerometer", "gyroscope", "gravity", "linear_acceleration", "rotation_vector", "magnetic_field")
    private val DEFAULT_SENSOR_TYPES = listOf("accelerometer", "gyroscope", "magnetic_field", "light", "proximity")
    private val DEFAULT_MOTION_HISTORY_SENSOR_TYPES = listOf("accelerometer", "gyroscope", "linear_acceleration", "rotation_vector", "magnetic_field")
    private val DEFAULT_MOTION_POSE_SENSOR_TYPES = listOf("accelerometer", "gravity", "magnetic_field", "rotation_vector", "gyroscope", "linear_acceleration")
    private const val DEFAULT_LIMIT = 5
    private const val MAX_LIMIT = 20
    private const val THERMAL_STATUS_UNSUPPORTED = -1
    private const val MAX_WIFI_RESULTS = 40
    private const val MAX_WIFI_CHANNEL_GRAPH_ROWS = 32
    private const val MAX_WIFI_CHANNEL_RATINGS = 24
    private const val MAX_WIFI_CHANNEL_UTILIZATION_ROWS = 32
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
    private const val WIFI_SCAN_MODE_AUTO = "auto"
    private const val WIFI_SCAN_MODE_PAUSED = "paused"
    private const val WIFI_SCAN_MODE_RESUMED = "resumed"
    private val WIFI_BAND_COVERAGE_BASE_BANDS = listOf("2.4GHz", "5GHz", "6GHz")
    private val WIFI_ATTENTION_SECURITY_MODES = setOf("Open", "WEP")
    private const val MAX_BLUETOOTH_RESULTS = 40
    private const val MAX_BLUETOOTH_SERVICE_UUIDS_PER_DEVICE = 8
    private const val MAX_BLUETOOTH_MANUFACTURER_IDS_PER_DEVICE = 8
    private const val MAX_BLUETOOTH_METADATA_SUMMARY_ROWS = 24
    private const val MAX_BLUETOOTH_SUMMARY_SAMPLES = 4
    private const val MAX_BLUETOOTH_HISTORY_DEVICES_PER_SCAN = 40
    private const val MAX_BLUETOOTH_HISTORY_DEVICES = 40
    private const val MAX_BLUETOOTH_HISTORY_ROWS = 16
    private const val MAX_BLUETOOTH_HISTORY_SAMPLES_PER_DEVICE = 12
    private const val MAX_BLUETOOTH_HISTORY_SERIES_POINTS = 8
    private const val BLUETOOTH_SIGNAL_HISTORY_PREFS = "hermes_bluetooth_signal_history"
    private const val BLUETOOTH_SIGNAL_HISTORY_KEY = "signal_history"
    private const val BLUETOOTH_SCAN_MODE_AUTO = "auto"
    private const val BLUETOOTH_SCAN_MODE_PAUSED = "paused"
    private const val BLUETOOTH_SCAN_MODE_RESUMED = "resumed"
    private const val MAX_MOTION_HISTORY_SENSORS_PER_SAMPLE = 8
    private const val MAX_MOTION_HISTORY_SENSORS = 12
    private const val MAX_MOTION_HISTORY_ROWS = 12
    private const val MAX_MOTION_HISTORY_SAMPLES_PER_SENSOR = 12
    private const val MAX_MOTION_HISTORY_SERIES_POINTS = 8
    private const val MOTION_SENSOR_HISTORY_PREFS = "hermes_motion_sensor_history"
    private const val MOTION_SENSOR_HISTORY_KEY = "motion_history"
    private const val STANDARD_GRAVITY = 9.80665
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
        "semantic_label",
        "security_risk_label",
        "estimated_distance_m",
        "passpoint_network",
        "80211mc_responder",
    )
    private val TIKTOK_PACKAGES = listOf("com.zhiliaoapp.musically", "com.ss.android.ugc.trill")
    private const val INSTAGRAM_PACKAGE = "com.instagram.android"
    private const val GMAIL_PACKAGE = "com.google.android.gm"
    private val BLUETOOTH_SERVICE_UUID_LABELS = mapOf(
        "0x1800" to "Generic Access",
        "0x1801" to "Generic Attribute",
        "0x1802" to "Immediate Alert",
        "0x1803" to "Link Loss",
        "0x1804" to "Tx Power",
        "0x1805" to "Current Time",
        "0x1809" to "Health Thermometer",
        "0x180A" to "Device Information",
        "0x180D" to "Heart Rate",
        "0x180F" to "Battery",
        "0x1810" to "Blood Pressure",
        "0x1812" to "Human Interface Device",
        "0x1814" to "Running Speed and Cadence",
        "0x1816" to "Cycling Speed and Cadence",
        "0x1819" to "Location and Navigation",
        "0x181A" to "Environmental Sensing",
        "0x181B" to "Body Composition",
        "0x181C" to "User Data",
        "0x181D" to "Weight Scale",
        "0x181F" to "Continuous Glucose Monitoring",
        "0x1822" to "Pulse Oximeter",
        "0x1826" to "Fitness Machine",
        "0x1827" to "Mesh Provisioning",
        "0x1828" to "Mesh Proxy",
        "0x183A" to "Insulin Delivery",
        "0x183B" to "Binary Sensor",
        "0x1843" to "Audio Input Control",
        "0x1844" to "Volume Control",
        "0x184E" to "Audio Stream Control",
        "0xFE2C" to "Google Fast Pair",
        "0xFEAA" to "Eddystone",
        "0xFD6F" to "Exposure Notification",
    )
    private val BLUETOOTH_COMPANY_ID_LABELS = mapOf(
        "0x0006" to "Microsoft",
        "0x000A" to "Qualcomm Technologies International",
        "0x000D" to "Texas Instruments",
        "0x000F" to "Broadcom",
        "0x001D" to "Qualcomm",
        "0x004C" to "Apple",
        "0x0059" to "Nordic Semiconductor",
        "0x0075" to "Samsung Electronics",
        "0x00E0" to "Google",
    )
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
