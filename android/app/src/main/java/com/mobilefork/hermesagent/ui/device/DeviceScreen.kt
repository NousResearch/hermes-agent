@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.mobilefork.hermesagent.ui.device

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.R
import com.mobilefork.hermesagent.device.HermesGlobalAction
import com.mobilefork.hermesagent.ui.i18n.AppLanguage
import com.mobilefork.hermesagent.ui.i18n.HermesStrings
import com.mobilefork.hermesagent.ui.i18n.LocalHermesStrings
import com.mobilefork.hermesagent.ui.shell.ShellActionItem

@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun DeviceScreen(
    modifier: Modifier = Modifier,
    viewModel: DeviceViewModel = viewModel(),
    extraBottomSpacing: Dp = 0.dp,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
) {
    val uiState by viewModel.uiState.collectAsState()
    val strings = LocalHermesStrings.current
    var pendingExportFile by remember { mutableStateOf<String?>(null) }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            viewModel.importDocument(uri)
        }
    }
    val sharedFolderLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            viewModel.rememberSharedFolder(uri)
        }
    }
    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("*/*")) { uri ->
        val fileName = pendingExportFile
        if (uri != null && !fileName.isNullOrBlank()) {
            viewModel.exportWorkspaceFile(fileName, uri)
        }
        pendingExportFile = null
    }
    val notificationPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        viewModel.refresh(strings.deviceNotificationPermissionStatus(granted))
    }
    val bluetoothPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        viewModel.refresh(strings.deviceBluetoothPermissionStatus(granted))
    }

    SideEffect {
        onContextActionsChanged(
            listOf(
                ShellActionItem(
                    label = strings.refresh.ifBlank { "Refresh" },
                    description = strings.deviceRefreshDescription(),
                    iconRes = R.drawable.ic_action_refresh,
                    onClick = viewModel::refresh,
                ),
                ShellActionItem(
                    label = strings.deviceGrantSharedFolderLabel(),
                    description = strings.deviceGrantSharedFolderDescription(),
                    iconRes = R.drawable.ic_nav_device,
                    onClick = { sharedFolderLauncher.launch(null) },
                ),
                ShellActionItem(
                    label = strings.deviceImportFileLabel(),
                    description = strings.deviceImportFileDescription(),
                    iconRes = R.drawable.ic_nav_device,
                    onClick = { importLauncher.launch(arrayOf("*/*")) },
                ),
                ShellActionItem(
                    label = strings.deviceNotificationSettingsLabel(),
                    description = strings.deviceNotificationSettingsDescription(),
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = { viewModel.performSystemAction("open_notification_settings") },
                ),
            )
        )
    }

    fun handleBluetoothAction() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !uiState.bluetoothPermissionGranted) {
            bluetoothPermissionLauncher.launch(Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            viewModel.performSystemAction("open_bluetooth_settings")
        }
    }

    fun handleNotificationAction() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && !uiState.notificationPermissionGranted) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            viewModel.performSystemAction("open_notification_settings")
        }
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .widthIn(max = 920.dp)
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                        .padding(bottom = extraBottomSpacing),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                DeviceGuideCard(workspacePath = uiState.workspacePath)
                LinuxSuiteCard(uiState = uiState)
                OperatorStandbyCard(uiState = uiState)
                ConnectivityCard(
                    uiState = uiState,
                    onOpenWifi = { viewModel.performSystemAction("open_wifi_panel") },
                    onOpenBluetooth = ::handleBluetoothAction,
                    onOpenConnectedDevices = { viewModel.performSystemAction("open_connected_devices_settings") },
                )
                RadioControlCard(
                    uiState = uiState,
                    onOpenMobileNetwork = { viewModel.performSystemAction("open_mobile_network_settings") },
                    onOpenDataUsage = { viewModel.performSystemAction("open_data_usage_settings") },
                    onOpenHotspot = { viewModel.performSystemAction("open_hotspot_settings") },
                    onOpenAirplaneMode = { viewModel.performSystemAction("open_airplane_mode_settings") },
                )
                InterfaceCard(
                    uiState = uiState,
                    onOpenNfc = { viewModel.performSystemAction("open_nfc_settings") },
                    onOpenConnectedDevices = { viewModel.performSystemAction("open_connected_devices_settings") },
                )
                PermissionsAndRuntimeCard(
                    uiState = uiState,
                    onNotifications = ::handleNotificationAction,
                    onOverlaySettings = { viewModel.performSystemAction("open_overlay_settings") },
                    onToggleRuntime = { enabled -> viewModel.setBackgroundPersistence(enabled) },
                    onToggleFloatingButton = { enabled -> viewModel.setFloatingButtonEnabled(enabled) },
                )
                WorkspaceAccessCard(
                    uiState = uiState,
                    onImportFile = { importLauncher.launch(arrayOf("*/*")) },
                    onGrantFolder = { sharedFolderLauncher.launch(null) },
                    onClearFolder = viewModel::clearSharedFolder,
                    onRefresh = viewModel::refresh,
                    onExport = { fileName ->
                        pendingExportFile = fileName
                        exportLauncher.launch(fileName)
                    },
                )
                AccessibilityCard(
                    uiState = uiState,
                    onOpenSettings = { viewModel.performSystemAction("open_accessibility_settings") },
                    onAction = viewModel::performGlobalAction,
                )
                if (uiState.status.isNotBlank()) {
                    Text(uiState.status, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
    }
}

@Composable
private fun OperatorStandbyCard(uiState: DeviceUiState) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(strings.operatorStandbyTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                strings.operatorStandbyStatus(
                    ready = uiState.operatorStandbyReady,
                    enabledCount = uiState.enabledAutomationCount,
                    externalCount = uiState.externalTriggerCount,
                ),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.operatorStandbyRunHistory(uiState.recentAutomationRunCount),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.operatorStandbyRemoteDispatch(uiState.remoteDispatchCount),
                style = MaterialTheme.typography.bodySmall,
            )
            if (uiState.lastAutomationRunLabel.isNotBlank() || uiState.lastAutomationRunResult.isNotBlank()) {
                Text(
                    strings.operatorStandbyLastRun(
                        label = uiState.lastAutomationRunLabel,
                        success = uiState.lastAutomationRunSuccess,
                        result = uiState.lastAutomationRunResult,
                    ),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (uiState.lastDispatchTaskName.isNotBlank() || uiState.lastDispatchSource.isNotBlank()) {
                Text(
                    strings.operatorStandbyLastDispatch(
                        taskName = uiState.lastDispatchTaskName,
                        source = uiState.lastDispatchSource,
                        channel = uiState.lastDispatchChannel,
                    ),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (uiState.operatorModelName.isNotBlank()) {
                Text(
                    strings.deviceModelRouting(
                        provider = uiState.operatorModelProvider.ifBlank { "Hermes" },
                        modelName = uiState.operatorModelName,
                        visionCapable = uiState.operatorVisionCapable,
                    ),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun DeviceGuideCard(workspacePath: String) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(strings.deviceGuideTitle(), style = MaterialTheme.typography.titleMedium)
            Text(strings.deviceGuideStep(1))
            Text(strings.deviceGuideStep(2))
            Text(strings.deviceGuideStep(3))
            Text(strings.deviceGuideStep(4))
            if (workspacePath.isNotBlank()) {
                Text(strings.deviceWorkspacePath(workspacePath), style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun LinuxSuiteCard(uiState: DeviceUiState) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(strings.deviceLinuxSuiteTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                if (uiState.linuxEnabled) {
                    strings.deviceLinuxSuiteReady()
                } else {
                    strings.deviceLinuxSuiteProvisioning()
                },
            )
            if (uiState.linuxAndroidAbi.isNotBlank() || uiState.linuxTermuxArch.isNotBlank()) {
                Text(
                    strings.deviceLinuxAbi(uiState.linuxAndroidAbi, uiState.linuxTermuxArch),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (uiState.linuxPrefixPath.isNotBlank()) {
                Text(strings.deviceLinuxPrefix(uiState.linuxPrefixPath), style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.linuxBashPath.isNotBlank()) {
                Text(strings.deviceLinuxBash(uiState.linuxBashPath), style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.linuxHomePath.isNotBlank()) {
                Text(strings.deviceLinuxHome(uiState.linuxHomePath), style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.linuxTmpPath.isNotBlank()) {
                Text(strings.deviceLinuxTemp(uiState.linuxTmpPath), style = MaterialTheme.typography.bodySmall)
            }
            Text(
                strings.deviceLinuxPackageCount(uiState.linuxPackageCount),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.deviceLinuxTerminalGuidance(),
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun ConnectivityCard(
    uiState: DeviceUiState,
    onOpenWifi: () -> Unit,
    onOpenBluetooth: () -> Unit,
    onOpenConnectedDevices: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(strings.deviceConnectivityTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                strings.deviceNetworkSummary(uiState.activeNetworkLabel, uiState.wifiEnabled),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.deviceBluetoothTitle(), style = MaterialTheme.typography.titleSmall,
            )
            Text(
                if (!uiState.bluetoothSupported) {
                    strings.deviceBluetoothUnavailable()
                } else if (!uiState.bluetoothPermissionGranted) {
                    strings.deviceBluetoothPermissionPrompt()
                } else {
                    strings.deviceBluetoothSummary(
                        enabled = uiState.bluetoothEnabled,
                        bondedDevices = uiState.pairedBluetoothDevices.joinToString(),
                    )
                },
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenWifi) {
                    Text(strings.deviceInternetPanelLabel())
                }
                Button(onClick = onOpenBluetooth) {
                    Text(strings.deviceBluetoothTitle())
                }
                Button(onClick = onOpenConnectedDevices) {
                    Text(strings.deviceConnectedDevicesLabel())
                }
            }
        }
    }
}

@Composable
private fun RadioControlCard(
    uiState: DeviceUiState,
    onOpenMobileNetwork: () -> Unit,
    onOpenDataUsage: () -> Unit,
    onOpenHotspot: () -> Unit,
    onOpenAirplaneMode: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    val title = when (strings.language) {
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.CHINESE -> "蜂窝网络与无线电控制"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Controles celulares y de radio"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Mobilfunk- und Funksteuerung"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Controles celulares e de rádio"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Contrôles cellulaires et radio"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Cellular + radio controls"
    }
    val summary = when (strings.language) {
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.CHINESE -> "当前网络：${uiState.activeNetworkLabel} · 计量网络：${if (uiState.activeNetworkMetered) "是" else "否"} · 省流模式：${if (uiState.dataSaverEnabled) "开" else "关"} · 飞行模式：${if (uiState.airplaneModeEnabled) "开" else "关"}。由于 Android 限制，Hermes 使用系统面板而不是不受支持的直接无线电切换。"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Red actual: ${uiState.activeNetworkLabel} · medida: ${if (uiState.activeNetworkMetered) "sí" else "no"} · ahorro de datos: ${if (uiState.dataSaverEnabled) "activo" else "inactivo"} · modo avión: ${if (uiState.airplaneModeEnabled) "activo" else "inactivo"}. Por las restricciones de Android, Hermes usa paneles del sistema en lugar de toggles directos no soportados."
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Aktives Netzwerk: ${uiState.activeNetworkLabel} · getaktet: ${if (uiState.activeNetworkMetered) "ja" else "nein"} · Datensparen: ${if (uiState.dataSaverEnabled) "aktiv" else "inaktiv"} · Flugmodus: ${if (uiState.airplaneModeEnabled) "aktiv" else "inaktiv"}. Wegen Android-Beschränkungen nutzt Hermes Systemansichten statt nicht unterstützter Direktumschaltungen."
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Rede atual: ${uiState.activeNetworkLabel} · limitada: ${if (uiState.activeNetworkMetered) "sim" else "não"} · economia de dados: ${if (uiState.dataSaverEnabled) "ativa" else "inativa"} · modo avião: ${if (uiState.airplaneModeEnabled) "ativo" else "inativo"}. Devido às restrições do Android, o Hermes usa painéis do sistema em vez de alternâncias diretas não suportadas."
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Réseau actif : ${uiState.activeNetworkLabel} · limité : ${if (uiState.activeNetworkMetered) "oui" else "non"} · économie de données : ${if (uiState.dataSaverEnabled) "active" else "inactive"} · mode avion : ${if (uiState.airplaneModeEnabled) "actif" else "inactif"}. En raison des limites Android, Hermes utilise des panneaux système plutôt que des bascules radio directes non prises en charge."
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Active network: ${uiState.activeNetworkLabel} · metered: ${if (uiState.activeNetworkMetered) "yes" else "no"} · data saver: ${if (uiState.dataSaverEnabled) "enabled" else "disabled"} · airplane mode: ${if (uiState.airplaneModeEnabled) "enabled" else "disabled"}. Because of Android platform limits, Hermes uses system panels instead of unsupported direct radio toggles."
    }
    val mobileNetworkLabel = when (strings.language) {
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.CHINESE -> "移动网络"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Red móvil"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Mobilfunk"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Rede móvel"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Réseau mobile"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Mobile network"
    }
    val dataUsageLabel = when (strings.language) {
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.CHINESE -> "数据使用"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Uso de datos"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Datennutzung"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Uso de dados"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Utilisation des données"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Data usage"
    }
    val hotspotLabel = when (strings.language) {
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.CHINESE -> "热点 / 共享"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Punto de acceso"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Hotspot / Tethering"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Hotspot / ancoragem"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Point d’accès / partage"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Hotspot / tethering"
    }
    val airplaneLabel = when (strings.language) {
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.CHINESE -> "飞行模式"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Modo avión"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Flugmodus"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Modo avião"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Mode avion"
        com.mobilefork.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Airplane mode"
    }
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(summary, style = MaterialTheme.typography.bodySmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenMobileNetwork) {
                    Text(mobileNetworkLabel)
                }
                Button(onClick = onOpenDataUsage) {
                    Text(dataUsageLabel)
                }
                Button(onClick = onOpenHotspot) {
                    Text(hotspotLabel)
                }
                Button(onClick = onOpenAirplaneMode) {
                    Text(airplaneLabel)
                }
            }
        }
    }
}

@Composable
private fun InterfaceCard(
    uiState: DeviceUiState,
    onOpenNfc: () -> Unit,
    onOpenConnectedDevices: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(strings.deviceInterfaceTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                if (uiState.usbHostSupported) {
                    strings.deviceUsbSummary(
                        usbDeviceCount = uiState.usbDeviceCount,
                        usbDevices = uiState.usbDevices.joinToString(),
                    )
                } else {
                    strings.deviceUsbUnavailable()
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                if (uiState.nfcSupported) {
                    strings.deviceNfcSummary(uiState.nfcEnabled)
                } else {
                    strings.deviceNfcUnavailable()
                },
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenConnectedDevices) {
                    Text(strings.deviceUsbDevicesLabel())
                }
                Button(onClick = onOpenNfc, enabled = uiState.nfcSupported) {
                    Text(strings.deviceNfcSettingsLabel())
                }
            }
        }
    }
}

@Composable
private fun PermissionsAndRuntimeCard(
    uiState: DeviceUiState,
    onNotifications: () -> Unit,
    onOverlaySettings: () -> Unit,
    onToggleRuntime: (Boolean) -> Unit,
    onToggleFloatingButton: (Boolean) -> Unit,
) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(strings.deviceRuntimeTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                strings.deviceRuntimeSummary(
                    notificationPermissionGranted = uiState.notificationPermissionGranted,
                    runtimeServiceRunning = uiState.runtimeServiceRunning,
                ),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.deviceOverlayPermissionTitle(), style = MaterialTheme.typography.titleSmall,
            )
            Text(
                if (uiState.overlayPermissionGranted) {
                    strings.deviceOverlayGranted()
                } else {
                    strings.deviceOverlayDisabled()
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.deviceFloatingButtonTitle(),
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                strings.deviceFloatingButtonSummary(
                    overlayPermissionGranted = uiState.overlayPermissionGranted,
                    floatingButtonEnabled = uiState.floatingButtonEnabled,
                    floatingButtonRunning = uiState.floatingButtonRunning,
                ),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                strings.deviceResizableWindowTitle(), style = MaterialTheme.typography.titleSmall,
            )
            Text(
                strings.deviceResizableWindowSummary(
                    resizableWindowSupport = uiState.resizableWindowSupport,
                    freeformWindowSupported = uiState.freeformWindowSupported,
                ),
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onNotifications) {
                    Text(if (uiState.notificationPermissionGranted) strings.deviceNotificationSettingsLabel() else strings.deviceEnableNotificationsLabel())
                }
                Button(onClick = onOverlaySettings) {
                    Text(strings.deviceOverlaySettingsLabel())
                }
                Button(onClick = { onToggleRuntime(!uiState.backgroundPersistenceEnabled) }) {
                    Text(if (uiState.backgroundPersistenceEnabled) strings.deviceStopBackgroundRuntimeLabel() else strings.deviceStartBackgroundRuntimeLabel())
                }
                Button(onClick = { onToggleFloatingButton(!uiState.floatingButtonEnabled) }) {
                    Text(if (uiState.floatingButtonEnabled) strings.deviceStopFloatingButtonLabel() else strings.deviceStartFloatingButtonLabel())
                }
            }
            Text(
                strings.deviceBackgroundRuntimeDescription(),
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun WorkspaceAccessCard(
    uiState: DeviceUiState,
    onImportFile: () -> Unit,
    onGrantFolder: () -> Unit,
    onClearFolder: () -> Unit,
    onRefresh: () -> Unit,
    onExport: (String) -> Unit,
) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(strings.deviceWorkspaceAccessTitle(), style = MaterialTheme.typography.titleMedium)
            Text(strings.deviceWorkspaceAccessDescription())
            Text(strings.deviceSharedFolderLabel(uiState.sharedFolderLabel), style = MaterialTheme.typography.bodySmall)
            if (uiState.sharedFolderUri.isNotBlank()) {
                Text(uiState.sharedFolderUri, style = MaterialTheme.typography.bodySmall)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onImportFile, modifier = Modifier.weight(1f)) {
                    Text(strings.deviceImportFileLabel())
                }
                Button(onClick = onGrantFolder, modifier = Modifier.weight(1f)) {
                    Text(strings.deviceGrantFolderLabel())
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onRefresh, modifier = Modifier.weight(1f)) {
                    Text(strings.refresh.ifBlank { "Refresh" })
                }
                Button(
                    onClick = onClearFolder,
                    enabled = uiState.sharedFolderUri.isNotBlank(),
                    modifier = Modifier.weight(1f),
                ) {
                    Text(strings.deviceClearFolderLabel())
                }
            }
            if (uiState.workspaceFiles.isEmpty()) {
                Text(strings.deviceNoWorkspaceFiles(), style = MaterialTheme.typography.bodySmall)
            } else {
                uiState.workspaceFiles.forEach { file ->
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Text(file.name, style = MaterialTheme.typography.titleSmall)
                            Text(
                                strings.deviceWorkspaceFileUpdated(file.sizeLabel, file.modifiedLabel),
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Button(onClick = { onExport(file.name) }) {
                                Text(strings.deviceExportLabel())
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
private fun AccessibilityCard(
    uiState: DeviceUiState,
    onOpenSettings: () -> Unit,
    onAction: (HermesGlobalAction) -> Unit,
) {
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(strings.deviceAccessibilityTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                if (uiState.accessibilityEnabled) {
                    if (uiState.accessibilityConnected) {
                        strings.deviceAccessibilityConnected()
                    } else {
                        strings.deviceAccessibilityEnabledWaiting()
                    }
                } else {
                    strings.deviceAccessibilityDisabled()
                },
            )
            Button(onClick = onOpenSettings) {
                Text(strings.deviceOpenAccessibilitySettingsLabel())
            }
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                HermesGlobalAction.values().forEach { action ->
                    Button(onClick = { onAction(action) }) {
                        Text(strings.deviceGlobalActionLabel(action))
                    }
                }
            }
        }
    }
}

private fun HermesStrings.deviceNotificationPermissionStatus(granted: Boolean): String = when (language) {
    AppLanguage.CHINESE -> if (granted) "已为 Hermes 运行时提醒启用通知" else "通知权限被拒绝"
    AppLanguage.SPANISH -> if (granted) "Notificaciones activadas para las alertas del runtime de Hermes" else "Se denegó el permiso de notificaciones"
    AppLanguage.GERMAN -> if (granted) "Benachrichtigungen für Hermes-Laufzeitwarnungen aktiviert" else "Benachrichtigungsberechtigung wurde verweigert"
    AppLanguage.PORTUGUESE -> if (granted) "Notificações ativadas para alertas do runtime Hermes" else "A permissão de notificação foi negada"
    AppLanguage.FRENCH -> if (granted) "Notifications activées pour les alertes du runtime Hermes" else "L’autorisation de notification a été refusée"
    AppLanguage.ENGLISH -> if (granted) "Notifications enabled for Hermes runtime alerts" else "Notification permission was denied"
}

private fun HermesStrings.deviceBluetoothPermissionStatus(granted: Boolean): String = when (language) {
    AppLanguage.CHINESE -> if (granted) "已授予蓝牙访问权限" else "蓝牙访问被拒绝"
    AppLanguage.SPANISH -> if (granted) "Acceso Bluetooth concedido" else "Se denegó el acceso Bluetooth"
    AppLanguage.GERMAN -> if (granted) "Bluetooth-Zugriff gewährt" else "Bluetooth-Zugriff wurde verweigert"
    AppLanguage.PORTUGUESE -> if (granted) "Acesso Bluetooth concedido" else "O acesso Bluetooth foi negado"
    AppLanguage.FRENCH -> if (granted) "Accès Bluetooth accordé" else "L’accès Bluetooth a été refusé"
    AppLanguage.ENGLISH -> if (granted) "Bluetooth access granted" else "Bluetooth access was denied"
}

private fun HermesStrings.deviceRefreshDescription(): String = when (language) {
    AppLanguage.CHINESE -> "重新加载共享文件夹、Linux 套件和手机控制状态。"
    AppLanguage.SPANISH -> "Recarga el estado de carpeta compartida, suite Linux y controles del teléfono."
    AppLanguage.GERMAN -> "Lädt Status von freigegebenem Ordner, Linux-Suite und Telefonsteuerung neu."
    AppLanguage.PORTUGUESE -> "Recarrega o status de pasta compartilhada, suíte Linux e controles do telefone."
    AppLanguage.FRENCH -> "Recharge l’état du dossier partagé, de la suite Linux et des contrôles du téléphone."
    AppLanguage.ENGLISH -> "Reload shared-folder, Linux suite, and phone-control status."
}

private fun HermesStrings.deviceGrantSharedFolderLabel(): String = when (language) {
    AppLanguage.CHINESE -> "授权共享文件夹"
    AppLanguage.SPANISH -> "Conceder carpeta compartida"
    AppLanguage.GERMAN -> "Freigegebenen Ordner gewähren"
    AppLanguage.PORTUGUESE -> "Conceder pasta compartilhada"
    AppLanguage.FRENCH -> "Autoriser un dossier partagé"
    AppLanguage.ENGLISH -> "Grant shared folder"
}

private fun HermesStrings.deviceGrantSharedFolderDescription(): String = when (language) {
    AppLanguage.CHINESE -> "选择真实的 Android 文件夹，让 Hermes 直接访问文件。"
    AppLanguage.SPANISH -> "Elige una carpeta real de Android para acceso directo de Hermes."
    AppLanguage.GERMAN -> "Wählt einen echten Android-Ordner für direkten Hermes-Dateizugriff."
    AppLanguage.PORTUGUESE -> "Escolha uma pasta Android real para acesso direto do Hermes."
    AppLanguage.FRENCH -> "Choisit un vrai dossier Android pour l’accès direct de Hermes aux fichiers."
    AppLanguage.ENGLISH -> "Pick a real Android folder for direct Hermes file access."
}

private fun HermesStrings.deviceImportFileLabel(): String = when (language) {
    AppLanguage.CHINESE -> "导入文件"
    AppLanguage.SPANISH -> "Importar archivo"
    AppLanguage.GERMAN -> "Datei importieren"
    AppLanguage.PORTUGUESE -> "Importar arquivo"
    AppLanguage.FRENCH -> "Importer un fichier"
    AppLanguage.ENGLISH -> "Import file"
}

private fun HermesStrings.deviceImportFileDescription(): String = when (language) {
    AppLanguage.CHINESE -> "将文件带入 Hermes 工作区用于临时编辑。"
    AppLanguage.SPANISH -> "Trae un archivo al espacio de trabajo de Hermes para ediciones temporales."
    AppLanguage.GERMAN -> "Bringt eine Datei für temporäre Bearbeitungen in den Hermes-Arbeitsbereich."
    AppLanguage.PORTUGUESE -> "Traz um arquivo para o workspace Hermes para edições temporárias."
    AppLanguage.FRENCH -> "Ajoute un fichier à l’espace de travail Hermes pour les modifications temporaires."
    AppLanguage.ENGLISH -> "Bring a file into the Hermes workspace for scratch edits."
}

private fun HermesStrings.deviceNotificationSettingsLabel(): String = when (language) {
    AppLanguage.CHINESE -> "通知设置"
    AppLanguage.SPANISH -> "Ajustes de notificaciones"
    AppLanguage.GERMAN -> "Benachrichtigungseinstellungen"
    AppLanguage.PORTUGUESE -> "Configurações de notificação"
    AppLanguage.FRENCH -> "Réglages des notifications"
    AppLanguage.ENGLISH -> "Notification settings"
}

private fun HermesStrings.deviceNotificationSettingsDescription(): String = when (language) {
    AppLanguage.CHINESE -> "打开 Hermes 通知设置和后台控制。"
    AppLanguage.SPANISH -> "Abre los ajustes de notificaciones y controles en segundo plano de Hermes."
    AppLanguage.GERMAN -> "Öffnet Hermes-Benachrichtigungseinstellungen und Hintergrundsteuerung."
    AppLanguage.PORTUGUESE -> "Abre notificações do Hermes e controles em segundo plano."
    AppLanguage.FRENCH -> "Ouvre les réglages de notifications Hermes et les contrôles en arrière-plan."
    AppLanguage.ENGLISH -> "Open Hermes notification settings and background controls."
}

private fun HermesStrings.deviceModelRouting(provider: String, modelName: String, visionCapable: Boolean): String {
    val mode = when (language) {
        AppLanguage.CHINESE -> if (visionCapable) "视觉就绪" else "文本优先"
        AppLanguage.SPANISH -> if (visionCapable) "listo para visión" else "texto primero"
        AppLanguage.GERMAN -> if (visionCapable) "bildfähig" else "text zuerst"
        AppLanguage.PORTUGUESE -> if (visionCapable) "pronto para visão" else "texto primeiro"
        AppLanguage.FRENCH -> if (visionCapable) "prêt pour la vision" else "texte d’abord"
        AppLanguage.ENGLISH -> if (visionCapable) "vision-ready" else "text-first"
    }
    return when (language) {
        AppLanguage.CHINESE -> "模型路由：$provider / $modelName（$mode）"
        AppLanguage.SPANISH -> "Ruta del modelo: $provider / $modelName ($mode)"
        AppLanguage.GERMAN -> "Modellrouting: $provider / $modelName ($mode)"
        AppLanguage.PORTUGUESE -> "Roteamento de modelo: $provider / $modelName ($mode)"
        AppLanguage.FRENCH -> "Routage du modèle : $provider / $modelName ($mode)"
        AppLanguage.ENGLISH -> "Model routing: $provider / $modelName ($mode)"
    }
}

private fun HermesStrings.deviceLinuxSuiteTitle(): String = when (language) {
    AppLanguage.CHINESE -> "Linux 命令套件"
    AppLanguage.SPANISH -> "Suite de comandos Linux"
    AppLanguage.GERMAN -> "Linux-Befehlssuite"
    AppLanguage.PORTUGUESE -> "Suíte de comandos Linux"
    AppLanguage.FRENCH -> "Suite de commandes Linux"
    AppLanguage.ENGLISH -> "Linux command suite"
}

private fun HermesStrings.deviceLinuxSuiteReady(): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 可以使用已解压的 Linux 套件，通过 terminal/process 在本机执行完整 CLI 命令。"
    AppLanguage.SPANISH -> "Hermes puede ejecutar comandos CLI completos localmente con terminal/process usando la suite Linux extraída."
    AppLanguage.GERMAN -> "Hermes kann vollständige CLI-Befehle lokal mit terminal/process über die extrahierte Linux-Suite ausführen."
    AppLanguage.PORTUGUESE -> "O Hermes pode executar comandos CLI completos localmente com terminal/process usando a suíte Linux extraída."
    AppLanguage.FRENCH -> "Hermes peut exécuter localement des commandes CLI complètes avec terminal/process via la suite Linux extraite."
    AppLanguage.ENGLISH -> "Hermes can execute full CLI commands locally with terminal/process using the extracted Linux suite."
}

private fun HermesStrings.deviceLinuxSuiteProvisioning(): String = when (language) {
    AppLanguage.CHINESE -> "Linux 命令套件仍在配置。后端完成启动后请重试 Hermes。"
    AppLanguage.SPANISH -> "La suite de comandos Linux aún se está preparando. Reintenta Hermes cuando el backend termine de arrancar."
    AppLanguage.GERMAN -> "Die Linux-Befehlssuite wird noch bereitgestellt. Versuche Hermes erneut, sobald das Backend gestartet ist."
    AppLanguage.PORTUGUESE -> "A suíte de comandos Linux ainda está sendo preparada. Tente o Hermes novamente quando o backend terminar de iniciar."
    AppLanguage.FRENCH -> "La suite de commandes Linux est encore en préparation. Réessayez Hermes lorsque le backend a fini de démarrer."
    AppLanguage.ENGLISH -> "Linux command suite is still provisioning. Retry Hermes once the backend finishes booting."
}

private fun HermesStrings.deviceLinuxAbi(androidAbi: String, termuxArch: String): String = when (language) {
    AppLanguage.CHINESE -> "ABI：$androidAbi · 套件架构：$termuxArch"
    AppLanguage.SPANISH -> "ABI: $androidAbi · arquitectura de la suite: $termuxArch"
    AppLanguage.GERMAN -> "ABI: $androidAbi · Suite-Architektur: $termuxArch"
    AppLanguage.PORTUGUESE -> "ABI: $androidAbi · arquitetura da suíte: $termuxArch"
    AppLanguage.FRENCH -> "ABI : $androidAbi · architecture de la suite : $termuxArch"
    AppLanguage.ENGLISH -> "ABI: $androidAbi · suite arch: $termuxArch"
}

private fun HermesStrings.deviceLinuxPrefix(path: String): String = when (language) {
    AppLanguage.CHINESE -> "前缀：$path"
    AppLanguage.SPANISH -> "Prefijo: $path"
    AppLanguage.GERMAN -> "Präfix: $path"
    AppLanguage.PORTUGUESE -> "Prefixo: $path"
    AppLanguage.FRENCH -> "Préfixe : $path"
    AppLanguage.ENGLISH -> "Prefix: $path"
}

private fun HermesStrings.deviceLinuxBash(path: String): String = when (language) {
    AppLanguage.CHINESE -> "Bash：$path"
    AppLanguage.SPANISH -> "Bash: $path"
    AppLanguage.GERMAN -> "Bash: $path"
    AppLanguage.PORTUGUESE -> "Bash: $path"
    AppLanguage.FRENCH -> "Bash : $path"
    AppLanguage.ENGLISH -> "Bash: $path"
}

private fun HermesStrings.deviceLinuxHome(path: String): String = when (language) {
    AppLanguage.CHINESE -> "主目录：$path"
    AppLanguage.SPANISH -> "Inicio: $path"
    AppLanguage.GERMAN -> "Home: $path"
    AppLanguage.PORTUGUESE -> "Home: $path"
    AppLanguage.FRENCH -> "Accueil : $path"
    AppLanguage.ENGLISH -> "Home: $path"
}

private fun HermesStrings.deviceLinuxTemp(path: String): String = when (language) {
    AppLanguage.CHINESE -> "临时目录：$path"
    AppLanguage.SPANISH -> "Temporal: $path"
    AppLanguage.GERMAN -> "Temp: $path"
    AppLanguage.PORTUGUESE -> "Temp: $path"
    AppLanguage.FRENCH -> "Temp : $path"
    AppLanguage.ENGLISH -> "Temp: $path"
}

private fun HermesStrings.deviceLinuxPackageCount(count: Int): String = when (language) {
    AppLanguage.CHINESE -> "包含的软件包数量：$count"
    AppLanguage.SPANISH -> "Cantidad de paquetes incluidos: $count"
    AppLanguage.GERMAN -> "Enthaltene Pakete: $count"
    AppLanguage.PORTUGUESE -> "Quantidade de pacotes incluídos: $count"
    AppLanguage.FRENCH -> "Nombre de paquets inclus : $count"
    AppLanguage.ENGLISH -> "Included package count: $count"
}

private fun HermesStrings.deviceLinuxTerminalGuidance(): String = when (language) {
    AppLanguage.CHINESE -> "可以让 Hermes 使用 terminal 在此套件中直接运行 git status、ls、curl、grep 或更长的 shell 管道。"
    AppLanguage.SPANISH -> "Pide a Hermes usar terminal para comandos como git status, ls, curl, grep o pipelines de shell más largos directamente en esta suite."
    AppLanguage.GERMAN -> "Bitte Hermes, terminal für Befehle wie git status, ls, curl, grep oder längere Shell-Pipelines direkt in dieser Suite zu nutzen."
    AppLanguage.PORTUGUESE -> "Peça ao Hermes para usar terminal em comandos como git status, ls, curl, grep ou pipelines de shell maiores diretamente nesta suíte."
    AppLanguage.FRENCH -> "Demandez à Hermes d’utiliser terminal pour git status, ls, curl, grep ou des pipelines shell plus longs directement dans cette suite."
    AppLanguage.ENGLISH -> "Ask Hermes to use terminal for commands like 'git status', 'ls', 'curl', 'grep', or longer shell pipelines directly in this suite."
}

private fun HermesStrings.deviceConnectivityTitle(): String = when (language) {
    AppLanguage.CHINESE -> "Wi-Fi 与连接"
    AppLanguage.SPANISH -> "Wi-Fi y conectividad"
    AppLanguage.GERMAN -> "Wi-Fi und Konnektivität"
    AppLanguage.PORTUGUESE -> "Wi-Fi e conectividade"
    AppLanguage.FRENCH -> "Wi-Fi et connectivité"
    AppLanguage.ENGLISH -> "Wi-Fi + connectivity"
}

private fun HermesStrings.deviceNetworkSummary(activeNetworkLabel: String, wifiEnabled: Boolean): String = when (language) {
    AppLanguage.CHINESE -> "网络：$activeNetworkLabel · Wi-Fi ${if (wifiEnabled) "已开启" else "已关闭"}。Hermes 使用 Android 安全设置面板，而不是不受支持的直接无线电切换。"
    AppLanguage.SPANISH -> "Red: $activeNetworkLabel · Wi-Fi ${if (wifiEnabled) "activado" else "desactivado"}. Hermes usa paneles seguros de Android en lugar de toggles directos no soportados."
    AppLanguage.GERMAN -> "Netzwerk: $activeNetworkLabel · Wi-Fi ist ${if (wifiEnabled) "ein" else "aus"}. Hermes nutzt Android-sichere Einstellungsansichten statt nicht unterstützter Direktumschaltungen."
    AppLanguage.PORTUGUESE -> "Rede: $activeNetworkLabel · Wi-Fi ${if (wifiEnabled) "ativado" else "desativado"}. O Hermes usa painéis seguros do Android em vez de alternâncias diretas não suportadas."
    AppLanguage.FRENCH -> "Réseau : $activeNetworkLabel · Wi-Fi ${if (wifiEnabled) "activé" else "désactivé"}. Hermes utilise des panneaux Android sûrs plutôt que des bascules radio directes non prises en charge."
    AppLanguage.ENGLISH -> "Network: $activeNetworkLabel · Wi-Fi is ${if (wifiEnabled) "on" else "off"}. Hermes uses Android-safe settings panels instead of unsupported direct radio toggles."
}

private fun HermesStrings.deviceBluetoothTitle(): String = when (language) {
    AppLanguage.CHINESE -> "蓝牙"
    AppLanguage.SPANISH -> "Bluetooth"
    AppLanguage.GERMAN -> "Bluetooth"
    AppLanguage.PORTUGUESE -> "Bluetooth"
    AppLanguage.FRENCH -> "Bluetooth"
    AppLanguage.ENGLISH -> "Bluetooth"
}

private fun HermesStrings.deviceBluetoothUnavailable(): String = when (language) {
    AppLanguage.CHINESE -> "此设备没有可用的蓝牙无线电。"
    AppLanguage.SPANISH -> "La radio Bluetooth no está disponible en este dispositivo."
    AppLanguage.GERMAN -> "Bluetooth-Funk ist auf diesem Gerät nicht verfügbar."
    AppLanguage.PORTUGUESE -> "O rádio Bluetooth não está disponível neste dispositivo."
    AppLanguage.FRENCH -> "La radio Bluetooth n’est pas disponible sur cet appareil."
    AppLanguage.ENGLISH -> "Bluetooth radio is not available on this device."
}

private fun HermesStrings.deviceBluetoothPermissionPrompt(): String = when (language) {
    AppLanguage.CHINESE -> "授予蓝牙访问权限后，Hermes 可在打开设置前读取已配对设备状态。"
    AppLanguage.SPANISH -> "Concede acceso Bluetooth para que Hermes lea el estado de dispositivos vinculados antes de abrir ajustes."
    AppLanguage.GERMAN -> "Gewähre Bluetooth-Zugriff, damit Hermes gekoppelte Geräte vor dem Öffnen der Einstellungen lesen kann."
    AppLanguage.PORTUGUESE -> "Conceda acesso Bluetooth para o Hermes ler o estado de dispositivos pareados antes de abrir configurações."
    AppLanguage.FRENCH -> "Autorisez Bluetooth afin que Hermes lise l’état des appareils associés avant d’ouvrir les réglages."
    AppLanguage.ENGLISH -> "Grant Bluetooth access so Hermes can read bonded-device state before opening settings."
}

private fun HermesStrings.deviceBluetoothSummary(enabled: Boolean, bondedDevices: String): String {
    val state = when (language) {
        AppLanguage.CHINESE -> if (enabled) "已启用" else "已禁用"
        AppLanguage.SPANISH -> if (enabled) "activado" else "desactivado"
        AppLanguage.GERMAN -> if (enabled) "aktiviert" else "deaktiviert"
        AppLanguage.PORTUGUESE -> if (enabled) "ativado" else "desativado"
        AppLanguage.FRENCH -> if (enabled) "activé" else "désactivé"
        AppLanguage.ENGLISH -> if (enabled) "enabled" else "disabled"
    }
    val devices = bondedDevices.ifBlank {
        when (language) {
            AppLanguage.CHINESE -> "无"
            AppLanguage.SPANISH -> "ninguno"
            AppLanguage.GERMAN -> "keine"
            AppLanguage.PORTUGUESE -> "nenhum"
            AppLanguage.FRENCH -> "aucun"
            AppLanguage.ENGLISH -> "none"
        }
    }
    return when (language) {
        AppLanguage.CHINESE -> "蓝牙$state。已配对设备：$devices"
        AppLanguage.SPANISH -> "Bluetooth $state. Dispositivos vinculados: $devices"
        AppLanguage.GERMAN -> "Bluetooth ist $state. Gekoppelte Geräte: $devices"
        AppLanguage.PORTUGUESE -> "Bluetooth $state. Dispositivos pareados: $devices"
        AppLanguage.FRENCH -> "Bluetooth $state. Appareils associés : $devices"
        AppLanguage.ENGLISH -> "Bluetooth is $state. Bonded devices: $devices"
    }
}

private fun HermesStrings.deviceInternetPanelLabel(): String = when (language) {
    AppLanguage.CHINESE -> "互联网面板"
    AppLanguage.SPANISH -> "Panel de Internet"
    AppLanguage.GERMAN -> "Internetansicht"
    AppLanguage.PORTUGUESE -> "Painel de internet"
    AppLanguage.FRENCH -> "Panneau Internet"
    AppLanguage.ENGLISH -> "Internet panel"
}

private fun HermesStrings.deviceConnectedDevicesLabel(): String = when (language) {
    AppLanguage.CHINESE -> "已连接设备"
    AppLanguage.SPANISH -> "Dispositivos conectados"
    AppLanguage.GERMAN -> "Verbundene Geräte"
    AppLanguage.PORTUGUESE -> "Dispositivos conectados"
    AppLanguage.FRENCH -> "Appareils connectés"
    AppLanguage.ENGLISH -> "Connected devices"
}

private fun HermesStrings.deviceInterfaceTitle(): String = when (language) {
    AppLanguage.CHINESE -> "USB 与 NFC"
    AppLanguage.SPANISH -> "USB y NFC"
    AppLanguage.GERMAN -> "USB und NFC"
    AppLanguage.PORTUGUESE -> "USB e NFC"
    AppLanguage.FRENCH -> "USB et NFC"
    AppLanguage.ENGLISH -> "USB + NFC"
}

private fun HermesStrings.deviceUsbSummary(usbDeviceCount: Int, usbDevices: String): String {
    val devices = usbDevices.ifBlank {
        when (language) {
            AppLanguage.CHINESE -> "当前未检测到 USB 设备。"
            AppLanguage.SPANISH -> "No se detectan dispositivos USB ahora."
            AppLanguage.GERMAN -> "Derzeit keine USB-Geräte erkannt."
            AppLanguage.PORTUGUESE -> "Nenhum dispositivo USB detectado agora."
            AppLanguage.FRENCH -> "Aucun appareil USB détecté pour l’instant."
            AppLanguage.ENGLISH -> "No USB devices detected right now."
        }
    }
    return when (language) {
        AppLanguage.CHINESE -> "USB 主机模式可用。已连接 USB 设备：$usbDeviceCount。$devices"
        AppLanguage.SPANISH -> "El modo host USB está disponible. Dispositivos USB conectados: $usbDeviceCount. $devices"
        AppLanguage.GERMAN -> "USB-Hostmodus ist verfügbar. Verbundene USB-Geräte: $usbDeviceCount. $devices"
        AppLanguage.PORTUGUESE -> "Modo host USB disponível. Dispositivos USB conectados: $usbDeviceCount. $devices"
        AppLanguage.FRENCH -> "Le mode hôte USB est disponible. Appareils USB connectés : $usbDeviceCount. $devices"
        AppLanguage.ENGLISH -> "USB host mode is available. Connected USB devices: $usbDeviceCount. $devices"
    }
}

private fun HermesStrings.deviceUsbUnavailable(): String = when (language) {
    AppLanguage.CHINESE -> "此设备构建未声明 USB 主机模式。"
    AppLanguage.SPANISH -> "Esta compilación del dispositivo no anuncia el modo host USB."
    AppLanguage.GERMAN -> "Dieser Geräte-Build meldet keinen USB-Hostmodus."
    AppLanguage.PORTUGUESE -> "Esta build do dispositivo não anuncia modo host USB."
    AppLanguage.FRENCH -> "Cette version de l’appareil ne déclare pas le mode hôte USB."
    AppLanguage.ENGLISH -> "USB host mode is not advertised on this device build."
}

private fun HermesStrings.deviceNfcSummary(enabled: Boolean): String = when (language) {
    AppLanguage.CHINESE -> "NFC ${if (enabled) "已启用" else "已禁用"}。Hermes 可以显示 NFC 状态并直接带你进入系统设置。"
    AppLanguage.SPANISH -> "NFC está ${if (enabled) "activado" else "desactivado"}. Hermes puede mostrar el estado NFC y llevarte directo a ajustes del sistema."
    AppLanguage.GERMAN -> "NFC ist ${if (enabled) "aktiviert" else "deaktiviert"}. Hermes kann den NFC-Status anzeigen und direkt zu den Systemeinstellungen führen."
    AppLanguage.PORTUGUESE -> "NFC está ${if (enabled) "ativado" else "desativado"}. O Hermes pode mostrar o estado NFC e levar direto às configurações do sistema."
    AppLanguage.FRENCH -> "NFC est ${if (enabled) "activé" else "désactivé"}. Hermes peut afficher l’état NFC et ouvrir directement les réglages système."
    AppLanguage.ENGLISH -> "NFC is ${if (enabled) "enabled" else "disabled"}. Hermes can surface NFC state and take you straight to system settings."
}

private fun HermesStrings.deviceNfcUnavailable(): String = when (language) {
    AppLanguage.CHINESE -> "此设备没有可用的 NFC 硬件。"
    AppLanguage.SPANISH -> "El hardware NFC no está disponible en este dispositivo."
    AppLanguage.GERMAN -> "NFC-Hardware ist auf diesem Gerät nicht verfügbar."
    AppLanguage.PORTUGUESE -> "Hardware NFC não disponível neste dispositivo."
    AppLanguage.FRENCH -> "Le matériel NFC n’est pas disponible sur cet appareil."
    AppLanguage.ENGLISH -> "NFC hardware is not available on this device."
}

private fun HermesStrings.deviceUsbDevicesLabel(): String = when (language) {
    AppLanguage.CHINESE -> "USB / 设备"
    AppLanguage.SPANISH -> "USB / dispositivos"
    AppLanguage.GERMAN -> "USB / Geräte"
    AppLanguage.PORTUGUESE -> "USB / dispositivos"
    AppLanguage.FRENCH -> "USB / appareils"
    AppLanguage.ENGLISH -> "USB / devices"
}

private fun HermesStrings.deviceNfcSettingsLabel(): String = when (language) {
    AppLanguage.CHINESE -> "NFC 设置"
    AppLanguage.SPANISH -> "Ajustes NFC"
    AppLanguage.GERMAN -> "NFC-Einstellungen"
    AppLanguage.PORTUGUESE -> "Configurações de NFC"
    AppLanguage.FRENCH -> "Réglages NFC"
    AppLanguage.ENGLISH -> "NFC settings"
}

private fun HermesStrings.deviceRuntimeTitle(): String = when (language) {
    AppLanguage.CHINESE -> "通知与后台运行时"
    AppLanguage.SPANISH -> "Notificaciones y runtime en segundo plano"
    AppLanguage.GERMAN -> "Benachrichtigungen und Hintergrundlaufzeit"
    AppLanguage.PORTUGUESE -> "Notificações e runtime em segundo plano"
    AppLanguage.FRENCH -> "Notifications et runtime en arrière-plan"
    AppLanguage.ENGLISH -> "Notifications + background runtime"
}

private fun HermesStrings.deviceRuntimeSummary(notificationPermissionGranted: Boolean, runtimeServiceRunning: Boolean): String = when (language) {
    AppLanguage.CHINESE -> "通知权限${if (notificationPermissionGranted) "已授予" else "未授予"}。Hermes 后台运行时${if (runtimeServiceRunning) "处于活动状态" else "未活动"}。"
    AppLanguage.SPANISH -> "Permiso de notificaciones ${if (notificationPermissionGranted) "concedido" else "no concedido"}. El runtime en segundo plano de Hermes está ${if (runtimeServiceRunning) "activo" else "inactivo"}."
    AppLanguage.GERMAN -> "Benachrichtigungsberechtigung ${if (notificationPermissionGranted) "erteilt" else "nicht erteilt"}. Hermes-Hintergrundlaufzeit ist ${if (runtimeServiceRunning) "aktiv" else "inaktiv"}."
    AppLanguage.PORTUGUESE -> "Permissão de notificação ${if (notificationPermissionGranted) "concedida" else "não concedida"}. O runtime em segundo plano do Hermes está ${if (runtimeServiceRunning) "ativo" else "inativo"}."
    AppLanguage.FRENCH -> "Autorisation de notification ${if (notificationPermissionGranted) "accordée" else "non accordée"}. Le runtime Hermes en arrière-plan est ${if (runtimeServiceRunning) "actif" else "inactif"}."
    AppLanguage.ENGLISH -> "Notification permission is ${if (notificationPermissionGranted) "granted" else "not granted"}. Hermes background runtime is ${if (runtimeServiceRunning) "active" else "inactive"}."
}

private fun HermesStrings.deviceOverlayPermissionTitle(): String = when (language) {
    AppLanguage.CHINESE -> "悬浮窗权限"
    AppLanguage.SPANISH -> "Permiso de superposición"
    AppLanguage.GERMAN -> "Overlay-Berechtigung"
    AppLanguage.PORTUGUESE -> "Permissão de sobreposição"
    AppLanguage.FRENCH -> "Autorisation de superposition"
    AppLanguage.ENGLISH -> "Overlay permission"
}

private fun HermesStrings.deviceOverlayGranted(): String = when (language) {
    AppLanguage.CHINESE -> "已授予未来浮动工具所需的悬浮窗权限。"
    AppLanguage.SPANISH -> "El permiso de superposición está concedido para futuras utilidades flotantes."
    AppLanguage.GERMAN -> "Overlay-Berechtigung ist für zukünftige schwebende Werkzeuge erteilt."
    AppLanguage.PORTUGUESE -> "Permissão de sobreposição concedida para futuras utilidades flutuantes."
    AppLanguage.FRENCH -> "L’autorisation de superposition est accordée pour les futurs outils flottants."
    AppLanguage.ENGLISH -> "Overlay permission is granted for future floating utilities."
}

private fun HermesStrings.deviceOverlayDisabled(): String = when (language) {
    AppLanguage.CHINESE -> "悬浮窗权限已禁用。如需未来浮动控制，请打开 Android 设置。"
    AppLanguage.SPANISH -> "El permiso de superposición está desactivado. Abre ajustes de Android si quieres futuros controles flotantes."
    AppLanguage.GERMAN -> "Overlay-Berechtigung ist deaktiviert. Öffne Android-Einstellungen für zukünftige schwebende Steuerungen."
    AppLanguage.PORTUGUESE -> "Permissão de sobreposição desativada. Abra as configurações do Android para futuros controles flutuantes."
    AppLanguage.FRENCH -> "L’autorisation de superposition est désactivée. Ouvrez les réglages Android pour les futurs contrôles flottants."
    AppLanguage.ENGLISH -> "Overlay permission is disabled. Open Android settings if you want future floating controls."
}

private fun HermesStrings.deviceFloatingButtonTitle(): String = when (language) {
    AppLanguage.CHINESE -> "浮动 Hermes 按钮"
    AppLanguage.SPANISH -> "Botón flotante de Hermes"
    AppLanguage.GERMAN -> "Schwebende Hermes-Schaltfläche"
    AppLanguage.PORTUGUESE -> "Botão flutuante do Hermes"
    AppLanguage.FRENCH -> "Bouton flottant Hermes"
    AppLanguage.ENGLISH -> "Floating Hermes button"
}

private fun HermesStrings.deviceFloatingButtonSummary(
    overlayPermissionGranted: Boolean,
    floatingButtonEnabled: Boolean,
    floatingButtonRunning: Boolean,
): String {
    if (!overlayPermissionGranted) {
        return floatingOverlayPermissionHint()
    }
    return when (language) {
        AppLanguage.CHINESE -> "浮动按钮${if (floatingButtonEnabled) "已启用" else "已关闭"}，服务${if (floatingButtonRunning) "正在运行" else "未运行"}。启用后，Hermes 会在主屏幕和其他应用上层保留一个可拖动入口。"
        AppLanguage.SPANISH -> "El botón flotante está ${if (floatingButtonEnabled) "activado" else "desactivado"} y el servicio está ${if (floatingButtonRunning) "en ejecución" else "detenido"}. Cuando está activo, Hermes mantiene un acceso arrastrable en Inicio y sobre otras apps."
        AppLanguage.GERMAN -> "Die schwebende Schaltfläche ist ${if (floatingButtonEnabled) "aktiviert" else "deaktiviert"} und der Dienst ${if (floatingButtonRunning) "läuft" else "läuft nicht"}. Aktiv hält Hermes einen ziehbaren Einstieg auf dem Startbildschirm und über anderen Apps bereit."
        AppLanguage.PORTUGUESE -> "O botão flutuante está ${if (floatingButtonEnabled) "ativado" else "desativado"} e o serviço está ${if (floatingButtonRunning) "em execução" else "parado"}. Ativo, o Hermes mantém um atalho arrastável na tela inicial e sobre outros apps."
        AppLanguage.FRENCH -> "Le bouton flottant est ${if (floatingButtonEnabled) "activé" else "désactivé"} et le service est ${if (floatingButtonRunning) "en cours" else "arrêté"}. Activé, Hermes garde un accès déplaçable sur l’accueil et au-dessus des autres apps."
        AppLanguage.ENGLISH -> "Floating button is ${if (floatingButtonEnabled) "enabled" else "off"} and the service is ${if (floatingButtonRunning) "running" else "stopped"}. When enabled, Hermes keeps a draggable entry point on Home and over other apps."
    }
}

private fun HermesStrings.deviceResizableWindowTitle(): String = when (language) {
    AppLanguage.CHINESE -> "可调整窗口支持"
    AppLanguage.SPANISH -> "Compatibilidad con ventana redimensionable"
    AppLanguage.GERMAN -> "Unterstützung für größenänderbare Fenster"
    AppLanguage.PORTUGUESE -> "Suporte a janela redimensionável"
    AppLanguage.FRENCH -> "Prise en charge des fenêtres redimensionnables"
    AppLanguage.ENGLISH -> "Resizable window support"
}

private fun HermesStrings.deviceResizableWindowSummary(resizableWindowSupport: Boolean, freeformWindowSupported: Boolean): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 声明可调整窗口支持：${if (resizableWindowSupport) "已启用" else "已禁用"}。此设备上的自由窗口/多窗口功能：${if (freeformWindowSupported) "可用" else "不可用"}。"
    AppLanguage.SPANISH -> "Hermes declara ventana redimensionable: ${if (resizableWindowSupport) "activada" else "desactivada"}. Función de ventana libre/multiventana disponible: ${if (freeformWindowSupported) "sí" else "no"}."
    AppLanguage.GERMAN -> "Hermes meldet größenänderbare Fenster: ${if (resizableWindowSupport) "aktiviert" else "deaktiviert"}. Freiform-/Mehrfensterfunktion auf diesem Gerät: ${if (freeformWindowSupported) "ja" else "nein"}."
    AppLanguage.PORTUGUESE -> "Hermes declara suporte a janela redimensionável: ${if (resizableWindowSupport) "ativado" else "desativado"}. Recurso de janela livre/multijanela neste dispositivo: ${if (freeformWindowSupported) "sim" else "não"}."
    AppLanguage.FRENCH -> "Hermes déclare la prise en charge des fenêtres redimensionnables : ${if (resizableWindowSupport) "activée" else "désactivée"}. Fenêtre libre/multifenêtre disponible sur cet appareil : ${if (freeformWindowSupported) "oui" else "non"}."
    AppLanguage.ENGLISH -> "Hermes declares resizable window support: ${if (resizableWindowSupport) "enabled" else "disabled"}. Freeform/multi-window feature available on this device: ${if (freeformWindowSupported) "yes" else "no"}."
}

private fun HermesStrings.deviceEnableNotificationsLabel(): String = when (language) {
    AppLanguage.CHINESE -> "启用通知"
    AppLanguage.SPANISH -> "Activar notificaciones"
    AppLanguage.GERMAN -> "Benachrichtigungen aktivieren"
    AppLanguage.PORTUGUESE -> "Ativar notificações"
    AppLanguage.FRENCH -> "Activer les notifications"
    AppLanguage.ENGLISH -> "Enable notifications"
}

private fun HermesStrings.deviceOverlaySettingsLabel(): String = when (language) {
    AppLanguage.CHINESE -> "悬浮窗设置"
    AppLanguage.SPANISH -> "Ajustes de superposición"
    AppLanguage.GERMAN -> "Overlay-Einstellungen"
    AppLanguage.PORTUGUESE -> "Configurações de sobreposição"
    AppLanguage.FRENCH -> "Réglages de superposition"
    AppLanguage.ENGLISH -> "Overlay settings"
}

private fun HermesStrings.deviceStopBackgroundRuntimeLabel(): String = when (language) {
    AppLanguage.CHINESE -> "停止后台运行时"
    AppLanguage.SPANISH -> "Detener runtime en segundo plano"
    AppLanguage.GERMAN -> "Hintergrundlaufzeit stoppen"
    AppLanguage.PORTUGUESE -> "Parar runtime em segundo plano"
    AppLanguage.FRENCH -> "Arrêter le runtime en arrière-plan"
    AppLanguage.ENGLISH -> "Stop background runtime"
}

private fun HermesStrings.deviceStartBackgroundRuntimeLabel(): String = when (language) {
    AppLanguage.CHINESE -> "启动后台运行时"
    AppLanguage.SPANISH -> "Iniciar runtime en segundo plano"
    AppLanguage.GERMAN -> "Hintergrundlaufzeit starten"
    AppLanguage.PORTUGUESE -> "Iniciar runtime em segundo plano"
    AppLanguage.FRENCH -> "Démarrer le runtime en arrière-plan"
    AppLanguage.ENGLISH -> "Start background runtime"
}

private fun HermesStrings.deviceStopFloatingButtonLabel(): String = when (language) {
    AppLanguage.CHINESE -> "停止浮动按钮"
    AppLanguage.SPANISH -> "Detener botón flotante"
    AppLanguage.GERMAN -> "Schaltfläche stoppen"
    AppLanguage.PORTUGUESE -> "Parar botão flutuante"
    AppLanguage.FRENCH -> "Arrêter le bouton flottant"
    AppLanguage.ENGLISH -> "Stop floating button"
}

private fun HermesStrings.deviceStartFloatingButtonLabel(): String = when (language) {
    AppLanguage.CHINESE -> "启动浮动按钮"
    AppLanguage.SPANISH -> "Iniciar botón flotante"
    AppLanguage.GERMAN -> "Schaltfläche starten"
    AppLanguage.PORTUGUESE -> "Iniciar botão flutuante"
    AppLanguage.FRENCH -> "Démarrer le bouton flottant"
    AppLanguage.ENGLISH -> "Start floating button"
}

private fun HermesStrings.deviceBackgroundRuntimeDescription(): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 后台运行时会在通知栏中保持本地后端就绪，以支持更长会话和后续 Android 窗口模式。"
    AppLanguage.SPANISH -> "El runtime en segundo plano de Hermes mantiene el backend local listo en la barra de notificaciones para sesiones largas y futuros modos de ventana Android."
    AppLanguage.GERMAN -> "Die Hermes-Hintergrundlaufzeit hält das lokale Backend in der Benachrichtigungsleiste für längere Sitzungen und spätere Android-Fenstermodi bereit."
    AppLanguage.PORTUGUESE -> "O runtime em segundo plano do Hermes mantém o backend local pronto na barra de notificações para sessões longas e futuros modos de janela Android."
    AppLanguage.FRENCH -> "Le runtime Hermes en arrière-plan garde le backend local prêt dans la barre de notifications pour les longues sessions et les futurs modes de fenêtre Android."
    AppLanguage.ENGLISH -> "Hermes background runtime keeps the local backend ready in the notification bar for longer sessions and later Android windowing modes."
}

private fun HermesStrings.deviceWorkspaceAccessTitle(): String = when (language) {
    AppLanguage.CHINESE -> "共享文件夹与工作区访问"
    AppLanguage.SPANISH -> "Carpeta compartida y acceso al espacio de trabajo"
    AppLanguage.GERMAN -> "Freigegebener Ordner und Arbeitsbereichszugriff"
    AppLanguage.PORTUGUESE -> "Pasta compartilhada e acesso ao workspace"
    AppLanguage.FRENCH -> "Dossier partagé et accès à l’espace de travail"
    AppLanguage.ENGLISH -> "Shared folder + workspace access"
}

private fun HermesStrings.deviceWorkspaceAccessDescription(): String = when (language) {
    AppLanguage.CHINESE -> "授权共享文件夹，让 Hermes 可直接读取和写入真实文件。需要副本时，导入的文件仍会进入 Hermes 工作区；terminal/process 现在覆盖通用 CLI 工作。"
    AppLanguage.SPANISH -> "Concede una carpeta compartida para que Hermes lea y escriba archivos reales directamente. Los archivos importados siguen en el espacio de trabajo de Hermes cuando quieras copias, mientras terminal/process cubre trabajo CLI general."
    AppLanguage.GERMAN -> "Gewähre einen freigegebenen Ordner, damit Hermes echte Dateien direkt lesen und schreiben kann. Importierte Dateien landen weiterhin im Hermes-Arbeitsbereich, wenn Kopien gewünscht sind; terminal/process deckt allgemeine CLI-Arbeit ab."
    AppLanguage.PORTUGUESE -> "Conceda uma pasta compartilhada para o Hermes ler e escrever arquivos reais diretamente. Arquivos importados ainda ficam no workspace Hermes quando você quiser cópias, enquanto terminal/process cobre trabalho CLI geral."
    AppLanguage.FRENCH -> "Autorisez un dossier partagé pour que Hermes lise et écrive directement les vrais fichiers. Les fichiers importés restent dans l’espace de travail Hermes lorsque vous voulez des copies, tandis que terminal/process couvre le travail CLI général."
    AppLanguage.ENGLISH -> "Grant a shared folder to let Hermes read and write the real files directly. Imported files still land in the Hermes workspace when you want copies instead, while terminal/process now cover general CLI work."
}

private fun HermesStrings.deviceSharedFolderLabel(label: String): String = when (language) {
    AppLanguage.CHINESE -> "共享文件夹：$label"
    AppLanguage.SPANISH -> "Carpeta compartida: $label"
    AppLanguage.GERMAN -> "Freigegebener Ordner: $label"
    AppLanguage.PORTUGUESE -> "Pasta compartilhada: $label"
    AppLanguage.FRENCH -> "Dossier partagé : $label"
    AppLanguage.ENGLISH -> "Shared folder: $label"
}

private fun HermesStrings.deviceGrantFolderLabel(): String = when (language) {
    AppLanguage.CHINESE -> "授权文件夹"
    AppLanguage.SPANISH -> "Conceder carpeta"
    AppLanguage.GERMAN -> "Ordner gewähren"
    AppLanguage.PORTUGUESE -> "Conceder pasta"
    AppLanguage.FRENCH -> "Autoriser le dossier"
    AppLanguage.ENGLISH -> "Grant folder"
}

private fun HermesStrings.deviceClearFolderLabel(): String = when (language) {
    AppLanguage.CHINESE -> "清除文件夹"
    AppLanguage.SPANISH -> "Borrar carpeta"
    AppLanguage.GERMAN -> "Ordner löschen"
    AppLanguage.PORTUGUESE -> "Limpar pasta"
    AppLanguage.FRENCH -> "Effacer le dossier"
    AppLanguage.ENGLISH -> "Clear folder"
}

private fun HermesStrings.deviceNoWorkspaceFiles(): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 工作区中还没有文件。"
    AppLanguage.SPANISH -> "Aún no hay archivos en el espacio de trabajo de Hermes."
    AppLanguage.GERMAN -> "Noch keine Dateien im Hermes-Arbeitsbereich."
    AppLanguage.PORTUGUESE -> "Ainda não há arquivos no workspace Hermes."
    AppLanguage.FRENCH -> "Aucun fichier dans l’espace de travail Hermes pour l’instant."
    AppLanguage.ENGLISH -> "No files in the Hermes workspace yet."
}

private fun HermesStrings.deviceWorkspaceFileUpdated(sizeLabel: String, modifiedLabel: String): String = when (language) {
    AppLanguage.CHINESE -> "$sizeLabel · 更新时间 $modifiedLabel"
    AppLanguage.SPANISH -> "$sizeLabel · actualizado $modifiedLabel"
    AppLanguage.GERMAN -> "$sizeLabel · aktualisiert $modifiedLabel"
    AppLanguage.PORTUGUESE -> "$sizeLabel · atualizado $modifiedLabel"
    AppLanguage.FRENCH -> "$sizeLabel · mis à jour $modifiedLabel"
    AppLanguage.ENGLISH -> "$sizeLabel · updated $modifiedLabel"
}

private fun HermesStrings.deviceExportLabel(): String = when (language) {
    AppLanguage.CHINESE -> "导出"
    AppLanguage.SPANISH -> "Exportar"
    AppLanguage.GERMAN -> "Exportieren"
    AppLanguage.PORTUGUESE -> "Exportar"
    AppLanguage.FRENCH -> "Exporter"
    AppLanguage.ENGLISH -> "Export"
}

private fun HermesStrings.deviceAccessibilityTitle(): String = when (language) {
    AppLanguage.CHINESE -> "无障碍控制"
    AppLanguage.SPANISH -> "Control de accesibilidad"
    AppLanguage.GERMAN -> "Bedienungshilfensteuerung"
    AppLanguage.PORTUGUESE -> "Controle de acessibilidade"
    AppLanguage.FRENCH -> "Contrôle d’accessibilité"
    AppLanguage.ENGLISH -> "Accessibility control"
}

private fun HermesStrings.deviceAccessibilityConnected(): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 无障碍已启用并已连接。Hermes 可使用 android_ui_snapshot 检查可见界面，并使用 android_ui_action 定位控件。"
    AppLanguage.SPANISH -> "La accesibilidad de Hermes está activada y conectada. Hermes puede inspeccionar la UI visible con android_ui_snapshot y apuntar controles con android_ui_action."
    AppLanguage.GERMAN -> "Hermes-Bedienungshilfe ist aktiviert und verbunden. Hermes kann die sichtbare UI mit android_ui_snapshot prüfen und Steuerelemente mit android_ui_action ansteuern."
    AppLanguage.PORTUGUESE -> "A acessibilidade do Hermes está ativada e conectada. O Hermes pode inspecionar a UI visível com android_ui_snapshot e mirar controles com android_ui_action."
    AppLanguage.FRENCH -> "L’accessibilité Hermes est activée et connectée. Hermes peut inspecter l’UI visible avec android_ui_snapshot et cibler les contrôles avec android_ui_action."
    AppLanguage.ENGLISH -> "Hermes accessibility is enabled and connected. Hermes can inspect the visible UI with android_ui_snapshot and target controls with android_ui_action."
}

private fun HermesStrings.deviceAccessibilityEnabledWaiting(): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 无障碍已启用，但 Android 尚未连接该服务。"
    AppLanguage.SPANISH -> "La accesibilidad de Hermes está activada, pero Android aún no ha conectado el servicio."
    AppLanguage.GERMAN -> "Hermes-Bedienungshilfe ist aktiviert, aber Android hat den Dienst noch nicht verbunden."
    AppLanguage.PORTUGUESE -> "A acessibilidade do Hermes está ativada, mas o Android ainda não conectou o serviço."
    AppLanguage.FRENCH -> "L’accessibilité Hermes est activée, mais Android n’a pas encore connecté le service."
    AppLanguage.ENGLISH -> "Hermes accessibility is enabled, but Android has not connected the service yet."
}

private fun HermesStrings.deviceAccessibilityDisabled(): String = when (language) {
    AppLanguage.CHINESE -> "Hermes 无障碍已禁用。在 Android 设置中启用后，可解锁快捷设备操作以及界面检查/操作定位。"
    AppLanguage.SPANISH -> "La accesibilidad de Hermes está desactivada. Actívala en ajustes de Android para desbloquear acciones rápidas del dispositivo e inspección/acción de UI."
    AppLanguage.GERMAN -> "Hermes-Bedienungshilfe ist deaktiviert. Aktiviere sie in den Android-Einstellungen für schnelle Geräteaktionen sowie UI-Prüfung/-Aktionen."
    AppLanguage.PORTUGUESE -> "A acessibilidade do Hermes está desativada. Ative nas configurações do Android para liberar ações rápidas do dispositivo e inspeção/ação de UI."
    AppLanguage.FRENCH -> "L’accessibilité Hermes est désactivée. Activez-la dans les réglages Android pour débloquer les actions rapides et l’inspection/action de l’UI."
    AppLanguage.ENGLISH -> "Hermes accessibility is disabled. Enable it in Android settings to unlock quick device actions plus UI inspection/action targeting."
}

private fun HermesStrings.deviceOpenAccessibilitySettingsLabel(): String = when (language) {
    AppLanguage.CHINESE -> "打开无障碍设置"
    AppLanguage.SPANISH -> "Abrir ajustes de accesibilidad"
    AppLanguage.GERMAN -> "Bedienungshilfen öffnen"
    AppLanguage.PORTUGUESE -> "Abrir configurações de acessibilidade"
    AppLanguage.FRENCH -> "Ouvrir les réglages d’accessibilité"
    AppLanguage.ENGLISH -> "Open accessibility settings"
}

private fun HermesStrings.deviceGlobalActionLabel(action: HermesGlobalAction): String = when (action) {
    HermesGlobalAction.Home -> when (language) {
        AppLanguage.CHINESE -> "主页"
        AppLanguage.SPANISH -> "Inicio"
        AppLanguage.GERMAN -> "Startseite"
        AppLanguage.PORTUGUESE -> "Início"
        AppLanguage.FRENCH -> "Accueil"
        AppLanguage.ENGLISH -> action.label
    }
    HermesGlobalAction.Back -> when (language) {
        AppLanguage.CHINESE -> "返回"
        AppLanguage.SPANISH -> "Atrás"
        AppLanguage.GERMAN -> "Zurück"
        AppLanguage.PORTUGUESE -> "Voltar"
        AppLanguage.FRENCH -> "Retour"
        AppLanguage.ENGLISH -> action.label
    }
    HermesGlobalAction.Recents -> when (language) {
        AppLanguage.CHINESE -> "最近任务"
        AppLanguage.SPANISH -> "Recientes"
        AppLanguage.GERMAN -> "Zuletzt verwendet"
        AppLanguage.PORTUGUESE -> "Recentes"
        AppLanguage.FRENCH -> "Récents"
        AppLanguage.ENGLISH -> action.label
    }
    HermesGlobalAction.Notifications -> when (language) {
        AppLanguage.CHINESE -> "通知"
        AppLanguage.SPANISH -> "Notificaciones"
        AppLanguage.GERMAN -> "Benachrichtigungen"
        AppLanguage.PORTUGUESE -> "Notificações"
        AppLanguage.FRENCH -> "Notifications"
        AppLanguage.ENGLISH -> action.label
    }
    HermesGlobalAction.QuickSettings -> when (language) {
        AppLanguage.CHINESE -> "快捷设置"
        AppLanguage.SPANISH -> "Ajustes rápidos"
        AppLanguage.GERMAN -> "Schnelleinstellungen"
        AppLanguage.PORTUGUESE -> "Configurações rápidas"
        AppLanguage.FRENCH -> "Réglages rapides"
        AppLanguage.ENGLISH -> action.label
    }
}
