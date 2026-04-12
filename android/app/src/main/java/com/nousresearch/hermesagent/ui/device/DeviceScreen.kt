@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.nousresearch.hermesagent.ui.device

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
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.device.HermesGlobalAction
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings
import com.nousresearch.hermesagent.ui.shell.ShellActionItem

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
        viewModel.refresh(if (granted) "Notifications enabled for Hermes runtime alerts" else "Notification permission was denied")
    }
    val bluetoothPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        viewModel.refresh(if (granted) "Bluetooth access granted" else "Bluetooth access was denied")
    }

    SideEffect {
        onContextActionsChanged(
            listOf(
                ShellActionItem(
                    label = strings.refresh.ifBlank { "Refresh" },
                    description = "Reload shared-folder, Linux suite, and phone-control status.",
                    iconRes = R.drawable.ic_action_refresh,
                    onClick = viewModel::refresh,
                ),
                ShellActionItem(
                    label = "Grant shared folder",
                    description = "Pick a real Android folder for direct Hermes file access.",
                    iconRes = R.drawable.ic_nav_device,
                    onClick = { sharedFolderLauncher.launch(null) },
                ),
                ShellActionItem(
                    label = "Import file",
                    description = "Bring a file into the Hermes workspace for scratch edits.",
                    iconRes = R.drawable.ic_nav_device,
                    onClick = { importLauncher.launch(arrayOf("*/*")) },
                ),
                ShellActionItem(
                    label = "Notification settings",
                    description = "Open Hermes notification settings and background controls.",
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
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Linux command suite", style = MaterialTheme.typography.titleMedium)
            Text(
                if (uiState.linuxEnabled) {
                    "Hermes can execute full CLI commands locally with terminal/process using the extracted Linux suite."
                } else {
                    "Linux command suite is still provisioning. Retry Hermes once the backend finishes booting."
                },
            )
            if (uiState.linuxAndroidAbi.isNotBlank() || uiState.linuxTermuxArch.isNotBlank()) {
                Text(
                    "ABI: ${uiState.linuxAndroidAbi} · suite arch: ${uiState.linuxTermuxArch}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (uiState.linuxPrefixPath.isNotBlank()) {
                Text("Prefix: ${uiState.linuxPrefixPath}", style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.linuxBashPath.isNotBlank()) {
                Text("Bash: ${uiState.linuxBashPath}", style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.linuxHomePath.isNotBlank()) {
                Text("Home: ${uiState.linuxHomePath}", style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.linuxTmpPath.isNotBlank()) {
                Text("Temp: ${uiState.linuxTmpPath}", style = MaterialTheme.typography.bodySmall)
            }
            Text(
                "Included package count: ${uiState.linuxPackageCount}",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Ask Hermes to use terminal for commands like 'git status', 'ls', 'curl', 'grep', or longer shell pipelines directly in this suite.",
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
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("Wi-Fi + connectivity", style = MaterialTheme.typography.titleMedium)
            Text(
                "Network: ${uiState.activeNetworkLabel} · Wi-Fi is ${if (uiState.wifiEnabled) "on" else "off"}. Hermes uses Android-safe settings panels instead of unsupported direct radio toggles.",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Bluetooth", style = MaterialTheme.typography.titleSmall,
            )
            Text(
                if (!uiState.bluetoothSupported) {
                    "Bluetooth radio is not available on this device."
                } else if (!uiState.bluetoothPermissionGranted) {
                    "Grant Bluetooth access so Hermes can read bonded-device state before opening settings."
                } else {
                    "Bluetooth is ${if (uiState.bluetoothEnabled) "enabled" else "disabled"}. Bonded devices: ${uiState.pairedBluetoothDevices.joinToString().ifBlank { "none" }}"
                },
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenWifi) {
                    Text("Internet panel")
                }
                Button(onClick = onOpenBluetooth) {
                    Text("Bluetooth")
                }
                Button(onClick = onOpenConnectedDevices) {
                    Text("Connected devices")
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
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.CHINESE -> "蜂窝网络与无线电控制"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Controles celulares y de radio"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Mobilfunk- und Funksteuerung"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Controles celulares e de rádio"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Contrôles cellulaires et radio"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Cellular + radio controls"
    }
    val summary = when (strings.language) {
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.CHINESE -> "当前网络：${uiState.activeNetworkLabel} · 计量网络：${if (uiState.activeNetworkMetered) "是" else "否"} · 省流模式：${if (uiState.dataSaverEnabled) "开" else "关"} · 飞行模式：${if (uiState.airplaneModeEnabled) "开" else "关"}。由于 Android 限制，Hermes 使用系统面板而不是不受支持的直接无线电切换。"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Red actual: ${uiState.activeNetworkLabel} · medida: ${if (uiState.activeNetworkMetered) "sí" else "no"} · ahorro de datos: ${if (uiState.dataSaverEnabled) "activo" else "inactivo"} · modo avión: ${if (uiState.airplaneModeEnabled) "activo" else "inactivo"}. Por las restricciones de Android, Hermes usa paneles del sistema en lugar de toggles directos no soportados."
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Aktives Netzwerk: ${uiState.activeNetworkLabel} · getaktet: ${if (uiState.activeNetworkMetered) "ja" else "nein"} · Datensparen: ${if (uiState.dataSaverEnabled) "aktiv" else "inaktiv"} · Flugmodus: ${if (uiState.airplaneModeEnabled) "aktiv" else "inaktiv"}. Wegen Android-Beschränkungen nutzt Hermes Systemansichten statt nicht unterstützter Direktumschaltungen."
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Rede atual: ${uiState.activeNetworkLabel} · limitada: ${if (uiState.activeNetworkMetered) "sim" else "não"} · economia de dados: ${if (uiState.dataSaverEnabled) "ativa" else "inativa"} · modo avião: ${if (uiState.airplaneModeEnabled) "ativo" else "inativo"}. Devido às restrições do Android, o Hermes usa painéis do sistema em vez de alternâncias diretas não suportadas."
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Réseau actif : ${uiState.activeNetworkLabel} · limité : ${if (uiState.activeNetworkMetered) "oui" else "non"} · économie de données : ${if (uiState.dataSaverEnabled) "active" else "inactive"} · mode avion : ${if (uiState.airplaneModeEnabled) "actif" else "inactif"}. En raison des limites Android, Hermes utilise des panneaux système plutôt que des bascules radio directes non prises en charge."
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Active network: ${uiState.activeNetworkLabel} · metered: ${if (uiState.activeNetworkMetered) "yes" else "no"} · data saver: ${if (uiState.dataSaverEnabled) "enabled" else "disabled"} · airplane mode: ${if (uiState.airplaneModeEnabled) "enabled" else "disabled"}. Because of Android platform limits, Hermes uses system panels instead of unsupported direct radio toggles."
    }
    val mobileNetworkLabel = when (strings.language) {
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.CHINESE -> "移动网络"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Red móvil"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Mobilfunk"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Rede móvel"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Réseau mobile"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Mobile network"
    }
    val dataUsageLabel = when (strings.language) {
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.CHINESE -> "数据使用"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Uso de datos"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Datennutzung"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Uso de dados"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Utilisation des données"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Data usage"
    }
    val hotspotLabel = when (strings.language) {
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.CHINESE -> "热点 / 共享"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Punto de acceso"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Hotspot / Tethering"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Hotspot / ancoragem"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Point d’accès / partage"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Hotspot / tethering"
    }
    val airplaneLabel = when (strings.language) {
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.CHINESE -> "飞行模式"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Modo avión"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.GERMAN -> "Flugmodus"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Modo avião"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Mode avion"
        com.nousresearch.hermesagent.ui.i18n.AppLanguage.ENGLISH -> "Airplane mode"
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
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("USB + NFC", style = MaterialTheme.typography.titleMedium)
            Text(
                if (uiState.usbHostSupported) {
                    "USB host mode is available. Connected USB devices: ${uiState.usbDeviceCount}. ${uiState.usbDevices.joinToString().ifBlank { "No USB devices detected right now." }}"
                } else {
                    "USB host mode is not advertised on this device build."
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                if (uiState.nfcSupported) {
                    "NFC is ${if (uiState.nfcEnabled) "enabled" else "disabled"}. Hermes can surface NFC state and take you straight to system settings."
                } else {
                    "NFC hardware is not available on this device."
                },
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenConnectedDevices) {
                    Text("USB / devices")
                }
                Button(onClick = onOpenNfc, enabled = uiState.nfcSupported) {
                    Text("NFC settings")
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
) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("Notifications + background runtime", style = MaterialTheme.typography.titleMedium)
            Text(
                "Notification permission is ${if (uiState.notificationPermissionGranted) "granted" else "not granted"}. Hermes background runtime is ${if (uiState.runtimeServiceRunning) "active" else "inactive"}.",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Overlay permission", style = MaterialTheme.typography.titleSmall,
            )
            Text(
                if (uiState.overlayPermissionGranted) {
                    "Overlay permission is granted for future floating utilities."
                } else {
                    "Overlay permission is disabled. Open Android settings if you want future floating controls."
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Resizable window support", style = MaterialTheme.typography.titleSmall,
            )
            Text(
                "Hermes declares resizable window support: ${if (uiState.resizableWindowSupport) "enabled" else "disabled"}. Freeform/multi-window feature available on this device: ${if (uiState.freeformWindowSupported) "yes" else "no"}.",
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onNotifications) {
                    Text(if (uiState.notificationPermissionGranted) "Notification settings" else "Enable notifications")
                }
                Button(onClick = onOverlaySettings) {
                    Text("Overlay settings")
                }
                Button(onClick = { onToggleRuntime(!uiState.backgroundPersistenceEnabled) }) {
                    Text(if (uiState.backgroundPersistenceEnabled) "Stop background runtime" else "Start background runtime")
                }
            }
            Text(
                "Hermes background runtime keeps the local backend ready in the notification bar for longer sessions and later Android windowing modes.",
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
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("Shared folder + workspace access", style = MaterialTheme.typography.titleMedium)
            Text("Grant a shared folder to let Hermes read and write the real files directly. Imported files still land in the Hermes workspace when you want copies instead, while terminal/process now cover general CLI work.")
            Text("Shared folder: ${uiState.sharedFolderLabel}", style = MaterialTheme.typography.bodySmall)
            if (uiState.sharedFolderUri.isNotBlank()) {
                Text(uiState.sharedFolderUri, style = MaterialTheme.typography.bodySmall)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onImportFile, modifier = Modifier.weight(1f)) {
                    Text("Import file")
                }
                Button(onClick = onGrantFolder, modifier = Modifier.weight(1f)) {
                    Text("Grant folder")
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onRefresh, modifier = Modifier.weight(1f)) {
                    Text("Refresh")
                }
                Button(
                    onClick = onClearFolder,
                    enabled = uiState.sharedFolderUri.isNotBlank(),
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Clear folder")
                }
            }
            if (uiState.workspaceFiles.isEmpty()) {
                Text("No files in the Hermes workspace yet.", style = MaterialTheme.typography.bodySmall)
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
                                "${file.sizeLabel} · updated ${file.modifiedLabel}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Button(onClick = { onExport(file.name) }) {
                                Text("Export")
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
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("Accessibility control", style = MaterialTheme.typography.titleMedium)
            Text(
                if (uiState.accessibilityEnabled) {
                    if (uiState.accessibilityConnected) {
                        "Hermes accessibility is enabled and connected. Hermes can inspect the visible UI with android_ui_snapshot and target controls with android_ui_action."
                    } else {
                        "Hermes accessibility is enabled, but Android has not connected the service yet."
                    }
                } else {
                    "Hermes accessibility is disabled. Enable it in Android settings to unlock quick device actions plus UI inspection/action targeting."
                },
            )
            Button(onClick = onOpenSettings) {
                Text("Open accessibility settings")
            }
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                HermesGlobalAction.values().forEach { action ->
                    Button(onClick = { onAction(action) }) {
                        Text(action.label)
                    }
                }
            }
        }
    }
}
