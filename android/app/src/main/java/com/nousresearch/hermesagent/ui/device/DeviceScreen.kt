package com.nousresearch.hermesagent.ui.device

import android.content.Intent
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.device.HermesGlobalAction
import com.nousresearch.hermesagent.ui.shell.ShellActionItem

@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun DeviceScreen(
    modifier: Modifier = Modifier,
    viewModel: DeviceViewModel = viewModel(),
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
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

    SideEffect {
        onContextActionsChanged(
            listOf(
                ShellActionItem(
                    label = "Refresh device state",
                    description = "Reload shared-folder, Linux suite, and accessibility status.",
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
                    label = "Accessibility settings",
                    description = "Open Android accessibility settings for Hermes controls.",
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = {
                        context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    },
                ),
            )
        )
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Device", style = MaterialTheme.typography.headlineSmall)
                DeviceGuideCard(workspacePath = uiState.workspacePath)
                LinuxSuiteCard(uiState = uiState)
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
                    onOpenSettings = {
                        context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    },
                    onAction = viewModel::performGlobalAction,
                )
                if (uiState.status.isNotBlank()) {
                    Text(uiState.status, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@Composable
private fun DeviceGuideCard(workspacePath: String) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("How to use this alpha", style = MaterialTheme.typography.titleMedium)
            Text("1. Hermes now ships a local Linux command suite inside the Android app. Ask Hermes to call android_device_status first, then use terminal/process for full CLI execution.")
            Text("2. Grant a shared folder from Android's native picker if you want Hermes to edit the real files in place with android_shared_folder_list/read/write.")
            Text("3. Import files into the workspace only when you want scratch copies or staging files.")
            Text("4. Enable Hermes accessibility if you want Hermes to inspect the visible UI and trigger targeted actions in addition to Home / Back / Recents / Notifications / Quick settings.")
            if (workspacePath.isNotBlank()) {
                Text("Workspace path: $workspacePath", style = MaterialTheme.typography.bodySmall)
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
