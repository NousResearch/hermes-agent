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
import androidx.compose.foundation.layout.navigationBarsPadding
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.device.HermesGlobalAction

@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun DeviceScreen(
    modifier: Modifier = Modifier,
    viewModel: DeviceViewModel = viewModel(),
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

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .navigationBarsPadding()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Device", style = MaterialTheme.typography.headlineSmall)
                DeviceGuideCard(workspacePath = uiState.workspacePath)
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
            Text("1. Import a file or grant a shared folder from Android's native pickers.")
            Text("2. Hermes now has workspace file tools enabled in the Android runtime.")
            Text("3. Ask Hermes to call android_device_status first, then use read_file / write_file / search_files / patch inside the workspace path below.")
            Text("4. Enable Hermes accessibility if you want quick Home, Back, Recents, Notifications, and Quick settings control.")
            if (workspacePath.isNotBlank()) {
                Text("Workspace path: $workspacePath", style = MaterialTheme.typography.bodySmall)
            }
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
            Text("Workspace + file access", style = MaterialTheme.typography.titleMedium)
            Text("Imported files land in the Hermes workspace so the local agent can read and edit them safely.")
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
                        "Hermes accessibility is enabled and connected."
                    } else {
                        "Hermes accessibility is enabled, but Android has not connected the service yet."
                    }
                } else {
                    "Hermes accessibility is disabled. Enable it in Android settings to unlock quick device actions."
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
