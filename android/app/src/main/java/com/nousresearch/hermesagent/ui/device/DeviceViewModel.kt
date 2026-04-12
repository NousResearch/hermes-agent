package com.nousresearch.hermesagent.ui.device

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import android.text.format.DateFormat
import android.text.format.Formatter
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.data.DeviceCapabilityStore
import com.nousresearch.hermesagent.device.DeviceStateWriter
import com.nousresearch.hermesagent.device.HermesAccessibilityController
import com.nousresearch.hermesagent.device.HermesGlobalAction
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.IOException

data class WorkspaceFileUi(
    val name: String,
    val sizeLabel: String,
    val modifiedLabel: String,
)

data class DeviceUiState(
    val workspacePath: String = "",
    val workspaceFiles: List<WorkspaceFileUi> = emptyList(),
    val sharedFolderLabel: String = "No shared folder granted yet",
    val sharedFolderUri: String = "",
    val linuxEnabled: Boolean = false,
    val linuxAndroidAbi: String = "",
    val linuxTermuxArch: String = "",
    val linuxPrefixPath: String = "",
    val linuxBashPath: String = "",
    val linuxHomePath: String = "",
    val linuxTmpPath: String = "",
    val linuxPackageCount: Int = 0,
    val accessibilityEnabled: Boolean = false,
    val accessibilityConnected: Boolean = false,
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
    val status: String = "",
)

class DeviceViewModel(application: Application) : AndroidViewModel(application) {
    private val capabilityStore = DeviceCapabilityStore(application)

    private val _uiState = MutableStateFlow(buildState())
    val uiState: StateFlow<DeviceUiState> = _uiState.asStateFlow()

    init {
        DeviceStateWriter.write(application)
    }

    fun refresh(status: String = _uiState.value.status) {
        val context = getApplication<Application>()
        DeviceStateWriter.write(context)
        _uiState.value = buildState(status)
    }

    fun importDocument(uri: Uri) {
        viewModelScope.launch {
            runCatching {
                val context = getApplication<Application>()
                val resolver = context.contentResolver
                val displayName = queryDisplayName(uri).ifBlank {
                    "import-${System.currentTimeMillis()}"
                }
                val target = uniqueDestination(DeviceStateWriter.workspaceDir(context), displayName)
                resolver.openInputStream(uri)?.use { input ->
                    target.outputStream().use { output ->
                        input.copyTo(output)
                    }
                } ?: throw IOException("Unable to open the selected document")
                refresh("Imported ${target.name} into the Hermes workspace")
            }.getOrElse { error ->
                refresh("Import failed: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun rememberSharedFolder(uri: Uri) {
        val context = getApplication<Application>()
        runCatching {
            context.contentResolver.takePersistableUriPermission(
                uri,
                android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
        }
        val label = DocumentFile.fromTreeUri(context, uri)?.name?.takeIf { it.isNotBlank() }
            ?: uri.lastPathSegment?.takeIf { it.isNotBlank() }
            ?: "Granted folder"
        capabilityStore.saveSharedFolder(uri.toString(), label)
        refresh("Saved shared folder access for $label")
    }

    fun clearSharedFolder() {
        val context = getApplication<Application>()
        val stored = capabilityStore.load()
        if (stored.sharedFolderUri.isNotBlank()) {
            runCatching {
                context.contentResolver.releasePersistableUriPermission(
                    Uri.parse(stored.sharedFolderUri),
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
                )
            }
        }
        capabilityStore.clearSharedFolder()
        refresh("Cleared shared folder permission")
    }

    fun exportWorkspaceFile(fileName: String, destinationUri: Uri) {
        viewModelScope.launch {
            runCatching {
                val context = getApplication<Application>()
                val source = File(DeviceStateWriter.workspaceDir(context), fileName)
                if (!source.isFile) {
                    throw IOException("Workspace file not found: $fileName")
                }
                context.contentResolver.openOutputStream(destinationUri)?.use { output ->
                    source.inputStream().use { input ->
                        input.copyTo(output)
                    }
                } ?: throw IOException("Unable to open export destination")
                refresh("Exported $fileName")
            }.getOrElse { error ->
                refresh("Export failed: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun performGlobalAction(action: HermesGlobalAction) {
        val context = getApplication<Application>()
        val succeeded = HermesAccessibilityController.performAction(action)
        refresh(
            if (succeeded) {
                "Ran ${action.label.lowercase()}"
            } else if (!HermesAccessibilityController.isServiceEnabled(context)) {
                "Enable Hermes accessibility in Android settings first"
            } else {
                "Hermes accessibility is enabled but not connected yet"
            },
        )
    }

    fun performSystemAction(action: String) {
        val context = getApplication<Application>()
        val result = HermesSystemControlBridge.performAction(context, action)
        refresh(result.message)
    }

    fun setBackgroundPersistence(enabled: Boolean) {
        performSystemAction(if (enabled) "start_background_runtime" else "stop_background_runtime")
    }

    private fun buildState(status: String = ""): DeviceUiState {
        val context = getApplication<Application>()
        val sharedFolder = capabilityStore.load()
        val linuxState = HermesLinuxSubsystemBridge.readState(context)
        val systemStatus = HermesSystemControlBridge.readStatus(context)
        val workspace = DeviceStateWriter.workspaceDir(context)
        val workspaceFiles = workspace
            .listFiles()
            .orEmpty()
            .filter { it.isFile }
            .sortedByDescending { it.lastModified() }
            .take(12)
            .map { file ->
                WorkspaceFileUi(
                    name = file.name,
                    sizeLabel = Formatter.formatShortFileSize(context, file.length()),
                    modifiedLabel = DateFormat.format("yyyy-MM-dd HH:mm", file.lastModified()).toString(),
                )
            }

        return DeviceUiState(
            workspacePath = workspace.absolutePath,
            workspaceFiles = workspaceFiles,
            sharedFolderLabel = sharedFolder.sharedFolderLabel.ifBlank { "No shared folder granted yet" },
            sharedFolderUri = sharedFolder.sharedFolderUri,
            linuxEnabled = linuxState?.optBoolean("enabled") == true,
            linuxAndroidAbi = linuxState?.optString("android_abi").orEmpty(),
            linuxTermuxArch = linuxState?.optString("termux_arch").orEmpty(),
            linuxPrefixPath = linuxState?.optString("prefix_path").orEmpty(),
            linuxBashPath = linuxState?.optString("bash_path").orEmpty(),
            linuxHomePath = linuxState?.optString("home_path").orEmpty(),
            linuxTmpPath = linuxState?.optString("tmp_path").orEmpty(),
            linuxPackageCount = linuxState?.optJSONArray("packages")?.length() ?: 0,
            accessibilityEnabled = HermesAccessibilityController.isServiceEnabled(context),
            accessibilityConnected = HermesAccessibilityController.isServiceConnected(),
            wifiEnabled = systemStatus.wifiEnabled,
            activeNetworkLabel = systemStatus.activeNetworkLabel,
            bluetoothSupported = systemStatus.bluetoothSupported,
            bluetoothEnabled = systemStatus.bluetoothEnabled,
            bluetoothPermissionGranted = systemStatus.bluetoothPermissionGranted,
            pairedBluetoothDevices = systemStatus.pairedBluetoothDevices,
            usbHostSupported = systemStatus.usbHostSupported,
            usbDeviceCount = systemStatus.usbDeviceCount,
            usbDevices = systemStatus.usbDevices,
            nfcSupported = systemStatus.nfcSupported,
            nfcEnabled = systemStatus.nfcEnabled,
            overlayPermissionGranted = systemStatus.overlayPermissionGranted,
            notificationPermissionGranted = systemStatus.notificationPermissionGranted,
            backgroundPersistenceEnabled = systemStatus.backgroundPersistenceEnabled,
            runtimeServiceRunning = systemStatus.runtimeServiceRunning,
            resizableWindowSupport = systemStatus.resizableWindowSupport,
            freeformWindowSupported = systemStatus.freeformWindowSupported,
            status = status,
        )
    }

    private fun queryDisplayName(uri: Uri): String {
        val context = getApplication<Application>()
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) {
                    return cursor.getString(nameIndex).orEmpty()
                }
            }
        }
        return DocumentFile.fromSingleUri(context, uri)?.name.orEmpty()
    }

    private fun uniqueDestination(directory: File, fileName: String): File {
        directory.mkdirs()
        val candidate = File(directory, fileName)
        if (!candidate.exists()) {
            return candidate
        }
        val dotIndex = fileName.lastIndexOf('.')
        val stem = if (dotIndex > 0) fileName.substring(0, dotIndex) else fileName
        val extension = if (dotIndex > 0) fileName.substring(dotIndex) else ""
        var suffix = 1
        while (true) {
            val next = File(directory, "$stem-$suffix$extension")
            if (!next.exists()) {
                return next
            }
            suffix += 1
        }
    }
}
