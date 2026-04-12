package com.nousresearch.hermesagent.ui.settings

import android.app.Application
import android.text.format.Formatter
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import com.nousresearch.hermesagent.data.SecureSecretsStore
import com.nousresearch.hermesagent.models.HermesModelDownloadManager
import com.nousresearch.hermesagent.models.ModelDownloadDraft
import com.nousresearch.hermesagent.models.ModelDownloadInspection
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class LocalModelDownloadItemUi(
    val id: String,
    val title: String,
    val runtimeFlavor: String,
    val progressLabel: String,
    val progressFraction: Float,
    val statusLabel: String,
    val statusMessage: String,
    val ramWarning: String,
    val isPreferred: Boolean,
    val localPath: String,
)

data class LocalModelDownloadsUiState(
    val repoOrUrl: String = "",
    val filePath: String = "",
    val revision: String = "main",
    val runtimeFlavor: String = "GGUF",
    val huggingFaceToken: String = "",
    val inspectionStatus: String = "",
    val candidateSummary: String = "",
    val candidateRamWarning: String = "",
    val downloads: List<LocalModelDownloadItemUi> = emptyList(),
)

class LocalModelDownloadsViewModel(application: Application) : AndroidViewModel(application) {
    private val secretsStore = SecureSecretsStore(application)
    private val downloadStore = LocalModelDownloadStore(application)

    private val _uiState = MutableStateFlow(loadInitialState())
    val uiState: StateFlow<LocalModelDownloadsUiState> = _uiState.asStateFlow()

    init {
        refreshDownloads()
        viewModelScope.launch {
            while (true) {
                delay(1800)
                if (_uiState.value.downloads.any { item -> item.statusLabel in setOf("queued", "downloading", "paused") }) {
                    refreshDownloads()
                }
            }
        }
    }

    private fun loadInitialState(): LocalModelDownloadsUiState {
        return LocalModelDownloadsUiState(
            huggingFaceToken = secretsStore.loadApiKey("huggingface"),
        )
    }

    fun updateRepoOrUrl(value: String) = _uiState.update { it.copy(repoOrUrl = value) }
    fun updateFilePath(value: String) = _uiState.update { it.copy(filePath = value) }
    fun updateRevision(value: String) = _uiState.update { it.copy(revision = value) }
    fun updateRuntimeFlavor(value: String) = _uiState.update { it.copy(runtimeFlavor = value) }
    fun updateHuggingFaceToken(value: String) = _uiState.update { it.copy(huggingFaceToken = value) }

    fun saveHuggingFaceToken() {
        val token = _uiState.value.huggingFaceToken.trim()
        secretsStore.saveApiKey("huggingface", token)
        _uiState.update {
            it.copy(
                inspectionStatus = if (token.isBlank()) {
                    "Cleared Hugging Face token"
                } else {
                    "Saved Hugging Face token for gated model downloads"
                },
            )
        }
    }

    fun inspectCandidate() {
        val context = getApplication<Application>()
        val state = _uiState.value
        viewModelScope.launch {
            runCatching {
                HermesModelDownloadManager.inspectCandidate(
                    context,
                    draft = ModelDownloadDraft(
                        repoOrUrl = state.repoOrUrl,
                        filePath = state.filePath,
                        revision = state.revision,
                        runtimeFlavor = state.runtimeFlavor,
                    ),
                    hfToken = state.huggingFaceToken,
                )
            }.onSuccess { inspection ->
                _uiState.update {
                    it.copy(
                        inspectionStatus = "Model candidate inspected",
                        candidateSummary = candidateSummary(context, inspection),
                        candidateRamWarning = inspection.ramWarning,
                    )
                }
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        inspectionStatus = error.message ?: error.javaClass.simpleName,
                        candidateSummary = "",
                        candidateRamWarning = "",
                    )
                }
            }
        }
    }

    fun startDownload(dataSaverMode: Boolean) {
        val context = getApplication<Application>()
        val state = _uiState.value
        viewModelScope.launch {
            runCatching {
                HermesModelDownloadManager.enqueueDownload(
                    context = context,
                    store = downloadStore,
                    draft = ModelDownloadDraft(
                        repoOrUrl = state.repoOrUrl,
                        filePath = state.filePath,
                        revision = state.revision,
                        runtimeFlavor = state.runtimeFlavor,
                    ),
                    hfToken = state.huggingFaceToken,
                    dataSaverMode = dataSaverMode,
                )
            }.onSuccess { record ->
                refreshDownloads()
                _uiState.update {
                    it.copy(
                        inspectionStatus = "Queued ${record.title} in Android DownloadManager",
                        candidateSummary = it.candidateSummary.ifBlank { record.statusMessage },
                        candidateRamWarning = record.ramWarning,
                    )
                }
            }.onFailure { error ->
                _uiState.update {
                    it.copy(inspectionStatus = error.message ?: error.javaClass.simpleName)
                }
            }
        }
    }

    fun refreshDownloads() {
        val context = getApplication<Application>()
        viewModelScope.launch {
            val refreshed = HermesModelDownloadManager.refreshDownloads(context, downloadStore)
            _uiState.update { it.copy(downloads = refreshed.toUiItems(context, downloadStore.preferredDownloadId())) }
        }
    }

    fun removeDownload(recordId: String) {
        HermesModelDownloadManager.removeDownload(getApplication(), downloadStore, recordId)
        refreshDownloads()
    }

    fun setPreferredDownload(recordId: String) {
        HermesModelDownloadManager.setPreferredDownload(downloadStore, recordId)
        refreshDownloads()
        _uiState.update { it.copy(inspectionStatus = "Marked this model as the preferred local runtime candidate") }
    }

    private fun candidateSummary(context: Application, inspection: ModelDownloadInspection): String {
        val resumeText = if (inspection.supportsResume) {
            "HTTP range resume is available"
        } else {
            "resume depends on server support"
        }
        return buildString {
            append("File: ")
            append(inspection.destinationFileName)
            append(" · Size: ")
            append(inspection.totalBytesLabel)
            append(" · Phone RAM: ")
            append(inspection.deviceRamLabel)
            append(" · ABIs: ")
            append(inspection.abiSummary)
            append(" · ")
            append(resumeText)
        }
    }

    private fun List<LocalModelDownloadRecord>.toUiItems(
        context: Application,
        preferredId: String,
    ): List<LocalModelDownloadItemUi> {
        return sortedByDescending { it.updatedAtEpochMs }.map { record ->
            val totalBytes = record.totalBytes.coerceAtLeast(0L)
            val downloadedBytes = record.downloadedBytes.coerceAtLeast(0L)
            val progressFraction = if (totalBytes > 0L) {
                (downloadedBytes.toDouble() / totalBytes.toDouble()).toFloat().coerceIn(0f, 1f)
            } else {
                0f
            }
            val progressLabel = if (totalBytes > 0L) {
                val percent = (progressFraction * 100).toInt().coerceIn(0, 100)
                "$percent% · ${Formatter.formatShortFileSize(context, downloadedBytes)} / ${Formatter.formatShortFileSize(context, totalBytes)}"
            } else {
                Formatter.formatShortFileSize(context, downloadedBytes)
            }
            LocalModelDownloadItemUi(
                id = record.id,
                title = record.title,
                runtimeFlavor = record.runtimeFlavor,
                progressLabel = progressLabel,
                progressFraction = progressFraction,
                statusLabel = record.status,
                statusMessage = record.statusMessage,
                ramWarning = record.ramWarning,
                isPreferred = preferredId == record.id,
                localPath = record.destinationPath,
            )
        }
    }
}
