package com.nousresearch.hermesagent.ui.settings

import android.app.Application
import android.app.DownloadManager
import android.content.ActivityNotFoundException
import android.content.Intent
import android.text.format.Formatter
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import com.nousresearch.hermesagent.data.SecureSecretsStore
import com.nousresearch.hermesagent.models.HermesModelDownloadManager
import com.nousresearch.hermesagent.models.ModelDownloadDraft
import com.nousresearch.hermesagent.models.ModelDownloadInspection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

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
    val canRestartOnMobileData: Boolean,
    val canOpenSystemDownloads: Boolean,
)

data class RecommendedLocalModelPreset(
    val id: String,
    val title: String,
    val description: String,
    val repoOrUrl: String,
    val filePath: String,
    val revision: String = "main",
    val runtimeFlavor: String,
    val testedLabel: String,
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
    val pendingAutoStartRecordId: String = "",
    val downloads: List<LocalModelDownloadItemUi> = emptyList(),
)

class LocalModelDownloadsViewModel(application: Application) : AndroidViewModel(application) {
    private val settingsStore = AppSettingsStore(application)
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
        val settings = settingsStore.load()
        val initialRuntimeFlavor = when (settings.onDeviceBackend) {
            "litert-lm" -> "LiteRT-LM"
            else -> "GGUF"
        }
        return LocalModelDownloadsUiState(
            huggingFaceToken = secretsStore.loadApiKey("huggingface"),
            runtimeFlavor = initialRuntimeFlavor,
        )
    }

    fun updateRepoOrUrl(value: String) = _uiState.update {
        it.copy(
            repoOrUrl = value,
            inspectionStatus = "",
            candidateSummary = "",
            candidateRamWarning = "",
        )
    }

    fun updateFilePath(value: String) = _uiState.update {
        it.copy(
            filePath = value,
            inspectionStatus = "",
            candidateSummary = "",
            candidateRamWarning = "",
        )
    }

    fun updateRevision(value: String) = _uiState.update {
        it.copy(
            revision = value,
            inspectionStatus = "",
            candidateSummary = "",
            candidateRamWarning = "",
        )
    }

    fun updateRuntimeFlavor(value: String) = _uiState.update {
        it.copy(
            runtimeFlavor = value,
            inspectionStatus = "",
            candidateSummary = "",
            candidateRamWarning = "",
        )
    }

    fun updateHuggingFaceToken(value: String) = _uiState.update {
        it.copy(
            huggingFaceToken = value,
            inspectionStatus = "",
            candidateSummary = "",
            candidateRamWarning = "",
        )
    }

    fun syncSelectedBackend(selectedBackend: String) {
        val runtimeFlavor = when (selectedBackend) {
            "llama.cpp" -> "GGUF"
            "litert-lm" -> "LiteRT-LM"
            else -> _uiState.value.runtimeFlavor
        }
        if (runtimeFlavor != _uiState.value.runtimeFlavor) {
            updateRuntimeFlavor(runtimeFlavor)
        } else {
            _uiState.update {
                it.copy(
                    inspectionStatus = "",
                    candidateSummary = "",
                    candidateRamWarning = "",
                )
            }
        }
    }

    fun saveHuggingFaceToken() {
        val token = _uiState.value.huggingFaceToken.trim()
        secretsStore.saveApiKey("huggingface", token)
        _uiState.update {
            it.copy(
                inspectionStatus = if (token.isBlank()) {
                    "Cleared Hugging Face token"
                } else {
                    "Saved Hugging Face token for private or gated model downloads"
                },
            )
        }
    }

    fun startRecommendedModelDownload(presetId: String, dataSaverMode: Boolean) {
        val preset = recommendedModelPresets.firstOrNull { it.id == presetId } ?: return
        val context = getApplication<Application>()
        _uiState.update {
            it.copy(
                repoOrUrl = preset.repoOrUrl,
                filePath = preset.filePath,
                revision = preset.revision,
                runtimeFlavor = preset.runtimeFlavor,
                inspectionStatus = "Preparing ${preset.title}…",
                candidateSummary = preset.description,
                candidateRamWarning = "",
            )
        }
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val refreshed = HermesModelDownloadManager.refreshDownloads(context, downloadStore)
                    val existing = refreshed.firstOrNull { record -> record.matchesPreset(preset) && record.status == "completed" }
                    if (existing != null) {
                        HermesModelDownloadManager.setPreferredDownload(downloadStore, existing.id)
                        existing
                    } else {
                        HermesModelDownloadManager.enqueueDownload(
                            context = context,
                            store = downloadStore,
                            draft = preset.toDraft(),
                            hfToken = _uiState.value.huggingFaceToken,
                            dataSaverMode = dataSaverMode,
                        )
                    }
                }
            }.onSuccess { record ->
                refreshDownloads()
                _uiState.update {
                    it.copy(
                        pendingAutoStartRecordId = record.id,
                        inspectionStatus = if (record.status == "completed") {
                            "${record.title} is already downloaded. Starting runtime…"
                        } else {
                            "Queued ${record.title}; Hermes will start it when Android finishes the download."
                        },
                        candidateSummary = it.candidateSummary.ifBlank { record.statusMessage },
                        candidateRamWarning = record.ramWarning,
                    )
                }
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        inspectionStatus = error.message ?: error.javaClass.simpleName,
                        pendingAutoStartRecordId = "",
                    )
                }
            }
        }
    }

    fun inspectCandidate(runtimeFlavorOverride: String? = null) {
        val context = getApplication<Application>()
        val state = _uiState.value
        val resolvedRuntimeFlavor = runtimeFlavorOverride?.ifBlank { null } ?: state.runtimeFlavor
        _uiState.update {
            it.copy(
                runtimeFlavor = resolvedRuntimeFlavor,
                inspectionStatus = "Inspecting model candidate…",
                candidateSummary = "",
                candidateRamWarning = "",
            )
        }
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    HermesModelDownloadManager.inspectCandidate(
                        context,
                        draft = ModelDownloadDraft(
                            repoOrUrl = state.repoOrUrl,
                            filePath = state.filePath,
                            revision = state.revision,
                            runtimeFlavor = resolvedRuntimeFlavor,
                        ),
                        hfToken = state.huggingFaceToken,
                    )
                }
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

    fun startDownload(dataSaverMode: Boolean, runtimeFlavorOverride: String? = null) {
        val context = getApplication<Application>()
        val state = _uiState.value
        val resolvedRuntimeFlavor = runtimeFlavorOverride?.ifBlank { null } ?: state.runtimeFlavor
        _uiState.update {
            it.copy(
                runtimeFlavor = resolvedRuntimeFlavor,
                inspectionStatus = "Preparing download…",
            )
        }
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    HermesModelDownloadManager.enqueueDownload(
                        context = context,
                        store = downloadStore,
                        draft = ModelDownloadDraft(
                            repoOrUrl = state.repoOrUrl,
                            filePath = state.filePath,
                            revision = state.revision,
                            runtimeFlavor = resolvedRuntimeFlavor,
                        ),
                        hfToken = state.huggingFaceToken,
                        dataSaverMode = dataSaverMode,
                    )
                }
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
        viewModelScope.launch(Dispatchers.IO) {
            val refreshed = HermesModelDownloadManager.refreshDownloads(context, downloadStore)
            _uiState.update { it.copy(downloads = refreshed.toUiItems(context, downloadStore.preferredDownloadId())) }
        }
    }

    fun removeDownload(recordId: String) {
        HermesModelDownloadManager.removeDownload(getApplication(), downloadStore, recordId)
        refreshDownloads()
    }

    fun restartDownloadOnMobileData(recordId: String) {
        val restarted = HermesModelDownloadManager.restartDownloadOnMobileData(
            context = getApplication(),
            store = downloadStore,
            recordId = recordId,
            hfToken = _uiState.value.huggingFaceToken,
        )
        refreshDownloads()
        _uiState.update {
            it.copy(
                inspectionStatus = if (restarted != null) {
                    "Restarted ${restarted.title} with mobile data and roaming allowed"
                } else {
                    "Unable to restart this download on mobile data"
                }
            )
        }
    }

    fun openSystemDownloads() {
        val intent = Intent(DownloadManager.ACTION_VIEW_DOWNLOADS).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        try {
            getApplication<Application>().startActivity(intent)
            _uiState.update { it.copy(inspectionStatus = "Opened Android Downloads") }
        } catch (_: ActivityNotFoundException) {
            _uiState.update { it.copy(inspectionStatus = "Android Downloads is not available on this device") }
        }
    }

    fun setPreferredDownload(recordId: String) {
        HermesModelDownloadManager.setPreferredDownload(downloadStore, recordId)
        refreshDownloads()
        _uiState.update { it.copy(inspectionStatus = "Marked this model as the preferred local runtime candidate") }
    }

    fun promoteDownloadedModelForAutoStart(recordId: String) {
        HermesModelDownloadManager.setPreferredDownload(downloadStore, recordId)
        refreshDownloads()
        _uiState.update {
            it.copy(
                pendingAutoStartRecordId = "",
                inspectionStatus = "Preferred model is ready. Starting Hermes runtime…",
            )
        }
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
            if (inspection.compatibilityHint.isNotBlank()) {
                append(" · ")
                append(inspection.compatibilityHint)
            }
        }
    }

    private fun RecommendedLocalModelPreset.toDraft(): ModelDownloadDraft {
        return ModelDownloadDraft(
            repoOrUrl = repoOrUrl,
            filePath = filePath,
            revision = revision,
            runtimeFlavor = runtimeFlavor,
        )
    }

    private fun LocalModelDownloadRecord.matchesPreset(preset: RecommendedLocalModelPreset): Boolean {
        val exactFileMatches = preset.filePath.isNotBlank() &&
            (filePath.equals(preset.filePath, ignoreCase = true) ||
                destinationFileName.equals(preset.filePath.substringAfterLast('/'), ignoreCase = true) ||
                destinationPath.substringAfterLast('/').equals(preset.filePath.substringAfterLast('/'), ignoreCase = true))
        val repoMatches = repoOrUrl.equals(preset.repoOrUrl, ignoreCase = true)
        return runtimeFlavor.equals(preset.runtimeFlavor, ignoreCase = true) &&
            (exactFileMatches || repoMatches)
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
            val transientStatus = record.status in setOf("queued", "paused", "downloading")
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
                canRestartOnMobileData = transientStatus && (!record.allowMetered || !record.allowRoaming),
                canOpenSystemDownloads = transientStatus,
            )
        }
    }

    companion object {
        val recommendedModelPresets = listOf(
            RecommendedLocalModelPreset(
                id = "qwen35-08b-q4km-gguf",
                title = "Qwen3.5 0.8B Q4_K_M (GGUF)",
                description = "Small GGUF model validated on the physical Hermes test phone for visible chat replies, file creation, deletion, and native tool calling.",
                repoOrUrl = "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
                filePath = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
                runtimeFlavor = "GGUF",
                testedLabel = "Tested phone tool-calling",
            ),
            RecommendedLocalModelPreset(
                id = "gemma4-e2b-litert-lm",
                title = "Gemma 4 E2B (LiteRT-LM)",
                description = "First-class Gemma 4 local runtime target for Hermes mobile chat, image-capable runtime plumbing, and Android agent tools.",
                repoOrUrl = "litert-community/gemma-4-E2B-it-litert-lm",
                filePath = "",
                runtimeFlavor = "LiteRT-LM",
                testedLabel = "Gemma 4 mobile path",
            ),
            RecommendedLocalModelPreset(
                id = "gemma3-1b-litert-lm",
                title = "Gemma 3 1B IT INT4 (LiteRT-LM)",
                description = "Small Gemma 3 compatibility target for lower-memory devices and fast local runtime bring-up.",
                repoOrUrl = "litert-community/Gemma3-1B-IT",
                filePath = "",
                runtimeFlavor = "LiteRT-LM",
                testedLabel = "Small compatibility path",
            ),
        )
    }
}
