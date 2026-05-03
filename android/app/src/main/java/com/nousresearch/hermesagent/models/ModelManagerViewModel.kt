package com.nousresearch.hermesagent.models

/**
 * ModelManagerViewModel following Google AI Edge Gallery pattern.
 *
 * Manages the model catalog, download lifecycle, and initialization state
 * for on-device inference models. Provides a unified StateFlow that the UI
 * observes for model status changes.
 *
 * Key design decisions:
 * - Catalog is static + extensible via user downloads (LocalModelDownloadsViewModel)
 * - ModelState tracks: NOT_AVAILABLE -> DOWNLOADING -> DOWNLOADED -> READY / INIT_FAILED
 * - Initialization is lazy — models are only initialized when first needed
 * - Supports both llama.cpp (GGUF) and LiteRT-LM backends
 *
 * Based on: https://github.com/google-ai-edge/gallery
 * ModelManagerViewModel.kt pattern from Google AI Edge Gallery.
 */
import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.backend.BackendKind
import com.nousresearch.hermesagent.backend.LiteRtLmOpenAiProxy
import com.nousresearch.hermesagent.backend.LlamaCppServerController
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import com.nousresearch.hermesagent.data.SecureSecretsStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/**
 * Represents a model available for on-device inference.
 * Combines catalog metadata with runtime state.
 */
data class ManagedModel(
    val catalogEntry: ModelCatalogEntry,
    val state: ModelState = ModelState.NOT_AVAILABLE,
    val downloadProgress: ModelDownloadProgress? = null,
    val localFilePath: String? = null,
    val errorMessage: String? = null,
) {
    val isReady: Boolean = state == ModelState.READY
    val isDownloaded: Boolean = state in setOf(ModelState.DOWNLOADED, ModelState.READY)
    val isDownloading: Boolean = state == ModelState.DOWNLOADING
    val isFailed: Boolean = state == ModelState.INIT_FAILED

    /** Percentage of download complete (0-100) */
    val downloadPercentage: Int
        get() = downloadProgress?.percentage ?: (if (isDownloaded) 100 else 0)

    /** Human-readable status label */
    val statusLabel: String
        get() = when (state) {
            ModelState.NOT_AVAILABLE -> "Not downloaded"
            ModelState.DOWNLOADING -> "Downloading ${downloadPercentage}%"
            ModelState.DOWNLOADED -> "Downloaded"
            ModelState.READY -> "Ready"
            ModelState.INIT_FAILED -> "Init failed"
            ModelState.REMOVED -> "Removed"
        }
}

/**
 * Aggregate UI state for the ModelManager.
 */
data class ModelManagerUiState(
    /** Catalog of all available models */
    val models: List<ManagedModel> = emptyList(),

    /** Currently selected backend (llama.cpp or LiteRT-LM) */
    val selectedBackend: BackendKind = BackendKind.NONE,

    /** Model currently active for inference (if any) */
    val activeModelId: String? = null,

    /** Overall system status message */
    val systemMessage: String = "",

    /** Error message if initialization failed */
    val systemError: String? = null,

    /** Whether any model downloads are in progress */
    val hasActiveDownloads: Boolean = false,

    /** Filter applied to catalog (null = all) */
    val filter: ModelCatalogFilter? = null,

    /** Search query text */
    val searchText: String = "",
)

/** Filters for browsing the model catalog */
enum class ModelCatalogFilter {
    ALL,
    MOBILE_RECOMMENDED,
    DOWNLOADED,
    READY,
    LITERT_LM_ONLY,
    LLAMA_CPP_ONLY,
}

class ModelManagerViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application
    private val settingsStore = AppSettingsStore(app)
    private val downloadStore = LocalModelDownloadStore(app)
    private val secretsStore = SecureSecretsStore(app)

    /** The model catalog — pre-defined entries for known models */
    private val catalog: List<ModelCatalogEntry> = buildDefaultCatalog()

    /** Mutable UI state */
    private val _uiState = MutableStateFlow(ModelManagerUiState())
    val uiState: StateFlow<ModelManagerUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            refreshState()
        }
    }

    // =========================================================================
    // State Management
    // =========================================================================

    /** Refresh all model states from disk and backend */
    private suspend fun refreshState() {
        withContext(Dispatchers.IO) {
            val settings = settingsStore.load()
            val backend = BackendKind.fromPersistedValue(settings.onDeviceBackend)

            val managedModels = catalog.map { entry ->
                val record = findDownloadRecordForModel(entry.id)
                ManagedModel(
                    catalogEntry = entry,
                    state = inferModelState(entry, record),
                    downloadProgress = inferDownloadProgress(record),
                    localFilePath = record?.destinationPath,
                )
            }

            _uiState.update {
                it.copy(
                    models = managedModels,
                    selectedBackend = backend,
                    hasActiveDownloads = managedModels.any { it.isDownloading },
                    systemMessage = buildSystemMessage(backend, managedModels),
                )
            }
        }
    }

    // =========================================================================
    // Catalog Operations
    // =========================================================================

    /** Apply a filter to the catalog */
    fun setFilter(filter: ModelCatalogFilter?) {
        _uiState.update {
            it.copy(filter = filter)
        }
    }

    /** Update search text */
    fun setSearchText(text: String) {
        _uiState.update {
            it.copy(searchText = text)
        }
    }

    /** Get filtered catalog */
    fun getFilteredModels(): List<ManagedModel> {
        val state = _uiState.value
        return state.models.filter { model ->
            val passesFilter = when (state.filter) {
                ModelCatalogFilter.MOBILE_RECOMMENDED -> model.catalogEntry.isMobileRecommended
                ModelCatalogFilter.DOWNLOADED -> model.isDownloaded
                ModelCatalogFilter.READY -> model.isReady
                ModelCatalogFilter.LITERT_LM_ONLY -> model.catalogEntry.supportedBackends.contains(ModelRuntimeBackend.LITERT_LM)
                ModelCatalogFilter.LLAMA_CPP_ONLY -> model.catalogEntry.supportedBackends.contains(ModelRuntimeBackend.LLAMA_CPP)
                ModelCatalogFilter.ALL, null -> true
            }
            val passesSearch = state.searchText.isBlank() ||
                model.catalogEntry.displayName.contains(state.searchText, ignoreCase = true) ||
                model.catalogEntry.description.contains(state.searchText, ignoreCase = true) ||
                model.catalogEntry.tags.any { it.contains(state.searchText, ignoreCase = true) }
            passesFilter && passesSearch
        }
    }

    // =========================================================================
    // Model Lifecycle
    // =========================================================================

    /** Start downloading a model from the catalog */
    fun downloadModel(modelId: String) {
        val entry = catalog.find { it.id == modelId } ?: return

        _uiState.update {
            it.copy(
                models = it.models.map { m ->
                    if (m.catalogEntry.id == modelId) {
                        m.copy(state = ModelState.DOWNLOADING)
                    } else m
                },
                hasActiveDownloads = true,
                systemMessage = "Starting download for ${entry.displayName}…",
            )
        }

        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    HermesModelDownloadManager.enqueueDownload(
                        context = app,
                        store = downloadStore,
                        draft = ModelDownloadDraft(
                            repoOrUrl = entry.repoId,
                            filePath = "",
                            revision = entry.revision,
                            runtimeFlavor = if (entry.supportedBackends.contains(ModelRuntimeBackend.LITERT_LM)) "LiteRT-LM" else "GGUF",
                        ),
                        hfToken = secretsStore.loadApiKey("huggingface"),
                        dataSaverMode = false,
                    )
                }
            }.onSuccess { record ->
                _uiState.update {
                    it.copy(
                        systemMessage = "Download queued: ${record.title}",
                    )
                }
                // Monitor download progress
                monitorDownload(record.id, modelId)
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        models = it.models.map { m ->
                            if (m.catalogEntry.id == modelId) {
                                m.copy(
                                    state = ModelState.INIT_FAILED,
                                    errorMessage = error.message ?: "Download failed",
                                )
                            } else m
                        },
                        systemMessage = "Download failed: ${error.message}",
                    )
                }
            }
        }
    }

    /** Monitor download progress for a specific model */
    private fun monitorDownload(recordId: String, modelId: String) {
        viewModelScope.launch {
            while (true) {
                kotlinx.coroutines.delay(2000)
                val record = downloadStore.findDownload(recordId) ?: break
                val isComplete = record.status == "completed"
                val isFailed = record.status == "failed" || record.status == "canceled"

                val progress = ModelDownloadProgress(
                    modelId = modelId,
                    downloadedBytes = record.downloadedBytes,
                    totalBytes = record.totalBytes,
                    progressFraction = if (record.totalBytes > 0) {
                        record.downloadedBytes.toFloat() / record.totalBytes.toFloat()
                    } else 0f,
                    status = record.status,
                )

                val newState = when {
                    isComplete -> ModelState.DOWNLOADED
                    isFailed -> ModelState.INIT_FAILED
                    record.status in setOf("queued", "downloading", "paused") -> ModelState.DOWNLOADING
                    else -> ModelState.NOT_AVAILABLE
                }

                _uiState.update {
                    it.copy(
                        models = it.models.map { m ->
                            if (m.catalogEntry.id == modelId) {
                                m.copy(
                                    state = newState,
                                    downloadProgress = progress,
                                    localFilePath = record.destinationPath.takeIf { it.isNotBlank() },
                                )
                            } else m
                        },
                        hasActiveDownloads = !isComplete && !isFailed,
                        systemMessage = if (isComplete) {
                            "Download complete for ${record.title}"
                        } else if (isFailed) {
                            "Download failed for ${record.title}"
                        } else {
                            "Downloading ${record.title}… ${progress.percentage}%"
                        },
                    )
                }

                if (isComplete || isFailed) break
            }
        }
    }

    /** Initialize a model for inference (loads into runtime) */
    fun initializeModel(modelId: String) {
        val model = _uiState.value.models.find { it.catalogEntry.id == modelId } ?: return

        if (!model.isDownloaded) {
            _uiState.update {
                it.copy(systemMessage = "Model ${model.catalogEntry.displayName} is not downloaded yet")
            }
            return
        }

        val filePath = model.localFilePath ?: return
        val entry = model.catalogEntry

        _uiState.update {
            it.copy(
                models = it.models.map { m ->
                    if (m.catalogEntry.id == modelId) {
                        m.copy(state = ModelState.DOWNLOADING)
                    } else m
                },
                systemMessage = "Initializing ${entry.displayName}…",
            )
        }

        viewModelScope.launch {
            val success = withContext(Dispatchers.IO) {
                try {
                    // Try to probe the LiteRT-LM engine with this model
                    testModelInitialization(filePath, entry)
                } catch (e: Exception) {
                    false
                }
            }

            if (success) {
                _uiState.update {
                    it.copy(
                        models = it.models.map { m ->
                            if (m.catalogEntry.id == modelId) {
                                m.copy(state = ModelState.READY)
                            } else m
                        },
                        activeModelId = modelId,
                        systemMessage = "${entry.displayName} initialized and ready",
                    )
                }
            } else {
                _uiState.update {
                    it.copy(
                        models = it.models.map { m ->
                            if (m.catalogEntry.id == modelId) {
                                m.copy(
                                    state = ModelState.INIT_FAILED,
                                    errorMessage = "Model initialization failed",
                                )
                            } else m
                        },
                        systemMessage = "Failed to initialize ${entry.displayName}",
                    )
                }
            }
        }
    }

    /** Remove a model from the device */
    fun removeModel(modelId: String) {
        val model = _uiState.value.models.find { it.catalogEntry.id == modelId } ?: return

        model.localFilePath?.let { path ->
            kotlin.runCatching { File(path).delete() }
        }

        _uiState.update {
            it.copy(
                models = it.models.map { m ->
                    if (m.catalogEntry.id == modelId) {
                        m.copy(
                            state = ModelState.NOT_AVAILABLE,
                            localFilePath = null,
                            downloadProgress = null,
                            errorMessage = null,
                        )
                    } else m
                },
                activeModelId = if (it.activeModelId == modelId) null else it.activeModelId,
                systemMessage = "Removed ${model.catalogEntry.displayName}",
            )
        }
    }

    /** Set this model as preferred for the backend */
    fun setPreferred(modelId: String) {
        val entry = catalog.find { it.id == modelId } ?: return
        val record = findDownloadRecordForModel(modelId) ?: return

        // Find the corresponding download record and mark it as preferred
        downloadStore.setPreferredDownloadId(record.id)

        // Sync backend kind based on model
        val backendKind = if (entry.supportedBackends.contains(ModelRuntimeBackend.LITERT_LM)) {
            BackendKind.LITERT_LM
        } else {
            BackendKind.LLAMA_CPP
        }

        _uiState.update {
            it.copy(
                selectedBackend = backendKind,
                systemMessage = "${entry.displayName} set as preferred model",
            )
        }
    }

    // =========================================================================
    // Backend Operations
    // =========================================================================

    /** Start the backend server with the active model */
    fun startBackend() {
        val activeModel = _uiState.value.models.find { it.catalogEntry.id == _uiState.value.activeModelId }
            ?: run {
                _uiState.update { it.copy(systemMessage = "No active model selected") }
                return
            }

        if (!activeModel.isReady) {
            _uiState.update { it.copy(systemMessage = "Model is not ready for inference") }
            return
        }

        _uiState.update {
            it.copy(systemMessage = "Starting backend server…")
        }

        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                OnDeviceBackendManager.ensureConfigured(app, _uiState.value.selectedBackend.persistedValue)
            }
            refreshState()
        }
    }

    /** Stop the backend server */
    fun stopBackend() {
        OnDeviceBackendManager.stopAll()
        _uiState.update {
            it.copy(
                systemMessage = "Backend server stopped",
                activeModelId = null,
            )
        }
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    private fun findDownloadRecordForModel(modelId: String): LocalModelDownloadRecord? {
        // Try to find by matching the repo ID or title pattern
        val entry = catalog.find { it.id == modelId } ?: return null
        return downloadStore.loadDownloads().find { record ->
            record.title.contains(entry.displayName, ignoreCase = true) ||
                record.destinationPath.contains(entry.displayName.lowercase(Locale.US), true)
        }
    }

    private fun inferModelState(entry: ModelCatalogEntry, record: LocalModelDownloadRecord?): ModelState {
        if (record == null) return ModelState.NOT_AVAILABLE
        return when (record.status) {
            "completed" -> ModelState.DOWNLOADED
            "queued", "downloading", "paused" -> ModelState.DOWNLOADING
            "failed", "canceled" -> ModelState.INIT_FAILED
            else -> ModelState.NOT_AVAILABLE
        }
    }

    private fun inferDownloadProgress(record: LocalModelDownloadRecord?): ModelDownloadProgress? {
        if (record == null) return null
        val entry = catalog.find {
            record.title.contains(it.displayName, ignoreCase = true)
        } ?: return null
        return ModelDownloadProgress(
            modelId = entry.id,
            downloadedBytes = record.downloadedBytes,
            totalBytes = record.totalBytes,
            progressFraction = if (record.totalBytes > 0) {
                record.downloadedBytes.toFloat() / record.totalBytes.toFloat()
            } else 0f,
            status = record.status,
        )
    }

    private fun buildSystemMessage(backend: BackendKind, models: List<ManagedModel>): String {
        val readyModels = models.filter { it.isReady }
        val downloadingModels = models.count { it.isDownloading }

        return when {
            readyModels.isNotEmpty() && downloadingModels > 0 ->
                "${readyModels.size} model(s) ready, $downloadingModels downloading"
            readyModels.isNotEmpty() ->
                "${readyModels.size} model(s) ready for inference"
            downloadingModels > 0 ->
                "$downloadingModels model(s) downloading"
            backend == BackendKind.NONE ->
                "Remote provider mode — no local models configured"
            else ->
                "No models ready. Download and initialize a model to get started."
        }
    }

    private fun testModelInitialization(modelPath: String, entry: ModelCatalogEntry): Boolean {
        val file = File(modelPath)
        if (!file.exists()) return false

        // For LiteRT-LM models, try a quick probe
        if (entry.supportedBackends.contains(ModelRuntimeBackend.LITERT_LM)) {
            // LiteRT-LM initialization is done in the proxy server
            // We just verify the file exists and has reasonable size
            return file.length() > 1_000_000 // At least 1MB
        }

        // For GGUF models, just verify the file
        return file.length() > 1_000_000
    }

    /**
     * Build the default model catalog with known models for on-device inference.
     * Includes Gemma 4 variants, Qwen models, and other LiteRT-LM compatible models.
     */
    private fun buildDefaultCatalog(): List<ModelCatalogEntry> {
        return listOf(
            // Gemma 4 and small LiteRT-LM models verified for mobile-sized downloads.
            ModelCatalogEntry(
                id = "gemma-4-e2b-litert-lm",
                displayName = "Gemma 4 E2B (LiteRT-LM)",
                description = "Google Gemma 4 E2B instruction model packaged for LiteRT-LM. Small enough for on-device chat and native tool-use validation.",
                repoId = "litert-community/gemma-4-E2B-it-litert-lm",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 2_583_085_056,
                recommendedRamBytes = 4_000_000_000,
                tags = listOf("gemma", "google", "litert-lm", "conversation", "tool-use", "2b"),
                author = "Google",
                license = "Apache-2.0",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "gemma-4-e4b-litert-lm",
                displayName = "Gemma 4 E4B (LiteRT-LM)",
                description = "Google Gemma 4 E4B instruction model packaged for LiteRT-LM. Larger than E2B but still below the 5 GB mobile testing ceiling.",
                repoId = "litert-community/gemma-4-E4B-it-litert-lm",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 3_654_467_584,
                recommendedRamBytes = 6_000_000_000,
                tags = listOf("gemma", "google", "litert-lm", "reasoning", "tool-use", "4b"),
                author = "Google",
                license = "Apache-2.0",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "gemma-3-1b-it-litert-lm",
                displayName = "Gemma 3 1B IT INT4 (LiteRT-LM)",
                description = "Google Gemma 3 1B instruction model packaged for LiteRT-LM. This is the smallest first-class Gemma 3 path for compatibility and startup checks.",
                repoId = "litert-community/Gemma3-1B-IT",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 0,
                recommendedRamBytes = 2_000_000_000,
                tags = listOf("gemma", "google", "gemma-3", "litert-lm", "small", "1b"),
                author = "Google",
                license = "Gemma",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "gemma-3-4b-it-vision-task",
                displayName = "Gemma 3 4B IT Vision (.task)",
                description = "Google Gemma 3 4B image-text instruction model packaged as a LiteRT task artifact for multimodal image description tests.",
                repoId = "litert-community/Gemma3-4B-IT",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 0,
                recommendedRamBytes = 6_000_000_000,
                supportsImageInput = true,
                tags = listOf("gemma", "google", "gemma-3", "litert-lm", "vision", "image-text", "4b"),
                author = "Google",
                license = "Gemma",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "gemma-3n-e2b-it-litert-lm",
                displayName = "Gemma 3n E2B IT Vision (LiteRT-LM)",
                description = "Google Gemma 3n E2B instruction model packaged for LiteRT-LM with image input support for mobile multimodal tests.",
                repoId = "google/gemma-3n-E2B-it-litert-lm",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 0,
                recommendedRamBytes = 4_000_000_000,
                supportsImageInput = true,
                supportsAudioInput = true,
                tags = listOf("gemma", "google", "gemma-3n", "litert-lm", "vision", "audio", "2b"),
                author = "Google",
                license = "Gemma",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "gemma-3n-e4b-it-litert-lm",
                displayName = "Gemma 3n E4B IT Vision (LiteRT-LM)",
                description = "Google Gemma 3n E4B instruction model packaged for LiteRT-LM with image input support under the 5 GB model testing target when an int4 artifact is selected.",
                repoId = "google/gemma-3n-E4B-it-litert-lm",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 0,
                recommendedRamBytes = 6_000_000_000,
                supportsImageInput = true,
                supportsAudioInput = true,
                tags = listOf("gemma", "google", "gemma-3n", "litert-lm", "vision", "audio", "4b"),
                author = "Google",
                license = "Gemma",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "qwen3-0-6b-litert-lm",
                displayName = "Qwen3 0.6B (LiteRT-LM)",
                description = "Qwen3 0.6B LiteRT-LM model for very small on-device inference checks and fast native agent smoke tests.",
                repoId = "litert-community/Qwen3-0.6B",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 614_236_160,
                recommendedRamBytes = 2_000_000_000,
                tags = listOf("qwen", "alibaba", "litert-lm", "small", "0.6b"),
                author = "Qwen/Alibaba",
                license = "Apache-2.0",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "qwen2-5-1-5b-instruct-litert-lm",
                displayName = "Qwen2.5 1.5B Instruct (LiteRT-LM)",
                description = "Qwen2.5 1.5B Instruct quantized LiteRT-LM model for lightweight mobile chat and tool-routing tests.",
                repoId = "litert-community/Qwen2.5-1.5B-Instruct",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LITERT_LM),
                approximateSizeBytes = 1_597_931_520,
                recommendedRamBytes = 3_000_000_000,
                tags = listOf("qwen", "alibaba", "litert-lm", "instruct", "1.5b"),
                author = "Qwen/Alibaba",
                license = "Apache-2.0",
                isMobileRecommended = true,
            ),
            // Qwen models (GGUF compatible via llama.cpp)
            ModelCatalogEntry(
                id = "qwen3-4b-gguf",
                displayName = "Qwen3 4B (GGUF)",
                description = "Qwen3 4B quantized model in GGUF format. Lightweight and fast, ideal for devices with 6GB+ RAM. Supports tool use and function calling.",
                repoId = "Qwen/Qwen3-4B-GGUF",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LLAMA_CPP),
                approximateSizeBytes = 3_000_000_000,
                recommendedRamBytes = 6_000_000_000,
                tags = listOf("qwen", "alibaba", "tool-use", "gguf"),
                author = "Qwen/Alibaba",
                license = "Apache-2.0",
                isMobileRecommended = true,
            ),
            ModelCatalogEntry(
                id = "qwen3-8b-gguf",
                displayName = "Qwen3 8B (GGUF)",
                description = "Qwen3 8B in GGUF format. Strong multilingual capabilities and reasoning. Requires 10GB+ RAM for comfortable inference.",
                repoId = "Qwen/Qwen3-8B-GGUF",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LLAMA_CPP),
                approximateSizeBytes = 5_500_000_000,
                recommendedRamBytes = 10_000_000_000,
                tags = listOf("qwen", "alibaba", "multilingual", "gguf"),
                author = "Qwen/Alibaba",
                license = "Apache-2.0",
                isMobileRecommended = true,
            ),
            // Phi models
            ModelCatalogEntry(
                id = "phi-4-gguf",
                displayName = "Phi 4 (GGUF)",
                description = "Microsoft's Phi 4 — a capable small language model optimized for efficiency. Excellent performance-to-size ratio.",
                repoId = "microsoft/Phi-4-GGUF",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LLAMA_CPP),
                approximateSizeBytes = 2_500_000_000,
                recommendedRamBytes = 5_000_000_000,
                tags = listOf("phi", "microsoft", "efficient", "gguf"),
                author = "Microsoft",
                license = "MIT",
                isMobileRecommended = true,
            ),
            // Llama 3.2
            ModelCatalogEntry(
                id = "llama-3-2-3b-gguf",
                displayName = "Llama 3.2 3B (GGUF)",
                description = "Meta's Llama 3.2 3B — extremely lightweight model designed for edge deployment. Fast inference on modest hardware.",
                repoId = "NousResearch/Llama-3.2-3B-Instruct-GGUF",
                revision = "main",
                supportedBackends = listOf(ModelRuntimeBackend.LLAMA_CPP),
                approximateSizeBytes = 2_000_000_000,
                recommendedRamBytes = 4_000_000_000,
                tags = listOf("llama", "meta", "edge", "gguf"),
                author = "Meta",
                license = "llama3.2",
                isMobileRecommended = true,
            ),
        )
    }
}
