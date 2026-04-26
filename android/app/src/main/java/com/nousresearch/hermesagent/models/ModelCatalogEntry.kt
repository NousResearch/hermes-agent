package com.nousresearch.hermesagent.models

/**
 * Model catalog entry following Google AI Edge Gallery pattern.
 * Represents a model available for on-device inference with metadata
 * for download, compatibility, and runtime selection.
 */
data class ModelCatalogEntry(
    /** Unique identifier for this model entry */
    val id: String,

    /** Human-readable display name */
    val displayName: String,

    /** Short description of the model and its capabilities */
    val description: String,

    /** Hugging Face repo ID (e.g., "litert-community/gemma-4-E2B-it-litert-lm") */
    val repoId: String,

    /** Branch or revision to download from */
    val revision: String = "main",

    /** Supported runtime backends */
    val supportedBackends: List<ModelRuntimeBackend>,

    /** Approximate model file size in bytes (0 if unknown) */
    val approximateSizeBytes: Long = 0L,

    /** Recommended minimum RAM for inference (in bytes) */
    val recommendedRamBytes: Long = 0L,

    /** Whether this model supports quantization */
    val supportsQuantization: Boolean = false,

    /** Whether this model is recommended for mobile devices */
    val isMobileRecommended: Boolean = false,

    /** Tags for filtering (e.g., "gemma", "qwen", "multilingual") */
    val tags: List<String> = emptyList(),

    /** Author/organization that created or published this model */
    val author: String = "",

    /** License identifier */
    val license: String = "",
)

/** Supported runtime backends for on-device inference */
enum class ModelRuntimeBackend(val identifier: String, val label: String) {
    LLAMA_CPP("llama.cpp", "llama.cpp (GGUF)"),
    LITERT_LM("litert-lm", "LiteRT-LM"),
}

/** Current state of a model in the device */
enum class ModelState {
    /** Model is not present on the device */
    NOT_AVAILABLE,

    /** Model download is in progress */
    DOWNLOADING,

    /** Model is downloaded but not yet initialized */
    DOWNLOADED,

    /** Model is initialized and ready for inference */
    READY,

    /** Model initialization failed */
    INIT_FAILED,

    /** Model was removed or is corrupted */
    REMOVED,
}

/** Download progress for a model */
data class ModelDownloadProgress(
    val modelId: String,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = 0L,
    val progressFraction: Float = 0f,
    val status: String = "queued",
) {
    val percentage: Int
        get() = (progressFraction * 100).toInt().coerceIn(0, 100)

    val isComplete: Boolean
        get() = status == "completed" || progressFraction >= 1f

    val isActive: Boolean
        get() = status in setOf("queued", "downloading", "paused")
}
