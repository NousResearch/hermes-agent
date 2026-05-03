package com.nousresearch.hermesagent.backend

import android.content.Context
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import java.io.File
import java.util.Locale

enum class BackendKind(val persistedValue: String) {
    NONE("none"),
    LLAMA_CPP("llama.cpp"),
    LITERT_LM("litert-lm"),
    AICORE("aicore");

    companion object {
        fun fromPersistedValue(value: String?): BackendKind {
            val normalized = value.orEmpty().trim().lowercase()
            return entries.firstOrNull { it.persistedValue == normalized } ?: NONE
        }
    }
}

data class LocalBackendStatus(
    val backendKind: BackendKind,
    val started: Boolean,
    val baseUrl: String = "",
    val modelName: String = "",
    val sourceModelPath: String = "",
    val statusMessage: String = "",
)

object OnDeviceBackendManager {
    const val LLAMA_CPP_PORT = 15435
    const val LITERT_LM_PORT = 15436

    @Volatile
    private var currentStatus: LocalBackendStatus = LocalBackendStatus(
        backendKind = BackendKind.NONE,
        started = false,
        statusMessage = "Remote provider mode",
    )

    fun currentStatus(): LocalBackendStatus = currentStatus

    @Synchronized
    fun ensureConfigured(context: Context, backendValue: String): LocalBackendStatus {
        return when (BackendKind.fromPersistedValue(backendValue)) {
            BackendKind.NONE -> {
                stopAll()
                currentStatus = LocalBackendStatus(
                    backendKind = BackendKind.NONE,
                    started = false,
                    statusMessage = "Remote provider mode",
                )
                currentStatus
            }
            BackendKind.LLAMA_CPP -> ensureLlamaCpp(context)
            BackendKind.LITERT_LM -> ensureLiteRtLm(context)
            BackendKind.AICORE -> ensureAICore(context)
        }
    }

    @Synchronized
    fun stopAll() {
        LlamaCppServerController.stop()
        LiteRtLmOpenAiProxy.stop()
    }

    fun preferredDownloadSummary(context: Context, backendValue: String): String {
        val preferred = preferredCompletedDownload(context)
        return if (preferred != null) {
            "Preferred local model: ${preferred.title}"
        } else {
            "No preferred local model is selected yet. Download any repo or file and mark it as preferred to let the selected backend try it."
        }
    }

    private fun ensureLlamaCpp(context: Context): LocalBackendStatus {
        LiteRtLmOpenAiProxy.stop()
        val preferred = preferredCompletedDownload(context)
            ?: run {
                LlamaCppServerController.stop()
                return LocalBackendStatus(
                    backendKind = BackendKind.LLAMA_CPP,
                    started = false,
                    statusMessage = "No preferred local model is ready for llama.cpp yet",
                ).also { currentStatus = it }
            }

        val modelFile = File(preferred.destinationPath)
        if (!modelFile.isFile) {
            LlamaCppServerController.stop()
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                statusMessage = "Preferred local model is missing on disk: ${preferred.destinationPath}",
                sourceModelPath = preferred.destinationPath,
            ).also { currentStatus = it }
        }
        if (!preferred.matchesBackendArtifact(BackendKind.LLAMA_CPP)) {
            LlamaCppServerController.stop()
            return incompatiblePreferredDownloadStatus(preferred, BackendKind.LLAMA_CPP)
        }

        val status = LlamaCppServerController.ensureRunning(
            context = context,
            modelPath = modelFile.absolutePath,
            requestedModelName = preferred.title,
            port = LLAMA_CPP_PORT,
        )
        currentStatus = status
        return status
    }

    private fun ensureLiteRtLm(context: Context): LocalBackendStatus {
        LlamaCppServerController.stop()
        val preferred = preferredCompletedDownload(context)
            ?: run {
                LiteRtLmOpenAiProxy.stop()
                return LocalBackendStatus(
                    backendKind = BackendKind.LITERT_LM,
                    started = false,
                    statusMessage = "No preferred local model is ready for LiteRT-LM yet",
                ).also { currentStatus = it }
            }

        val modelFile = File(preferred.destinationPath)
        if (!modelFile.isFile) {
            LiteRtLmOpenAiProxy.stop()
            return LocalBackendStatus(
                backendKind = BackendKind.LITERT_LM,
                started = false,
                statusMessage = "Preferred local model is missing on disk: ${preferred.destinationPath}",
                sourceModelPath = preferred.destinationPath,
            ).also { currentStatus = it }
        }
        if (!preferred.matchesBackendArtifact(BackendKind.LITERT_LM)) {
            LiteRtLmOpenAiProxy.stop()
            return incompatiblePreferredDownloadStatus(preferred, BackendKind.LITERT_LM)
        }

        val status = LiteRtLmOpenAiProxy.ensureRunning(
            context = context,
            modelPath = modelFile.absolutePath,
            requestedModelName = preferred.title,
            port = LITERT_LM_PORT,
            inferenceConfig = inferenceConfigFor(preferred),
        )
        currentStatus = status
        return status
    }

    /**
     * Ensure AICore backend is running with GPU/CPU fallback.
     * AICore requires API 35+ and NPU hardware; gracefully falls back to GPU/CPU.
     */
    private fun ensureAICore(context: Context): LocalBackendStatus {
        LlamaCppServerController.stop()
        val preferred = preferredCompletedDownload(context)
            ?: run {
                LiteRtLmOpenAiProxy.stop()
                return LocalBackendStatus(
                    backendKind = BackendKind.AICORE,
                    started = false,
                    statusMessage = "No preferred local model is ready for AICore yet",
                ).also { currentStatus = it }
            }

        val modelFile = java.io.File(preferred.destinationPath)
        if (!modelFile.isFile) {
            LiteRtLmOpenAiProxy.stop()
            return LocalBackendStatus(
                backendKind = BackendKind.AICORE,
                started = false,
                statusMessage = "Preferred local model is missing on disk: ${preferred.destinationPath}",
                sourceModelPath = preferred.destinationPath,
            ).also { currentStatus = it }
        }
        if (!preferred.matchesBackendArtifact(BackendKind.AICORE)) {
            LiteRtLmOpenAiProxy.stop()
            return incompatiblePreferredDownloadStatus(preferred, BackendKind.AICORE)
        }

        // AICore uses same port as LiteRT-LM but with AICore-appropriate inference config
        val inferenceConfig = AICoreBackendController.createAICoreInferenceConfig()
            .copy(
                supportImage = preferred.supportsImageInput(),
                supportAudio = preferred.supportsAudioInput(),
            )
        val status = LiteRtLmOpenAiProxy.ensureRunning(
            context = context,
            modelPath = modelFile.absolutePath,
            requestedModelName = preferred.title,
            port = AICoreBackendController.AICORE_PORT,
            inferenceConfig = inferenceConfig,
        )

        // Update status message to reflect actual backend used
        val actualBackend = AICoreBackendController.getBackendDescription()
        val finalStatus = status.copy(
            backendKind = BackendKind.AICORE,
            statusMessage = "AICore mode active: $actualBackend",
        )
        currentStatus = finalStatus
        return finalStatus
    }

    private fun preferredCompletedDownload(context: Context): LocalModelDownloadRecord? {
        val store = LocalModelDownloadStore(context)
        val preferredId = store.preferredDownloadId().ifBlank { return null }
        val preferred = store.findDownload(preferredId) ?: return null
        return preferred.takeIf { it.status == "completed" }
    }

    private fun LocalModelDownloadRecord.matchesBackendArtifact(backendKind: BackendKind): Boolean {
        val lower = destinationPath.lowercase(Locale.US)
        return when (backendKind) {
            BackendKind.LLAMA_CPP -> lower.endsWith(".gguf")
            BackendKind.LITERT_LM -> lower.endsWith(".litertlm") || lower.endsWith(".task")
            BackendKind.AICORE -> lower.endsWith(".litertlm") || lower.endsWith(".task")
            BackendKind.NONE -> true
        }
    }

    private fun inferenceConfigFor(preferred: LocalModelDownloadRecord): LiteRtLmOpenAiProxy.InferenceConfig {
        return LiteRtLmOpenAiProxy.InferenceConfig(
            supportImage = preferred.supportsImageInput(),
            supportAudio = preferred.supportsAudioInput(),
        )
    }

    private fun LocalModelDownloadRecord.supportsImageInput(): Boolean {
        val lower = modelIdentityText()
        return "gemma-3n" in lower ||
            "gemma3-4b" in lower ||
            "gemma-3-4b" in lower ||
            "vision" in lower ||
            "image-text" in lower
    }

    private fun LocalModelDownloadRecord.supportsAudioInput(): Boolean {
        val lower = modelIdentityText()
        return "gemma-3n" in lower || "audio" in lower
    }

    private fun LocalModelDownloadRecord.modelIdentityText(): String {
        return listOf(title, repoOrUrl, filePath, destinationFileName, destinationPath)
            .joinToString(" ")
            .lowercase(Locale.US)
    }

    private fun incompatiblePreferredDownloadStatus(
        preferred: LocalModelDownloadRecord,
        backendKind: BackendKind,
    ): LocalBackendStatus {
        val requiredExtension = when (backendKind) {
            BackendKind.LLAMA_CPP -> ".gguf"
            BackendKind.LITERT_LM -> ".litertlm or .task"
            BackendKind.AICORE -> ".litertlm or .task"
            BackendKind.NONE -> "supported"
        }
        val backendLabel = when (backendKind) {
            BackendKind.LLAMA_CPP -> "llama.cpp"
            BackendKind.LITERT_LM -> "LiteRT-LM"
            BackendKind.AICORE -> "AICore (NPU)"
            BackendKind.NONE -> "the selected backend"
        }
        return LocalBackendStatus(
            backendKind = backendKind,
            started = false,
            sourceModelPath = preferred.destinationPath,
            statusMessage = "Preferred local model ${preferred.destinationFileName} is not a $requiredExtension file, so $backendLabel cannot load it. Download a $requiredExtension artifact and mark it as preferred first.",
        ).also { currentStatus = it }
    }
}
