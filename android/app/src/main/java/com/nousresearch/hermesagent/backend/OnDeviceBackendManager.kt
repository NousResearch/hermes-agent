package com.nousresearch.hermesagent.backend

import android.content.Context
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import java.io.File

enum class BackendKind(val persistedValue: String) {
    NONE("none"),
    LLAMA_CPP("llama.cpp"),
    LITERT_LM("litert-lm");

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
        }
    }

    @Synchronized
    fun stopAll() {
        LlamaCppServerController.stop()
        LiteRtLmOpenAiProxy.stop()
    }

    fun preferredDownloadSummary(context: Context, backendValue: String): String {
        val preferred = preferredDownload(context, BackendKind.fromPersistedValue(backendValue))
        return if (preferred != null) {
            "Preferred local model: ${preferred.title}"
        } else {
            "No compatible local model is selected yet. Download one and mark it as preferred first."
        }
    }

    private fun ensureLlamaCpp(context: Context): LocalBackendStatus {
        LiteRtLmOpenAiProxy.stop()
        val preferred = preferredDownload(context, BackendKind.LLAMA_CPP)
            ?: run {
                LlamaCppServerController.stop()
                return LocalBackendStatus(
                    backendKind = BackendKind.LLAMA_CPP,
                    started = false,
                    statusMessage = "No preferred GGUF model is ready for llama.cpp yet",
                ).also { currentStatus = it }
            }

        val modelFile = File(preferred.destinationPath)
        if (!modelFile.isFile) {
            LlamaCppServerController.stop()
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                statusMessage = "Preferred GGUF model is missing on disk: ${preferred.destinationPath}",
                sourceModelPath = preferred.destinationPath,
            ).also { currentStatus = it }
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
        val preferred = preferredDownload(context, BackendKind.LITERT_LM)
            ?: run {
                LiteRtLmOpenAiProxy.stop()
                return LocalBackendStatus(
                    backendKind = BackendKind.LITERT_LM,
                    started = false,
                    statusMessage = "No preferred LiteRT-LM model is ready yet",
                ).also { currentStatus = it }
            }

        val modelFile = File(preferred.destinationPath)
        if (!modelFile.isFile) {
            LiteRtLmOpenAiProxy.stop()
            return LocalBackendStatus(
                backendKind = BackendKind.LITERT_LM,
                started = false,
                statusMessage = "Preferred LiteRT-LM model is missing on disk: ${preferred.destinationPath}",
                sourceModelPath = preferred.destinationPath,
            ).also { currentStatus = it }
        }

        val status = LiteRtLmOpenAiProxy.ensureRunning(
            context = context,
            modelPath = modelFile.absolutePath,
            requestedModelName = preferred.title,
            port = LITERT_LM_PORT,
        )
        currentStatus = status
        return status
    }

    private fun preferredDownload(context: Context, backendKind: BackendKind): LocalModelDownloadRecord? {
        val store = LocalModelDownloadStore(context)
        val preferredId = store.preferredDownloadId().ifBlank { return null }
        val preferred = store.findDownload(preferredId) ?: return null
        if (preferred.status != "completed") {
            return null
        }
        return when (backendKind) {
            BackendKind.NONE -> null
            BackendKind.LLAMA_CPP -> if (preferred.matchesGguf()) preferred else null
            BackendKind.LITERT_LM -> if (preferred.matchesLiteRtLm()) preferred else null
        }
    }

    private fun LocalModelDownloadRecord.matchesGguf(): Boolean {
        return runtimeFlavor.equals("GGUF", ignoreCase = true) || destinationPath.endsWith(".gguf", ignoreCase = true)
    }

    private fun LocalModelDownloadRecord.matchesLiteRtLm(): Boolean {
        return runtimeFlavor.contains("litert", ignoreCase = true) ||
            destinationPath.endsWith(".litertlm", ignoreCase = true)
    }
}
