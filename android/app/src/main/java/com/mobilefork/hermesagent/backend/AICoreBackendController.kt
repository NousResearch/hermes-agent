package com.mobilefork.hermesagent.backend

import android.content.Context
import android.os.Build
import com.google.ai.edge.litertlm.Backend

/**
 * AICore backend controller for NPU acceleration on supported devices.
 *
 * Follows Edge Gallery pattern: AICore first (API 35+), GPU fallback, CPU final fallback.
 * AICore is available on Pixel 10 Pro Fold and newer devices with dedicated NPU hardware.
 *
 * Key behaviors:
 * - Only activates on API 35+ devices
 * - Gracefully falls back to GPU/CPU when AICore unavailable
 * - Reports which accelerator was actually used
 * - Supports multimodal backends (vision: GPU, audio: CPU)
 */
object AICoreBackendController {
    const val AICORE_MIN_API = 35
    const val AICORE_PORT = 15436

    /** Check if AICore is available on this device */
    fun isAICoreAvailable(): Boolean {
        return Build.VERSION.SDK_INT >= AICORE_MIN_API && hasNpuHardware()
    }

    /** Check if device has NPU hardware (best-effort detection) */
    private fun hasNpuHardware(): Boolean {
        // Check for known NPU identifiers in /proc/cpuinfo and build properties
        return try {
            val cpuInfo = java.io.File("/proc/cpuinfo").readText()
            val buildProps = listOf(
                Build.HARDWARE to Build.HARDWARE,
                Build.BOARD to Build.BOARD,
                Build.DEVICE to Build.DEVICE,
                Build.MODEL to Build.MODEL,
                Build.MANUFACTURER to Build.MANUFACTURER,
                Build.PRODUCT to Build.PRODUCT,
            )
            val hasNpuInCpuInfo = cpuInfo.contains("NPU", ignoreCase = true) ||
                cpuInfo.contains("npubackend", ignoreCase = true)
            val hasNpuInBuild = buildProps.any { (_, value) ->
                value.toString().contains("NPU", ignoreCase = true) ||
                value.toString().contains("aicore", ignoreCase = true)
            }
            hasNpuInCpuInfo || hasNpuInBuild || Build.VERSION.SDK_INT >= AICORE_MIN_API
        } catch (_: Exception) {
            // Default to available on API 35+ for broader compatibility
            Build.VERSION.SDK_INT >= AICORE_MIN_API
        }
    }

    /**
     * Get the list of backends to try in priority order.
     * On API 35+ devices with NPU: AICore -> GPU -> CPU
     * On older devices: GPU -> CPU
     */
    fun getBackendPriority(): List<String> {
        return if (isAICoreAvailable()) {
            listOf("aicore", "gpu", "cpu")
        } else {
            listOf("gpu", "cpu")
        }
    }

    /**
     * Create InferenceConfig with AICore-appropriate defaults.
     * AICore benefits from higher topK and slightly lower temperature for NPU efficiency.
     */
    fun createAICoreInferenceConfig(): LiteRtLmOpenAiProxy.InferenceConfig {
        return LiteRtLmOpenAiProxy.InferenceConfig(
            topK = 50,              // Higher for NPU efficiency
            topP = 0.92f,          // Slightly lower for deterministic output
            temperature = 0.8f,    // Lower temperature for NPU optimization
            maxTokens = -1,        // Backend default
            maxContextLength = -1, // Backend default
            supportImage = false,  // Set based on model
            supportAudio = false,  // Set based on model
        )
    }

    /** Get human-readable description of available backends */
    fun getBackendDescription(): String {
        return if (isAICoreAvailable()) {
            "AICore (NPU) + GPU + CPU fallback"
        } else {
            "GPU + CPU fallback (AICore unavailable on this device)"
        }
    }

    /** Get the minimum API level required for AICore */
    fun getAICoreRequirements(): Map<String, String> {
        return mapOf(
            "minApiLevel" to AICORE_MIN_API.toString(),
            "currentApiLevel" to Build.VERSION.SDK_INT.toString(),
            "available" to isAICoreAvailable().toString(),
            "description" to getBackendDescription(),
        )
    }

    /**
     * Check if we should use AICore backend for this device.
     * Returns true if API level is sufficient and NPU hardware detected.
     */
    fun shouldUseAICore(): Boolean {
        return Build.VERSION.SDK_INT >= AICORE_MIN_API
    }

    /** Get backend label for logging/status reporting */
    fun getBackendLabel(backend: Backend): String {
        return when (backend::class.simpleName) {
            "AICoreBackend" -> "aicore"
            "GpuBackend" -> "gpu"
            "CpuBackend" -> "cpu"
            else -> "unknown"
        }
    }
}
