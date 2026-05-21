package com.mobilefork.hermesagent.device

import java.util.Locale

internal object HermesAndroidHardwareProfile {
    data class Profile(
        val socFamily: String,
        val socLabel: String,
        val gpuFamily: String,
        val gpuLabel: String,
    )

    fun classify(values: List<String>): Profile {
        val socFamily = socFamily(values)
        val gpuFamily = gpuFamily(values)
        return Profile(
            socFamily = socFamily,
            socLabel = socLabel(socFamily),
            gpuFamily = gpuFamily,
            gpuLabel = gpuLabel(gpuFamily),
        )
    }

    fun socFamily(values: List<String>): String = when {
        isLikelyMediatekSoc(values) -> "mediatek"
        isLikelySnapdragonSoc(values) -> "qualcomm_snapdragon"
        isLikelyGoogleTensorSoc(values) -> "google_tensor"
        isLikelyExynosSoc(values) -> "samsung_exynos"
        isLikelyUnisocSoc(values) -> "unisoc"
        else -> "unknown"
    }

    fun gpuFamily(values: List<String>): String {
        val normalized = normalized(values)
        return when {
            "xclipse" in normalized || "amd rdna" in normalized -> "xclipse"
            "immortalis" in normalized -> "mali_immortalis"
            "mali" in normalized || "valhall" in normalized || "bifrost" in normalized -> "mali"
            "adreno" in normalized -> "adreno"
            "powervr" in normalized || "power vr" in normalized || "imgtec" in normalized || "rogue" in normalized -> "powervr_img"
            else -> "unknown"
        }
    }

    fun accelerationLabel(profile: Profile): String {
        val socPart = if (profile.socFamily == "unknown") "Android" else profile.socLabel
        val gpuPart = if (profile.gpuFamily == "unknown") "" else "/${profile.gpuLabel}"
        return "ARM $socPart$gpuPart"
    }

    fun nativeAbiCandidates(supportedAbis: List<String>): List<String> {
        return supportedAbis.map { it.trim() }.filter { it.isNotBlank() }.distinct()
    }

    fun nativeAbiStrategy(supportedAbis: List<String>): String {
        val candidates = nativeAbiCandidates(supportedAbis)
        val primary = candidates.firstOrNull().orEmpty().ifBlank { "unknown" }
        return when {
            candidates.any { isX86Abi(it) } ->
                "CPU-only emulator/device path for primary ABI $primary; do not infer phone GPU support from x86."
            candidates.any { isArmAbi(it) } ->
                "Select packaged native artifacts from Build.SUPPORTED_ABIS starting at $primary, then probe LiteRT-LM GPU support with CPU fallback across Adreno, Mali, Immortalis, Xclipse, and PowerVR/IMG devices."
            else ->
                "No ARM ABI advertised; use CPU/runtime fallbacks until a compatible native artifact is packaged."
        }
    }

    fun isArmAbi(abi: String): Boolean {
        val normalized = abi.lowercase(Locale.US)
        return normalized.startsWith("arm") || normalized == "aarch64"
    }

    fun isX86Abi(abi: String): Boolean = abi.lowercase(Locale.US).startsWith("x86")

    fun isLikelyMediatekSoc(values: List<String>): Boolean {
        val normalized = normalized(values)
        return listOf("mediatek", "mtk", "dimensity", "helio").any { it in normalized } ||
            Regex("""\bmt[0-9]{4,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    fun isLikelySnapdragonSoc(values: List<String>): Boolean {
        val normalized = normalized(values)
        return listOf("qualcomm", "snapdragon", "qcom", "msm", "sdm").any { it in normalized } ||
            Regex("""\b(sm|sdm|msm)[0-9]{3,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    fun isLikelyGoogleTensorSoc(values: List<String>): Boolean {
        val normalized = normalized(values)
        return "google tensor" in normalized ||
            Regex("""\bgs[0-9]{3,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    fun isLikelyExynosSoc(values: List<String>): Boolean {
        val normalized = normalized(values)
        return "exynos" in normalized ||
            Regex("""\bs5e[0-9]{3,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    fun isLikelyUnisocSoc(values: List<String>): Boolean {
        val normalized = normalized(values)
        return "unisoc" in normalized || "spreadtrum" in normalized ||
            Regex("""\bums[0-9]{3,}[a-z0-9_+-]*\b""").containsMatchIn(normalized)
    }

    private fun socLabel(family: String): String = when (family) {
        "mediatek" -> "MediaTek"
        "qualcomm_snapdragon" -> "Qualcomm Snapdragon"
        "google_tensor" -> "Google Tensor"
        "samsung_exynos" -> "Samsung Exynos"
        "unisoc" -> "Unisoc"
        else -> "unknown"
    }

    private fun gpuLabel(family: String): String = when (family) {
        "adreno" -> "Adreno"
        "mali" -> "Mali"
        "mali_immortalis" -> "Mali Immortalis"
        "powervr_img" -> "PowerVR/IMG"
        "xclipse" -> "Xclipse"
        else -> "unknown"
    }

    private fun normalized(values: List<String>): String {
        return values.joinToString(" ").lowercase(Locale.US)
    }
}
