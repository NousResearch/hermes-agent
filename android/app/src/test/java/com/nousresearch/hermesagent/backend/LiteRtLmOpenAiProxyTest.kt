package com.nousresearch.hermesagent.backend

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.io.path.pathString

class LiteRtLmOpenAiProxyTest {
    @Test
    fun validateModelArtifact_acceptsLiteRtLmHeader() {
        val file = tempModelFile("gemma-4-E2B-it.litertlm", "LITERTLM".toByteArray())

        assertNull(validateModelArtifact(file))
    }

    @Test
    fun validateModelArtifact_rejectsWebTaskFlatBufferBeforeEngineStart() {
        val file = tempModelFile(
            "gemma-4-E2B-it-web.task",
            byteArrayOf(0, 0, 0, 0, 'T'.code.toByte(), 'F'.code.toByte(), 'L'.code.toByte(), '3'.code.toByte()),
        )

        val error = validateModelArtifact(file).orEmpty()

        assertTrue(error, error.contains("web/browser .task FlatBuffer"))
        assertTrue(error, error.contains("download the .litertlm artifact instead"))
    }

    @Test
    fun validateModelArtifact_rejectsBrokenLiteRtLmFileWithZipHeader() {
        val file = tempModelFile("gemma-4-E4B-it.litertlm", byteArrayOf('P'.code.toByte(), 'K'.code.toByte(), 3, 4, 0, 0, 0, 0))

        val error = validateModelArtifact(file)

        assertEquals(
            "gemma-4-E4B-it.litertlm is not a valid LiteRT-LM bundle. Download the .litertlm artifact from the LiteRT-LM repo.",
            error,
        )
    }

    @Test
    fun memorySafeModalityDecision_keepsSmallModelMultimodalEnabled() {
        val decision = LiteRtLmOpenAiProxy.memorySafeModalityDecision(
            totalRamBytes = 4_000_000_000L,
            modelBytes = 1_000_000_000L,
            requestedImage = true,
            requestedAudio = true,
        )

        assertTrue(decision.supportImage)
        assertTrue(decision.supportAudio)
        assertTrue(decision.policy, decision.policy.contains("requested image and audio"))
    }

    @Test
    fun memorySafeModalityDecision_startsLargeGemma4TextOnlyOnFourGbDevice() {
        val decision = LiteRtLmOpenAiProxy.memorySafeModalityDecision(
            totalRamBytes = 4_000_000_000L,
            modelBytes = 2_583_085_056L,
            requestedImage = true,
            requestedAudio = true,
        )

        assertFalse(decision.supportImage)
        assertFalse(decision.supportAudio)
        assertTrue(decision.policy, decision.policy.startsWith("text-only memory guard"))
        assertTrue(decision.policy, decision.policy.contains("8.0GB RAM recommended"))
    }

    @Test
    fun memorySafeModalityDecision_keepsGemma4E2bMultimodalOnEightGbDevice() {
        val decision = LiteRtLmOpenAiProxy.memorySafeModalityDecision(
            totalRamBytes = 8_000_000_000L,
            modelBytes = 2_583_085_056L,
            requestedImage = true,
            requestedAudio = true,
        )

        assertTrue(decision.supportImage)
        assertTrue(decision.supportAudio)
    }

    @Test
    fun memorySafeModalityDecision_requiresMoreRamForE4bMultimodal() {
        val decision = LiteRtLmOpenAiProxy.memorySafeModalityDecision(
            totalRamBytes = 10_000_000_000L,
            modelBytes = 3_654_467_584L,
            requestedImage = true,
            requestedAudio = true,
        )

        assertFalse(decision.supportImage)
        assertFalse(decision.supportAudio)
        assertTrue(decision.policy, decision.policy.contains("12.0GB RAM recommended"))
    }

    @Test
    fun speculativeDecodingDecision_autoEnablesCapabilityBackedGemma4OnArm64() {
        val decision = LiteRtLmOpenAiProxy.decideSpeculativeDecoding(
            capabilitiesSupported = true,
            modelName = "gemma-4-E2B-it.litertlm",
            modelBytes = 2_583_085_056L,
            totalRamBytes = 8_000_000_000L,
            isX86Device = false,
            mode = LiteRtLmOpenAiProxy.SpeculativeDecodingMode.AUTO,
        )

        assertTrue(decision.supported)
        assertTrue(decision.enabled)
        assertTrue(decision.policy, decision.policy.contains("capabilities advertise"))
    }

    @Test
    fun speculativeDecodingDecision_usesGemma4FilenameFallbackWhenCapabilitiesProbeFails() {
        val decision = LiteRtLmOpenAiProxy.decideSpeculativeDecoding(
            capabilitiesSupported = false,
            modelName = "gemma-4-E2B-it.litertlm",
            modelBytes = 2_583_085_056L,
            totalRamBytes = 8_000_000_000L,
            isX86Device = false,
            mode = LiteRtLmOpenAiProxy.SpeculativeDecodingMode.AUTO,
        )

        assertTrue(decision.supported)
        assertTrue(decision.enabled)
        assertTrue(decision.policy, decision.policy.contains("filename fallback"))
    }

    @Test
    fun speculativeDecodingDecision_keepsMtpOffOnX86Emulator() {
        val decision = LiteRtLmOpenAiProxy.decideSpeculativeDecoding(
            capabilitiesSupported = true,
            modelName = "gemma-4-E2B-it.litertlm",
            modelBytes = 2_583_085_056L,
            totalRamBytes = 16_000_000_000L,
            isX86Device = true,
            mode = LiteRtLmOpenAiProxy.SpeculativeDecodingMode.AUTO,
        )

        assertTrue(decision.supported)
        assertFalse(decision.enabled)
        assertEquals("disabled: x86 emulator/device build", decision.policy)
    }

    @Test
    fun speculativeDecodingDecision_runtimeDisabledOverridesSupportedModel() {
        val decision = LiteRtLmOpenAiProxy.decideSpeculativeDecoding(
            capabilitiesSupported = true,
            modelName = "gemma-4-E2B-it.litertlm",
            modelBytes = 2_583_085_056L,
            totalRamBytes = 16_000_000_000L,
            isX86Device = false,
            mode = LiteRtLmOpenAiProxy.SpeculativeDecodingMode.DISABLED,
        )

        assertTrue(decision.supported)
        assertFalse(decision.enabled)
        assertEquals("disabled: runtime setting disabled Gemma 4 MTP", decision.policy)
    }

    @Test
    fun speculativeDecodingDecision_rejectsUnsupportedNonGemmaModel() {
        val decision = LiteRtLmOpenAiProxy.decideSpeculativeDecoding(
            capabilitiesSupported = false,
            modelName = "qwen3-0.6b-it.litertlm",
            modelBytes = 800_000_000L,
            totalRamBytes = 16_000_000_000L,
            isX86Device = false,
            mode = LiteRtLmOpenAiProxy.SpeculativeDecodingMode.ENABLED,
        )

        assertFalse(decision.supported)
        assertFalse(decision.enabled)
        assertTrue(decision.policy, decision.policy.contains("does not advertise support"))
    }

    private fun validateModelArtifact(file: File): String? {
        val method = LiteRtLmOpenAiProxy::class.java.getDeclaredMethod(
            "validateModelArtifact",
            String::class.java,
        )
        method.isAccessible = true
        return method.invoke(LiteRtLmOpenAiProxy, file.absolutePath) as String?
    }

    private fun tempModelFile(name: String, header: ByteArray): File {
        val dir = File(createTempDirectory(prefix = "hermes-litertlm-test-").pathString)
        return File(dir, name).apply {
            writeBytes(header + ByteArray(16) { 1 })
            deleteOnExit()
            dir.deleteOnExit()
        }
    }
}
