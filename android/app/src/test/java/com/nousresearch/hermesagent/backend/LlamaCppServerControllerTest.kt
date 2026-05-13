package com.nousresearch.hermesagent.backend

import org.junit.Assert.assertTrue
import org.junit.Test

class LlamaCppServerControllerTest {
    @Test
    fun launchOptionsUseCompactContextForTinyQwenGguf() {
        val options = LlamaCppServerController.launchOptionsForModel(
            modelPath = "/models/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
            availableProcessors = 8,
        )

        assertTrue(options, "--ctx-size 1024" in options)
        assertTrue(options, "--threads 4" in options)
        assertTrue(options, "--batch-size 64" in options)
        assertTrue(options, "--ubatch-size 64" in options)
        assertTrue(options, "--no-warmup" in options)
    }

    @Test
    fun launchOptionsUseMobileDefaultForLargerGguf() {
        val options = LlamaCppServerController.launchOptionsForModel(
            modelPath = "/models/model-4b-q4_k_m.gguf",
            availableProcessors = 2,
        )

        assertTrue(options, "--ctx-size 2048" in options)
        assertTrue(options, "--threads 2" in options)
        assertTrue(options, "--parallel 1" in options)
    }
}
