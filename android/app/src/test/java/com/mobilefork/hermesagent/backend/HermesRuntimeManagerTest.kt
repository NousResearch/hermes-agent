package com.mobilefork.hermesagent.backend

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class HermesRuntimeManagerTest {
    @Test
    fun currentState_defaultsToNotStarted() {
        assertFalse(HermesRuntimeManager.currentState().started)
    }

    @Test
    fun localBackendFallbackWarning_isBlankForRemoteMode() {
        val warning = HermesRuntimeManager.localBackendFallbackWarning(
            selectedLocalBackend = BackendKind.NONE,
            localBackendStatus = LocalBackendStatus(
                backendKind = BackendKind.NONE,
                started = false,
                statusMessage = "Remote provider mode",
            ),
        )

        assertNull(warning)
    }

    @Test
    fun localBackendFallbackWarning_preservesReasonForMissingModel() {
        val warning = HermesRuntimeManager.localBackendFallbackWarning(
            selectedLocalBackend = BackendKind.LLAMA_CPP,
            localBackendStatus = LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                statusMessage = "No preferred local model is ready for llama.cpp yet",
            ),
        )

        assertEquals(
            "Local llama.cpp backend unavailable: No preferred local model is ready for llama.cpp yet. " +
                "Using saved remote provider.",
            warning,
        )
    }

    @Test
    fun withLocalBackendWarning_appendsWarningToProbeText() {
        val warning = "Local litert-lm backend unavailable: model missing. Using saved remote provider."

        assertEquals(
            "python-ok\n$warning",
            with(HermesRuntimeManager) {
                "python-ok".withLocalBackendWarning(warning)
            },
        )
    }
}
