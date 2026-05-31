package com.mobilefork.hermesagent.ui.boot

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BootViewModelTest {
    @Test
    fun bootUiState_defaultsToOpeningShellWithoutProbeState() {
        val state = BootUiState()

        assertEquals("Opening Hermes…", state.status)
        assertFalse(state.ready)
        assertEquals("", state.probeResult)
        assertEquals("", state.baseUrl)
        assertEquals("", state.error)
    }

    @Test
    fun bootUiState_readyStateDoesNotRequireHealthProbe() {
        val state = BootUiState(status = "Hermes shell ready", ready = true)

        assertEquals("Hermes shell ready", state.status)
        assertTrue(state.ready)
        assertEquals("", state.probeResult)
        assertEquals("", state.baseUrl)
        assertEquals("", state.error)
    }
}
