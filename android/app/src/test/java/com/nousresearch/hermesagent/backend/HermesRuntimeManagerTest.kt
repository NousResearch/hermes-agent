package com.nousresearch.hermesagent.backend

import org.junit.Assert.assertFalse
import org.junit.Test

class HermesRuntimeManagerTest {
    @Test
    fun currentState_defaultsToNotStarted() {
        assertFalse(HermesRuntimeManager.currentState().started)
    }
}
