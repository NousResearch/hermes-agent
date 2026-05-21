package com.mobilefork.hermesagent.ui.boot

import org.junit.Assert.assertEquals
import org.junit.Test

class BootViewModelTest {
    @Test
    fun hermesHealthUrl_handlesOpenAiBaseUrl() {
        assertEquals(
            "http://127.0.0.1:15436/health",
            hermesHealthUrl("http://127.0.0.1:15436/v1"),
        )
    }

    @Test
    fun hermesHealthUrl_keepsRootBaseUrl() {
        assertEquals(
            "http://127.0.0.1:15436/health",
            hermesHealthUrl("http://127.0.0.1:15436/"),
        )
    }
}
