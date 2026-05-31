package com.mobilefork.hermesagent.ui.shell

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AppShellNavigationDrawerTest {
    @Test
    fun shellDrawerKeepsAllDestinationsReachable() {
        val sections = shellDrawerNavigationSections()

        assertEquals(AppSection.Hermes, sections.first())
        assertTrue(sections.contains(AppSection.Accounts))
        assertTrue(sections.contains(AppSection.NousPortal))
        assertTrue(sections.contains(AppSection.Device))
        assertTrue(sections.contains(AppSection.Settings))
        assertEquals(sections, sections.distinct())
    }
}
