package com.mobilefork.hermesagent.ui.chat

import com.mobilefork.hermesagent.ui.shell.AppSection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatScreenNavigationDrawerTest {
    @Test
    fun chatDrawerKeepsShellDestinationsReachable() {
        val sections = chatDrawerNavigationSections()

        assertEquals(AppSection.Hermes, sections.first())
        assertTrue(sections.contains(AppSection.Accounts))
        assertTrue(sections.contains(AppSection.NousPortal))
        assertTrue(sections.contains(AppSection.Device))
        assertTrue(sections.contains(AppSection.Settings))
        assertEquals(sections, sections.distinct())
    }
}
