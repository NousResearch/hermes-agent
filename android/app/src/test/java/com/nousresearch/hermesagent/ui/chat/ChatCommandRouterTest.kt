package com.nousresearch.hermesagent.ui.chat

import com.nousresearch.hermesagent.ui.shell.AppSection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatCommandRouterTest {
    @Test
    fun signinRuntimeProvidersPrepareApiKeySetupInSettings() {
        val host = RecordingCommandHost()

        val result = ChatCommandRouter.execute("/signin dashscope", host.asHost())

        assertTrue(result.handled)
        assertEquals("qwen", host.startedAuthMethods.single())
        assertEquals(AppSection.Settings, host.sections.single())
        assertEquals(
            "Prepared qwen API-key setup in Settings. Paste the provider key there to power Hermes.",
            result.feedback,
        )
    }

    @Test
    fun signinZaiAliasesPrepareApiKeySetupInSettings() {
        val host = RecordingCommandHost()

        val result = ChatCommandRouter.execute("/signin glm", host.asHost())

        assertTrue(result.handled)
        assertEquals("zai", host.startedAuthMethods.single())
        assertEquals(AppSection.Settings, host.sections.single())
        assertEquals(
            "Prepared zai API-key setup in Settings. Paste the provider key there to power Hermes.",
            result.feedback,
        )
    }

    @Test
    fun signinAppAccountsOpenAccountsInsteadOfSettings() {
        val host = RecordingCommandHost()

        val result = ChatCommandRouter.execute("/signin google", host.asHost())

        assertTrue(result.handled)
        assertEquals("google", host.startedAuthMethods.single())
        assertEquals(AppSection.Accounts, host.sections.single())
        assertEquals(
            "Opened Corr3xt app sign-in for google. Complete it in your browser, then come back to Hermes.",
            result.feedback,
        )
    }

    @Test
    fun signinRejectedAuthDoesNotNavigate() {
        val host = RecordingCommandHost(startAuthResult = false)

        val result = ChatCommandRouter.execute("/signin qwen", host.asHost())

        assertTrue(result.handled)
        assertEquals("qwen", host.startedAuthMethods.single())
        assertTrue(host.sections.isEmpty())
        assertEquals(
            "Could not start sign-in for 'qwen'. Configure a reachable Corr3xt URL in Accounts, or use provider API keys in Settings.",
            result.feedback,
        )
    }

    @Test
    fun nonCommandInputFallsThrough() {
        val result = ChatCommandRouter.execute("hello", RecordingCommandHost().asHost())

        assertEquals(false, result.handled)
        assertNull(result.feedback)
    }

    private class RecordingCommandHost(
        private val startAuthResult: Boolean = true,
    ) {
        val sections = mutableListOf<AppSection>()
        val startedAuthMethods = mutableListOf<String>()

        fun asHost(): ChatCommandHost = ChatCommandHost(
            openHistory = {},
            newConversation = {},
            clearConversation = {},
            navigateToSection = { sections += it },
            applyProvider = { true },
            applyModel = { true },
            startAuthMethod = {
                startedAuthMethods += it
                startAuthResult
            },
            speakLastReply = { true },
        )
    }
}
