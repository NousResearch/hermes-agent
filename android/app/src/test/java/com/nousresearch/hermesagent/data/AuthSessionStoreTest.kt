package com.nousresearch.hermesagent.data

import android.net.Uri
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AuthSessionStoreTest {
    @Test
    fun evaluateAuthCallback_rejectsMissingPendingRequest() {
        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=google&state=expected&email=user@example.com"),
            pending = null,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertFalse(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: no pending sign-in request", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_rejectsExpiredPendingRequest() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "google",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 1L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=google&state=expected&email=user@example.com"),
            pending = pending,
            nowEpochMs = (16 * 60 * 1000L),
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback expired. Start sign-in again.", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_rejectsMethodMismatch() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "google",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=email&state=expected&email=user@example.com"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertEquals("google", result.session?.methodId)
        assertEquals("Auth callback rejected: method mismatch", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_rejectsProviderMismatchForRuntimeProvider() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "chatgpt",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=chatgpt&state=expected&provider=anthropic&access_token=token"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: provider mismatch", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_requiresProviderCredentials() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "chatgpt",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=chatgpt&state=expected&provider=chatgpt-web"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: no provider credentials were returned", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_requiresAppIdentity() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "google",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse("${AuthSessionStore.CALLBACK_URI}?method=google&state=expected"),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertFalse(result.session?.signedIn ?: true)
        assertEquals("Auth callback rejected: no account identity returned", result.session?.status)
    }

    @Test
    fun evaluateAuthCallback_acceptsRuntimeProviderCredentials() {
        val pending = PendingAuthRequest(
            state = "expected",
            methodId = "chatgpt",
            startUrl = "https://auth.corr3xt.com/oauth/start",
            createdAtEpochMs = 10L,
        )

        val result = AuthSessionStore.evaluateAuthCallback(
            Uri.parse(
                "${AuthSessionStore.CALLBACK_URI}?method=chatgpt&state=expected&provider=chatgpt-web&access_token=access&session_token=session&base_url=https%3A%2F%2Fchatgpt.com%2Fbackend-api%2Ff%3Fignored%3D1&model=gpt-5-thinking"
            ),
            pending = pending,
            nowEpochMs = 100L,
        )

        assertTrue(result.consumed)
        assertTrue(result.clearPending)
        assertTrue(result.session?.signedIn ?: false)
        assertEquals("chatgpt-web", result.session?.runtimeProvider)
        assertEquals("https://chatgpt.com/backend-api/f", result.session?.baseUrl)
        assertEquals("gpt-5-thinking", result.session?.model)
        assertEquals("Signed in with ChatGPT", result.session?.status)
    }
}
