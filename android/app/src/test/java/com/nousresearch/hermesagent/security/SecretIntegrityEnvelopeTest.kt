package com.nousresearch.hermesagent.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class SecretIntegrityEnvelopeTest {
    @Test
    fun open_roundTripsSealedSecret() {
        val sealed = SecretIntegrityEnvelope.seal(
            value = "session-token",
            associatedData = "auth_session:chatgpt",
            keyBytes = testKey(),
        )

        val opened = SecretIntegrityEnvelope.open(
            storedValue = sealed,
            associatedData = "auth_session:chatgpt",
            keyBytes = testKey(),
        )

        assertEquals(SecretIntegrityEnvelope.Opened("session-token"), opened)
    }

    @Test
    fun open_returnsLegacyForUnsealedValues() {
        val opened = SecretIntegrityEnvelope.open(
            storedValue = "legacy-api-key",
            associatedData = "api_key:openrouter",
            keyBytes = testKey(),
        )

        assertEquals(SecretIntegrityEnvelope.Legacy("legacy-api-key"), opened)
    }

    @Test
    fun open_rejectsAssociatedDataSwap() {
        val sealed = SecretIntegrityEnvelope.seal(
            value = "access-token",
            associatedData = "auth_session:chatgpt",
            keyBytes = testKey(),
        )

        val opened = SecretIntegrityEnvelope.open(
            storedValue = sealed,
            associatedData = "auth_session:openrouter",
            keyBytes = testKey(),
        )

        assertTrue(opened is SecretIntegrityEnvelope.Invalid)
    }

    @Test
    fun open_rejectsPayloadTampering() {
        val sealed = SecretIntegrityEnvelope.seal(
            value = "access-token",
            associatedData = "auth_session:chatgpt",
            keyBytes = testKey(),
        )
        val tampered = sealed.dropLast(2) + "AA"

        val opened = SecretIntegrityEnvelope.open(
            storedValue = tampered,
            associatedData = "auth_session:chatgpt",
            keyBytes = testKey(),
        )

        assertTrue(opened is SecretIntegrityEnvelope.Invalid)
    }

    @Test
    fun open_rejectsWrongIntegrityKey() {
        val sealed = SecretIntegrityEnvelope.seal(
            value = "access-token",
            associatedData = "auth_session:chatgpt",
            keyBytes = testKey(),
        )

        val opened = SecretIntegrityEnvelope.open(
            storedValue = sealed,
            associatedData = "auth_session:chatgpt",
            keyBytes = otherTestKey(),
        )

        assertTrue(opened is SecretIntegrityEnvelope.Invalid)
    }

    private fun testKey(): ByteArray = ByteArray(SecretIntegrityEnvelope.MIN_KEY_BYTES) { index ->
        (index + 1).toByte()
    }

    private fun otherTestKey(): ByteArray = ByteArray(SecretIntegrityEnvelope.MIN_KEY_BYTES) { index ->
        (index + 11).toByte()
    }
}
