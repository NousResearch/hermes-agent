"""Megolm ciphertext padding tolerance in the Matrix E2EE shim (#138108).

libolm only accepts Matrix's unpadded base64, so clients that emit padded
Megolm ciphertext (eM Client) fail decryption with BAD_SIGNATURE. The wrapper
normalizes the ciphertext before mautrix hands it to libolm; signature, MAC
and replay checks still run inside libolm on the decoded bytes."""
import base64

import pytest

from plugins.platforms.matrix.adapter_crypto import with_megolm_padding_tolerance


class _RecordingDecryptor:
    """Stands in for mautrix's OlmMachine.decrypt_megolm_event."""

    def __init__(self):
        self.seen = None

    async def decrypt_megolm_event(self, evt, *args, **kwargs):
        self.seen = getattr(getattr(evt, "content", None), "ciphertext", None)
        return "decrypted"


class _Content:
    def __init__(self, ciphertext):
        self.ciphertext = ciphertext


class _Event:
    def __init__(self, ciphertext):
        self.content = _Content(ciphertext)


def _machine():
    return with_megolm_padding_tolerance(_RecordingDecryptor())


class TestMegolmPaddingTolerance:

    @pytest.mark.asyncio
    async def test_padded_ciphertext_is_normalized_before_decrypt(self):
        machine = _machine()
        evt = _Event("AbCdEf==")
        assert await machine.decrypt_megolm_event(evt, "extra", kw="arg") == "decrypted"
        assert machine.seen == "AbCdEf"
        assert evt.content.ciphertext == "AbCdEf"

    @pytest.mark.asyncio
    async def test_unpadded_ciphertext_passes_through_unchanged(self):
        machine = _machine()
        evt = _Event("AbCdEf")
        await machine.decrypt_megolm_event(evt)
        assert machine.seen == "AbCdEf"
        assert evt.content.ciphertext == "AbCdEf"

    @pytest.mark.asyncio
    async def test_non_string_ciphertext_is_left_alone(self):
        machine = _machine()
        evt = _Event(None)
        await machine.decrypt_megolm_event(evt)
        assert machine.seen is None

    def test_wrapper_returns_the_same_machine(self):
        inner = _RecordingDecryptor()
        machine = with_megolm_padding_tolerance(inner)
        assert machine is inner


def test_padded_ciphertext_decrypts_in_libolm_after_normalization():
    """End-to-end against real libolm: padded input fails, normalized input decrypts.

    Replicates the standalone reproducer from #138108 in memory — no homeserver
    or user keys needed. Skipped where python-olm is not installed (Linux-only)."""
    olm = pytest.importorskip("olm")
    pad_lengths = set()
    for size in range(1, 13):
        payload = "x" * size
        out = olm.OutboundGroupSession()
        original = out.encrypt(payload)
        packet = base64.b64decode(original + "=" * (-len(original) % 4))
        padded = base64.b64encode(packet).decode()
        assert padded.rstrip("=") == original
        pad_lengths.add(len(padded) - len(original))
        with pytest.raises(olm.OlmGroupSessionError):
            olm.InboundGroupSession(out.session_key).decrypt(padded)
        text, _ = olm.InboundGroupSession(out.session_key).decrypt(padded.rstrip("="))
        assert text == payload
    assert pad_lengths == {1, 2}, "fixture must cover both base64 padding lengths"
