"""Unit tests for the AES-256-GCM envelope and encrypted-file detection."""

from __future__ import annotations

import os

import pytest

from hermes_crypto import detect, envelope
from hermes_crypto.errors import DecryptionError


def _dek() -> bytes:
    return os.urandom(32)


def test_binary_round_trip():
    dek = _dek()
    plaintext = b'{"api_key": "sk-secret-value"}'
    blob = envelope.encrypt(plaintext, dek)
    assert blob.startswith(b"HRMSENC")
    assert envelope.decrypt(blob, dek) == plaintext


def test_env_round_trip():
    dek = _dek()
    plaintext = b"OPENAI_API_KEY=sk-abc\nFOO=bar\n"
    blob = envelope.encrypt_env(plaintext, dek)
    assert blob.startswith(b"#HERMES-ENCRYPTED-V1")
    assert envelope.decrypt_env(blob, dek) == plaintext


def test_is_encrypted_true_for_both_forms():
    dek = _dek()
    assert detect.is_encrypted(envelope.encrypt(b"x", dek))
    assert detect.is_encrypted(envelope.encrypt_env(b"x=1\n", dek))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"OPENAI_API_KEY=sk-plaintext\n",
        b'{"providers": {}}',
        b"security:\n  encryption:\n    enabled: false\n",
        b"# a comment\n",
    ],
)
def test_is_encrypted_false_for_plaintext(data):
    assert not detect.is_encrypted(data)


def test_empty_plaintext_round_trips():
    dek = _dek()
    assert envelope.decrypt(envelope.encrypt(b"", dek), dek) == b""


def test_nonce_is_unique_per_encryption():
    dek = _dek()
    a = envelope.encrypt(b"same plaintext", dek)
    b = envelope.encrypt(b"same plaintext", dek)
    assert a != b  # fresh random nonce each time


def test_wrong_key_fails():
    blob = envelope.encrypt(b"secret", _dek())
    with pytest.raises(DecryptionError):
        envelope.decrypt(blob, _dek())


def test_tampered_ciphertext_fails():
    dek = _dek()
    blob = bytearray(envelope.encrypt(b"secret payload", dek))
    blob[-1] ^= 0x01  # flip a bit in the GCM tag
    with pytest.raises(DecryptionError):
        envelope.decrypt(bytes(blob), dek)


def test_tampered_header_fails():
    dek = _dek()
    blob = bytearray(envelope.encrypt(b"secret", dek))
    blob[7] = 0x99  # corrupt the version byte (it is part of the AAD)
    with pytest.raises(DecryptionError):
        envelope.decrypt(bytes(blob), dek)


def test_truncated_envelope_fails():
    dek = _dek()
    blob = envelope.encrypt(b"secret", dek)
    with pytest.raises(DecryptionError):
        envelope.decrypt(blob[:10], dek)


@pytest.mark.parametrize("length", [7, 8, 9, 19, 20])
def test_magic_only_and_header_only_envelopes_fail_cleanly(length):
    # A blob cut anywhere inside the header/nonce must raise DecryptionError,
    # never IndexError — a 7-byte magic-only file previously hit the
    # version-byte read out of bounds.
    dek = _dek()
    blob = envelope.encrypt(b"secret", dek)
    with pytest.raises(DecryptionError):
        envelope.decrypt(blob[:length], dek)


def test_env_framed_magic_only_envelope_fails_cleanly():
    import base64

    body = base64.b64encode(b"HRMSENC").decode("ascii")
    framed = f"{detect.ENV_MARKER}\n{body}\n".encode("ascii")
    with pytest.raises(DecryptionError):
        envelope.decrypt_env(framed, _dek())


def test_decrypt_rejects_non_envelope():
    with pytest.raises(DecryptionError):
        envelope.decrypt(b"just some plaintext bytes", _dek())


def test_encrypt_rejects_bad_key_length():
    with pytest.raises(ValueError):
        envelope.encrypt(b"data", b"too-short")
