"""Coverage for QQBot scan-to-configure credential decryption."""

from __future__ import annotations

import base64
import os

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.platforms.qqbot.crypto import decrypt_secret, generate_bind_key


def _encrypt_secret(plaintext: str, key: bytes) -> str:
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def test_generate_bind_key_returns_base64_encoded_32_bytes():
    key = generate_bind_key()

    assert base64.b64decode(key)
    assert len(base64.b64decode(key)) == 32


def test_decrypt_secret_round_trips_utf8():
    key = os.urandom(32)
    plaintext = "secret-\u6d4b\u8bd5"

    assert (
        decrypt_secret(
            _encrypt_secret(plaintext, key), base64.b64encode(key).decode("ascii")
        )
        == plaintext
    )


def test_decrypt_secret_rejects_wrong_key():
    encrypted = _encrypt_secret("client-secret", os.urandom(32))
    wrong_key = base64.b64encode(os.urandom(32)).decode("ascii")

    with pytest.raises(InvalidTag):
        decrypt_secret(encrypted, wrong_key)


def test_decrypt_secret_rejects_tampered_ciphertext():
    key = os.urandom(32)
    raw = bytearray(base64.b64decode(_encrypt_secret("client-secret", key)))
    raw[-1] ^= 0x01

    with pytest.raises(InvalidTag):
        decrypt_secret(
            base64.b64encode(raw).decode("ascii"), base64.b64encode(key).decode("ascii")
        )
