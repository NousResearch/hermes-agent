# -*- coding: utf-8 -*-
"""AES-256-GCM utilities for QQBot scan-to-configure credential decryption."""

from __future__ import annotations

import base64
import os


def generate_bind_key() -> str:
    """Generate a 256-bit random AES key encoded as base64.

    The key is passed to ``create_bind_task`` so the server can encrypt
    the bot credentials before returning them.  Only this client holds
    the key, ensuring the secret never travels in plaintext.

    :returns: Base64-encoded 32-byte random key.
    """
    return base64.b64encode(os.urandom(32)).decode()


def decrypt_secret(encrypted_base64: str, key_base64: str) -> str:
    """Decrypt a base64-encoded AES-256-GCM ciphertext.

    Ciphertext layout (after base64-decoding)::

        IV (12 bytes) | ciphertext (N bytes) | AuthTag (16 bytes)

    :param encrypted_base64: The ``bot_encrypt_secret`` value from
        ``poll_bind_result``.
    :param key_base64: The base64 AES key from :func:`generate_bind_key`.
    :returns: Decrypted *client_secret* as a UTF-8 string.
    :raises ValueError: If decryption fails (wrong key or tampered data).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = base64.b64decode(key_base64)
    raw = base64.b64decode(encrypted_base64)

    # AESGCM expects ciphertext + tag concatenated after the IV
    iv = raw[:12]
    ciphertext_with_tag = raw[12:]

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, None)
    return plaintext.decode("utf-8")
