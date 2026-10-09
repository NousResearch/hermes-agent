"""AES-256-GCM utilities for QQBot scan-to-configure credential decryption."""

from __future__ import annotations

import base64
import os


def generate_bind_key() -> str:
    """Random 256-bit AES key (base64) passed to ``create_bind_task`` so the server
    encrypts the bot's *client_secret*; only this CLI holds the key."""
    return base64.b64encode(os.urandom(32)).decode()


def decrypt_secret(encrypted_base64: str, key_base64: str) -> str:
    """Decrypt ``bot_encrypt_secret`` (base64 of ``IV(12) ‖ ciphertext ‖ tag(16)``) to a UTF-8 string."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = base64.b64decode(encrypted_base64)  # AESGCM expects ciphertext + tag concatenated
    return AESGCM(base64.b64decode(key_base64)).decrypt(raw[:12], raw[12:], None).decode("utf-8")
