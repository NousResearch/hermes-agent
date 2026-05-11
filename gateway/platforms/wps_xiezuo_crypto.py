"""
KSO-1 handshake + HMAC-SHA256 signature + AES-256-CBC encryption for WPS Xiezuo.

Ported from the JS implementation in event-crypto.js.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as crypto_padding
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

@dataclass
class WpsFrame:
    """Parsed WPS Xiezuo WebSocket frame."""
    opcode: int
    data: Any


def parse_frame(raw: dict) -> WpsFrame:
    """Parse a raw WebSocket frame dict into a WpsFrame.

    Raises ValueError if required fields are missing.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Frame must be a dict, got {type(raw)}")
    if "opcode" not in raw or "payload" not in raw:
        raise ValueError("Frame missing 'opcode' or 'payload' field")

    opcode = raw["opcode"]
    payload = raw["payload"]

    # Try to parse payload as JSON
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload
    else:
        data = payload

    return WpsFrame(opcode=opcode, data=data)


# ---------------------------------------------------------------------------
# KSO-1 Handshake
# ---------------------------------------------------------------------------

def build_handshake(app_id: str, app_secret: str, nonce: str) -> dict:
    """Build a KSO-1 handshake frame.

    Returns a dict with:
      - opcode: 1
      - payload: JSON string with app_id, signature, nonce, timestamp
    """
    timestamp = int(time.time())
    signature = compute_signature(
        app_id=app_id,
        app_secret=app_secret,
        topic="",
        nonce=nonce,
        timestamp=timestamp,
        encrypted_data="",
    )

    payload = {
        "app_id": app_id,
        "signature": signature,
        "nonce": nonce,
        "timestamp": timestamp,
    }

    return {
        "opcode": 1,
        "payload": json.dumps(payload),
    }


# ---------------------------------------------------------------------------
# HMAC-SHA256 Signature
# ---------------------------------------------------------------------------

def compute_signature(
    app_id: str,
    app_secret: str,
    topic: str,
    nonce: str,
    timestamp: int,
    encrypted_data: str,
) -> str:
    """Compute HMAC-SHA256 signature for WPS event verification.

    The signed content is: "app_id:topic:nonce:timestamp:encrypted_data"
    The signature is base64url-encoded with trailing '=' stripped.
    """
    content = f"{app_id}:{topic}:{nonce}:{timestamp}:{encrypted_data}"
    mac = hmac.new(
        app_secret.encode("utf-8"),
        content.encode("utf-8"),
        hashlib.sha256,
    )
    return base64.urlsafe_b64encode(mac.digest()).decode("utf-8").rstrip("=")


def verify_signature(
    signature: str,
    app_id: str,
    app_secret: str,
    topic: str,
    nonce: str,
    timestamp: int,
    encrypted_data: str,
) -> bool:
    """Verify a WPS event signature.

    Returns True if the computed signature matches the provided one.
    """
    expected = compute_signature(
        app_id=app_id,
        app_secret=app_secret,
        topic=topic,
        nonce=nonce,
        timestamp=timestamp,
        encrypted_data=encrypted_data,
    )
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# AES-256-CBC Encryption / Decryption
# ---------------------------------------------------------------------------

def _derive_key(app_secret: str) -> bytes:
    """Derive the AES-256 key from the app secret.

    key = MD5(app_secret) as hex string, interpreted as UTF-8 bytes (32 bytes).
    """
    md5_hex = hashlib.md5(app_secret.encode("utf-8")).hexdigest()
    return md5_hex.encode("utf-8")  # 32 bytes for AES-256


def _derive_iv(nonce: str) -> bytes:
    """Derive the AES-CBC IV from the nonce.

    IV = first 16 bytes of the nonce string, UTF-8 encoded.
    """
    return nonce.encode("utf-8")[:16]


def decrypt_event(encrypted_data: str, app_secret: str, nonce: str) -> str:
    """Decrypt an encrypted WPS event body.

    Args:
        encrypted_data: Base64-encoded ciphertext.
        app_secret: The app secret for key derivation.
        nonce: The nonce used for IV derivation.

    Returns:
        Decrypted plaintext JSON string.

    Raises:
        ValueError: If decryption fails (bad padding, wrong key, etc.).
    """
    key = _derive_key(app_secret)
    iv = _derive_iv(nonce)
    ciphertext = base64.b64decode(encrypted_data)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove PKCS7 padding
    unpadder = crypto_padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()

    return plaintext.decode("utf-8")


def encrypt_event(plaintext: str, app_secret: str, nonce: str) -> str:
    """Encrypt a WPS event body (for round-trip testing).

    Args:
        plaintext: JSON string to encrypt.
        app_secret: The app secret for key derivation.
        nonce: The nonce used for IV derivation.

    Returns:
        Base64-encoded ciphertext.
    """
    key = _derive_key(app_secret)
    iv = _derive_iv(nonce)

    # Apply PKCS7 padding
    padder = crypto_padding.PKCS7(128).padder()
    padded = padder.update(plaintext.encode("utf-8")) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    return base64.b64encode(ciphertext).decode("utf-8")
