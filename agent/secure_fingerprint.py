"""Stable PBKDF2 fingerprints for non-verifier identity and cache boundaries.

These values are identifiers, not password verifiers: callers use them to keep
credentials from appearing in cache keys, telemetry, or sanitized metadata.
These values must never be used as authentication material or password
verifiers.
"""

from __future__ import annotations

import hashlib
from typing import Any


_FINGERPRINT_SALT = b"hermes-non-verifier-fingerprint-v3"
_FINGERPRINT_ITERATIONS = 100_000


def stable_fingerprint(value: Any, *, length: int = 16) -> str:
    """Return a stable PBKDF2 fingerprint truncated to *length* hex chars."""
    text = "" if value is None else str(value)
    if not text:
        return ""
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        text.encode("utf-8", errors="surrogatepass"),
        _FINGERPRINT_SALT,
        _FINGERPRINT_ITERATIONS,
        dklen=32,
    ).hex()
    return digest[:length]
