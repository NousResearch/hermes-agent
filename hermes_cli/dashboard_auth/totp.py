"""RFC 6238 TOTP (stdlib only) for the dashboard's second factor.

The defaults (SHA-1, 6 digits, 30 s step, base32 secret) are the only profile every
authenticator app supports (Microsoft Authenticator, Google Authenticator, Authy, 1Password,
…), so they are fixed rather than configurable. ``verify_totp`` accepts the current step plus
one step either side (clock skew) and reports WHICH step matched so the caller can refuse a
replayed code (a TOTP is single-use within its window).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from typing import Optional
from urllib.parse import quote, urlencode

TOTP_DIGITS = 6
TOTP_STEP_SECONDS = 30
TOTP_SKEW_STEPS = 1
_SECRET_BYTES = 20  # 160-bit, the RFC 4226 recommendation; 32 base32 chars


def generate_totp_secret() -> str:
    """Fresh base32 secret (no padding) — the form authenticator apps take as a manual key."""
    return base64.b32encode(secrets.token_bytes(_SECRET_BYTES)).decode().rstrip("=")


def decode_totp_secret(secret: str) -> bytes:
    """Decode a base32 secret as an operator would paste it (any case, spaces, no padding).
    Raises ``ValueError`` when it is not base32."""
    cleaned = "".join(secret.split()).upper().rstrip("=")
    if not cleaned:
        raise ValueError("TOTP secret is empty")
    padded = cleaned + "=" * (-len(cleaned) % 8)
    try:
        return base64.b32decode(padded, casefold=True)
    except Exception as exc:
        raise ValueError("TOTP secret is not valid base32") from exc


def totp_code(secret: bytes, *, counter: int) -> str:
    """The zero-padded code for one time step (RFC 4226 HOTP with the RFC 6238 counter)."""
    mac = hmac.new(secret, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    binary = struct.unpack(">I", mac[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(binary % (10 ** TOTP_DIGITS)).zfill(TOTP_DIGITS)


def totp_counter(now: Optional[float] = None) -> int:
    return int((time.time() if now is None else now) // TOTP_STEP_SECONDS)


def normalize_totp_input(code: str) -> str:
    """What a user types: digits only, ignoring spaces (``123 456``)."""
    return "".join(ch for ch in code if ch.isdigit())


def verify_totp(secret: bytes, code: str, *, now: Optional[float] = None,
                skew: int = TOTP_SKEW_STEPS) -> Optional[int]:
    """Return the time-step counter the code is valid for, or ``None``. Constant-time compare
    on every candidate (no early return) so a near-miss is indistinguishable from a miss."""
    digits = normalize_totp_input(code)
    if len(digits) != TOTP_DIGITS:
        return None
    center = totp_counter(now)
    matched: Optional[int] = None
    for counter in range(center - skew, center + skew + 1):
        if hmac.compare_digest(totp_code(secret, counter=counter).encode(), digits.encode()):
            matched = counter
    return matched


def totp_provisioning_uri(secret: str, *, account: str, issuer: str = "Hermes Dashboard") -> str:
    """``otpauth://`` URI (the QR payload). ``issuer`` is in both the label and the query, which is
    what the Key Uri Format expects and what Microsoft Authenticator uses for the account title."""
    label = quote(f"{issuer}:{account}", safe="")
    query = urlencode({
        "secret": "".join(secret.split()).upper().rstrip("="), "issuer": issuer,
        "algorithm": "SHA1", "digits": TOTP_DIGITS, "period": TOTP_STEP_SECONDS})
    return f"otpauth://totp/{label}?{query}"
