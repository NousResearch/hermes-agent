"""Gateway-side relay authentication primitives. EXPERIMENTAL.

WS upgrade auth (gateway -> connector): ``Authorization: Bearer <token>`` on the
``/relay`` upgrade, ``token = make_upgrade_token(gateway_id, secret)``. Wire bytes
must match the connector's ``relayAuthToken.ts`` ``makeToken`` exactly:
``base64url(f"{payload}:{exp}:{sig}")`` with ``sig = HMAC_SHA256(f"{payload}:{exp}",
secret).hexdigest()``. The connector verifies against a multi-secret rotation list.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

_DEFAULT_UPGRADE_TTL_SECONDS = 300  # connector makeUpgradeToken default


def sign(payload: str, secret: str) -> str:
    """HMAC-SHA256 hex digest (UTF-8) — the connector's ``sign``."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def make_token(payload: str, secret: str, ttl_seconds: int = 0) -> str:
    """``base64url(f"{payload}:{exp}:{sig}")``; ``exp`` unix seconds (0 = never).

    base64url is unpadded to match Node's ``Buffer.toString("base64url")``.
    """
    exp = int(time.time()) + ttl_seconds if ttl_seconds > 0 else 0
    signed = f"{payload}:{exp}"
    raw = f"{signed}:{sign(signed, secret)}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_upgrade_token(
    gateway_id: str, secret: str, ttl_seconds: int = _DEFAULT_UPGRADE_TTL_SECONDS
) -> str:
    """WS-upgrade bearer: ``payload = gateway_id`` (the connector peeks it to index its verify list)."""
    return make_token(gateway_id, secret, ttl_seconds)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
from typing import Sequence  # noqa: F401,E402

DELIVERY_SIG_HEADER = "x-relay-signature"

DELIVERY_TS_HEADER = "x-relay-timestamp"

_DEFAULT_MAX_SKEW_SECONDS = 300

def _hmac_hex(payload: str, secret: str) -> str:
    """HMAC-SHA256 hex digest of ``payload`` under ``secret`` (UTF-8)."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

def verify_signature(payload: str, sig_hex: str, secrets: Sequence[str]) -> bool:
    """Constant-time check that ``sig_hex`` is a valid HMAC of ``payload`` under
    ANY of ``secrets`` (rotation window). Length-mismatched candidates are
    skipped without a timing leak. Mirrors ``verifySignature``.
    """
    try:
        sig_buf = bytes.fromhex(sig_hex)
    except (ValueError, TypeError):
        return False
    if len(sig_buf) == 0:
        return False
    for secret in secrets:
        if not secret:
            continue
        expected = bytes.fromhex(_hmac_hex(payload, secret))
        if len(expected) != len(sig_buf):
            continue
        if hmac.compare_digest(sig_buf, expected):
            return True
    return False

def _delivery_payload(ts: int, body_json: str) -> str:
    """Signed material for an inbound delivery: ``f"{ts}.{body_json}"``."""
    return f"{ts}.{body_json}"

def verify_delivery_signature(
    body_json: str,
    timestamp: Optional[str],
    signature: Optional[str],
    verify_keys: Sequence[str],
    max_skew_seconds: int = _DEFAULT_MAX_SKEW_SECONDS,
    *,
    now: Optional[int] = None,
) -> bool:
    """Verify a connector→gateway inbound delivery signature.

    ``body_json`` MUST be the exact request body bytes decoded as UTF-8 — the
    connector signs over the literal serialized body, so the gateway verifies
    over the literal received body (no re-serialization). Checks the timestamp
    is within ``max_skew_seconds`` of now and the HMAC matches any key in the
    rotation verify list. Mirrors the connector's ``verifyDeliverySignature``.
    """
    if not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False
    current = now if now is not None else int(time.time())
    if abs(current - ts) > max_skew_seconds:
        return False
    return verify_signature(_delivery_payload(ts, body_json), signature, verify_keys)

def verify_token(token: str, secrets: Sequence[str]) -> Optional[str]:
    """Verify a token built by ``make_token``; return the payload or None.

    Splits from the right so a payload may itself contain colons (mirrors the
    connector's ``verifyToken``). Rejects an expired token and any signature
    that doesn't match a secret in the verify list.
    """
    try:
        # base64url decode with padding restored.
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, TypeError):
        return None
    parts = decoded.split(":")
    if len(parts) < 3:
        return None
    sig = parts[-1]
    try:
        exp = int(parts[-2])
    except ValueError:
        return None
    payload = ":".join(parts[:-2])
    if exp != 0 and int(time.time()) > exp:
        return None
    signed = f"{payload}:{exp}"
    return payload if verify_signature(signed, sig, secrets) else None
# ---- END PLUGIN-COMPAT ----
