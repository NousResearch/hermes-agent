"""Gateway-side relay authentication primitives. EXPERIMENTAL.

The connector⇄gateway channel is authenticated because a gateway may be
customer-managed and internet-exposed (see the connector repo
``docs/connector-gateway-auth-design.md``). This module is the **gateway half**
of two HMAC schemes whose wire bytes must match the connector's TypeScript
exactly:

1. **WS upgrade auth** (gateway → connector): the gateway presents
   ``Authorization: Bearer <token>`` on the ``/relay`` WebSocket upgrade, where
   ``token = make_upgrade_token(gateway_id, secret)``. Mirrors the connector's
   ``relayAuthToken.ts`` ``makeToken`` (``src/core/relayAuthToken.ts``):
   ``base64url(f"{payload}:{exp}:{sig}")`` with
   ``sig = HMAC_SHA256(f"{payload}:{exp}", secret).hexdigest()`` and
   ``payload == gateway_id``.

2. **Inbound delivery signature** (connector → gateway): the connector signs
   each inbound POST with the per-tenant *delivery key*, carried as
   ``x-relay-timestamp`` + ``x-relay-signature`` headers; the gateway verifies
   before accepting the event. Mirrors the connector's ``deliverySigning.ts``:
   ``sig = HMAC_SHA256(f"{ts}.{body_json}", key).hexdigest()`` over the EXACT
   request body bytes, with a replay-window skew check.

Both schemes use a **multi-secret verify list** (primary first, then a secondary
during a rotation window), exactly like ``api/src/handlers/stats_oauth.ts`` — so
a secret rotation doesn't invalidate outstanding tokens.

EXPERIMENTAL: may change without a deprecation cycle until ≥2 Class-1 platforms
validate the relay contract.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional, Sequence

# Header names the connector uses for inbound delivery signatures
# (connector ``src/core/deliverySigning.ts`` — DELIVERY_TS_HEADER / SIG_HEADER).
DELIVERY_TS_HEADER = "x-relay-timestamp"
DELIVERY_SIG_HEADER = "x-relay-signature"

# Default replay window for an inbound delivery signature (connector default).
_DEFAULT_MAX_SKEW_SECONDS = 300
# Default TTL for an upgrade token (connector ``makeUpgradeToken`` default).
_DEFAULT_UPGRADE_TTL_SECONDS = 300


def _hmac_hex(payload: str, secret: str) -> str:
    """HMAC-SHA256 hex digest of ``payload`` under ``secret`` (UTF-8)."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def sign(payload: str, secret: str) -> str:
    """HMAC-SHA256 hex digest — the connector's ``sign`` (relayAuthToken.ts)."""
    return _hmac_hex(payload, secret)


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


def make_token(payload: str, secret: str, ttl_seconds: int = 0) -> str:
    """Build a signed, optionally-expiring token — the connector's ``makeToken``.

    ``base64url(f"{payload}:{exp}:{sig}")`` where ``exp`` is a unix-seconds
    expiry (0 = never) and ``sig = HMAC_SHA256(f"{payload}:{exp}", secret)``.
    base64url is unpadded to match Node's ``Buffer.toString("base64url")``.
    """
    exp = int(time.time()) + ttl_seconds if ttl_seconds > 0 else 0
    signed = f"{payload}:{exp}"
    sig = _hmac_hex(signed, secret)
    raw = f"{signed}:{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_upgrade_token(
    gateway_id: str, secret: str, ttl_seconds: int = _DEFAULT_UPGRADE_TTL_SECONDS
) -> str:
    """The WS-upgrade bearer token a gateway sends: ``payload = gateway_id``.

    The connector peeks ``gateway_id`` (the payload head) to index its secret
    verify list, then verifies the signature against that gateway's stored
    secret(s). Mirrors the connector's ``makeUpgradeToken``.
    """
    return make_token(gateway_id, secret, ttl_seconds)


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
