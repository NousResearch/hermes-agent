"""Authentication helpers for trusted gateway proxy scope forwarding."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Mapping

PROXY_SCOPE_KEY_ENV = "GATEWAY_PROXY_SCOPE_KEY"
PROXY_SCOPE_SIGNATURE_HEADER = "X-Hermes-Proxy-Scope-Signature"
PROXY_SCOPE_TIMESTAMP_HEADER = "X-Hermes-Proxy-Scope-Timestamp"
PROXY_SCOPE_SIGNATURE_VERSION = "v1"
PROXY_SCOPE_MAX_CLOCK_SKEW_SECONDS = 300


def get_proxy_scope_key() -> str:
    """Return the shared secret used to authenticate proxy scope metadata."""
    return os.getenv(PROXY_SCOPE_KEY_ENV, "").strip()


def canonicalize_proxy_scope(proxy_scope: Mapping[str, Any]) -> str:
    """Serialize proxy scope metadata into the signed wire representation."""
    return json.dumps(proxy_scope, sort_keys=True, separators=(",", ":"))


def sign_proxy_scope(
    proxy_scope: Mapping[str, Any],
    secret: str,
    timestamp: int | None = None,
) -> tuple[str, str]:
    """Return ``(timestamp, signature)`` headers for trusted proxy scope metadata."""
    ts = str(int(time.time() if timestamp is None else timestamp))
    payload = f"{ts}.{canonicalize_proxy_scope(proxy_scope)}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return ts, f"{PROXY_SCOPE_SIGNATURE_VERSION}={digest}"


def verify_proxy_scope_signature(
    proxy_scope: Mapping[str, Any],
    secret: str,
    timestamp: str | None,
    signature: str | None,
    *,
    now: int | None = None,
) -> bool:
    """Return whether the supplied signature authenticates the proxy scope.

    Returns ``False`` when no secret is configured so that proxy scope metadata
    is rejected unless a shared key has been provisioned on both sides.
    """
    if not secret:
        return False
    if not timestamp or not signature:
        return False
    try:
        ts_int = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = int(time.time() if now is None else now)
    if abs(current - ts_int) > PROXY_SCOPE_MAX_CLOCK_SKEW_SECONDS:
        return False
    expected_timestamp, expected_signature = sign_proxy_scope(proxy_scope, secret, ts_int)
    return hmac.compare_digest(timestamp, expected_timestamp) and hmac.compare_digest(
        signature, expected_signature
    )
