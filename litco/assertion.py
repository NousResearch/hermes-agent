"""Host secret and user-assertion checks for the turn server.

The wire format is the one LitKit already mints
(``src/lib/agent-daemon/user-assertion.ts``)::

    v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>.<macBase64Url>

The MAC is HMAC-SHA256 over ``v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>``,
keyed on the host secret. ``issuedAt`` and ``ttl`` are milliseconds, matching
the TypeScript implementation byte for byte.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Callable, Optional

VERSION = "v1"
DEFAULT_TTL_MS = 5 * 60 * 1000
SKEW_MS = 60 * 1000


def secret_matches(presented: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time comparison of the ``X-Host-Secret`` header against the configured secret.

    An unset or empty expected secret never matches, so a host without a secret refuses everything.
    """
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _payload(user_id: str, matter_id: str, issued_at_ms: int, ttl_ms: int) -> str:
    return ".".join([VERSION, user_id, matter_id, str(issued_at_ms), str(ttl_ms)])


def mint_user_assertion(user_id: str, matter_id: str, secret: str, *, ttl_ms: int = DEFAULT_TTL_MS,
                        now_ms: Optional[int] = None) -> str:
    """Mint an assertion (tests and local tooling; LitKit mints the real ones)."""
    if not secret:
        raise ValueError("mint_user_assertion: secret is required")
    issued = int(time.time() * 1000) if now_ms is None else int(now_ms)
    payload = _payload(user_id, matter_id, issued, ttl_ms)
    mac = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return f"{payload}.{_b64url(mac)}"


@dataclass(frozen=True)
class AssertionResult:
    ok: bool
    user_id: Optional[str] = None
    reason: Optional[str] = None  # malformed | version | bad_mac | expired | matter_mismatch | missing


def verify_user_assertion(assertion: Optional[str], *, matter_id: str, secret: str,
                          now_ms: Optional[Callable[[], int]] = None) -> AssertionResult:
    """Verify an assertion. The MAC is checked before any field is trusted."""
    if not assertion or not isinstance(assertion, str):
        return AssertionResult(False, reason="missing")
    if not secret:
        return AssertionResult(False, reason="bad_mac")
    parts = assertion.split(".")
    if len(parts) != 6:
        return AssertionResult(False, reason="malformed")
    version, user_id, asserted_matter, issued_raw, ttl_raw, mac_b64 = parts
    if version != VERSION:
        return AssertionResult(False, reason="version")
    if not user_id or not asserted_matter:
        return AssertionResult(False, reason="malformed")
    try:
        issued = int(issued_raw, 10)
        ttl = int(ttl_raw, 10)
    except ValueError:
        return AssertionResult(False, reason="malformed")
    if ttl <= 0:
        return AssertionResult(False, reason="malformed")
    expected = hmac.new(secret.encode("utf-8"), _payload(user_id, asserted_matter, issued, ttl).encode("utf-8"),
                        hashlib.sha256).digest()
    try:
        presented = _b64url_decode(mac_b64)
    except Exception:
        return AssertionResult(False, reason="bad_mac")
    if not hmac.compare_digest(expected, presented):
        return AssertionResult(False, reason="bad_mac")
    now = (now_ms or (lambda: int(time.time() * 1000)))()
    if now > issued + ttl or now < issued - SKEW_MS:
        return AssertionResult(False, reason="expired")
    if matter_id and asserted_matter != matter_id:
        return AssertionResult(False, reason="matter_mismatch")
    return AssertionResult(True, user_id=user_id)
