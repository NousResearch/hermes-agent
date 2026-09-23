"""Short-lived, single-use handles for model-blind SMS/email verification codes.

Trusted retrieval paths (a skill script, ``hermes codes put``) park the raw code
server-side and return only an opaque ``otp_…`` handle. ``browser_vault_enter_code``
consumes the handle, injects the code over the supervisor socket, and never lets
the bytes enter model context, tool results, logs, or session storage.

Shaped after ``agent/redact.py``'s vault-value registry: profile-scoped, thread
locked, bounded per profile, memory-only. On mint the code is also registered
with the vault redaction boundary so any transient echo is scrubbed by every
model-egress door.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from typing import Any, Optional

#: Default lifetime: one SMS/email delivery window.
CODE_TTL_S = 300
MIN_TTL_S = 30
MAX_TTL_S = 900
#: Per-profile bound; oldest entries evicted first.
MAX_PER_PROFILE = 16
HANDLE_PREFIX = "otp_"

_LOCK = threading.Lock()
# profile home → ordered {handle → entry}
_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}

_CODE_KEYWORD = re.compile(r"\b(code|otp|pin|verification|verify)\b", re.I)
_DIGIT_RUN = re.compile(r"\b(\d{4,8})\b")
#: Alphanumeric OTPs (e.g. GitHub's 8-char codes): mixed case/digits, 6-10 chars.
_ALNUM_RUN = re.compile(
    r"\b(?=[A-Za-z0-9]{6,10}\b)(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9]{6,10}\b"
)


def extract_verification_code(text: str) -> Optional[str]:
    """Pull a likely OTP out of an SMS/email body. Never returns the body.

    Prefers a 4–8 digit run on a line that mentions code/otp/verify/pin, then
    any 4–8 digit run, then an alphanumeric run near a keyword. Returns
    ``None`` when nothing looks like a code.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    for line in text.splitlines():
        if _CODE_KEYWORD.search(line):
            m = _DIGIT_RUN.search(line)
            if m:
                return m.group(1)
            m = _ALNUM_RUN.search(line)
            if m:
                return m.group(0)
    m = _DIGIT_RUN.search(text)
    if m:
        return m.group(1)
    if _CODE_KEYWORD.search(text):
        m = _ALNUM_RUN.search(text)
        if m:
            return m.group(0)
    return None


def _scope() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _clamp_ttl(ttl_s: float) -> float:
    try:
        ttl = float(ttl_s)
    except (TypeError, ValueError):
        return float(CODE_TTL_S)
    return min(max(ttl, float(MIN_TTL_S)), float(MAX_TTL_S))


def mint(
    code: str,
    *,
    origin: Optional[str] = None,
    source: str = "",
    ttl_s: float = CODE_TTL_S,
) -> dict[str, Any]:
    """Park ``code`` server-side and return an opaque handle for the model.

    Registers the raw bytes with the vault redaction boundary before returning,
    so a later browser/terminal egress that echoes them is scrubbed. The raw
    ``code`` is never part of the returned dict.
    """
    from agent.redact import register_vault_redaction_value

    if not isinstance(code, str) or not code.strip():
        raise ValueError("code is required")
    raw = code.strip()

    normalized_origin: Optional[str] = None
    if origin:
        from agent.vault_store import normalize_origin

        normalized_origin = normalize_origin(origin)

    register_vault_redaction_value(raw)
    handle = HANDLE_PREFIX + secrets.token_urlsafe(16)
    expires_at = time.time() + _clamp_ttl(ttl_s)
    entry = {
        "code": raw,
        "origin": normalized_origin,
        "source": (source or "")[:32],
        "expires_at": expires_at,
    }
    with _LOCK:
        bucket = _REGISTRY.setdefault(_scope(), {})
        bucket[handle] = entry
        while len(bucket) > MAX_PER_PROFILE:
            del bucket[next(iter(bucket))]
    return {
        "code_handle": handle,
        "expires_at": expires_at,
        "source": entry["source"],
        "origin": normalized_origin,
    }


def consume(handle: str, *, origin: Optional[str] = None) -> Optional[str]:
    """Return the raw code for ``handle`` once, or ``None``.

    Single-use: a successful consume removes the entry. Expired, unknown, or
    origin-mismatched handles return ``None`` without revealing which failed.
    """
    if not isinstance(handle, str) or not handle:
        return None
    now = time.time()
    with _LOCK:
        bucket = _REGISTRY.get(_scope())
        if not bucket:
            return None
        entry = bucket.get(handle)
        if entry is None:
            return None
        # Always drop: expired entries must not linger, and a wrong-origin
        # consume must not leave a live handle for a retry against another site.
        del bucket[handle]
        if now > float(entry.get("expires_at") or 0):
            return None
        bound = entry.get("origin")
        if bound:
            if not origin:
                return None
            from agent.vault_store import normalize_origin

            try:
                if normalize_origin(origin) != bound:
                    return None
            except Exception:
                return None
        return entry.get("code") or None


def peek_source(handle: str) -> Optional[str]:
    """Mint-time source label for a still-live handle (tests / diagnostics)."""
    if not isinstance(handle, str) or not handle:
        return None
    now = time.time()
    with _LOCK:
        entry = _REGISTRY.get(_scope(), {}).get(handle)
        if not entry or now > float(entry.get("expires_at") or 0):
            return None
        return entry.get("source") or None


def clear_codes() -> None:
    """Drop the current profile's parked codes (profile teardown / tests)."""
    with _LOCK:
        _REGISTRY.pop(_scope(), None)
