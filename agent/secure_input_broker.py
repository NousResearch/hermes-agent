"""Ephemeral secure input broker.

The model may ask a UI surface to collect a secret, but the raw value must not
enter the LLM transcript.  This module stores such values in process memory and
returns opaque ``secret://...`` references that tools can consume under a narrow
scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import threading
import time
from typing import Iterable


_SECRET_REF_PREFIX = "secret://session/"
_DEFAULT_TTL_SECONDS = 600
_MAX_TTL_SECONDS = 3600


@dataclass
class SecretRecord:
    value: str
    purpose: str
    created_at: float
    expires_at: float
    allowed_consumers: set[str] = field(default_factory=set)
    single_use: bool = True
    consumed: bool = False
    label: str = ""


_lock = threading.Lock()
_records: dict[str, SecretRecord] = {}
_redaction_values: dict[str, float] = {}


class SecretBrokerError(ValueError):
    """Raised when a secure-input reference cannot be used."""


def _now() -> float:
    return time.time()


def _normalize_consumers(consumers: Iterable[str] | None) -> set[str]:
    values = {str(c).strip() for c in (consumers or []) if str(c).strip()}
    return values or {"terminal"}


def _clamp_ttl(ttl_seconds: int | float | None) -> int:
    try:
        ttl = int(ttl_seconds or _DEFAULT_TTL_SECONDS)
    except (TypeError, ValueError):
        ttl = _DEFAULT_TTL_SECONDS
    return max(1, min(ttl, _MAX_TTL_SECONDS))


def _purge_expired_locked(now: float | None = None) -> None:
    now = _now() if now is None else now
    for ref, rec in list(_records.items()):
        if rec.expires_at <= now:
            _records.pop(ref, None)
    for value, expires_at in list(_redaction_values.items()):
        if expires_at <= now:
            _redaction_values.pop(value, None)


def register_secret(
    value: str,
    *,
    purpose: str,
    allowed_consumers: Iterable[str] | None = None,
    ttl_seconds: int | float | None = None,
    single_use: bool = True,
    label: str = "",
) -> dict:
    """Store *value* temporarily and return redacted metadata with a ref."""
    if not isinstance(value, str) or not value:
        raise SecretBrokerError("secure input was empty or cancelled")

    ttl = _clamp_ttl(ttl_seconds)
    now = _now()
    ref = _SECRET_REF_PREFIX + secrets.token_urlsafe(18)
    rec = SecretRecord(
        value=value,
        purpose=str(purpose or "secure_input"),
        created_at=now,
        expires_at=now + ttl,
        allowed_consumers=_normalize_consumers(allowed_consumers),
        single_use=bool(single_use),
        label=str(label or ""),
    )
    with _lock:
        _purge_expired_locked(now)
        _records[ref] = rec
        # Keep exact-value redaction alive until expiry even after single-use
        # consumption, because the child process may echo the value after the
        # broker has marked it consumed.
        _redaction_values[value] = rec.expires_at

    return {
        "secret_ref": ref,
        "purpose": rec.purpose,
        "expires_at": rec.expires_at,
        "ttl_seconds": ttl,
        "single_use": rec.single_use,
        "allowed_consumers": sorted(rec.allowed_consumers),
        "redacted": True,
    }


def consume_secret(secret_ref: str, *, consumer: str) -> str:
    """Return the secret value for an authorized tool consumer.

    The returned value is intentionally only available to trusted tool runtime
    code. Tool schemas should pass refs, never raw values.
    """
    ref = str(secret_ref or "").strip()
    consumer = str(consumer or "").strip()
    if not ref.startswith(_SECRET_REF_PREFIX):
        raise SecretBrokerError("invalid secure input reference")
    if not consumer:
        raise SecretBrokerError("missing secure input consumer")

    now = _now()
    with _lock:
        _purge_expired_locked(now)
        rec = _records.get(ref)
        if rec is None:
            raise SecretBrokerError("secure input reference not found or expired")
        if consumer not in rec.allowed_consumers:
            allowed = ", ".join(sorted(rec.allowed_consumers)) or "none"
            raise SecretBrokerError(
                f"secure input reference is not scoped for {consumer!r} (allowed: {allowed})"
            )
        if rec.single_use and rec.consumed:
            raise SecretBrokerError("secure input reference has already been consumed")
        rec.consumed = True
        return rec.value


def describe_secret(secret_ref: str) -> dict | None:
    """Return non-sensitive metadata for *secret_ref*, or None."""
    now = _now()
    with _lock:
        _purge_expired_locked(now)
        rec = _records.get(secret_ref)
        if rec is None:
            return None
        return {
            "secret_ref": secret_ref,
            "purpose": rec.purpose,
            "expires_at": rec.expires_at,
            "single_use": rec.single_use,
            "consumed": rec.consumed,
            "allowed_consumers": sorted(rec.allowed_consumers),
            "redacted": True,
        }


def redact_known_secrets(text: str) -> str:
    """Replace exact broker-held secret values in text with a sentinel."""
    if not text:
        return text
    with _lock:
        _purge_expired_locked()
        values = list(_redaction_values)
    for value in values:
        if value:
            text = text.replace(value, "[REDACTED SECURE INPUT]")
    return text


def clear_all() -> None:
    """Test helper: clear every in-memory secret."""
    with _lock:
        _records.clear()
        _redaction_values.clear()
