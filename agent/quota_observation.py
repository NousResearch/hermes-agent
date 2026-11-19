"""Host-owned, per-attempt quota observation for plugin hooks.

Built from the exact request attempt the ``credential_scope`` snapshot binds to,
and carries only non-secret operational signals:

- ``outcome`` — coarse class: ``ok`` / ``auth_failure`` / ``rate_limited`` /
  ``provider_error`` / ``timeout`` / ``unknown``. Deliberately no
  ``quota_exhausted``: nothing on the error path proves a monthly allowance was
  exhausted, so the host never claims one. A ``429`` is a provider rate-limit
  signal, never an "account has no quota" verdict.
- ``status_code`` — when known.
- ``retry_after_seconds`` — when the provider sent a parseable ``Retry-After``
  (seconds or HTTP-date). Absent means unknown, never "no wait required".
- ``rate_limit`` — the ``x-ratelimit-*`` windows exactly as sent (missing
  windows are absent; nothing is zero-filled). Only windows the provider
  actually sent appear, so ``200`` is never read as proof of remaining budget
  and a missing header is never read as zero.

The observation describes the observation, never the entitlement: a generic 200,
a 429 and a Retry-After are reported as-is; interpretation belongs to the
consumer. Raw credentials and authorization headers never appear.
"""

from __future__ import annotations

import email.utils
import math
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from agent.credential_scope import attempt_credential_scope
from agent.rate_limit_tracker import _BUCKET_TAGS, has_rate_limit_headers, lower_headers

QUOTA_OBSERVATION_CAPABILITY = "llm.quota_observation"
SCHEMA_VERSION = 1

OUTCOME_OK = "ok"
OUTCOME_AUTH_FAILURE = "auth_failure"
OUTCOME_RATE_LIMITED = "rate_limited"
OUTCOME_PROVIDER_ERROR = "provider_error"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_UNKNOWN = "unknown"

MAX_RETRY_AFTER_SECONDS = 7 * 24 * 3600.0


def _headers_of(http_response: Any) -> Optional[Mapping[str, str]]:
    headers = getattr(http_response, "headers", None)
    if headers is None:
        return None
    try:
        items = list(headers.items())
    except Exception:  # health: allow BLE001 -- unreadable headers are unknown, never fatal
        return None
    out: dict[str, str] = {}
    for name, value in items:
        if isinstance(name, str) and isinstance(value, (str, int, float)):
            out[name] = str(value)
    return out or None


def _retry_after_seconds(lowered: Optional[Mapping[str, str]], now: float) -> Optional[float]:
    if not lowered:
        return None
    raw = lowered.get("retry-after")
    if raw is None:
        return None
    raw = raw.strip()
    value: Optional[float] = None
    try:
        value = float(raw)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                value = parsed.timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    if value is None or not math.isfinite(value) or value < 0.0:
        # A non-finite Retry-After is unknown, not an invented seven-day wait.
        return None
    return min(value, MAX_RETRY_AFTER_SECONDS)


def _rate_limit_windows(lowered: Optional[Mapping[str, str]]) -> Optional[dict]:
    """Only the windows the provider actually sent, raw and unsanitized."""
    if not lowered or not has_rate_limit_headers(lowered):
        return None

    def number(name: str, *, count: bool = False) -> Any:
        raw = lowered.get(name)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        # Provider headers are untrusted: NaN/Infinity must not raise when
        # converted to int, and negative/fractional counts are not evidence.
        if not math.isfinite(value) or value < 0.0:
            return None
        if count:
            return int(value) if value.is_integer() else None
        return value

    windows: dict[str, dict] = {}
    for attr, tag in _BUCKET_TAGS:
        names = {
            "limit": f"x-ratelimit-limit-{tag}",
            "remaining": f"x-ratelimit-remaining-{tag}",
            "reset_seconds": f"x-ratelimit-reset-{tag}",
        }
        present = {key: names[key] in lowered for key in names}
        if not any(present.values()):
            continue
        limit = number(names["limit"], count=True)
        remaining = number(names["remaining"], count=True)
        reset = number(names["reset_seconds"])
        windows[attr] = {
            "limit": limit if present["limit"] else None,
            "remaining": remaining if present["remaining"] else None,
            "reset_seconds": reset if present["reset_seconds"] else None,
        }
    return windows or None


def classify_outcome(status_code: Optional[int], reason: Optional[str] = None) -> str:
    if isinstance(status_code, int):
        if 200 <= status_code < 300:
            return OUTCOME_OK
        if status_code in (401, 403):
            return OUTCOME_AUTH_FAILURE
        if status_code == 429:
            return OUTCOME_RATE_LIMITED
        if status_code >= 500:
            return OUTCOME_PROVIDER_ERROR
        return OUTCOME_UNKNOWN
    if isinstance(reason, str) and reason.strip().lower() == "timeout":
        return OUTCOME_TIMEOUT
    return OUTCOME_UNKNOWN


def build_quota_observation(agent: Any, *, http_response: Any = None,
                            status_code: Optional[int] = None,
                            reason: Optional[str] = None,
                            api_request_id: Any = None) -> dict:
    """The observation for one attempt. Never raises; gaps become explicit unknowns."""
    now = time.time()
    scope = attempt_credential_scope(agent, api_request_id)
    headers = _headers_of(http_response)
    lowered = lower_headers(headers) if headers else None
    sc = status_code if isinstance(status_code, int) else getattr(http_response, "status_code", None)
    if not isinstance(sc, int):
        sc = None
    return {
        "schema_version": SCHEMA_VERSION,
        "provider_id": scope.provider_id,
        "credential_scope_id": scope.credential_scope_id,
        "identity_status": scope.identity_status,
        "observed_at": now,
        "outcome": classify_outcome(sc, reason),
        "status_code": sc,
        "retry_after_seconds": _retry_after_seconds(lowered, now),
        "rate_limit": _rate_limit_windows(lowered),
    }
