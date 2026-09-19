"""Exhaustion cooldown windows and reset-time parsing for pooled credentials.

A benched credential comes back when a wall-clock instant passes: the
provider's own ``reset_at`` when it sent one, otherwise the failing HTTP
status sized against the TTLs below. Both halves are pure functions of
persisted error state, so they live beside the pool that stores it.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from agent.retry_utils import reset_delay_from_message


# Cooldowns before retrying an exhausted credential. Transient 401s cool down
# briefly so single-key setups recover; 429/402/other take an hour.
# Provider-supplied reset_at timestamps override these defaults.
EXHAUSTED_TTL_401_SECONDS = 5 * 60
EXHAUSTED_TTL_429_SECONDS = 60 * 60
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60
# When the offending key is the sole non-DEAD entry, an hour-long bench means
# an hour of hard failures. Throttles (429/403/5xx) reset in seconds, so a sole
# credential cools down briefly instead.
EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS = 60

# ``FailoverReason.billing`` as a bare string: the pool persists classified
# failure semantics to JSON and must not import the classifier.
FAILURE_REASON_BILLING = "billing"

# Billing verdict resting on an ambiguous body (#82154): Anthropic's "out of
# extra usage" 400 is returned both for genuine overage and for a server-side
# content-filter rejection, which leaves the credential healthy. Unverified
# billing gets the short transient cooldown; genuine depletion re-latches.
FAILURE_REASON_BILLING_UNVERIFIED = "billing_unverified"


def _exhausted_ttl(
    error_code: Optional[int],
    *,
    sole_credential: bool = False,
    failure_reason: Optional[str] = None,
) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion.

    *sole_credential*: the pool has nothing to rotate to, so transient
    throttles (429 and the catch-all default covering 403/5xx/unknown) are
    capped to a brief cooldown; 401 keeps its own already-short TTL.

    *failure_reason* is the classifier verdict: an OpenRouter ``key limit
    exceeded`` and an xAI spending block both arrive as 403 but are billing,
    and a 60s retry on a spent account just re-fails. Billing keeps the full
    bench regardless of status; 402 is billing by definition.
    Unverified billing (#82154) gets the short cooldown regardless of pool
    size (the credential may be healthy), unless the status is a true 402.
    """
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    base = EXHAUSTED_TTL_429_SECONDS if error_code == 429 else EXHAUSTED_TTL_DEFAULT_SECONDS
    if failure_reason == FAILURE_REASON_BILLING_UNVERIFIED and error_code != 402:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    is_billing = error_code == 402 or failure_reason == FAILURE_REASON_BILLING
    if sole_credential and not is_billing:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    return base


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse of epoch seconds / epoch ms / ISO-8601 into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _normalize_error_context(error_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key in ("reason", "message"):
        value = error_context.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    reset_at = (
        error_context.get("reset_at")
        or error_context.get("resets_at")
        or error_context.get("retry_until")
    )
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
    message = error_context.get("message")
    if parsed_reset_at is None and isinstance(message, str):
        retry_delay_seconds = reset_delay_from_message(message)
        if retry_delay_seconds is not None:
            parsed_reset_at = time.time() + retry_delay_seconds
    if parsed_reset_at is not None:
        normalized["reset_at"] = parsed_reset_at
    return normalized
