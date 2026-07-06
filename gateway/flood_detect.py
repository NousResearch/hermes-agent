"""Canonical flood-control / rate-limit detection for adapter send/edit results.

ONE predicate, used by every gateway surface that must decide whether a failed
``SendResult`` was a flood/rate reject. A flood/rate rejection means the
platform DEFINITIVELY did not deliver the message, so a re-seed/retry after one
cannot create a duplicate — unlike an AMBIGUOUS failure (a network drop after
the bytes left, an unknown error) that MIGHT have landed and must not be
retried blindly. That known-not-delivered vs maybe-delivered split is the
dividing line the roster seed path and the stream consumer both rely on.

Extracted (mirrors the gateway.duration_format extraction) so the token set is
authoritative in ONE place for the two SHARED predicates that classify a
``SendResult`` failure — ``subagent_roster.is_flood_error`` and
``GatewayStreamConsumer._is_flood_error`` — instead of each carrying its own
substring list, and so tightening it (e.g. dropping the old bare ``"rate"``
substring that false-matched accurate/moderate/separate) happens once. NOTE:
two intentionally-narrow inline ``"flood"``-only backoff checks are NOT routed
through here — ``gateway/run.py`` progress-edit backoff and
``gateway/todo_card.py`` (its own comment pins it to the ``"flood"`` subset);
both are non-seed, non-dup-sensitive paths that never used the bare ``"rate"``.
"""

from __future__ import annotations

from typing import Any

# Precise flood / rate-limit substrings, lower-cased. Deliberately NOT a bare
# ``"rate"`` (which false-matches "accurate" / "moderate" / "separate" /
# "generate"). Covers the real shapes seen across adapters:
#   - Telegram: "flood_control:{wait}", "Flood control exceeded.",
#     "Too Many Requests: retry after N" (HTTP 429).
#   - generic / other adapters: explicit rate-limit phrasings + the 429 code.
_FLOOD_TOKENS = (
    "flood",
    "retry after",
    "too many requests",
    "429",
    "rate limit",
    "ratelimit",
    "rate_limited",
    "rate-limited",
)


def is_flood_error(result: Any) -> bool:
    """True if a failed adapter send/edit result is flood-control / rate-limit.

    ``retryable=True`` (Telegram short floods, <=5s) short-circuits to True.
    Long floods (>5s) come back ``retryable=False`` with
    ``error="flood_control:{wait}"``, so the ``"flood"`` token is load-bearing.
    ``None`` / missing / non-flood errors return False (ambiguous → caller
    latches instead of retrying).
    """
    if result is None:
        return False
    if getattr(result, "retryable", False):
        return True
    err = (getattr(result, "error", "") or "").lower()
    if not err:
        return False
    return any(tok in err for tok in _FLOOD_TOKENS)
