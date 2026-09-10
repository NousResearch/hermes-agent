"""Primary rate-limit cooldown arming, shared by fallback switches."""
import logging
import time

from agent.error_classifier import FailoverReason

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


def _arm_rate_limit_cooldown(agent, reason: "FailoverReason | None") -> int | None:
    """Arm the primary's exponential cooldown (60s → 2m → ... → 4h cap) on CONSECUTIVE rate-limits;
    restore_primary_runtime resets the counter. Only when leaving the primary: chain-switching from
    an active fallback means the primary was not the 429 source, so its cooldown is left alone.
    Return the armed cooldown in seconds, or None when no cooldown was armed."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return None
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
    if getattr(agent, "_fallback_activated", False) and not (primary_provider and current_provider == primary_provider):
        return None
    backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
    agent._rate_limit_backoff_count = backoff_count + 1
    backoff_seconds = min(60 * (2 ** backoff_count), 14400)
    agent._rate_limited_until = time.monotonic() + backoff_seconds
    logging.info("Rate-limit backoff level %d: cooldown %d s (%.1f min, backoff#%d)", backoff_count, backoff_seconds, backoff_seconds / 60, backoff_count + 1)
    return backoff_seconds


