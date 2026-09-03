"""Rate-limit retry policy — config-gated 429 handling for the turn loop.

Config shape (all keys optional):

    agent:
      rate_limit_retry:
        max_retries: -1    # -1 = unlimited, 0 = off (default), N = N policy retries
        backoff_base: 10   # seconds, first wait
        backoff_max: 600   # seconds, ceiling

When enabled (``max_retries != 0``), the conversation loop holds the primary
provider on long exponential backoff for rate-limit 429s instead of eagerly
failing over to the fallback chain — the right posture when every fallback
rides the same throttled upstream (z.ai, single-tenant aggregators). Billing
429s are exempt: backoff cannot recover an empty account.

Disabled by default. An absent, malformed, or empty section must produce the
same result as legacy behavior (``None`` → loop code takes the legacy path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RateLimitRetryPolicy:
    """Resolved ``agent.rate_limit_retry`` config values.

    ``max_retries`` semantics: 0 = disabled, -1 = unlimited,
    positive N = at most N policy retries per API-call block.
    """

    max_retries: int = 0
    backoff_base: float = 10.0
    backoff_max: float = 600.0

    @property
    def enabled(self) -> bool:
        """Return True when the policy is active (max_retries != 0)."""
        return self.max_retries != 0


def resolve_rate_limit_retry_policy(raw: Any) -> Optional[RateLimitRetryPolicy]:
    """Resolve the raw ``agent.rate_limit_retry`` config value.

    Returns None for anything disabled or malformed so callers can use a
    simple ``if policy and policy.enabled`` gate. Never raises — a bad config
    entry must not break agent startup.
    """
    if not isinstance(raw, dict):
        return None

    def _int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            val = int(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, val))

    def _float(key: str, default: float, lo: float, hi: float) -> float:
        try:
            val = float(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, val))

    policy = RateLimitRetryPolicy(
        max_retries=_int("max_retries", 0, -1, 100000),
        backoff_base=_float("backoff_base", 10.0, 0.1, 3600.0),
        backoff_max=_float("backoff_max", 600.0, 1.0, 86400.0),
    )
    if not policy.enabled:
        return None
    # backoff_max below backoff_base is a config error; clamp so the
    # schedule is monotonically capped rather than shrinking.
    if policy.backoff_max < policy.backoff_base:
        policy = RateLimitRetryPolicy(
            max_retries=policy.max_retries,
            backoff_base=policy.backoff_base,
            backoff_max=policy.backoff_base,
        )
    return policy


def policy_backoff_wait(
    policy: RateLimitRetryPolicy,
    retry_index: int,
) -> float:
    """Exponential backoff wait for policy retry ``retry_index`` (1-based).

    Adds light uniform jitter (±10%) to decorrelate concurrent sessions
    without making status messages unreadable. The result never exceeds
    ``policy.backoff_max``.
    """
    import random

    exponent = max(0, retry_index - 1)
    delay = min(policy.backoff_base * (2 ** min(exponent, 20)), policy.backoff_max)
    jitter = random.uniform(0.9, 1.1)
    return min(delay * jitter, policy.backoff_max)
