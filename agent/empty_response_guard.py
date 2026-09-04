"""Deterministic-empty detection and cost-aware retry budgets (NS-503).

When a provider returns an empty completion, the agent loop retries up to
3 times and then walks the fallback chain. Every attempt re-sends the full
conversation input — at large context on paid routes this bills the user
repeatedly for a turn that produces no text (the "charged ~$2.33 for an
empty answer" incident class).

Signaled refusals (``finish_reason="content_filter"``, Anthropic
``stop_reason="refusal"``, Bedrock guardrails) are already terminal and
never reach the empty-retry loop. This module addresses the *unsignaled*
empties: the provider reports a successful completion with zero output
tokens and a generic finish reason (portal-proxied refusals commonly look
like this).

Two independent guards, both failing OPEN to today's behaviour:

1. **Deterministic-empty detection** — two consecutive empty attempts from
   the same (model, provider, finish_reason) are treated as deterministic
   when usage proves zero output, or when usage is absent and the assembled
   responses contain neither content nor reasoning. Remaining retries are
   skipped and the loop proceeds straight to the fallback chain (a different
   model may behave differently). Mixed evidence or any generated tokens keep
   the full retry budget.

2. **Cost-aware retry budget** — when the estimated input cost of a
   single empty attempt exceeds the configured threshold (default
   $0.25), the empty-retry budget for this streak drops from 3 to 1.
   Unknown pricing, missing usage, or included/subscription routes
   leave the budget untouched.

Configured via the additive ``agent.empty_response_guard`` section in
``config.yaml`` (resolved once at agent init by ``agent_init``)::

    agent:
      empty_response_guard:
        enabled: true            # false = legacy fixed 3-retry behaviour
        cost_threshold_usd: 0.25 # per-attempt cost that halves the budget

Per project policy, no ``HERMES_*`` environment variables are involved —
``.env`` is reserved for credentials; behavioural settings live in
``config.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_EMPTY_RETRY_BUDGET = 3
REDUCED_EMPTY_RETRY_BUDGET = 1
DEFAULT_COST_THRESHOLD_USD = Decimal("0.25")
DEFAULT_GUARD_ENABLED = True

# Attribute names stashed on the agent object. State is scoped to one
# consecutive empty streak: it is cleared whenever a streak starts
# (``_empty_content_retries == 0`` at record time), which transparently
# honours every existing reset site (turn start, compaction, tool
# success, fallback activation) without touching them.
_ATTEMPTS_ATTR = "_empty_attempt_history"
_STREAK_COST_ATTR = "_empty_streak_cost_usd"
_ENABLED_ATTR = "_empty_guard_enabled"
_THRESHOLD_ATTR = "_empty_guard_cost_threshold_usd"


@dataclass(frozen=True)
class EmptyAttempt:
    """One observed empty completion within the current streak."""

    model: str
    provider: str
    finish_reason: str
    usage_present: bool
    zero_output: bool
    observed_generation: bool

    @property
    def signature(self) -> tuple:
        return (self.model, self.provider, self.finish_reason)


def resolve_guard_settings(section: Any) -> Tuple[bool, Decimal]:
    """Resolve ``agent.empty_response_guard`` config into (enabled, threshold).

    Tolerant of malformed input: anything that isn't a well-formed dict
    (or well-formed values within it) falls back to the schema defaults.
    Called once per agent at init; the resolved values are stashed on the
    agent object so the hot loop never re-reads config.
    """
    if not isinstance(section, dict):
        return (DEFAULT_GUARD_ENABLED, DEFAULT_COST_THRESHOLD_USD)

    enabled_raw = section.get("enabled", DEFAULT_GUARD_ENABLED)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    elif isinstance(enabled_raw, str):
        # YAML quoting can turn true/false into strings.
        enabled = enabled_raw.strip().lower() not in ("0", "false", "no", "off")
    else:
        enabled = DEFAULT_GUARD_ENABLED

    threshold = DEFAULT_COST_THRESHOLD_USD
    threshold_raw = section.get("cost_threshold_usd")
    if threshold_raw is not None and not isinstance(threshold_raw, bool):
        try:
            candidate = Decimal(str(threshold_raw))
            if candidate > 0:
                threshold = candidate
        except Exception:  # noqa: BLE001 — malformed config must not break init
            logger.debug(
                "empty-guard: invalid cost_threshold_usd %r, using default",
                threshold_raw,
            )
    return (enabled, threshold)


def guard_enabled(agent: Any) -> bool:
    """Whether the guard is enabled for this agent (config-resolved).

    Agents built before the config was threaded through (tests, embedded
    callers) simply get the default: enabled.
    """
    value = getattr(agent, _ENABLED_ATTR, DEFAULT_GUARD_ENABLED)
    return value if isinstance(value, bool) else DEFAULT_GUARD_ENABLED


def _cost_threshold_usd(agent: Any) -> Decimal:
    value = getattr(agent, _THRESHOLD_ATTR, None)
    if isinstance(value, Decimal) and value > 0:
        return value
    return DEFAULT_COST_THRESHOLD_USD


def _attempts(agent: Any) -> List[EmptyAttempt]:
    attempts = getattr(agent, _ATTEMPTS_ATTR, None)
    if attempts is None:
        attempts = []
        setattr(agent, _ATTEMPTS_ATTR, attempts)
    return attempts


def _estimate_attempt_cost(agent: Any, response: Any) -> Optional[Decimal]:
    """Best-effort USD estimate for one attempt. None when unknown."""
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return None
    try:
        from agent.usage_pricing import estimate_usage_cost, normalize_usage

        canonical = normalize_usage(
            raw_usage,
            provider=getattr(agent, "provider", None),
            api_mode=getattr(agent, "api_mode", None),
        )
        result = estimate_usage_cost(
            getattr(agent, "model", "") or "",
            canonical,
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
            api_key=getattr(agent, "api_key", None),
        )
    except Exception:  # noqa: BLE001 — pricing must never break the loop
        logger.debug("empty-guard: cost estimation failed", exc_info=True)
        return None
    return getattr(result, "amount_usd", None)


def _zero_output(agent: Any, response: Any) -> tuple:
    """Return (usage_present, zero_output) for a response, failing open."""
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return (False, False)
    try:
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(
            raw_usage,
            provider=getattr(agent, "provider", None),
            api_mode=getattr(agent, "api_mode", None),
        )
    except Exception:  # noqa: BLE001
        logger.debug("empty-guard: usage normalization failed", exc_info=True)
        return (False, False)
    output = getattr(canonical, "output_tokens", None)
    if output is None:
        return (False, False)
    # A present-but-empty usage object (some proxies emit usage with no
    # fields) normalizes to all zeros. A genuine completion always has
    # input tokens — without them the usage is not evidence, fail open.
    if getattr(canonical, "prompt_tokens", 0) <= 0:
        return (False, False)
    # Reasoning tokens count as real generation — a reasoning-only
    # response is NOT a deterministic empty (the prefill-continuation
    # path upstream owns that case).
    reasoning = getattr(canonical, "reasoning_tokens", 0) or 0
    return (True, (output + reasoning) == 0)


def record_empty_attempt(
    agent: Any,
    *,
    finish_reason: str,
    response: Any,
    observed_generation: bool = True,
) -> None:
    """Record one empty completion in the current streak.

    Must be called before ``_empty_content_retries`` is incremented for
    this attempt: a counter of 0 marks the start of a new streak and
    clears prior history (this transparently follows every existing
    counter-reset site).
    """
    attempts = _attempts(agent)
    if getattr(agent, "_empty_content_retries", 0) == 0:
        attempts.clear()
        setattr(agent, _STREAK_COST_ATTR, Decimal("0"))

    usage_present, zero_output = _zero_output(agent, response)
    attempts.append(
        EmptyAttempt(
            model=str(getattr(agent, "model", "") or ""),
            provider=str(getattr(agent, "provider", "") or ""),
            finish_reason=str(finish_reason or ""),
            usage_present=usage_present,
            zero_output=zero_output,
            observed_generation=bool(observed_generation),
        )
    )

    cost = _estimate_attempt_cost(agent, response)
    if cost is not None and cost > 0:
        prior = getattr(agent, _STREAK_COST_ATTR, Decimal("0")) or Decimal("0")
        setattr(agent, _STREAK_COST_ATTR, prior + cost)


def deterministic_empty(agent: Any) -> bool:
    """True when the current streak looks deterministic.

    Requires >= 2 consecutive attempts with an identical (model, provider,
    finish_reason) signature. Usage-backed attempts must all prove zero output.
    Usage-absent attempts must all have no observed content or reasoning. Mixed
    evidence fails open so ambiguous transients keep their retries.
    """
    if not guard_enabled(agent):
        return False
    attempts = getattr(agent, _ATTEMPTS_ATTR, None) or []
    if len(attempts) < 2:
        return False
    first = attempts[0]
    same_signature = all(a.signature == first.signature for a in attempts)
    usage_proves_empty = all(a.usage_present and a.zero_output for a in attempts)
    response_proves_empty = all(
        not a.usage_present and not a.observed_generation for a in attempts
    )
    return same_signature and (usage_proves_empty or response_proves_empty)


def empty_retry_budget(agent: Any, response: Any) -> int:
    """Empty-retry budget for the current streak (3, or 1 when a single
    attempt is estimated to cost more than the configured threshold)."""
    if not guard_enabled(agent):
        return DEFAULT_EMPTY_RETRY_BUDGET
    cost = _estimate_attempt_cost(agent, response)
    if cost is None:
        return DEFAULT_EMPTY_RETRY_BUDGET
    if cost >= _cost_threshold_usd(agent):
        return REDUCED_EMPTY_RETRY_BUDGET
    return DEFAULT_EMPTY_RETRY_BUDGET


def streak_cost_usd(agent: Any) -> Optional[Decimal]:
    """Accumulated estimated cost of the current empty streak, if known."""
    cost = getattr(agent, _STREAK_COST_ATTR, None)
    if cost is None or cost <= 0:
        return None
    return cost
