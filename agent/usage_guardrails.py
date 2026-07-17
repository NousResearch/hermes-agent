"""Per-turn model-call and input-usage guardrails.

Gross prompt traffic remains an observability and per-request safety measure.
Subscription-included Codex routes use a separate weighted cumulative budget:
uncached input + cache writes + 10% of cache reads.  The 10% factor is Hermes
policy only; it is not a statement about provider allowance accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_CACHE_READ_WEIGHT = 0.10
_DEFAULTS = {
    "interactive": {
        "max_model_calls": 44,
        "max_subscription_weighted_input_tokens": 1_120_000,
        "max_request_input_tokens": 140_000,
        # 3 is reachable and stops sustained 100k+ prompts while leaving a
        # normal one-off large context request available.
        "max_consecutive_large_requests": 3,
        "large_request_tokens": 100_000,
    },
    "cron": {
        "max_model_calls": 2,
        "max_subscription_weighted_input_tokens": 75_000,
        "max_request_input_tokens": 75_000,
        "max_consecutive_large_requests": 3,
        "large_request_tokens": 60_000,
    },
}

GOVERNOR_STOP_REASONS = frozenset({
    "model_call_limit",
    "subscription_weighted_input_token_limit",
    "request_input_token_limit",
    "consecutive_large_request_limit",
})


def is_governor_stop(turn_result: Any) -> bool:
    if not isinstance(turn_result, Mapping):
        return False
    return bool(turn_result.get("governor_stop")) or str(
        turn_result.get("turn_exit_reason") or ""
    ) in GOVERNOR_STOP_REASONS


def default_usage_guardrails_config() -> dict[str, Any]:
    return {"operator_authorized": False, **{k: dict(v) for k, v in _DEFAULTS.items()}}


@dataclass(frozen=True)
class UsageLimits:
    max_model_calls: int
    max_subscription_weighted_input_tokens: int
    max_request_input_tokens: int
    max_consecutive_large_requests: int
    large_request_tokens: int


class UsageGuardrail:
    """Fail-closed turn budget; subscription enforcement is explicitly opt-in."""

    def __init__(self, limits: UsageLimits, *, scope: str, subscription_included: bool = False) -> None:
        self.limits = limits
        self.scope = scope
        self.subscription_included = subscription_included
        self.session_usage = self._empty_usage()
        self._reset_turn_usage()

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "uncached_input_tokens": 0, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "gross_input_tokens": 0,
            "weighted_input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "model_calls": 0,
        }

    def _reset_turn_usage(self) -> None:
        self.calls = 0
        self.uncached_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.gross_input_tokens = 0
        self.weighted_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.consecutive_large_requests = 0
        self.receipts: list[dict[str, int | str]] = []
        self.stop_reason: str | None = None
        self._request_stop_details: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None, *, scope: str,
        subscription_included: bool = False,
    ) -> "UsageGuardrail":
        defaults = _DEFAULTS[scope]
        raw = (config or {}).get("usage_guardrails", {})
        raw = raw if isinstance(raw, Mapping) else {}
        candidate = raw.get(scope, {})
        candidate = candidate if isinstance(candidate, Mapping) else {}
        authorized = raw.get("operator_authorized") is True
        # Legacy gross cumulative configuration is intentionally ignored: it
        # named the wrong quantity for subscription enforcement.
        resolved: dict[str, int] = {}
        for key, default in defaults.items():
            try:
                value = int(candidate.get(key, default))
            except (TypeError, ValueError):
                value = default
            resolved[key] = value if value <= default or authorized else default
        return cls(UsageLimits(**resolved), scope=scope, subscription_included=subscription_included)

    def begin_user_turn(self) -> None:
        """Reset this explicit turn only; lifetime session observability remains."""
        self._reset_turn_usage()

    def request_input_limit_exceeded(self, estimate: int) -> bool:
        return max(0, int(estimate or 0)) > self.limits.max_request_input_tokens

    def before_request(self, estimated_gross_input_tokens: int) -> str | None:
        estimate = max(0, int(estimated_gross_input_tokens or 0))
        if self.calls >= self.limits.max_model_calls:
            return self._stop("model_call_limit")
        if self.request_input_limit_exceeded(estimate):
            return self._stop("request_input_token_limit")
        # The exact cache mix is known only after a provider response. Do not
        # use a gross preflight estimate as a weighted allowance surrogate;
        # record_usage() enforces the cumulative weighted budget from actual
        # uncached/cache-read/cache-write buckets.
        if self.subscription_included and (
            self.weighted_input_tokens >= self.limits.max_subscription_weighted_input_tokens
        ):
            return self._stop("subscription_weighted_input_token_limit")
        if self.consecutive_large_requests >= self.limits.max_consecutive_large_requests:
            return self._stop("consecutive_large_request_limit")
        return None

    def stop_for_request_input_limit(
        self, *, estimated_request_tokens: int, context_window: int | None,
        compaction_attempted: bool, post_compaction_estimate: int | None,
    ) -> str:
        self._request_stop_details = {
            "estimated_request_tokens": max(0, int(estimated_request_tokens or 0)),
            "request_limit": self.limits.max_request_input_tokens,
            "context_window": context_window if isinstance(context_window, int) and context_window > 0 else None,
            "compaction_attempted": bool(compaction_attempted),
            "post_compaction_estimate": post_compaction_estimate,
        }
        return self._stop("request_input_token_limit")

    def begin_request(self, estimated_gross_input_tokens: int) -> str | None:
        stop = self.before_request(estimated_gross_input_tokens)
        if stop is None:
            self.calls += 1
            self.session_usage["model_calls"] += 1
        return stop

    def record_usage(
        self, *, input_tokens: int, cache_read_tokens: int, cache_write_tokens: int = 0,
        output_tokens: int = 0, reasoning_tokens: int = 0,
    ) -> None:
        uncached = max(0, int(input_tokens or 0))
        cached_read = max(0, int(cache_read_tokens or 0))
        cached_write = max(0, int(cache_write_tokens or 0))
        output = max(0, int(output_tokens or 0))
        reasoning = max(0, int(reasoning_tokens or 0))
        gross = uncached + cached_read + cached_write
        weighted = uncached + cached_write + round(cached_read * _CACHE_READ_WEIGHT)
        for target in (self.__dict__, self.session_usage):
            target["uncached_input_tokens"] = target.get("uncached_input_tokens", 0) + uncached
            target["cache_read_tokens"] = target.get("cache_read_tokens", 0) + cached_read
            target["cache_write_tokens"] = target.get("cache_write_tokens", 0) + cached_write
            target["gross_input_tokens"] = target.get("gross_input_tokens", 0) + gross
            target["weighted_input_tokens"] = target.get("weighted_input_tokens", 0) + weighted
            target["output_tokens"] = target.get("output_tokens", 0) + output
            target["reasoning_tokens"] = target.get("reasoning_tokens", 0) + reasoning
        self.consecutive_large_requests = self.consecutive_large_requests + 1 if gross >= self.limits.large_request_tokens else 0
        self.receipts.append({
            "call": self.calls, "uncached_input_tokens": uncached,
            "cache_read_tokens": cached_read, "cache_write_tokens": cached_write,
            "gross_input_tokens": gross, "weighted_input_tokens": weighted,
            "output_tokens": output, "reasoning_tokens": reasoning,
        })
        if self.subscription_included and self.weighted_input_tokens >= self.limits.max_subscription_weighted_input_tokens:
            self._stop("subscription_weighted_input_token_limit")

    def turn_usage_summary(self) -> dict[str, Any]:
        largest = max((int(r.get("gross_input_tokens", 0)) for r in self.receipts), default=0)
        return {
            "model_calls": self.calls, "max_model_calls": self.limits.max_model_calls,
            "uncached_input_tokens": self.uncached_input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "gross_input_tokens": self.gross_input_tokens,
            "weighted_input_tokens": self.weighted_input_tokens,
            "max_subscription_weighted_input_tokens": self.limits.max_subscription_weighted_input_tokens,
            "output_tokens": self.output_tokens, "reasoning_tokens": self.reasoning_tokens,
            "largest_request_tokens": largest, "stop_reason": self.stop_reason,
            "subscription_included": self.subscription_included,
            "session_usage": dict(self.session_usage),
        }

    def checkpoint(self) -> str:
        reason = self.stop_reason or "budget_limit"
        text = (
            f"Usage guardrail stopped this {self.scope} run: {reason}. "
            f"Checkpoint: {self.calls}/{self.limits.max_model_calls} model calls; "
            f"{self.gross_input_tokens:,} gross input tokens; "
            f"{self.weighted_input_tokens:,}/{self.limits.max_subscription_weighted_input_tokens:,} weighted subscription input tokens."
        )
        if self._request_stop_details is not None:
            detail = self._request_stop_details
            context = detail["context_window"] if detail["context_window"] is not None else "unknown"
            post = detail["post_compaction_estimate"]
            post_text = "not available" if post is None else f"{int(post):,}"
            text += (
                f" Request estimate={detail['estimated_request_tokens']:,}; "
                f"request limit={detail['request_limit']:,}; context window={context}; "
                f"compaction attempted={detail['compaction_attempted']}; "
                f"post-compaction estimate={post_text}."
            )
        return text

    def _stop(self, reason: str) -> str:
        self.stop_reason = reason
        return self.checkpoint()
