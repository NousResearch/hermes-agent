"""What a keep-alive costs against the rewrite it prevents (no absolute dollar floor)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

#: No entry exists below this prefix: the smallest documented minimum (Anthropic 1,024 for Sonnet/Opus,
#: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching ; OpenAI 1,024,
#: https://developers.openai.com/api/docs/guides/prompt-caching ; Qwen 1,024,
#: https://www.alibabacloud.com/help/en/model-studio/context-cache).
MIN_CACHEABLE_PREFIX_TOKENS = 1024


@dataclass(frozen=True)
class CacheWarmingDecision:
    """Inputs and outcome of one warm-or-stop decision."""

    warm_cost: float
    miss_cost: float
    expected_savings: float
    economics_available: bool
    action: str  # "warm" | "stop"
    within_miss: bool = True

    @property
    def stop_reason(self) -> str:
        if not self.economics_available:
            return "cache economics unavailable"
        if not self.within_miss:
            return "refreshes would cost more than the cache write they prevent"
        return "a refresh would cost more than the cache write it prevents"


def _per_token(rate: Optional[Decimal]) -> float:
    return float(rate) / 1_000_000 if rate is not None else 0.0


def evaluate_economics(
    pricing: Any, cached_tokens: int, uncached_tokens: int = 0, *,
    spent_usd: float = 0.0, observed_warm_usd: float = 0.0, output_tokens: int = 1,
) -> CacheWarmingDecision:
    """Price one refresh against the miss it prevents using a ``PricingEntry``-shaped object
    (``input/output/cache_read/cache_write_cost_per_million``).

    ``cached_tokens`` is the marked prefix the entry holds (read on a refresh, re-written on a
    miss); ``uncached_tokens`` is the tail past the last marker, billed at the input rate either
    way, so it adds to the refresh's cost and nothing to the miss it prevents. A refresh is
    reserved at no less than ``observed_warm_usd`` (what the last one was actually billed), and
    ``spent_usd`` (what the run's refreshes were billed so far) plus that reservation may not
    exceed the miss: past it, warming is dearer than letting the entry expire once. There is no
    absolute dollar floor: a cheap route warms whenever the read is cheaper than the rewrite; a prefix
    below :data:`MIN_CACHEABLE_PREFIX_TOKENS` has no entry to keep."""
    read = _per_token(getattr(pricing, "cache_read_cost_per_million", None))
    write = _per_token(getattr(pricing, "cache_write_cost_per_million", None))
    inp = _per_token(getattr(pricing, "input_cost_per_million", None))
    out = _per_token(getattr(pricing, "output_cost_per_million", None))
    fee = float(getattr(pricing, "request_cost", None) or 0)  # a per-request fee the refresh pays on top
    tail_cost = uncached_tokens * inp
    hit_cost = cached_tokens * read
    miss_full = cached_tokens * (write if write > 0 else inp)
    warm_cost = max(hit_cost + tail_cost + output_tokens * out + fee, observed_warm_usd)
    miss_cost = max(0.0, miss_full - hit_cost)
    economics_available = cached_tokens > 0 and (hit_cost > 0 or miss_full > 0)
    expected = miss_cost - warm_cost
    within_miss = spent_usd + warm_cost <= miss_cost
    economics_available = economics_available and cached_tokens >= MIN_CACHEABLE_PREFIX_TOKENS
    action = "warm" if economics_available and within_miss and expected > 0 else "stop"
    return CacheWarmingDecision(
        warm_cost=warm_cost, miss_cost=miss_cost, expected_savings=expected,
        economics_available=economics_available, action=action, within_miss=within_miss,
    )
