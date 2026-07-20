"""Session prompt-cache efficiency metrics (OpenAI + Anthropic shapes).

Tracks cache hit ratios without mutating conversation context. Safe to call
from /status and Model Desk — never touches the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CacheMetrics:
    """Accumulated cache read/write stats for a session."""

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    prompt_tokens: int = 0
    api_calls: int = 0
    hits: int = 0  # calls with cache_read > 0

    def record(
        self,
        *,
        cache_read: int = 0,
        cache_write: int = 0,
        prompt: int = 0,
    ) -> None:
        self.cache_read_tokens += max(0, int(cache_read or 0))
        self.cache_write_tokens += max(0, int(cache_write or 0))
        self.prompt_tokens += max(0, int(prompt or 0))
        self.api_calls += 1
        if cache_read and int(cache_read) > 0:
            self.hits += 1

    @property
    def hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cache_read_tokens / self.prompt_tokens

    @property
    def hit_rate_pct(self) -> float:
        return round(self.hit_ratio * 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "prompt_tokens": self.prompt_tokens,
            "api_calls": self.api_calls,
            "hits": self.hits,
            "hit_rate_pct": self.hit_rate_pct,
            "efficiency": (
                "excellent"
                if self.hit_rate_pct >= 70
                else "good"
                if self.hit_rate_pct >= 40
                else "low"
                if self.api_calls > 0
                else "n/a"
            ),
        }


def metrics_from_agent(agent: Any) -> CacheMetrics:
    """Build metrics from an AIAgent's session accumulators."""
    m = CacheMetrics(
        cache_read_tokens=int(getattr(agent, "session_cache_read_tokens", 0) or 0),
        cache_write_tokens=int(getattr(agent, "session_cache_write_tokens", 0) or 0),
        prompt_tokens=int(getattr(agent, "session_input_tokens", 0) or 0),
        api_calls=int(getattr(agent, "session_api_calls", 0) or 0),
    )
    # Approximate hits from read>0 when we don't track call-level hits
    if m.cache_read_tokens > 0 and m.api_calls == 0:
        m.api_calls = 1
        m.hits = 1
    elif m.cache_read_tokens > 0:
        m.hits = max(1, m.hits)
    return m


def format_cache_status_line(agent: Any) -> str:
    """One-line cache summary for /status."""
    m = metrics_from_agent(agent)
    if m.prompt_tokens <= 0 and m.cache_read_tokens <= 0:
        return "Cache: n/a (no usage yet)"
    return (
        f"Cache: {m.cache_read_tokens:,} read / {m.prompt_tokens:,} prompt "
        f"({m.hit_rate_pct}% hit, {m.cache_write_tokens:,} written) [{m.to_dict()['efficiency']}]"
    )


def cache_efficiency_report(agent: Any = None, **token_kwargs: int) -> Dict[str, Any]:
    """Full report for desk / doctor / tests."""
    if agent is not None:
        m = metrics_from_agent(agent)
    else:
        m = CacheMetrics()
        m.record(
            cache_read=token_kwargs.get("cache_read", 0),
            cache_write=token_kwargs.get("cache_write", 0),
            prompt=token_kwargs.get("prompt", 0),
        )
    data = m.to_dict()
    data["ok"] = True
    data["recommendations"] = []
    if m.api_calls > 2 and m.hit_rate_pct < 20:
        data["recommendations"].append(
            "Low cache hit rate — avoid mid-conversation system/toolset swaps; "
            "prefer /focus and volatile context injection."
        )
    if m.cache_write_tokens > m.cache_read_tokens * 2 and m.api_calls > 1:
        data["recommendations"].append(
            "High cache writes vs reads — session prefix may be unstable; "
            "check compression/focus toggles that rewrite early messages."
        )
    return data
