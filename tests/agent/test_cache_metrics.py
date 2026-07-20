"""Tests for cache metrics + prompt caching layout helpers."""

from __future__ import annotations

from types import SimpleNamespace

from agent.cache_metrics import (
    CacheMetrics,
    cache_efficiency_report,
    format_cache_status_line,
    metrics_from_agent,
)
from agent.prompt_caching import (
    ANTHROPIC_MAX_CACHE_BREAKPOINTS,
    apply_anthropic_cache_control,
    count_cache_markers,
    describe_cache_layout,
)


class TestCacheMetrics:
    def test_hit_ratio(self):
        m = CacheMetrics()
        m.record(cache_read=75, cache_write=10, prompt=100)
        assert m.hit_rate_pct == 75.0
        assert m.to_dict()["efficiency"] == "excellent"

    def test_low_efficiency_recommendation(self):
        report = cache_efficiency_report(
            cache_read=5, cache_write=50, prompt=100
        )
        # Force api_calls path
        m = CacheMetrics()
        for _ in range(3):
            m.record(cache_read=1, cache_write=20, prompt=100)
        data = m.to_dict()
        assert data["hit_rate_pct"] < 20
        full = cache_efficiency_report(
            SimpleNamespace(
                session_cache_read_tokens=m.cache_read_tokens,
                session_cache_write_tokens=m.cache_write_tokens,
                session_input_tokens=m.prompt_tokens,
                session_api_calls=m.api_calls,
            )
        )
        assert full["ok"] is True
        assert any("Low cache" in r for r in full["recommendations"])

    def test_format_status_line(self):
        agent = SimpleNamespace(
            session_cache_read_tokens=40,
            session_cache_write_tokens=5,
            session_input_tokens=100,
            session_api_calls=2,
        )
        line = format_cache_status_line(agent)
        assert "Cache:" in line
        assert "40" in line

    def test_metrics_from_agent_empty(self):
        agent = SimpleNamespace()
        m = metrics_from_agent(agent)
        assert m.cache_read_tokens == 0


class TestPromptCachingLayout:
    def test_describe_layout(self):
        info = describe_cache_layout()
        assert info["ok"] is True
        assert info["max_breakpoints"] == ANTHROPIC_MAX_CACHE_BREAKPOINTS
        assert info["cache_safe_rules"]

    def test_apply_respects_cap(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
        out = apply_anthropic_cache_control(msgs)
        assert count_cache_markers(out) <= ANTHROPIC_MAX_CACHE_BREAKPOINTS

    def test_max_breakpoints_kwarg_capped(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        # Requesting more than Anthropic allows still caps at 4
        out = apply_anthropic_cache_control(msgs, max_breakpoints=99)
        assert count_cache_markers(out) <= 4

    def test_empty_messages(self):
        assert apply_anthropic_cache_control([]) == []
        assert count_cache_markers([]) == 0
