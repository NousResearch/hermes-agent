"""/context reports when the session's cached prompt prefix expires (port of deepagents#7005)."""

from types import SimpleNamespace

from agent.context_breakdown import render_context_breakdown_lines
from agent.prompt_cache_window import prompt_cache_window, stamp_prompt_cache_activity


def _agent(**kw):
    base = dict(model="claude-sonnet-4-5", provider="anthropic", _use_prompt_caching=True, _cache_ttl="5m")
    return SimpleNamespace(**{**base, **kw})


def test_window_follows_the_last_cache_hit_and_dies_with_the_route(monkeypatch):
    import agent.prompt_cache_window as pcw

    monkeypatch.setattr(pcw.time, "time", lambda: 1_000.0)
    agent = _agent(_api_attempt_started_at=996.0)  # the successful attempt's send time, not the retry loop's
    dest = {"model": agent.model, "provider": agent.provider}
    stamp_prompt_cache_activity(agent, SimpleNamespace(cache_read_tokens=0, cache_write_tokens=0), **dest)
    assert prompt_cache_window(agent) is None  # nothing cached (prompt under the cacheable minimum)

    stamp_prompt_cache_activity(agent, SimpleNamespace(cache_read_tokens=9_000, cache_write_tokens=0), **dest)
    assert prompt_cache_window(agent) == {"ttl": "5m", "ttl_seconds": 300, "refreshed_at": 996.0, "expires_at": 1_296.0}

    # MoA: the TTL is clamped for the aggregator that holds the cache (Qwen on Alibaba caps 1h at 5m).
    moa = _agent(model="my-preset", provider="moa", _cache_ttl="1h", _api_attempt_started_at=996.0)
    stamp_prompt_cache_activity(moa, SimpleNamespace(cache_read_tokens=9_000, cache_write_tokens=0),
                                model="qwen3-max", provider="alibaba")
    assert prompt_cache_window(moa)["ttl"] == "5m"

    agent.model = "gpt-5.5"  # /model switch: a different cache, the old stamp must not carry over
    assert prompt_cache_window(agent) is None
    agent.model, agent._use_prompt_caching = "claude-sonnet-4-5", False  # unpublished retention: no guess
    assert prompt_cache_window(agent) is None


def test_context_view_shows_time_left_then_expiry(monkeypatch):
    import agent.prompt_cache_window as pcw

    payload = {"categories": [], "context_max": 200_000, "context_used": 120_000, "context_percent": 60,
               "context_source": "provider_usage", "estimated_total": 0,
               "prompt_cache": {"ttl": "5m", "ttl_seconds": 300, "refreshed_at": 1_000.0, "expires_at": 1_300.0}}
    monkeypatch.setattr(pcw.time, "time", lambda: 1_048.0)
    assert "Prompt cache: warm, expires in 4m 12s (5m retention)." in render_context_breakdown_lines(payload, grid=False)
    monkeypatch.setattr(pcw.time, "time", lambda: 2_200.0)
    lines = render_context_breakdown_lines(payload, grid=False)
    assert any(line.startswith("Prompt cache: expired 15m ago (5m retention)") for line in lines)
    assert not any("Prompt cache" in line for line in render_context_breakdown_lines({**payload, "prompt_cache": None}, grid=False))
