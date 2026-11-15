"""The prompt-cache clock: one shared answer to "when does this session's cached prompt expire?".

Behaviour contracts per route: what was sent on the wire, what the response proved, and how the
window survives (or does not survive) reads, route switches, compaction and a restart.
"""

from types import SimpleNamespace

import pytest

from agent.prompt_cache_clock import (
    cache_window, record_attempt_sent, record_cache_response, render_prompt_cache_line,
)
from agent.usage_pricing import normalize_usage


def _agent(provider, base_url, model, *, api_mode="chat_completions", cache_ttl="1h", session_id="s1"):
    return SimpleNamespace(
        provider=provider, base_url=base_url, model=model, api_mode=api_mode, session_id=session_id,
        _cache_ttl=cache_ttl, _use_prompt_caching=True, _session_db=None,
        context_compressor=SimpleNamespace(compression_count=0),
    )


def _marked_request(ttl):
    marker = {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}
    return {"messages": [{"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": marker}]},
                         {"role": "user", "content": "hi"}]}


def _turn(agent, request, raw_usage, *, sent_at, monkeypatch):
    import agent.prompt_cache_clock as clock

    monkeypatch.setattr(clock.time, "time", lambda: sent_at)
    record_attempt_sent(agent, request)
    usage = normalize_usage(raw_usage, provider=agent.provider, api_mode=agent.api_mode)
    return record_cache_response(agent, usage, model=agent.model, provider=agent.provider,
                                 base_url=agent.base_url, api_mode=agent.api_mode)


def test_anthropic_split_proves_the_tier_and_a_read_refreshes_it(monkeypatch):
    agent = _agent("anthropic", "https://api.anthropic.com", "claude-sonnet-4-6", api_mode="anthropic_messages")
    write = {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 4096,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4096}}
    _turn(agent, _marked_request("1h"), write, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.tier, window.confidence, window.expires_at) == ("1h", "proven", 4_600.0)

    # A pure read refreshes the entry with the tier it was WRITTEN at, still proven.
    read = {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 4096}
    _turn(agent, _marked_request("1h"), read, sent_at=2_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.tier, window.confidence, window.expires_at) == ("1h", "proven", 5_600.0)

    # The server granted 5m although 1h was asked for: the clock believes the response.
    downgraded = {**write, "cache_creation": {"ephemeral_5m_input_tokens": 4096, "ephemeral_1h_input_tokens": 0}}
    _turn(agent, _marked_request("1h"), downgraded, sent_at=3_000.0, monkeypatch=monkeypatch)
    assert (cache_window(agent).tier, cache_window(agent).expires_at) == ("5m", 3_300.0)


def test_the_cache_moves_with_the_route_and_dies_at_compaction(monkeypatch):
    agent = _agent("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6")
    usage = {"prompt_tokens": 5000, "completion_tokens": 3,
             "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4800}}
    _turn(agent, _marked_request("1h"), usage, sent_at=1_000.0, monkeypatch=monkeypatch)
    assert (cache_window(agent).tier, cache_window(agent).confidence) == ("1h", "declared")

    agent.context_compressor.compression_count = 1
    assert (cache_window(agent).confidence, cache_window(agent).reason) == ("unknown", "compacted")
    agent.context_compressor.compression_count = 0
    agent.model = "openai/gpt-5.6-luna"  # /model switch or fallback: a different cache
    assert (cache_window(agent).confidence, cache_window(agent).reason) == ("unknown", "route_changed")

    # A response with no cache activity leaves the previous window standing.
    agent.model = "anthropic/claude-sonnet-4.6"
    _turn(agent, _marked_request("1h"), {"prompt_tokens": 10, "completion_tokens": 1}, sent_at=1_500.0,
          monkeypatch=monkeypatch)
    assert cache_window(agent).expires_at == 4_600.0


def test_nous_ceiling_is_data_on_the_wire_and_in_the_clock(tmp_path, monkeypatch):
    """Nous Portal drops 1h upstream: Hermes asks for 5m there and the clock says 5m. Lifting the
    ceiling is a config entry, not a code change, and a ``custom`` route at the Portal host counts."""
    from agent.prompt_caching import effective_cache_ttl

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    portal = "https://inference-api.nousresearch.com/v1"
    for provider in ("nous", "custom"):
        assert effective_cache_ttl("1h", provider=provider, model="anthropic/claude-sonnet-4.6", base_url=portal) == "5m"
    assert effective_cache_ttl("1h", provider="openrouter", model="anthropic/claude-sonnet-4.6",
                               base_url="https://openrouter.ai/api/v1") == "1h"

    agent = _agent("nous", portal, "anthropic/claude-sonnet-4.6")
    usage = {"prompt_tokens": 5000, "completion_tokens": 3,
             "prompt_tokens_details": {"cached_tokens": 4800, "cache_write_tokens": 0}}
    _turn(agent, _marked_request("5m"), usage, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.tier, window.confidence, window.expires_at, window.reason) == ("5m", "declared", 1_300.0,
                                                                                 "route_ceiling_5m")

    (tmp_path / "config.yaml").write_text("prompt_caching:\n  route_ttl_ceilings:\n    nous: 1h\n", encoding="utf-8")
    assert effective_cache_ttl("1h", provider="nous", model="anthropic/claude-sonnet-4.6", base_url=portal) == "1h"


@pytest.mark.parametrize(("model", "details", "expected"), [
    ("gpt-5.6-luna", {"cached_tokens": 0, "cache_write_tokens": 3640,
                      "cache_write_tokens_details": {"ttl_30m_tokens": 3640, "ttl_12h_tokens": 0}},
     ("30m", "proven", 2_800.0)),
    ("gpt-5.6-luna", {"cached_tokens": 3640, "cache_write_tokens": 0}, ("30m", "declared", 2_800.0)),
    ("gpt-5.4-nano", {"cached_tokens": 2816, "cache_write_tokens": 0}, ("30m", "assumed", 2_800.0)),
])
def test_openai_tier_follows_the_model_family_and_the_write_details(model, details, expected, monkeypatch):
    agent = _agent("openai-api", "https://api.openai.com/v1", model, api_mode="codex_responses")
    usage = {"input_tokens": 4000, "output_tokens": 2, "input_tokens_details": details}
    _turn(agent, {"input": []}, usage, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.tier, window.confidence, window.expires_at) == expected


def test_implicit_caches_report_unknown_rather_than_a_guess(monkeypatch):
    agent = _agent("deepseek", "https://api.deepseek.com/v1", "deepseek-v4")
    agent._use_prompt_caching = False
    usage = {"prompt_tokens": 4000, "completion_tokens": 2, "prompt_cache_hit_tokens": 3000}
    _turn(agent, {"messages": [{"role": "user", "content": "hi"}]}, usage, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.confidence, window.expires_at) == ("unknown", None)
    assert render_prompt_cache_line({"expires_at": None, "confidence": "unknown", "reason": "implicit_cache"}) == (
        "Prompt cache: unknown (this provider does not publish its cache retention).")


def test_context_line_shows_the_wall_clock_expiry_then_how_long_ago(monkeypatch):
    import agent.prompt_cache_clock as clock
    from agent.context_breakdown import render_context_breakdown_lines

    monkeypatch.setattr(clock.time, "localtime", lambda ts: clock.time.gmtime(ts))
    window = {"expires_at": 4_600.0, "floor_at": 4_600.0, "tier": "1h", "confidence": "proven",
              "reason": "anthropic_usage_split", "refreshed_at": 1_000.0}
    payload = {"categories": [], "context_max": 200_000, "context_used": 1_000, "context_percent": 1,
               "context_source": "provider_usage", "estimated_total": 0, "prompt_cache": window}
    monkeypatch.setattr(clock.time, "time", lambda: 1_048.0)
    assert "Prompt cache: warm until 01:16 (59m 12s left; 1h, proven)." in render_context_breakdown_lines(payload, grid=False)
    monkeypatch.setattr(clock.time, "time", lambda: 5_500.0)
    assert any(line.startswith("Prompt cache: expired 15m ago (1h, proven)")
               for line in render_context_breakdown_lines(payload, grid=False))
    assert not any("Prompt cache" in line for line in render_context_breakdown_lines({**payload, "prompt_cache": None},
                                                                                     grid=False))
