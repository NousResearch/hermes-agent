"""The prompt-cache clock: one shared answer to "until when is this session's cached prompt warm?".

Behaviour contracts per route: what was sent on the wire, what the response proved, how coverage
deadlines move on a read (Anthropic restarts the tier; OpenAI keeps max(existing, read + 30m) and
never restarts 12h), the guaranteed vs estimated ends, and how the window dies on a route switch
or compaction.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.prompt_cache_clock import (
    cache_certainly_cold, cache_certainly_warm, cache_window, record_attempt_sent, record_cache_response,
    render_prompt_cache_line,
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


def _openai_request(ttl):
    return {"input": [], "extra_body": {"prompt_cache_options": {"ttl": ttl}}}


def _turn(agent, request, raw_usage, *, sent_at, monkeypatch):
    import agent.prompt_cache_clock as clock

    monkeypatch.setattr(clock.time, "time", lambda: sent_at)
    record_attempt_sent(agent, request)
    usage = normalize_usage(raw_usage, provider=agent.provider, api_mode=agent.api_mode)
    return record_cache_response(agent, usage, model=agent.model, provider=agent.provider,
                                 base_url=agent.base_url, api_mode=agent.api_mode)


def _ends(agent):
    window = cache_window(agent)
    return window.tier, window.confidence, window.guaranteed_until, window.estimated_until


def test_anthropic_split_proves_the_tier_and_a_read_restarts_it(monkeypatch):
    agent = _agent("anthropic", "https://api.anthropic.com", "claude-sonnet-4-6", api_mode="anthropic_messages")
    write = {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 4096,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4096}}
    _turn(agent, _marked_request("1h"), write, sent_at=1_000.0, monkeypatch=monkeypatch)
    assert _ends(agent) == ("1h", "proven", 4_600.0, 4_600.0)  # Anthropic tiers are exact: both ends agree

    read = {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 4096}
    _turn(agent, _marked_request("1h"), read, sent_at=2_000.0, monkeypatch=monkeypatch)
    assert _ends(agent) == ("1h", "proven", 5_600.0, 5_600.0)

    # The server granted 5m although 1h was asked for: the clock believes the response.
    downgraded = {**write, "cache_creation": {"ephemeral_5m_input_tokens": 4096, "ephemeral_1h_input_tokens": 0}}
    _turn(agent, _marked_request("1h"), downgraded, sent_at=3_000.0, monkeypatch=monkeypatch)
    assert _ends(agent)[0] == "5m" and _ends(agent)[2] == 3_300.0


def test_mixed_tiers_in_one_response_guarantee_only_the_shorter(monkeypatch):
    agent = _agent("anthropic", "https://api.anthropic.com", "claude-sonnet-4-6", api_mode="anthropic_messages")
    mixed = {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 6000,
             "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 4000}}
    _turn(agent, _marked_request("1h"), mixed, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert window.coverage == {"5m": 1_300.0, "1h": 4_600.0}  # both recorded
    assert (window.tier, window.guaranteed_until, window.estimated_until) == ("5m", 1_300.0, 4_600.0)


def test_openai_12h_deadline_is_fixed_at_write_and_a_read_only_extends_the_30m_floor(monkeypatch):
    """OpenAI: a 30m read "preserves the later of the existing expiry and 30 minutes after the read.
    It does not restart a 12-hour clock"."""
    agent = _agent("openai-api", "https://api.openai.com/v1", "gpt-5.6-luna", api_mode="codex_responses")
    write = {"input_tokens": 4000, "output_tokens": 2, "input_tokens_details": {
        "cached_tokens": 0, "cache_write_tokens": 3640,
        "cache_write_tokens_details": {"ttl_30m_tokens": 0, "ttl_12h_tokens": 3640}}}
    _turn(agent, _openai_request("12h"), write, sent_at=1_000.0, monkeypatch=monkeypatch)
    assert _ends(agent)[:3] == ("12h", "proven", 44_200.0)

    read = {"input_tokens": 4000, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 3640}}
    _turn(agent, _openai_request("12h"), read, sent_at=2_000.0, monkeypatch=monkeypatch)
    assert _ends(agent)[:3] == ("12h", "proven", 44_200.0)
    _turn(agent, _openai_request("12h"), read, sent_at=44_000.0, monkeypatch=monkeypatch)
    assert _ends(agent)[2] == 45_800.0  # near the 12h deadline the read's 30 minutes are the later one


def test_openai_minimum_is_not_an_expiry(monkeypatch):
    """GPT-5.6+ 30m is a documented MINIMUM ("may retain entries longer"; 24h ceiling): warm before it,
    may still be warm after it, certainly cold only past the retention ceiling."""
    agent = _agent("openai-api", "https://api.openai.com/v1", "gpt-5.6-luna", api_mode="codex_responses")
    write = {"input_tokens": 4000, "output_tokens": 2, "input_tokens_details": {
        "cached_tokens": 0, "cache_write_tokens": 3640, "cache_write_tokens_details": {"ttl_30m_tokens": 3640}}}
    _turn(agent, {"input": []}, write, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.guaranteed_until, window.estimated_until) == (2_800.0, 87_400.0)
    assert cache_certainly_warm(window, now=2_000.0) and not cache_certainly_cold(window, now=2_000.0)
    assert not cache_certainly_warm(window, now=3_000.0) and not cache_certainly_cold(window, now=3_000.0)
    assert cache_certainly_cold(window, now=87_400.0)
    assert render_prompt_cache_line(dict(window.__dict__), now=3_000.0).startswith("Prompt cache: may still be warm until")

    # Pre-5.6 retention: no guarantee at all, only a ceiling.
    old = _agent("openai-api", "https://api.openai.com/v1", "gpt-5.4-nano", api_mode="codex_responses")
    _turn(old, {"input": []}, {"input_tokens": 4000, "output_tokens": 2,
                               "input_tokens_details": {"cached_tokens": 2816}}, sent_at=1_000.0, monkeypatch=monkeypatch)
    assert _ends(old)[1:] == ("assumed", None, 87_400.0)
    assert not cache_certainly_warm(cache_window(old), now=1_001.0)


def test_the_cache_moves_with_the_full_route_and_dies_at_compaction(monkeypatch):
    agent = _agent("custom", "http://10.0.0.5:8000/v1", "anthropic/claude-sonnet-4.6")
    usage = {"prompt_tokens": 5000, "completion_tokens": 3,
             "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4800}}
    _turn(agent, _marked_request("5m"), usage, sent_at=1_000.0, monkeypatch=monkeypatch)
    assert _ends(agent)[:2] == ("5m", "declared")

    agent.context_compressor.compression_count = 1
    assert (cache_window(agent).confidence, cache_window(agent).reason) == ("unknown", "compacted")
    agent.context_compressor.compression_count = 0
    for moved in ("http://10.0.0.5:8001/v1", "http://10.0.0.5:8000/other/v1", "https://10.0.0.5:8000/v1"):
        agent.base_url = moved  # same host, different server: a different cache
        assert cache_window(agent).reason == "route_changed", moved
    agent.base_url = "http://10.0.0.5:8000/v1/"  # a trailing slash is the same endpoint
    assert cache_window(agent).confidence == "declared"

    # A response with no cache activity leaves the previous window standing.
    _turn(agent, _marked_request("5m"), {"prompt_tokens": 10, "completion_tokens": 1}, sent_at=1_200.0,
          monkeypatch=monkeypatch)
    assert _ends(agent)[2] == 1_300.0


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
    assert _ends(agent) == ("5m", "declared", 1_300.0, 1_300.0)
    assert cache_window(agent).reason == "route_ceiling_5m"

    (tmp_path / "config.yaml").write_text("prompt_caching:\n  route_ttl_ceilings:\n    nous: 1h\n", encoding="utf-8")
    assert effective_cache_ttl("1h", provider="nous", model="anthropic/claude-sonnet-4.6", base_url=portal) == "1h"


def test_a_configured_ceiling_applies_on_the_measured_provider_path_too(tmp_path, monkeypatch):
    """opencode-go returns early as a measured 1h route; a configured 5m ceiling must still win there."""
    from agent.prompt_caching import effective_cache_ttl

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert effective_cache_ttl("1h", provider="opencode-go", model="qwen3.6-plus") == "1h"
    (tmp_path / "config.yaml").write_text("prompt_caching:\n  route_ttl_ceilings:\n    opencode-go: 5m\n",
                                          encoding="utf-8")
    assert effective_cache_ttl("1h", provider="opencode-go", model="qwen3.6-plus") == "5m"


def test_moa_records_the_resolved_aggregator_route_and_payload(monkeypatch):
    """The MoA agent shows the virtual preset; the clock must use the aggregator's resolved route
    (OpenRouter Claude: 1h declared) and the payload the MoA client really sent."""
    agent = _agent("moa", "moa://local", "closed")
    agent.client = SimpleNamespace(
        last_aggregator_route={"provider": "openrouter", "model": "anthropic/claude-opus-4.8",
                               "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"},
        last_aggregator_attempt={"sent_at": 1_000.0, "marker_ttl": "1h", "openai_ttl": None, "retention": None})
    usage = normalize_usage({"prompt_tokens": 5000, "completion_tokens": 3,
                             "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4800}},
                            provider="openrouter", api_mode="chat_completions")
    event = record_cache_response(agent, usage, model="closed", provider="moa", base_url="moa://local",
                                  api_mode="chat_completions")
    assert (event.route_class, event.tier, event.confidence) == ("openrouter_claude", "1h", "declared")
    assert _ends(agent)[2] == 4_600.0


def test_implicit_caches_report_unknown_rather_than_a_guess(monkeypatch):
    agent = _agent("deepseek", "https://api.deepseek.com/v1", "deepseek-v4")
    agent._use_prompt_caching = False
    usage = {"prompt_tokens": 4000, "completion_tokens": 2, "prompt_cache_hit_tokens": 3000}
    _turn(agent, {"messages": [{"role": "user", "content": "hi"}]}, usage, sent_at=1_000.0, monkeypatch=monkeypatch)
    window = cache_window(agent)
    assert (window.confidence, window.estimated_until) == ("unknown", None)
    assert not cache_certainly_warm(window) and not cache_certainly_cold(window)
    assert render_prompt_cache_line({"estimated_until": None, "confidence": "unknown", "reason": "implicit_cache"}) == (
        "Prompt cache: unknown (this provider does not publish its cache retention).")


@pytest.mark.parametrize(("now", "expected"), [
    (1_048.0, "Prompt cache: warm until 01:16 (59m 12s left; 1h, proven)."),
    (5_500.0, "Prompt cache: expired 15m ago (1h, proven); the next message re-processes the full prompt."),
])
def test_context_line_shows_the_wall_clock_expiry_then_how_long_ago(now, expected, monkeypatch):
    import agent.prompt_cache_clock as clock
    from agent.context_breakdown import render_context_breakdown_lines

    monkeypatch.setattr(clock.time, "localtime", lambda ts: clock.time.gmtime(ts))
    window = {"guaranteed_until": 4_600.0, "estimated_until": 4_600.0, "tier": "1h", "confidence": "proven",
              "reason": "anthropic_usage_split", "refreshed_at": 1_000.0, "coverage": {"1h": 4_600.0}}
    payload = {"categories": [], "context_max": 200_000, "context_used": 1_000, "context_percent": 1,
               "context_source": "provider_usage", "estimated_total": 0, "prompt_cache": window}
    monkeypatch.setattr(clock.time, "time", lambda: now)
    assert expected in render_context_breakdown_lines(payload, grid=False)
    assert not any("Prompt cache" in line for line in render_context_breakdown_lines({**payload, "prompt_cache": None},
                                                                                     grid=False))


def test_a_1h_write_over_an_older_5m_prefix_is_priced_at_the_1h_rate(monkeypatch):
    """The clock's tier is the earliest deadline over the whole prefix (5m here); the new writes of a
    1h OpenRouter Claude request still went to 1h and bill at 2x input, not 1.25x."""
    from agent.turn_usage import _with_declared_write_tier
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    agent = _agent("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6")
    agent._prompt_cache_clock_restored = True
    for at, ttl, usage in ((1_000.0, "5m", CanonicalUsage(cache_write_tokens=100_000)),
                           (1_060.0, "1h", CanonicalUsage(cache_read_tokens=100_000, cache_write_tokens=1_000_000))):
        agent._prompt_cache_attempt = {"sent_at": at, "marker_ttl": ttl}
        event = record_cache_response(agent, usage, model=agent.model, provider=agent.provider,
                                      base_url=agent.base_url, api_mode=agent.api_mode)
    assert event.tier == "5m"  # the shorter, older prefix still bounds the window
    priced = _with_declared_write_tier(usage, event)
    assert priced.cache_write_by_ttl == (("1h", 1_000_000),)
    # Priced on the native Anthropic table ($3/M input) to keep the test offline: 2x vs 1.25x input.
    as_1h = estimate_usage_cost("claude-sonnet-4-6", priced, provider="anthropic").amount_usd
    as_5m = estimate_usage_cost("claude-sonnet-4-6", usage, provider="anthropic").amount_usd
    assert as_1h - as_5m == Decimal("2.25")
