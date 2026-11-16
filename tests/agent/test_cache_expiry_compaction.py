"""Compaction before the prompt cache expires: an idle chat compacts while its 1h cache is still
warm (at 0.9 of the window), never on a 5m window, never for a small context or a machine-paced
session; and a 1h cache write is billed at the 1h rate."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.cache_expiry_compaction import cache_expiry_compaction_due, claim_cache_expiry_compaction
from agent.prompt_cache_clock import record_attempt_sent, record_cache_response
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

_SENT = 1_000_000.0
_ONE_HOUR_WRITE = {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 150_000,
                   "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 150_000}}


@pytest.fixture(autouse=True)
def _interactive(monkeypatch):
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)


def _idle_chat(monkeypatch, *, provider="anthropic", base_url="https://api.anthropic.com", ttl="1h",
               tokens=150_000, platform="cli", raw=None):
    import agent.prompt_cache_clock as clock

    agent = SimpleNamespace(
        provider=provider, base_url=base_url, model="claude-sonnet-4-6", api_mode="anthropic_messages",
        session_id="s1", platform=platform, _cache_ttl=ttl, _use_prompt_caching=True, _session_db=None,
        compression_enabled=True, _last_activity_ts=_SENT, _delegate_depth=0,
        context_compressor=SimpleNamespace(compression_count=0, last_prompt_tokens=tokens, threshold_tokens=500_000,
                                           summary_target_ratio=0.2, awaiting_real_usage_after_compression=False,
                                           get_active_compression_failure_cooldown=lambda: None),
    )
    marker = {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}
    monkeypatch.setattr(clock.time, "time", lambda: _SENT)
    record_attempt_sent(agent, {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": marker}]},
        {"role": "user", "content": "hi"}]})
    usage = normalize_usage(raw or _ONE_HOUR_WRITE, provider=provider, api_mode=agent.api_mode)
    record_cache_response(agent, usage, model=agent.model, provider=provider, base_url=base_url,
                          api_mode=agent.api_mode)
    return agent


def test_idle_chat_compacts_at_ninety_percent_of_a_warm_one_hour_window(monkeypatch):
    agent = _idle_chat(monkeypatch)
    assert cache_expiry_compaction_due(agent, now=_SENT + 30 * 60) == (False, "recently_active")
    assert cache_expiry_compaction_due(agent, now=_SENT + 50 * 60) == (False, "not_yet")
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (True, "due")
    # Once claimed, an attempt that committed nothing is not retried every tick in the same window.
    assert claim_cache_expiry_compaction(agent, now=_SENT + 54 * 60) is True
    assert cache_expiry_compaction_due(agent, now=_SENT + 55 * 60) == (False, "attempted")
    # Past the deadline the cache is already cold: compacting now saves nothing.
    assert cache_expiry_compaction_due(agent, now=_SENT + 61 * 60)[0] is False


@pytest.mark.parametrize("case, reason", [
    ({"provider": "nous", "base_url": "https://inference-api.nousresearch.com/v1"}, "short_window:5m"),
    ({"ttl": "5m", "raw": {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 150_000}},
     "short_window:5m"),
    ({"tokens": 40_000}, "small_context"),
    ({"platform": "cron"}, "machine_paced"),
])
def test_never_fires_on_short_windows_small_contexts_or_machine_paced_sessions(monkeypatch, case, reason):
    agent = _idle_chat(monkeypatch, **case)
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (False, reason)


def test_one_hour_cache_write_is_billed_at_twice_input():
    split = CanonicalUsage(cache_write_tokens=1_000_000, cache_write_by_ttl=(("1h", 1_000_000),))
    five_minute = CanonicalUsage(cache_write_tokens=1_000_000)
    one_hour = estimate_usage_cost("claude-sonnet-4-6", split, provider="anthropic").amount_usd
    default = estimate_usage_cost("claude-sonnet-4-6", five_minute, provider="anthropic").amount_usd
    assert (one_hour, default) == (Decimal("6.00"), Decimal("3.75"))  # $3/M input: 2x vs 1.25x


def test_a_chat_still_large_after_compacting_waits_for_new_content(monkeypatch):
    """A compaction that left 140k (protected history) is not re-run each idle hour on the same 140k."""
    agent = _idle_chat(monkeypatch)
    agent._session_messages = [{"role": "user", "content": "short follow-up"}]
    agent.context_compressor.last_compression_rough_tokens = 140_000
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (False, "no_growth")
    agent._session_messages = [{"role": "user", "content": "x" * 4 * 250_000}]  # a floor's worth more
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (True, "due")
