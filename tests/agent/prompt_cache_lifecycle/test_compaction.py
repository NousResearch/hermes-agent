"""Step 2 of the prompt-cache lifecycle: once its warm budget is over, an idle chat compacts on any
proven or declared window (warm or not: the summary is a fresh request), never for a context under
half the compaction threshold or a machine-paced session, and not again until it grew; an ``auto``
chat compacted on 5m moves to 1h; a 1h cache write is billed at the 1h rate."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
from agent.prompt_cache_lifecycle.tier import maybe_upgrade_cache_tier
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
        context_compressor=SimpleNamespace(compression_count=0, last_prompt_tokens=tokens, threshold_tokens=200_000,
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


def _spend_budget(agent, *, due_at=_SENT + 54 * 60):
    """The warm budget ran out with the entry still warm (what the controller records)."""
    from agent.prompt_cache_clock import cache_window, context_witness
    from agent.prompt_caching import TTL_SECONDS

    window = cache_window(agent)
    warmer = pcl.get_prompt_cache_warmer(agent)
    warmer._spent = pcl.edge._Spent(due_at, TTL_SECONDS.get(window.tier or "", 0), context_witness(agent))
    return agent


def cache_expiry_compaction_due(agent, *, now):
    reason = pcl.get_prompt_cache_warmer(agent).edge_decision(now=now).compact
    return reason == "due", reason


def test_a_spent_budget_compacts_once_per_idle_stretch(monkeypatch):
    agent = _idle_chat(monkeypatch)
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (False, "budget_not_spent")
    _spend_budget(agent)
    assert cache_expiry_compaction_due(agent, now=_SENT + 53 * 60) == (False, "not_yet")
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (True, "due")
    # Past the deadline still due: the summary never replays the cached prefix.
    assert cache_expiry_compaction_due(agent, now=_SENT + 61 * 60) == (True, "due")
    # Claimed once: an attempt that committed nothing is not retried in the same idle stretch.
    assert pcl.get_prompt_cache_warmer(agent).claim_edge(now=_SENT + 54 * 60) == "compact"
    assert cache_expiry_compaction_due(agent, now=_SENT + 55 * 60) == (False, "compacted_this_stretch")


@pytest.mark.parametrize("case, reason", [
    ({"tokens": 99_000}, "small_context"),  # under half of the 200k threshold
    ({"platform": "cron"}, "machine_paced"),
])
def test_never_fires_on_small_contexts_or_machine_paced_sessions(monkeypatch, case, reason):
    agent = _spend_budget(_idle_chat(monkeypatch, **case))
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (False, reason)


def test_a_5m_window_compacts_when_its_keep_alives_end_and_the_claim_keeps_the_tier(tmp_path, monkeypatch):
    """Nous Portal keeps 1h like Anthropic; a ``{nous: 5m}`` ceiling (or a 5m tier) compacts too once its
    keep-alives stop covering the pause. Claiming that compaction does not move an ``auto`` chat to 1h:
    the next message does, once its 5m entry is gone."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    nous = {"provider": "nous", "base_url": "https://inference-api.nousresearch.com/v1"}
    agent = _spend_budget(_idle_chat(monkeypatch, **nous))
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (True, "due")

    (tmp_path / "config.yaml").write_text("prompt_caching:\n  route_ttl_ceilings:\n    nous: 5m\n", encoding="utf-8")
    agent = _spend_budget(_idle_chat(monkeypatch, **nous))
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (True, "due")

    five = {"ttl": "5m", "raw": {"input_tokens": 5, "output_tokens": 1, "cache_creation_input_tokens": 150_000}}
    agent = _spend_budget(_idle_chat(monkeypatch, **five), due_at=_SENT + 280)
    agent._cache_ttl_adaptive = agent._cache_ttl_auto = True
    assert pcl.get_prompt_cache_warmer(agent).claim_edge(now=_SENT + 280) == "compact"
    # The claim decides nothing about the tier: a compaction the owner then skips leaves the entry warm,
    # and a return inside it must still read it on 5m.
    assert agent._cache_ttl == "5m"
    assert not maybe_upgrade_cache_tier(agent, now=_SENT + 290) and agent._cache_ttl == "5m"


def test_one_hour_cache_write_is_billed_at_twice_input():
    split = CanonicalUsage(cache_write_tokens=1_000_000, cache_write_by_ttl=(("1h", 1_000_000),))
    five_minute = CanonicalUsage(cache_write_tokens=1_000_000)
    one_hour = estimate_usage_cost("claude-sonnet-4-6", split, provider="anthropic").amount_usd
    default = estimate_usage_cost("claude-sonnet-4-6", five_minute, provider="anthropic").amount_usd
    assert (one_hour, default) == (Decimal("6.00"), Decimal("3.75"))  # $3/M input: 2x vs 1.25x


def test_a_chat_still_large_after_compacting_waits_for_new_content(monkeypatch):
    """A compaction that left 140k (protected history) is not re-run each idle hour on the same 140k."""
    agent = _spend_budget(_idle_chat(monkeypatch))
    agent._session_messages = [{"role": "user", "content": "short follow-up"}]
    agent.context_compressor.last_compression_rough_tokens = 140_000
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (False, "no_growth")
    agent._session_messages = [{"role": "user", "content": "x" * 4 * 250_000}]  # a floor's worth more
    assert cache_expiry_compaction_due(agent, now=_SENT + 54 * 60) == (True, "due")
