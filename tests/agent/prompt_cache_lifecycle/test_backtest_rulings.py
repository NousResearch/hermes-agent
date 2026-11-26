"""Teknium's rulings after the backtest on his real usage, one invariant each.

1. A keep-alive miss served by another upstream than the one that cached the prefix (Nous serves one
   Claude model from ``chatcmpl-*`` and ``gen-*`` upstreams), or one that still read the stable system
   prefix, does not end warming; real misses end it only on the second in a row.
2. Under ``cache_ttl: auto`` a person's chat sends 5m markers until its 5m entry first expires and 1h
   from that request on, for the rest of the session; the tier never changes while the 5m entry is
   still warm (a 1h request then re-writes the whole prompt; a keep-alive still in flight counts as a
   touch), and idle keep-alives run on the 5m window.
3. The route store learns a downgrade only from samples in two different sessions, never from an
   upstream switch, and never a lifetime shorter than the longest hit it ever saw.

Timers never start: ``_fire`` runs the armed callback inline, so no test sleeps.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
import agent.prompt_cache_routes as routes

_OPUS = SimpleNamespace(input_cost_per_million=5, output_cost_per_million=25,
                        cache_read_cost_per_million=0.5, cache_write_cost_per_million=10)
_NOUS = "https://inference-api.nousresearch.com/v1"


class _Usage:
    def __init__(self, inp, out, read, write):
        self.input_tokens, self.output_tokens = inp, out
        self.cache_read_input_tokens, self.cache_creation_input_tokens = read, write


class _Client:
    """Anthropic-shaped double answering each keep-alive with the next ``(id, read, write)``."""

    def __init__(self, replies):
        self.calls, self.replies = [], list(replies)
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def effective_output_cap(*, max_tokens=None, **_):
        return max_tokens

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        rid, read, write = self.replies.pop(0)
        return SimpleNamespace(id=rid, usage=_Usage(3, 1, read, write), content=[], stop_reason="max_tokens")


class _InertTimer:
    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args, self.daemon, self.name = delay, fn, args, True, ""

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    from collections import OrderedDict

    from agent.prompt_cache_lifecycle import tier
    from agent.usage_pricing import PricingEntry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    entry = PricingEntry(**{k: Decimal(str(v)) for k, v in vars(_OPUS).items()}, source="official_docs_snapshot")
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: entry)
    monkeypatch.setattr(pcl.PromptCacheWarmer, "_pricing", lambda self: _OPUS)
    monkeypatch.setattr(pcl.warmer.threading, "Timer", _InertTimer)
    monkeypatch.setattr(tier, "_UPGRADED", OrderedDict())


def _chat(client=None, **over):
    agent = SimpleNamespace(
        api_mode="anthropic_messages", _use_prompt_caching=True, _cache_disabled=False, _cache_ttl="1h",
        _interrupt_requested=False, _anthropic_client=client, model="claude-opus-4-7",
        provider="anthropic", base_url="https://api.anthropic.com", api_key="", session_api_calls=7,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0, session_input_tokens=0,
        session_output_tokens=0, session_cache_read_tokens=0, session_cache_write_tokens=0, session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0, _session_db=None, session_id="s", log_prefix="", platform="cli",
        compression_enabled=True, _user_turn_count=1, _executing_tools=False,
        context_compressor=SimpleNamespace(
            compression_count=0, last_prompt_tokens=150_000, threshold_tokens=200_000, summary_target_ratio=0.2,
            last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
            get_active_compression_failure_cooldown=lambda: None),
    )
    for key, value in over.items():
        setattr(agent, key, value)
    return agent


def _request(ttl="1h"):
    mark = {"type": "ephemeral", "ttl": ttl} if ttl == "1h" else {"type": "ephemeral"}
    return {"model": "claude-opus-4-7", "max_tokens": 8192, "stream": True,
            "system": [{"type": "text", "text": "You are Hermes. " * 400, "cache_control": mark}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": mark}]}]}


_WRITE = SimpleNamespace(id="chatcmpl-1", usage=_Usage(50, 30, 0, 119_950))


def _respond(agent, request=None, response=_WRITE, at=None):
    from agent.prompt_cache_clock import attempt_facts, record_cache_response, response_upstream
    from agent.usage_pricing import normalize_usage

    request = request or _request()
    pcl.cancel_prompt_cache_warming(agent, "new request in flight")
    agent.session_api_calls += 1
    agent._prompt_cache_attempt = dict(attempt_facts(request), **({"sent_at": at} if at else {}))
    record_cache_response(agent, normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode),
                          model=agent.model, provider=agent.provider, base_url=agent.base_url,
                          api_mode=agent.api_mode, upstream=response_upstream(response))
    return pcl.get_prompt_cache_warmer(agent).start(request, response)


def _fire(warmer):
    run = warmer._run
    assert run is not None and run.timer is not None, warmer.status
    warmer._refresh(run)


# ── 1. Upstream-aware misses ─────────────────────────────────────────────────────────────────────────

def test_1_an_upstream_switch_or_partial_read_is_not_a_miss_and_two_real_misses_end_warming():
    """d4b21395 (a 24 s gap, 406k prefix): the reply came from the other upstream and read nothing. A
    partial read (the system prefix only) is the same blip. Neither ends warming; a real miss is retried
    once, and the second real miss in a row ends warming for the session."""
    client = _Client([
        ("gen-a", 0, 119_950),        # another upstream: nothing read, everything re-written
        ("gen-b", 119_997, 0),        # a hit on the upstream that now holds it
        ("gen-c", 6_000, 113_950),    # same upstream, read the system prefix only
        ("gen-d", 0, 119_950),        # same upstream, nothing read: one real miss, retried
        ("gen-e", 0, 119_950),        # the second real miss in a row
    ])
    agent = _chat(client=client)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    agent._executing_tools = True  # a long tool call: keep-alives run past the idle three
    for sent in range(1, 5):
        _fire(warmer)
        assert not pcl.prompt_cache_warming_disabled(agent), f"warming ended after keep-alive #{sent}"
        assert warmer._run is not None
    _fire(warmer)
    assert len(client.calls) == 5 and warmer._run is None
    assert pcl.prompt_cache_warming_disabled(agent)


# ── 2. Start human chats on 5m, upgrade to 1h at the first idle gap past 5m ─────────────────────────

def _auto_chat(**over):
    from agent.prompt_cache_lifecycle.tier import adaptive_start_tier

    agent = _chat(client=_Client([("chatcmpl-k", 119_997, 0)] * 10), _cache_ttl_auto=True, **over)
    agent._cache_ttl = adaptive_start_tier(agent, "1h")
    return agent


def _markers(agent):
    from agent.prompt_caching import apply_anthropic_cache_control

    msgs = apply_anthropic_cache_control([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
                                         cache_ttl=agent._cache_ttl)
    return {part["cache_control"].get("ttl", "5m") for m in msgs for part in m["content"]
            if isinstance(part, dict) and "cache_control" in part}


def test_2_a_human_auto_chat_sends_5m_until_its_first_long_pause_then_1h_for_good(monkeypatch):
    from agent.prompt_cache_lifecycle.tier import carry_cache_tier, maybe_upgrade_cache_tier

    import time

    t0 = time.time()  # the warmer arms on the real clock
    agent = _auto_chat()
    assert _markers(agent) == {"5m"}
    assert _respond(agent, _request("5m"), at=t0) is None
    pcl.end_turn_prompt_cache_warming(agent)
    run = pcl.get_prompt_cache_warmer(agent)._run
    assert run is not None and run.ttl_s == 300, "idle keep-alives run on the 5m window before the upgrade"

    assert not maybe_upgrade_cache_tier(agent, now=t0 + 240)  # a quick reply: the 5m entry is still warm
    assert _markers(agent) == {"5m"}
    _respond(agent, _request("5m"), at=t0 + 240)
    assert not maybe_upgrade_cache_tier(agent, now=t0 + 240 + 299), "the TTL changed while the 5m entry was warm"
    warmer = pcl.get_prompt_cache_warmer(agent)
    warmer.last_sent_at = t0 + 240 + 250  # a keep-alive whose reply is still due
    assert not maybe_upgrade_cache_tier(agent, now=t0 + 240 + 301), "the TTL changed under an in-flight refresh"

    def _unreachable(_run):
        raise ConnectionError("refresh never reached the provider")
    monkeypatch.setattr(warmer, "_send", _unreachable)
    with monkeypatch.context() as at_send:
        at_send.setattr(time, "time", lambda: t0 + 240 + 250)
        _fire(warmer)  # that keep-alive fails: it renewed nothing, so it stops counting as a touch

    assert maybe_upgrade_cache_tier(agent, now=t0 + 240 + 301)  # the first pause past five minutes
    assert _markers(agent) == {"1h"}
    assert _respond(agent, _request("1h"), at=t0 + 600) is None
    pcl.end_turn_prompt_cache_warming(agent)
    run = pcl.get_prompt_cache_warmer(agent)._run
    assert run is not None and run.ttl_s == 3600, "idle keep-alives resume once on 1h, on the 1h window"
    assert maybe_upgrade_cache_tier(agent, now=t0 + 610) and _markers(agent) == {"1h"}  # stays 1h

    rebuilt = _auto_chat()  # the gateway rebuilt the agent for the same session
    assert maybe_upgrade_cache_tier(rebuilt, now=t0 + 620) and rebuilt._cache_ttl == "1h"
    carry_cache_tier("s", "s-child")  # a compaction moved the chat to a child session
    assert maybe_upgrade_cache_tier(_auto_chat(session_id="s-child"), now=t0 + 630)

    never_paused = _auto_chat(session_id="busy")
    for i in range(20):  # back-to-back turns, never 5 minutes apart
        maybe_upgrade_cache_tier(never_paused, now=t0 + 60 * i + 1)
        _respond(never_paused, _request("5m"), at=t0 + 60 * i)
    assert never_paused._cache_ttl == "5m"


# ── 3. Learning-store guards ─────────────────────────────────────────────────────────────────────────

def _obs(model, *, read=0, write=0, gap, session, tier=3600.0):
    routes.observe(provider="nous", base_url=_NOUS, model=model, explicit=True, read=read, write=write,
                   gap_s=gap, tier_s=tier, now=1_000_000.0, session_id=session)


def test_3_a_downgrade_needs_two_sessions_no_upstream_switch_and_never_undercuts_the_longest_hit(monkeypatch):
    from agent.prompt_cache_clock import record_cache_response
    from agent.usage_pricing import CanonicalUsage

    monkeypatch.setattr(routes.time, "time", lambda: 1_000_000.0)
    rule = lambda m: routes.rule_for("nous", _NOUS, m, True, tier_s=3600)

    _obs("claude-a", read=20_000, gap=10, session="one")
    for gap in (40, 46):  # one chat's routing streak
        _obs("claude-a", gap=gap, session="one")
    assert rule("claude-a") is None, "two misses from ONE session downgraded the route"
    _obs("claude-a", gap=50, session="two")
    assert rule("claude-a") == "short-lived"

    _obs("claude-b", read=20_000, gap=1_800, session="one")  # a hit after a 30-minute gap
    for i in range(10):  # later hits that re-wrote a big tail: the recent hit gaps no longer show 1_800
        _obs("claude-b", read=20_000, write=15_000, gap=5, session=f"h{i}")
    for session in ("x", "y"):
        _obs("claude-b", gap=600, session=session)
    assert rule("claude-b") is None, "learned a lifetime shorter than a hit it saw"

    agent = SimpleNamespace(provider="nous", base_url=_NOUS, model="claude-c", api_mode="chat_completions",
                            session_id="s", _session_db=None, _prompt_cache_clock_restored=True,
                            context_compressor=SimpleNamespace(compression_count=0))
    for session in ("p", "q"):  # each chat: cached on one upstream, the next reply from the other
        agent.session_id, agent._prompt_cache_clock = session, None
        for at, upstream in ((1_000_000.0, "chatcmpl"), (1_000_005.0, "gen")):
            agent._prompt_cache_attempt = {"sent_at": at, "marker_ttl": "1h"}
            record_cache_response(agent, CanonicalUsage(input_tokens=3, cache_write_tokens=6013), model=agent.model,
                                  provider="nous", base_url=_NOUS, api_mode="chat_completions", upstream=upstream)
    assert rule("claude-c") is None, "an upstream switch was learned as a miss"
