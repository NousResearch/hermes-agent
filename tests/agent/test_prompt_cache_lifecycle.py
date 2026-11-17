"""The prompt-cache lifecycle loop (agent/prompt_cache_lifecycle.py): an idle human chat warms its
cache up to three times per user message, then compacts (on a window of 30 minutes or more) and sends
nothing more until the next user message, or expires and offloads. A miss ends warming at once and
goes straight to that decision; a machine session never warms idle; an unknown clock makes nothing act.

Timers never start: ``_fire`` runs the armed callback inline, so no test sleeps."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl

_OPUS = SimpleNamespace(input_cost_per_million=5, output_cost_per_million=25,
                        cache_read_cost_per_million=0.5, cache_write_cost_per_million=10)


class _Usage:
    def __init__(self, inp, out, read, write):
        self.input_tokens, self.output_tokens = inp, out
        self.cache_read_input_tokens, self.cache_creation_input_tokens = read, write


class _Client:
    def __init__(self, read=119_997, write=0):
        self.calls, self.read, self.write = [], read, write
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def effective_output_cap(*, max_tokens=None, **_):
        return max_tokens  # an SDK-shaped double: sends the cap it is given

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(usage=_Usage(3, 1, self.read, self.write), content=[], stop_reason="max_tokens")


class _InertTimer:
    """Stands in for ``threading.Timer``: the test fires the callback itself."""

    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args, self.daemon, self.name = delay, fn, args, True, ""

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    from agent.usage_pricing import PricingEntry

    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    entry = PricingEntry(**{k: Decimal(str(v)) for k, v in vars(_OPUS).items()}, source="official_docs_snapshot")
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: entry)
    monkeypatch.setattr(pcl.PromptCacheWarmer, "_pricing", lambda self: _OPUS)
    monkeypatch.setattr(pcl.threading, "Timer", _InertTimer)


def _chat(*, platform="cli", client=None, tokens=150_000, provider="anthropic", base_url="https://api.anthropic.com",
          **over):
    agent = SimpleNamespace(
        api_mode="anthropic_messages", _use_prompt_caching=True, _cache_disabled=False, _cache_ttl="1h",
        _interrupt_requested=False, _anthropic_client=client or _Client(), model="claude-opus-4-7",
        provider=provider, base_url=base_url, api_key="", session_api_calls=7, session_prompt_tokens=0,
        session_completion_tokens=0, session_total_tokens=0, session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0, session_estimated_cost_usd=0.0,
        _session_db=None, session_id="s", log_prefix="", platform=platform, compression_enabled=True,
        context_compressor=SimpleNamespace(
            compression_count=0, last_prompt_tokens=tokens, threshold_tokens=200_000, summary_target_ratio=0.2,
            last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
            get_active_compression_failure_cooldown=lambda: None),
    )
    for key, value in over.items():
        setattr(agent, key, value)
    return agent


def _request(ttl="1h"):
    marker = {"type": "ephemeral", "ttl": ttl} if ttl == "1h" else {"type": "ephemeral"}
    return {"model": "claude-opus-4-7", "max_tokens": 8192, "stream": True,
            "system": [{"type": "text", "text": "sys", "cache_control": marker}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": marker}]}]}


_WRITE_1H = SimpleNamespace(usage=SimpleNamespace(
    input_tokens=50, output_tokens=30, cache_read_input_tokens=0, cache_creation_input_tokens=119_950,
    cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=119_950)))
_WRITE_5M = SimpleNamespace(usage=_Usage(50, 30, 0, 119_950))


def _respond(agent, request, response):
    """What the loop does per response: fold it into the clock, then arm the controller."""
    from agent.prompt_cache_clock import attempt_facts, record_cache_response
    from agent.usage_pricing import normalize_usage

    agent._prompt_cache_attempt = attempt_facts(request)
    record_cache_response(agent, normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode),
                          model=agent.model, provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode)
    return pcl.get_prompt_cache_warmer(agent).start(request, response)


def _fire(warmer):
    run = warmer._run
    assert run is not None and run.timer is not None, warmer.status
    warmer._refresh(run)


def _warm_until_spent(agent):
    warmer = pcl.get_prompt_cache_warmer(agent)
    sent = 0
    while warmer._run is not None and sent < 10:
        _fire(warmer)
        sent += 1
    return sent


def test_after_a_cache_edge_compaction_nothing_is_sent_until_the_next_user_message():
    """Ruling A: no priming write, no post-compaction keep-alives; the next user message starts a budget."""
    agent = _chat(_user_turn_count=1)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request(), _WRITE_1H) is None
    pcl.end_turn_prompt_cache_warming(agent)  # the turn settled; a person may come back
    assert warmer._run is not None, "a human chat keeps warming while idle"

    assert _warm_until_spent(agent) == pcl.WARM_BUDGET
    due_at = warmer._spent.due_at
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=due_at - 1).compact == "not_yet"
    assert pcl.get_prompt_cache_warmer(agent).claim_edge(now=due_at) == "compact"
    agent.context_compressor.compression_count = 1  # the surface committed the compaction
    assert warmer._run is None and not hasattr(warmer, "prime")
    assert len(agent._anthropic_client.calls) == pcl.WARM_BUDGET  # nothing after the compaction
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=due_at + 60).compact == "compacted_this_stretch"

    agent._user_turn_count = 2  # the next user message: a fresh budget on the compacted chat
    agent.session_api_calls += 1
    assert _respond(agent, _request(), _WRITE_1H) is None and warmer._run.warm_count == 0


def test_tool_rounds_inside_one_user_turn_share_one_budget():
    """The idle budget restarts on a new user message only: keep-alives fired between the requests of one
    turn while no tool is executing share three. (Changed by ruling 3: while a tool executes they are
    uncounted, see ``test_prompt_cache_rulings.py``.)"""
    agent = _chat(_user_turn_count=1)
    warmer = pcl.get_prompt_cache_warmer(agent)
    for _round in range(5):  # five tool-round requests in the same turn, each long enough to warm once
        agent.session_api_calls += 1
        if _respond(agent, _request(), _WRITE_1H) is None:
            _fire(warmer)
    assert len(agent._anthropic_client.calls) == pcl.WARM_BUDGET
    agent._user_turn_count = 2
    agent.session_api_calls += 1
    assert _respond(agent, _request(), _WRITE_1H) is None  # a new user message: a new budget


def test_keep_alives_fire_at_five_sixths_of_the_ttl_and_compaction_decides_there():
    assert [pcl.cache_warming_delay_s(t) for t in (300, 1800, 3600)] == [250, 1500, 3000]
    assert pcl.cache_warming_delay_s(12) == 2  # the 10 s margin wins on a tiny TTL
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    _respond(agent, _request(), _WRITE_1H)
    run = warmer._run
    assert run.deadline_at - run.next_warm_at == pytest.approx(600, rel=0, abs=1e-6)


def test_a_five_minute_window_warms_three_times_then_expires_without_compacting():
    agent = _chat(_cache_ttl="5m")
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request("5m"), _WRITE_5M) is None
    pcl.end_turn_prompt_cache_warming(agent)
    assert _warm_until_spent(agent) == pcl.WARM_BUDGET
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=warmer._spent.due_at).compact == "short_window:300s"


def test_a_keep_alive_that_misses_ends_warming():
    agent = _chat(client=_Client(read=0, write=119_950))
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request(), _WRITE_1H) is None
    _fire(warmer)
    assert warmer._run is None and warmer.status["reason"].startswith("cache refresh missed")
    assert len(agent._anthropic_client.calls) == 1


def test_a_miss_goes_straight_to_the_compact_or_offload_decision():
    """Ruling B: the miss ends the budget, so step 2 decides now; a small chat goes to step 3 instead."""
    agent = _chat(client=_Client(read=0, write=119_950))
    warmer = pcl.get_prompt_cache_warmer(agent)
    _respond(agent, _request(), _WRITE_1H)
    _fire(warmer)
    assert warmer._spent is not None
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=warmer._spent.due_at).compact == "due"

    small = _chat(client=_Client(read=0, write=119_950), tokens=50_000)
    small.session_id = "small"  # another chat: the first one's miss is its own
    w2 = pcl.get_prompt_cache_warmer(small)
    _respond(small, _request(), _WRITE_1H)
    _fire(w2)
    assert pcl.get_prompt_cache_warmer(small).edge_decision(now=w2._spent.due_at).compact == "small_context"
    after_window = small._prompt_cache_clock.estimated_until + 1
    assert pcl.get_prompt_cache_warmer(small).edge_decision(now=after_window).offload == "due"


def test_a_compaction_decision_needs_no_still_warm_cache():
    """Ruling E: the summary is a fresh request; a budget spent past the window still compacts."""
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    _respond(agent, _request(), _WRITE_1H)
    pcl.end_turn_prompt_cache_warming(agent)
    _warm_until_spent(agent)
    late = warmer._spent.due_at + warmer._spent.tier_s / 6 + 300  # the gateway's 300 s housekeeping, past the window
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=late).compact == "due"


def test_an_idle_chat_past_its_window_is_offloaded_once_by_its_owner():
    """Ruling F: the clock's window passed with no compaction: step 3 offloads now, once per stretch."""
    agent = _chat(tokens=50_000)
    warmer = pcl.get_prompt_cache_warmer(agent)
    _respond(agent, _request(), _WRITE_1H)
    pcl.end_turn_prompt_cache_warming(agent)
    _warm_until_spent(agent)
    now = agent._prompt_cache_clock.estimated_until - 1
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=now).offload == "not_cold"
    now += 2
    assert pcl.get_prompt_cache_warmer(agent).claim_edge(now=now) == "offload"
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(now=now).offload == "offloaded_this_stretch"
    assert warmer._run is None


@pytest.mark.parametrize("over", [{"platform": "cron"}, {"platform": "api_server"}, {"_delegate_depth": 1},
                                  {"side_agent": True}])
def test_a_machine_session_warms_only_while_its_tool_runs(over):
    agent = _chat(**over)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request(), _WRITE_1H) is None, "a running tool is still kept warm"
    pcl.end_turn_prompt_cache_warming(agent)
    assert warmer._run is None and warmer.status["reason"] == "agent run settled"
    assert agent._anthropic_client.calls == []


def test_an_unknown_clock_makes_nothing_act():
    from agent.turn_proactive_prune import prompt_cache_is_cold

    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert warmer.start(_request(), _WRITE_1H) == "cache window not guaranteed warm"  # nothing folded
    assert pcl.get_prompt_cache_warmer(agent).edge_decision().compact == "budget_not_spent"
    assert prompt_cache_is_cold(agent) is False
    assert agent._anthropic_client.calls == []


def test_a_release_ends_a_live_budget_and_decides_step_two_then():
    """The owner releasing the agent ends the budget even with keep-alives left (the gateway evicts an
    idle 1h chat after an hour, before 3 keep-alives at ~54 minutes run out): step 2 is decided now and
    the release cancels warming, so no keep-alive follows; the stretch never compacts twice."""
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request(), _WRITE_1H) is None
    pcl.end_turn_prompt_cache_warming(agent)
    assert warmer._run is not None and warmer._run.warm_count == 0  # three keep-alives still to go
    assert pcl.get_prompt_cache_warmer(agent).edge_decision().compact == "budget_not_spent"  # the tick alone would wait ~3h

    assert pcl.get_prompt_cache_warmer(agent).claim_edge(releasing=True) == "compact"
    assert warmer._run is None and agent._anthropic_client.calls == []
    assert pcl.get_prompt_cache_warmer(agent).edge_decision(releasing=True).compact == "compacted_this_stretch"

    small = _chat(tokens=50_000)  # under half the threshold: released as before, nothing compacted
    assert _respond(small, _request(), _WRITE_1H) is None
    assert pcl.get_prompt_cache_warmer(small).edge_decision(releasing=True).compact == "small_context"


def test_cheap_routes_warm_with_no_dollar_floor_but_never_below_the_cacheable_minimum():
    """Ruling C: a read cheaper than the rewrite it prevents is worth sending, however few cents."""
    cheap = SimpleNamespace(input_cost_per_million=0.09, output_cost_per_million=0.18,
                            cache_read_cost_per_million=0.018, cache_write_cost_per_million=0.09)
    assert pcl.evaluate_economics(cheap, 20_000).action == "warm"  # saves ~$0.0014
    assert pcl.evaluate_economics(cheap, 900).action == "stop"  # no entry below 1,024 tokens
