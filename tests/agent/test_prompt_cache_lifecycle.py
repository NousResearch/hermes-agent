"""The prompt-cache lifecycle loop (agent/prompt_cache_lifecycle.py): an idle human chat warms its
cache up to three times, then compacts (on a window of 30 minutes or more), then warms the compacted
chat up to three more times, then lets it expire. A miss ends warming at once; a machine session
never warms idle; an unknown clock makes nothing act.

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


def test_idle_chat_warms_three_times_compacts_then_warms_three_more_then_expires():
    agent = _chat()
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request(), _WRITE_1H) is None
    pcl.end_turn_prompt_cache_warming(agent)  # the turn settled; a person may come back
    assert warmer._run is not None, "a human chat keeps warming while idle"

    assert _warm_until_spent(agent) == pcl.WARM_BUDGET
    assert warmer.status["reason"] == "warm budget spent"
    assert warmer._spent is not None
    due_at = warmer._spent.due_at
    assert pcl.compaction_due(agent, now=due_at - 1) == (False, "not_yet")
    assert pcl.claim_compaction(agent, now=due_at) is True
    assert pcl.compaction_due(agent, now=due_at) == (False, "compacted_this_stretch")

    # The surface commits the compaction (a new prompt generation), then primes the smaller chat.
    agent.context_compressor.compression_count = 1
    calls_before = len(agent._anthropic_client.calls)
    compacted = _request()
    compacted["messages"][0]["content"][0]["text"] = "compacted handoff " * 20_000  # ~60k tokens
    assert warmer.prime(compacted) is None
    _fire(warmer)  # the priming write
    assert len(agent._anthropic_client.calls) == calls_before + 1
    assert _warm_until_spent(agent) == pcl.WARM_BUDGET
    assert len(agent._anthropic_client.calls) == calls_before + 1 + pcl.WARM_BUDGET
    # The fresh budget is spent: the cache expires; no second compaction this idle stretch.
    assert pcl.compaction_due(agent, now=warmer._spent.due_at) == (False, "compacted_this_stretch")
    assert pcl.claim_compaction(agent, now=warmer._spent.due_at) is False


def test_a_five_minute_window_warms_three_times_then_expires_without_compacting():
    agent = _chat(_cache_ttl="5m")
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request("5m"), _WRITE_5M) is None
    pcl.end_turn_prompt_cache_warming(agent)
    assert _warm_until_spent(agent) == pcl.WARM_BUDGET
    assert pcl.compaction_due(agent, now=warmer._spent.due_at) == (False, "short_window:300s")


def test_a_keep_alive_that_misses_ends_warming_and_never_compacts():
    agent = _chat(client=_Client(read=0, write=119_950))
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent, _request(), _WRITE_1H) is None
    _fire(warmer)
    assert warmer._run is None and warmer.status["reason"].startswith("cache refresh missed")
    assert len(agent._anthropic_client.calls) == 1
    assert pcl.compaction_due(agent) == (False, "budget_not_spent")


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
    assert pcl.compaction_due(agent) == (False, "budget_not_spent")
    assert prompt_cache_is_cold(agent) is False
    assert agent._anthropic_client.calls == []
