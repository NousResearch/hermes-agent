"""Prompt-cache warming (agent/prompt_cache_warmer.py, port of earendil-works/pi#9668).

The timer callback is invoked directly after the armed timer is cancelled, so no test sleeps.
"""

from types import SimpleNamespace

import pytest

import agent.prompt_cache_warmer as pcw

_OPUS = SimpleNamespace(  # claude-opus-4-7 official rates, $/M
    input_cost_per_million=5, output_cost_per_million=25,
    cache_read_cost_per_million=0.5, cache_write_cost_per_million=6.25,
)


class _Usage:
    def __init__(self, inp, out, read, write):
        self.input_tokens, self.output_tokens = inp, out
        self.cache_read_input_tokens, self.cache_creation_input_tokens = read, write


class _Client:
    def __init__(self):
        self.calls = []
        self.options = None
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, **kw):
        self.options = kw
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(usage=_Usage(3, 1, 119_997, 0), content=[], stop_reason="max_tokens")


def _agent(mode="streaming", **over):
    base = dict(
        api_mode="anthropic_messages", _use_prompt_caching=True, _cache_disabled=False, _cache_ttl="5m",
        _cache_warming_mode=mode, _interrupt_requested=False, _anthropic_client=_Client(),
        model="claude-opus-4-7", provider="anthropic", base_url="https://api.anthropic.com", api_key="",
        session_api_calls=7, session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_estimated_cost_usd=0.0, _session_db=None, session_id="s",
        log_prefix="",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _request():
    return {
        "model": "claude-opus-4-7", "max_tokens": 8192, "stream": True, "thinking": {"type": "adaptive"},
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "terminal", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
    }


_RESPONSE = SimpleNamespace(usage=_Usage(50, 30, 0, 119_950))


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setattr(pcw.PromptCacheWarmer, "_pricing", lambda self: _OPUS)


def _fire(warmer):
    """Simulate the armed timer expiring: cancel the real one, run its callback inline."""
    run = warmer._run
    assert run is not None and run.timer is not None, warmer.status
    run.timer.cancel()
    warmer._refresh(run)


def test_replay_is_the_original_request_capped_at_one_token(priced):
    agent = _agent()
    warmer = pcw.get_prompt_cache_warmer(agent)
    request = _request()
    assert warmer.start(request, _RESPONSE) is None
    assert warmer.status["state"] == "scheduled" and warmer.status["phase"] == "streaming"
    # The loop redecorates message dicts in place before the next request; the replay must
    # still send what wrote the cache entry.
    request["messages"][0]["content"][0].pop("cache_control")

    _fire(warmer)

    client = agent._anthropic_client
    assert client.options == {"max_retries": 0, "timeout": pcw.WARM_REQUEST_TIMEOUT_S}
    (sent,) = client.calls
    assert sent["max_tokens"] == 1 and "stream" not in sent
    assert sent["system"] == _request()["system"] and sent["tools"] == _request()["tools"]
    assert sent["messages"] == _request()["messages"], "replay must carry the original cache markers"
    # Accounted as tokens + dollars, never as a conversation request (that counter is the
    # warmer's own currency token, so bumping it would end warming after one refresh).
    assert agent.session_cache_read_tokens == 119_997 and agent.session_cache_warm_calls == 1
    assert agent.session_api_calls == 7
    assert agent.session_estimated_cost_usd == pytest.approx(119_997 * 0.5e-6 + 3 * 5e-6 + 25e-6)
    assert warmer.status["state"] == "scheduled" and warmer.status["warm_count"] == 1


def test_settle_ends_streaming_mode_but_idle_mode_keeps_warming(priced):
    agent = _agent("streaming")
    warmer = pcw.get_prompt_cache_warmer(agent)
    warmer.start(_request(), _RESPONSE)
    warmer.on_turn_settled()
    assert warmer.status == {"state": "inactive", "reason": "agent run settled", "decision": warmer.status["decision"]}

    agent._cache_warming_mode = "idle"
    warmer.start(_request(), SimpleNamespace(usage=_Usage(50, 30, 0, 400_000)))
    warmer.on_turn_settled()
    assert warmer.status["phase"] == "idle"
    _fire(warmer)
    assert len(agent._anthropic_client.calls) == 1
    # Idle economics: 15% continuation chance must still clear the $0.05 floor; at 120k Opus
    # tokens it does not, so the same idle run stops before spending.
    warmer.start(_request(), _RESPONSE)
    warmer.on_turn_settled()
    _fire(warmer)
    assert len(agent._anthropic_client.calls) == 1
    assert warmer.status["reason"] == "expected savings below threshold"


@pytest.mark.parametrize("mutate, reason", [
    (lambda a, r: setattr(a, "_cache_warming_mode", "off"), "cache warming disabled"),
    (lambda a, r: setattr(a, "api_mode", "chat_completions"), "only native Anthropic requests are warmed"),
    (lambda a, r: setattr(a, "_cache_ttl", None), "cache lifetime unavailable"),
    (lambda a, r: r.update(thinking={"type": "enabled", "budget_tokens": 2048}), "request cannot be replayed safely (budget thinking)"),
    (lambda a, r: (r.pop("system"), r["messages"][0].update(content="hi")), "request carries no cache markers"),
    (lambda a, r: r.update(_moa_prepared_request={}), "MoA requests are not replayable"),
])
def test_requests_that_cannot_be_warmed_arm_nothing(priced, mutate, reason):
    agent, request = _agent(), _request()
    mutate(agent, request)
    warmer = pcw.get_prompt_cache_warmer(agent)
    assert warmer.start(request, _RESPONSE) == reason
    assert warmer._run is None and agent._anthropic_client.calls == []


def test_unpriced_or_cheap_prompts_never_pay_for_a_refresh(monkeypatch):
    monkeypatch.setattr(pcw.PromptCacheWarmer, "_pricing", lambda self: None)
    warmer = pcw.get_prompt_cache_warmer(_agent())
    assert warmer.start(_request(), _RESPONSE) == "cache economics unavailable"
    monkeypatch.setattr(pcw.PromptCacheWarmer, "_pricing", lambda self: _OPUS)
    warmer = pcw.get_prompt_cache_warmer(_agent())
    assert warmer.start(_request(), SimpleNamespace(usage=_Usage(3_000, 30, 0, 0))) == "expected savings below threshold"
    # The pure economics: pi's decision rule (p * miss - warm >= $0.05).
    d = pcw.evaluate_economics(_OPUS, 100_000, "streaming")
    assert (round(d.warm_cost, 4), round(d.miss_cost, 4), d.action) == (0.05, 0.575, "warm")
    assert pcw.evaluate_economics(_OPUS, 100_000, "idle").action == "stop"


def test_a_new_real_request_or_a_changed_conversation_stops_the_run(priced):
    agent = _agent()
    warmer = pcw.get_prompt_cache_warmer(agent)
    warmer.start(_request(), _RESPONSE)
    assert warmer._run is not None and warmer._run.timer is not None
    armed = warmer._run.timer
    pcw.cancel_prompt_cache_warming(agent, "new request in flight")
    armed.join(2)
    assert not armed.is_alive() and warmer.status["reason"] == "new request in flight"

    warmer.start(_request(), _RESPONSE)
    agent.session_api_calls += 1  # another response landed: this entry is no longer the live prefix
    _fire(warmer)
    assert agent._anthropic_client.calls == [] and warmer.status["reason"] == "conversation context changed"


def test_timer_delay_keeps_a_ten_second_margin():
    assert pcw.cache_warming_delay_s(300) == 270 and pcw.cache_warming_delay_s(3600) == 3240
    assert pcw.cache_warming_delay_s(12) == 2 and pcw.cache_warming_delay_s(10) is None
    assert pcw.normalize_cache_warming_mode(" Idle ") == "idle" and pcw.normalize_cache_warming_mode(True) == "off"
