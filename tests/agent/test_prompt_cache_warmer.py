"""Prompt-cache warming (agent/prompt_cache_warmer.py, port of earendil-works/pi#9668).

The timer callback is invoked directly after the armed timer is cancelled, so no test sleeps.
"""

import json
import logging
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kw):
        self.options = kw
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(usage=_Usage(3, 1, 119_997, 0), content=[], stop_reason="max_tokens")


def _agent(**over):
    base = dict(
        api_mode="anthropic_messages", _use_prompt_caching=True, _cache_disabled=False, _cache_ttl="5m",
        _interrupt_requested=False, _anthropic_client=_Client(),
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
    assert warmer.status["state"] == "scheduled"
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


def test_turn_end_stops_warming_and_a_run_never_outspends_the_miss_it_prevents(priced):
    """No warming between turns (the next request would be a guess), and a tool call that
    outlasts ``miss / warm`` refreshes stops warming rather than pay more than one cache write."""
    agent = _agent()
    warmer = pcw.get_prompt_cache_warmer(agent)
    warmer.start(_request(), _RESPONSE)
    pcw.cancel_prompt_cache_warming(agent, "agent run settled")
    assert warmer._run is None and warmer.status["reason"] == "agent run settled"

    warmer.start(_request(), _RESPONSE)
    d = pcw.evaluate_economics(_OPUS, 120_000)  # _RESPONSE's prompt size
    affordable = int(d.miss_cost // d.warm_cost)
    for _ in range(affordable + 3):
        if warmer._run is None:
            break
        _fire(warmer)
    assert len(agent._anthropic_client.calls) == affordable
    assert warmer.status["reason"] == "refreshes would cost more than the cache write they prevent"
    assert agent.session_estimated_cost_usd <= d.miss_cost


@pytest.mark.parametrize("mutate, reason", [
    (lambda a, r: setattr(a, "_skip_prompt_cache_warming", True), "the parent agent keeps this prefix warm"),
    (lambda a, r: setattr(a, "api_mode", "codex_responses"), "only Anthropic Messages and Chat Completions requests are warmed"),
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
    assert warmer.start(_request(), SimpleNamespace(usage=_Usage(30, 30, 0, 3_000))) == "expected savings below threshold"
    # The pure economics: pi's decision rule (p * miss - warm >= $0.05).
    d = pcw.evaluate_economics(_OPUS, 100_000)
    assert (round(d.warm_cost, 4), round(d.miss_cost, 4), d.action) == (0.05, 0.575, "warm")


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


def test_chat_completions_gateway_is_warmed_on_the_five_minute_schedule(priced):
    """A Claude request through a Chat Completions gateway (Nous Portal, OpenRouter) is replayed on
    that wire, and its ``1h`` tier is not trusted: the gateway was measured dropping it at 7 min."""
    agent = _agent(api_mode="chat_completions", provider="nous", _cache_ttl="1h", client=_Client())
    request = {
        "model": "anthropic/claude-opus-5.5", "max_tokens": 8192, "stream": True,
        "stream_options": {"include_usage": True}, "extra_body": {"reasoning": {"effort": "medium"}},
        "messages": [{"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}],
    }
    warmer = pcw.get_prompt_cache_warmer(agent)
    assert warmer.start(request, _RESPONSE) is None
    assert warmer._run.delay_s == pcw.cache_warming_delay_s(300)
    _fire(warmer)
    (sent,) = agent.client.calls
    assert sent["max_tokens"] == 1 and "stream" not in sent and "stream_options" not in sent
    assert sent["messages"] == request["messages"] and sent["extra_body"] == request["extra_body"]
    assert sent["extra_headers"]["X-OpenRouter-Cache"] == "false"  # a response-cache hit refreshes nothing
    assert pcw.warm_ttl_seconds(_agent(_cache_ttl="1h")) == 3600  # native Anthropic keeps its tier


def test_warming_needs_no_config_and_slow_responses_refresh_sooner(priced, tmp_path, monkeypatch):
    """With an empty config.yaml warming arms (no knob), and a response that took 200 s to
    finish refreshes 200 s sooner: the entry's lifetime started when the request was sent."""
    import time as _time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n")
    warmer = pcw.get_prompt_cache_warmer(_agent())
    sent = _time.time() - 200
    assert warmer.start(_request(), _RESPONSE, sent_at=sent) is None
    assert warmer._run.next_warm_at == pytest.approx(sent + pcw.cache_warming_delay_s(300))
    warmer.cancel("done")
    assert warmer.start(_request(), _RESPONSE, sent_at=_time.time() - 400) == "cache entry already expired"


@contextmanager
def _chat_completions_stub(usage):
    """A local Chat Completions endpoint that answers every request with ``usage``, driven
    through the real OpenAI client (the warmer reads what the SDK parses, not a fake object)."""
    from openai import OpenAI

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append(body)
            data = json.dumps({
                "id": "r", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
                "usage": usage,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = OpenAI(api_key="local-stub", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
    try:
        yield client, srv.requests
    finally:
        client.close()
        srv.shutdown()
        srv.server_close()


def _cc_usage(read, write, uncached):
    """OpenAI-shaped totals plus the Anthropic fields a Claude gateway passes through (the
    agent fixture is ``provider=anthropic``, which normalizes on the Anthropic fields)."""
    return {
        "prompt_tokens": read + write + uncached, "completion_tokens": 1,
        "total_tokens": read + write + uncached + 1,
        "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": write},
        "input_tokens": uncached, "output_tokens": 1,
        "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
    }


def _cc_agent(client):
    return _agent(api_mode="chat_completions", client=client, _anthropic_client=None)


def _cc_request():
    return {"model": "claude-opus-4-7", "max_tokens": 100,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]}


def test_a_refresh_that_misses_the_cache_stops_warming_and_says_why(priced, caplog):
    """A refresh billed as a cache WRITE (entry already gone, or the replay re-wrote it) ends the
    run at once: every further refresh would pay write prices, not the cheap read assumed."""
    with _chat_completions_stub(_cc_usage(read=0, write=120_000, uncached=0)) as (client, requests):
        agent = _cc_agent(client)
        warmer = pcw.get_prompt_cache_warmer(agent)
        assert warmer.start(_cc_request(), _RESPONSE) is None
        with caplog.at_level(logging.INFO, logger=pcw.__name__):
            _fire(warmer)
        assert warmer._run is None and len(requests) == 1
    reason = warmer.status["reason"]
    assert reason.startswith("cache refresh missed (read=0, write=120000")
    assert agent.session_cache_write_tokens == 120_000  # the miss is still accounted
    assert any("prompt cache warming stopped: cache refresh missed" in r.getMessage()
               and r.levelno == logging.INFO for r in caplog.records)


def test_the_spend_cap_counts_what_refreshes_were_billed_not_an_assumed_read(priced):
    """Refreshes that hit but bill a large uncached tail cost far more than ``prompt * read``;
    the cap stops the run once their summed estimated cost would pass the miss they prevent."""
    with _chat_completions_stub(_cc_usage(read=60_000, write=0, uncached=60_000)) as (client, requests):
        agent = _cc_agent(client)
        warmer = pcw.get_prompt_cache_warmer(agent)
        assert warmer.start(_cc_request(), _RESPONSE) is None
        for _ in range(12):
            if warmer._run is None:
                break
            _fire(warmer)
        d = pcw.evaluate_economics(_OPUS, 120_000)
        billed = agent.session_estimated_cost_usd
        assert len(requests) == 2, "two refreshes at ~$0.33 each, not the 11 an assumed $0.06 read allows"
        assert billed + d.warm_cost > d.miss_cost >= billed
    assert warmer.status["reason"] == "refreshes would cost more than the cache write they prevent"


def test_every_stop_reason_is_logged_once_at_info(priced, caplog):
    agent = _agent(_cache_ttl=None)
    warmer = pcw.get_prompt_cache_warmer(agent)
    with caplog.at_level(logging.INFO, logger=pcw.__name__):
        for _ in range(3):
            warmer.start(_request(), _RESPONSE)
    stops = [r for r in caplog.records if "prompt cache warming stopped" in r.getMessage()]
    assert len(stops) == 1 and stops[0].levelno == logging.INFO
    assert "cache lifetime unavailable" in stops[0].getMessage()


def test_an_uncached_tail_is_priced_at_the_input_rate(priced):
    """A refresh re-sends the tail past the last cache marker at the full input rate: a request
    with a small cached prefix and a large tail costs more to refresh than the miss it prevents."""
    d = pcw.evaluate_economics(_OPUS, 10_000, 110_000)
    assert d.warm_cost == pytest.approx(10_000 * 0.5e-6 + 110_000 * 5e-6 + 25e-6) and d.action == "stop"
    warmer = pcw.get_prompt_cache_warmer(_agent())
    assert (warmer.start(_request(), SimpleNamespace(usage=_Usage(110_000, 30, 10_000, 0)))
            == "refreshes would cost more than the cache write they prevent")


def test_a_refresh_in_flight_when_the_session_rotates_is_booked_where_it_was_sent(priced):
    """Compression (``in_place: false``) can rotate ``session_id`` while the timer thread waits
    for the reply; the refresh's tokens and dollars belong to the session that sent it."""
    booked = []
    agent = _agent(_session_db=SimpleNamespace(queue_token_counts=lambda sid, **kw: booked.append((sid, kw))))
    client = agent._anthropic_client
    real_create = client._create

    def _create_then_rotate(**kw):
        agent.session_id = "s-after-compression"
        return real_create(**kw)

    client.messages = SimpleNamespace(create=_create_then_rotate)
    warmer = pcw.get_prompt_cache_warmer(agent)
    assert warmer.start(_request(), _RESPONSE) is None
    _fire(warmer)
    ((sid, kw),) = booked
    assert sid == "s" and kw["task"] == "cache_warm" and kw["cache_read_tokens"] == 119_997
    assert agent.session_cache_read_tokens == 0, "the new session's in-memory totals stay untouched"
