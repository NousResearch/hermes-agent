"""Prompt-cache keep-alive races against real state: a real storage interface handle on a scratch state.db and the
real OpenAI SDK posting to a local HTTP endpoint that can hold its reply. Local-wire proofs of the
warmer's own state machine, not of any provider's cache TTL."""

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcw
from agent.prompt_cache_clock import attempt_facts, cache_window, record_cache_response
from agent.usage_pricing import normalize_usage

_OPUS = SimpleNamespace(input_cost_per_million=5, output_cost_per_million=25,
                        cache_read_cost_per_million=0.5, cache_write_cost_per_million=6.25)
_READ = 120_000


@pytest.fixture(autouse=True)
def priced(monkeypatch):
    monkeypatch.setattr(pcw.PromptCacheWarmer, "_pricing", lambda self: _OPUS)


@contextmanager
def _endpoint():
    """Chat Completions endpoint answering a cache read of ``_READ`` tokens; ``release`` gates replies."""
    from openai import OpenAI

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append(body)
            self.server.entered.set()
            assert self.server.release.wait(10)
            usage = {"prompt_tokens": _READ, "completion_tokens": 1, "total_tokens": _READ + 1,
                     "input_tokens": 0, "output_tokens": 1,
                     "cache_read_input_tokens": _READ, "cache_creation_input_tokens": 0}
            data = json.dumps({"id": "r", "object": "chat.completion", "created": 1, "model": body["model"],
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                                            "finish_reason": "length"}], "usage": usage}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.entered, srv.release = [], threading.Event(), threading.Event()
    srv.release.set()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = OpenAI(api_key="local-stub", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
    try:
        yield client, srv
    finally:
        srv.release.set()
        client.close()
        srv.shutdown()
        srv.server_close()


def _armed(tmp_path, client, sid="s1", *, expect=None):
    """An agent whose first response (a 120k-token cache write) is folded into the clock and armed."""
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    db = open_persistence(profile_scope(home=tmp_path), eager=True)
    seed_session(db, sid, "cli")
    seed_messages(db, sid, {"role": "user", "content": "before"}, {"role": "assistant", "content": "old prefix"})
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="anthropic", base_url="https://api.anthropic.com",
        model="claude-opus-4-7", api_key="", _use_prompt_caching=True, _cache_disabled=False,
        _interrupt_requested=False, session_id=sid, session_api_calls=1, _session_db=db, persistence=db, client=client,
        context_compressor=None, log_prefix="", session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
    )
    request = {"model": agent.model, "max_tokens": 100, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "OLD PREFIX", "cache_control": {"type": "ephemeral"}}]}]}
    usage = SimpleNamespace(input_tokens=0, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=_READ)
    record_cache_response(agent, normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode),
                          model=agent.model, provider=agent.provider, base_url=agent.base_url,
                          api_mode=agent.api_mode, attempt=attempt_facts(request))
    warmer = pcw.get_prompt_cache_warmer(agent)
    assert warmer.start(request, SimpleNamespace(usage=usage)) == expect
    run = warmer._run
    if run is not None:
        run.timer.cancel()
    return agent, warmer, run, db


def _compact(agent):
    from persistence.conversation.messages import NewMessage
    agent._session_db.messages.compact(agent.session_id, [NewMessage.from_live({"role": "user", "content": "NEW COMPACTED PREFIX"})])
    assert cache_window(agent).reason == "compacted"


@pytest.mark.parametrize("when", ["before_send", "in_flight"])
def test_a_refresh_of_a_compacted_prefix_never_certifies_the_new_one(tmp_path, when):
    """A compaction committed before the timer fires sends nothing; one committed while the
    refresh is on the wire still books its usage, but the clock stays unknown for the rewritten
    prefix and no further refresh is scheduled from it."""
    with _endpoint() as (client, srv):
        agent, warmer, run, db = _armed(tmp_path, client)
        if when == "before_send":
            _compact(agent)
            warmer._refresh(run)
            assert srv.requests == []
        else:
            srv.release.clear()
            worker = threading.Thread(target=warmer._refresh, args=(run,))
            worker.start()
            assert srv.entered.wait(5)
            _compact(agent)
            srv.release.set()
            worker.join(10)
            assert not worker.is_alive()
            db.usage.flush_usage()
            assert db.sessions.get(agent.session_id).cache_read_tokens == _READ  # billed where sent
        window = cache_window(agent)
        assert window.confidence == "unknown" and window.reason == "compacted"
        assert warmer._run is None and warmer.status["reason"] == "conversation context changed"
        db.close()


def test_a_cancel_that_returned_admits_no_later_request(tmp_path):
    """Pause the timer thread after it fired and before admission (inside pricing), cancel on the
    loop thread, resume: no HTTP request reaches the endpoint and nothing is billed."""
    with _endpoint() as (client, srv):
        agent, warmer, run, db = _armed(tmp_path, client)
        paused, resume = threading.Event(), threading.Event()

        def barrier(frame, event, arg):  # a scheduler pause only: no return value or branch is patched
            if event == "call" and frame.f_code is pcw.PromptCacheWarmer._evaluate.__code__:
                paused.set()
                assert resume.wait(10)

        def timer_thread():
            sys.settrace(barrier)
            try:
                warmer._refresh(run)
            finally:
                sys.settrace(None)

        worker = threading.Thread(target=timer_thread)
        worker.start()
        assert paused.wait(5)
        warmer.cancel("agent run settled")
        assert warmer.status["state"] == "inactive"
        resume.set()
        worker.join(10)
        assert not worker.is_alive()
        assert srv.requests == [] and agent.session_estimated_cost_usd == 0.0
        db.close()


def test_a_refresh_from_before_a_reset_stays_out_of_the_new_sessions_totals(tmp_path):
    """Armed at API call 1, the refresh is held on the wire; ``/new`` resets the counters and the
    new session's first response raises its count back to 1. The old reply is booked on the old
    session's row and never enters the new session's running totals."""
    from run_agent import AIAgent

    with _endpoint() as (client, srv):
        agent, warmer, run, db = _armed(tmp_path, client, sid="old")
        srv.release.clear()
        worker = threading.Thread(target=warmer._refresh, args=(run,))
        worker.start()
        assert srv.entered.wait(5)
        warmer.cancel("agent run settled")
        agent.session_id = "new"
        from tests.fixtures.storage_seed import seed_session
        seed_session(db, "new", "cli")
        agent._transition_context_engine_session = lambda **kw: None  # no context engine on this agent
        agent.session_cache_warm_calls = 3
        AIAgent.reset_session_state(agent)
        assert agent.session_cache_warm_calls == 0
        agent.session_api_calls = 1  # the new session's first response
        srv.release.set()
        worker.join(10)
        assert not worker.is_alive()
        db.usage.flush_usage()
        assert agent.session_cache_read_tokens == 0 and agent.session_estimated_cost_usd == 0.0
        assert agent.session_cache_warm_calls == 0
        assert db.sessions.get("old").cache_read_tokens == _READ
        assert db.sessions.get("new").cache_read_tokens == 0
        db.close()


def test_every_keep_alive_is_bracketed_by_the_auxiliary_call_events(tmp_path, monkeypatch):
    """Plugins that meter auxiliary spend see each refresh: ``pre_auxiliary_call`` carries the one-token
    request actually posted, ``post_auxiliary_call`` its cache read, both ``aux_task="cache_warm"``."""
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    mgr = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    ctx = PluginContext(PluginManifest(name="spend-meter", source="user"), mgr)
    fired = []
    for name in ("pre_auxiliary_call", "post_auxiliary_call", "pre_api_request", "post_api_request"):
        ctx.register_hook(name, lambda _name=name, **kw: fired.append((_name, kw)))
    with _endpoint() as (client, srv):
        _agent, warmer, run, db = _armed(tmp_path, client)
        warmer._refresh(run)
        (posted,) = srv.requests
        db.close()
    assert [name for name, _ in fired] == ["pre_auxiliary_call", "post_auxiliary_call"]
    pre, post = fired[0][1], fired[1][1]
    assert pre["aux_task"] == post["aux_task"] == "cache_warm" and pre["api_request_id"] == post["api_request_id"]
    assert pre["request"]["body"]["max_tokens"] == posted["max_tokens"] == 1
    assert post["error"] is None and post["usage"]["cache_read_tokens"] == _READ


def test_a_route_without_cache_prices_is_never_warmed(tmp_path, monkeypatch):
    """Input and output priced, cache read and write not: the refresh's cost and the saving are
    unknown, so nothing is armed and no request is sent (missing rates are not free)."""
    from decimal import Decimal

    from agent.usage_pricing import PricingEntry

    partial = PricingEntry(input_cost_per_million=Decimal(5), output_cost_per_million=Decimal(25),
                           source="provider_models_api")
    monkeypatch.setattr(pcw.PromptCacheWarmer, "_pricing", lambda self: partial)
    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", lambda *a, **k: partial)
    with _endpoint() as (client, srv):
        _agent, _warmer, run, db = _armed(tmp_path, client, expect="cache economics unavailable")
        assert run is None and srv.requests == []
        db.close()


def test_a_reset_never_lands_between_the_epoch_check_and_the_add(tmp_path):
    """Pause the timer thread after its reply passed the reset-epoch check and before it adds to the
    running totals; run ``/new`` meanwhile. The reset waits for the add and then zeroes it, so the
    fresh session's totals never carry the old refresh."""
    from run_agent import AIAgent

    with _endpoint() as (client, _srv):
        agent, warmer, run, db = _armed(tmp_path, client)
        agent._transition_context_engine_session = lambda **kw: None
        paused, resume, reached, inside = (threading.Event() for _ in range(4))
        order: list[str] = []

        def worker_trace(frame, event, arg):  # a scheduler pause only: no return value or branch is patched
            if frame.f_code is pcw.PromptCacheWarmer._add_to_session_totals.__code__:
                if event == "call":
                    paused.set()
                    assert resume.wait(10)
                elif event == "return":
                    order.append("add")
            return worker_trace

        def reset_trace(frame, event, arg):
            if frame.f_code is pcw.session_totals_lock.__code__ and event == "return":
                reached.set()  # the reset holds the lock object and is entering it
            elif frame.f_code is AIAgent.reset_session_state.__code__ and event == "line" and reached.is_set():
                if not inside.is_set():  # the first statement past the ``with``: the lock was acquired
                    order.append("reset")
                    inside.set()
            return reset_trace

        def traced(target, trace, *args):
            sys.settrace(trace)
            try:
                target(*args)
            finally:
                sys.settrace(None)

        worker = threading.Thread(target=traced, args=(warmer._refresh, worker_trace, run))
        worker.start()
        assert paused.wait(5)
        reset = threading.Thread(target=traced, args=(AIAgent.reset_session_state, reset_trace, agent))
        reset.start()
        assert reached.wait(5)  # the reset reached the contested lock while the add is paused inside it
        # Without the shared lock the reset's body runs now; with it, it cannot until the add returns.
        assert not inside.wait(2.0)
        resume.set()
        worker.join(10)
        reset.join(10)
        assert not worker.is_alive() and not reset.is_alive()
        assert order == ["add", "reset"]
        assert agent.session_cache_read_tokens == 0 and agent.session_estimated_cost_usd == 0.0
        assert agent.session_cache_warm_calls == 0
        db.close()
