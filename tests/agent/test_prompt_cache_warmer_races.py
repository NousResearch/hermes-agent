"""Prompt-cache keep-alive races against real state: a real ``SessionDB`` in a scratch dir and the
real OpenAI SDK posting to a local HTTP endpoint that can hold its reply. Local-wire proofs of the
warmer's own state machine, not of any provider's cache TTL."""

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import agent.prompt_cache_warmer as pcw
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


def _armed(tmp_path, client, sid="s1"):
    """An agent whose first response (a 120k-token cache write) is folded into the clock and armed."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(sid, "cli")
    db.append_message(sid, "user", "before")
    db.append_message(sid, "assistant", "old prefix")
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="anthropic", base_url="https://api.anthropic.com",
        model="claude-opus-4-7", api_key="", _use_prompt_caching=True, _cache_disabled=False,
        _interrupt_requested=False, session_id=sid, session_api_calls=1, _session_db=db, client=client,
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
    assert warmer.start(request, SimpleNamespace(usage=usage)) is None
    run = warmer._run
    run.timer.cancel()
    return agent, warmer, run, db


def _compact(agent):
    agent._session_db.archive_and_compact(agent.session_id, [{"role": "user", "content": "NEW COMPACTED PREFIX"}])
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
            db.flush_token_counts()
            assert db.get_session(agent.session_id)["cache_read_tokens"] == _READ  # billed where sent
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
        db.create_session("new", "cli")
        agent._transition_context_engine_session = lambda **kw: None  # no context engine on this agent
        agent.session_cache_warm_calls = 3
        AIAgent.reset_session_state(agent)
        assert agent.session_cache_warm_calls == 0
        agent.session_api_calls = 1  # the new session's first response
        srv.release.set()
        worker.join(10)
        assert not worker.is_alive()
        db.flush_token_counts()
        assert agent.session_cache_read_tokens == 0 and agent.session_estimated_cost_usd == 0.0
        assert agent.session_cache_warm_calls == 0
        assert db.get_session("old")["cache_read_tokens"] == _READ
        assert db.get_session("new")["cache_read_tokens"] == 0
        db.close()
