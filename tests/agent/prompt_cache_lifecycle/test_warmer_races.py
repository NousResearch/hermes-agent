"""Prompt-cache keep-alive races against real state: a real storage interface handle on a scratch state.db and the
real OpenAI SDK posting to a local HTTP endpoint that can hold its reply. Local-wire proofs of the
warmer's own state machine, not of any provider's cache TTL."""

import json
import sys
import threading
import uuid
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


def _adaptive_5m(agent):
    agent.platform = "cli"
    agent._cache_ttl = "5m"
    agent._cache_ttl_adaptive = agent._cache_ttl_auto = True


def test_a_committed_5m_edge_compaction_upgrades_the_next_message_even_right_after_the_edge(tmp_path, monkeypatch):
    """The edge falls ~250 s after the last keep-alive; a compaction committed there replaced the prompt, so
    nothing warm matches it and the return asks for 1h at once (a skipped one keeps 5m: test_compaction)."""
    import time

    from agent.context_compressor import ContextCompressor
    from agent.prompt_cache_lifecycle.tier import maybe_upgrade_cache_tier

    with _endpoint() as (client, _server):
        agent, warmer, run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)  # the upgrade is kept per session id
        _adaptive_5m(agent)
        agent.compression_enabled = True
        agent.context_compressor = ContextCompressor(model=agent.model)
        agent.context_compressor.last_prompt_tokens, agent.context_compressor.threshold_tokens = 150_000, 200_000
        t0 = cache_window(agent).refreshed_at
        try:
            for step in (1, 2, 3):
                with monkeypatch.context() as clock:
                    clock.setattr(time, "time", lambda step=step: t0 + step * 250)
                    warmer._refresh(run)
                    if run.timer:
                        run.timer.cancel()
            due = warmer._spent.due_at
            assert warmer.claim_edge(now=due) == "compact"
            _compact(agent)
            assert maybe_upgrade_cache_tier(agent, now=due + 1) and agent._cache_ttl == "1h"
        finally:
            warmer.cancel()
            db.close()


def test_a_keep_alive_awaiting_its_reply_keeps_the_entry_warm_for_the_tier_and_the_cold_prune(tmp_path, monkeypatch):
    """Past the old expiry with a refresh in flight, the next message neither upgrades nor prunes: the
    refresh renewed that prefix."""
    import time

    from agent.context_compressor import ContextCompressor
    from agent.prompt_cache_lifecycle.tier import maybe_upgrade_cache_tier
    from agent.turn_proactive_prune import run_proactive_prune
    from persistence.conversation.messages import ConversationRequest
    from tests.agent.test_proactive_prune_cold_cache import _history
    from tests.fixtures.storage_read import conversation
    from tests.fixtures.storage_seed import seed_messages

    with _endpoint() as (client, server):
        agent, warmer, run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)  # the upgrade is kept per session id
        _adaptive_5m(agent)
        seed_messages(db, agent.session_id, *_history(20, big=set(range(18)), chars=24_000, start=1))
        history = conversation(db, agent.session_id, ConversationRequest(include_row_ids=True))
        compressor = ContextCompressor(model=agent.model, quiet_mode=True)
        compressor.bind_session_state(session_db=db, session_id=agent.session_id, persistence=db)
        agent.context_compressor, agent.compression_enabled = compressor, True
        t0 = cache_window(agent).refreshed_at
        now = [t0 + 250]
        server.release.clear()
        with monkeypatch.context() as clock:
            clock.setattr(time, "time", lambda: now[0])
            worker = threading.Thread(target=warmer._refresh, args=(run,))
            worker.start()
            try:
                assert server.entered.wait(5)
                now[0] = t0 + 301
                assert not maybe_upgrade_cache_tier(agent)
                assert run_proactive_prune(agent, history, 120_000, None) is history
            finally:
                server.release.set()
                worker.join(10)
                warmer.cancel()
                db.close()


def test_the_tier_check_never_reads_the_old_clock_and_a_cleared_in_flight_mark(tmp_path, monkeypatch):
    """The refresh's reply folds and clears its in-flight mark while the turn is mid-check: the check
    still sees one consistent snapshot, so it never upgrades over the entry that reply renewed."""
    import time

    from agent.prompt_cache_clock import cache_certainly_warm
    from agent.prompt_cache_lifecycle.tier import maybe_upgrade_cache_tier

    with _endpoint() as (client, server):
        agent, warmer, run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)  # the upgrade is kept per session id
        _adaptive_5m(agent)
        t0 = cache_window(agent).refreshed_at
        now = [t0 + 250]
        server.release.clear()
        paused, resume, result = threading.Event(), threading.Event(), []

        def trace(frame, event, _arg):
            if event == "call" and frame.f_code is cache_certainly_warm.__code__:
                paused.set()
                assert resume.wait(10)

        def user_turn():
            sys.settrace(trace)
            try:
                result.append(maybe_upgrade_cache_tier(agent))
            finally:
                sys.settrace(None)

        with monkeypatch.context() as clock:
            clock.setattr(time, "time", lambda: now[0])
            worker = threading.Thread(target=warmer._refresh, args=(run,))
            worker.start()
            try:
                assert server.entered.wait(5)
                now[0] = t0 + 301
                user = threading.Thread(target=user_turn)
                user.start()
                assert paused.wait(5)  # the turn is past its clock read
                server.release.set()
                worker.join(10)
                assert not worker.is_alive() and cache_certainly_warm(cache_window(agent))
                resume.set()
                user.join(10)
                assert result == [False]
            finally:
                server.release.set()
                resume.set()
                worker.join(10)
                warmer.cancel()
                db.close()


def _next_request(agent):
    return {"model": agent.model, "max_tokens": 100, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "OLD PREFIX " * 2000}]},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": [{"type": "text", "text": "NEXT REQUEST", "cache_control": {"type": "ephemeral"}}]}]}


def test_a_reply_generating_past_the_entrys_lifetime_keeps_its_own_prompt_warm(tmp_path, monkeypatch):
    """A request whose reply takes longer than the 5m entry lives (a long generation): a keep-alive
    replaying that same request goes out while the reply streams, and the reply landing afterwards
    keeps the renewal instead of moving the window back to its own send (issue #160322)."""
    import time

    from agent.prompt_cache_clock import cache_certainly_warm, record_attempt_sent

    with _endpoint() as (client, server):
        agent, warmer, run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)
        try:
            warmer.cancel("new request in flight")
            request = _next_request(agent)
            t0 = time.time()
            record_attempt_sent(agent, request)
            pcw.start_in_flight_prompt_cache_warming(agent, request)
            run = warmer._run
            assert run is not None and run.timer is not None
            run.timer.cancel()
            with monkeypatch.context() as clock:
                clock.setattr(time, "time", lambda: t0 + 250)
                warmer._refresh(run)
            assert server.requests[-1]["messages"][-1]["content"][0]["text"] == "NEXT REQUEST"
            usage = SimpleNamespace(input_tokens=0, output_tokens=9_000, cache_read_input_tokens=_READ,
                                    cache_creation_input_tokens=400)
            record_cache_response(agent, normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode),
                                  model=agent.model, provider=agent.provider, base_url=agent.base_url,
                                  api_mode=agent.api_mode)  # the loop's fold: from the agent's attempt
            window = cache_window(agent)
            assert window.refreshed_at == pytest.approx(t0 + 250, abs=1)
            assert cache_certainly_warm(window, now=t0 + 390)
        finally:
            warmer.cancel()
            db.close()


def test_an_automatic_cache_that_never_proved_itself_arms_nothing_for_the_request_in_flight(tmp_path):
    """No markers (an OpenAI-spec automatic cache) and no reply on the clock yet: nothing says this
    request is cached, so nothing is kept warm; a marked request declares its tier and is."""
    with _endpoint() as (client, _server):
        agent, warmer, _run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)
        try:
            warmer.cancel("new request in flight")
            agent._prompt_cache_clock, agent._prompt_cache_clock_restored = None, True  # no reply has cached yet
            agent._prompt_cache_attempt = None
            unmarked = {"model": agent.model, "max_tokens": 100,
                        "messages": [{"role": "user", "content": "OLD PREFIX " * 2000}]}
            assert warmer.start_in_flight(unmarked) == "no cache window to keep warm (no markers, nothing proven)"
            assert warmer._run is None
            assert warmer.start_in_flight(_next_request(agent)) is None
        finally:
            warmer.cancel()
            db.close()


def test_a_keep_alive_reply_landing_after_its_requests_own_reply_still_renews_the_window(tmp_path, monkeypatch):
    """The request's reply lands (and is folded, and replaces the run) while its keep-alive is still on
    the wire: the keep-alive's reply folds after it as a later read of the same prompt, so the window
    counts from the keep-alive, not from the request's send."""
    import time

    from agent.prompt_cache_clock import cache_certainly_warm, record_attempt_sent
    from agent.prompt_cache_lifecycle.warmer import fold_reply

    with _endpoint() as (client, server):
        agent, warmer, run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)
        try:
            warmer.cancel("new request in flight")
            request = _next_request(agent)
            t0 = time.time()
            now = [t0]
            monkeypatch.setattr(time, "time", lambda: now[0])
            record_attempt_sent(agent, request)
            pcw.start_in_flight_prompt_cache_warming(agent, request)
            run = warmer._run
            run.timer.cancel()
            server.release.clear()
            now[0] = t0 + 250
            worker = threading.Thread(target=warmer._refresh, args=(run,))
            worker.start()
            assert server.entered.wait(5)
            now[0] = t0 + 255  # the request's own reply lands while the keep-alive is on the wire
            usage = SimpleNamespace(input_tokens=0, output_tokens=9_000, cache_read_input_tokens=_READ,
                                    cache_creation_input_tokens=400)
            fold_reply(agent, lambda: record_cache_response(
                agent, normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode), model=agent.model,
                provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode))
            warmer.cancel("the reply replaced the run")
            server.release.set()
            worker.join(10)
            assert not worker.is_alive()
            window = cache_window(agent)
            assert window.refreshed_at == pytest.approx(t0 + 250, abs=1)
            assert cache_certainly_warm(window, now=t0 + 390)
        finally:
            server.release.set()
            warmer.cancel()
            db.close()


def test_an_unmarked_tail_past_the_last_cache_marker_is_priced_as_input(tmp_path):
    """A small marked prefix and a large tool result past the last marker (a route that leaves tool
    results unmarked): every replay re-bills that tail as plain input, so the request in flight is not
    kept warm; with the marker on the last message the same prompt is."""
    with _endpoint() as (client, _server):
        agent, warmer, _run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)
        try:
            warmer.cancel("new request in flight")
            prefix = {"role": "user", "content": [{"type": "text", "text": "SMALL PREFIX " * 300,
                                                   "cache_control": {"type": "ephemeral"}}]}
            tail = "TOOL RESULT " * 60_000
            unmarked = {"model": agent.model, "max_tokens": 100, "messages": [
                prefix, {"role": "assistant", "content": "x"}, {"role": "user", "content": [{"type": "text", "text": tail}]}]}
            assert warmer.start_in_flight(unmarked) is not None
            assert warmer._run is None
            marked = {"model": agent.model, "max_tokens": 100, "messages": [
                prefix, {"role": "assistant", "content": "x"},
                {"role": "user", "content": [{"type": "text", "text": tail, "cache_control": {"type": "ephemeral"}}]}]}
            assert warmer.start_in_flight(marked) is None
        finally:
            warmer.cancel()
            db.close()


def test_a_late_keep_alive_reply_never_rewinds_a_newer_requests_fold(tmp_path, monkeypatch):
    """The request's reply folds, then the next request is sent and folds too, all while the old
    keep-alive is still on the wire: its late reply is older than what the clock holds and is dropped,
    so the newer request's coverage stands."""
    import time

    from agent.prompt_cache_clock import record_attempt_sent
    from agent.prompt_cache_lifecycle.warmer import fold_reply

    def fold(agent):
        usage = SimpleNamespace(input_tokens=0, output_tokens=9_000, cache_read_input_tokens=_READ,
                                cache_creation_input_tokens=400)
        fold_reply(agent, lambda: record_cache_response(
            agent, normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode), model=agent.model,
            provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode))

    with _endpoint() as (client, server):
        agent, warmer, run, db = _armed(tmp_path, client, sid=uuid.uuid4().hex)
        try:
            warmer.cancel("new request in flight")
            request = _next_request(agent)
            t0 = time.time()
            now = [t0]
            monkeypatch.setattr(time, "time", lambda: now[0])
            record_attempt_sent(agent, request)
            pcw.start_in_flight_prompt_cache_warming(agent, request)
            run = warmer._run
            run.timer.cancel()
            server.release.clear()
            now[0] = t0 + 250
            worker = threading.Thread(target=warmer._refresh, args=(run,))
            worker.start()
            assert server.entered.wait(5)
            now[0] = t0 + 255
            fold(agent)
            warmer.cancel("the reply replaced the run")
            now[0] = t0 + 260  # the next request is sent and its reply folds first
            record_attempt_sent(agent, _next_request(agent))
            fold(agent)
            server.release.set()
            worker.join(10)
            assert not worker.is_alive()
            assert cache_window(agent).refreshed_at == pytest.approx(t0 + 260, abs=1)
        finally:
            server.release.set()
            warmer.cancel()
            db.close()
