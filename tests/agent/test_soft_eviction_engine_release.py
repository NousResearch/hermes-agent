"""Soft eviction (release_clients) releases an agent-owned context engine exactly once.

Gateway cache eviction (gateway/run_agent_cache.py::_release_evicted_agent_soft) releases cached
agents through release_clients(), not close(). That soft path never touched the agent's
context-engine clone, so its SQLite connection + WAL/SHM fds lived until GC or process exit
(#147087, the gateway-side half of #87328's leak class). These tests pin the soft path to
agent.context_engine.shutdown_context_engine() — idempotent through that module's once-flag —
scoped to agent-owned engines only, and ordered: a session-end claim, taken before the gateway
queues its hard cleanup, keeps on_session_end ahead of a later soft eviction's shutdown().

The #147087 review (100yenadmin, 11-05) found the soft path could shut the engine down while
someone was still using it: eviction lands on the gateway's memory-manager thread, so it races a
running turn (finding 1) and a background-review fork / out-of-process ``/compress`` (finding 2).
Soft release is now PENDING while an engine request is in flight: the requester pairs
``begin_engine_request``/``end_engine_request`` (``engine_request()`` contextmanager) and the LAST
one out plays the deferred shutdown exactly once.
"""

import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

# Warm the module AIAgent.__init__ imports lazily: agent_init binds ContextCompressor through a
# module-level ``from agent.context_compressor import`` (agent/agent_init.py:25), so a FIRST import
# landing inside test_compress_mutation_defers_soft_release bakes that test's patched fake into
# agent_init's own binding. monkeypatch restores agent.context_compressor afterwards but not that
# copy, so every AIAgent built later in the process gets the fake (#147087 review round 2).
import agent.agent_init

from agent.context_engine import (
    begin_engine_request,
    claim_engine_session_end,
    end_engine_request,
    engine_request,
)
from run_agent import AIAgent


class _OrderingEngine:
    """Records the lifecycle order an LCM-style engine depends on."""

    name = "lcm"

    def __init__(self):
        self.calls = []

    def on_session_end(self, session_id, messages):
        self.calls.append("on_session_end")

    def shutdown(self):
        self.calls.append("shutdown")


def _bare_agent(**attrs):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sid"
    agent._memory_manager = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._context_engine_shutdown_lock = threading.Lock()
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def test_release_clients_shuts_down_owned_engine_once():
    engine = Mock()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.release_clients()
    agent.release_clients()
    engine.shutdown.assert_called_once_with()


def test_release_clients_never_touches_shared_registered_engine():
    engine = Mock()
    # Ownership is opt-OUT: a path holding the process-shared registered engine must say so.
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=False)
    agent.release_clients()
    engine.shutdown.assert_not_called()


def test_release_clients_releases_engine_before_gc_of_agent():
    # The fd leak was fd-level: prove shutdown runs while the agent is still alive.
    calls = []

    class _Engine:
        name = "lcm"

        def shutdown(self):
            calls.append(1)

    agent = _bare_agent(context_compressor=_Engine(), _context_engine_owned=True)
    agent.release_clients()
    assert calls == [1]


def test_close_after_release_clients_is_idempotent_for_engine():
    engine = Mock()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.close()
    agent.release_clients()
    engine.shutdown.assert_called_once_with()


def test_session_end_claim_defers_soft_eviction_until_the_hard_path():
    # The /new shape: claim taken before the hard cleanup is queued; the 30 s wait expires; the
    # agent is soft-evicted while the job is queued or running — the soft path must not touch the
    # engine, and the hard path's on_session_end lands before its finally shuts it down.
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    assert claim_engine_session_end(agent) is True
    agent.release_clients()
    assert engine.calls == []
    agent.shutdown_memory_provider([])
    assert engine.calls == ["on_session_end", "shutdown"]


def test_soft_release_first_still_shuts_down_exactly_once():
    # No claim: the soft eviction owns the shutdown; a hard teardown afterwards must not
    # double-release (plugins have no idempotency contract on shutdown()).
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.release_clients()
    # round-2 item 1: the soft release takes over the session-end delivery.
    assert engine.calls == ["on_session_end", "shutdown"]
    agent.shutdown_memory_provider([])
    assert engine.calls.count("shutdown") == 1


def test_claim_re_take_wins_session_end_over_a_won_soft_release():
    agent = _bare_agent(context_compressor=Mock(), _context_engine_owned=True)
    assert claim_engine_session_end(agent) is True
    assert claim_engine_session_end(agent) is True  # session-end callers may re-claim
    other = _bare_agent(context_compressor=Mock(), _context_engine_owned=True)
    other.release_clients()  # soft release claims once for itself
    assert claim_engine_session_end(other) is False  # the engine is already down


def test_child_soft_close_releases_child_engine_not_the_parents():
    child_engine, parent_engine = Mock(), Mock()
    child = _bare_agent(context_compressor=child_engine, _context_engine_owned=True)
    parent = _bare_agent(context_compressor=parent_engine, _context_engine_owned=True)
    parent._active_children = [child]
    parent._close_active_children(soft=True)
    child_engine.shutdown.assert_called_once_with()
    parent_engine.shutdown.assert_not_called()


def test_retired_agent_release_shuts_down_its_engine():
    # hermes_cli cli_agent_setup_mixin._retire_agent and managed_worker.retire_agent both drop
    # the instance right after release_clients(); the engine must go with it.
    engine = Mock()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.release_clients()
    engine.shutdown.assert_called_once_with()


def test_soft_release_defers_while_turn_inflight():
    # Finding 1: the worker thread is mid-turn (facade run_conversation holds a request) when the
    # gateway's memory-manager thread evicts. shutdown() must NOT land under the running turn.
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    begin_engine_request(agent)
    agent.release_clients()
    assert engine.calls == []
    assert getattr(agent, "_engine_inflight", 0) == 1
    assert getattr(agent, "_engine_release_pending", False) is True
    # A second eviction of the same (still busy) agent cannot add a second pending release.
    agent.release_clients()
    assert engine.calls == []


def test_pending_release_fires_at_turn_end_exactly_once():
    # The last user honours the latch: the shutdown the deferred eviction owed happens once, when
    # the turn's finally balances the count back to zero.
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    with engine_request(agent):
        agent.release_clients()
        assert engine.calls == []
    assert engine.calls == ["on_session_end", "shutdown"]
    assert getattr(agent, "_engine_inflight", 0) == 0
    assert getattr(agent, "_engine_release_pending", False) is False
    # The engine is down: a later eviction (or a replayed finally) must never re-shut it.
    agent.release_clients()
    end_engine_request(agent)
    assert engine.calls.count("shutdown") == 1


def test_parent_eviction_keeps_inflight_child_engine_alive():
    # Finding 1's sibling path: release_clients() soft-closes active children too. A child still
    # running its own turn keeps its engine open until that turn exits; the parent's engine goes.
    parent_engine, child_engine = _OrderingEngine(), _OrderingEngine()
    child = _bare_agent(context_compressor=child_engine, _context_engine_owned=True)
    parent = _bare_agent(context_compressor=parent_engine, _context_engine_owned=True)
    parent._active_children = [child]
    begin_engine_request(child)
    parent.release_clients()
    assert parent_engine.calls == ["on_session_end", "shutdown"]
    assert child_engine.calls == []
    end_engine_request(child)
    assert child_engine.calls == ["on_session_end", "shutdown"]


def test_fork_request_end_closes_engine_once():
    # Finding 2: a background-review fork (_register_background_review -> run_conversation, released
    # by _release_fork_clients) is a requester of ITS OWN engine. Reentrant pairing (nested requests)
    # must not close the engine early, and the shutdown must land exactly once at the last end.
    engine = _OrderingEngine()
    fork = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    with engine_request(fork):
        with engine_request(fork):  # nested user; the inner end must not trigger anything
            fork.release_clients()
            assert engine.calls == []
            assert getattr(fork, "_engine_inflight", 0) == 2  # outer turn + nested user both live
            assert getattr(fork, "_engine_release_pending", False) is True
        # The inner end balanced only its own claim: one holder is left, so the latch still waits.
        assert engine.calls == []
        assert getattr(fork, "_engine_inflight", 0) == 1
        assert getattr(fork, "_engine_release_pending", False) is True
    assert engine.calls == ["on_session_end", "shutdown"]
    assert engine.calls.count("shutdown") == 1


def test_fork_join_window_defers_parent_soft_eviction(monkeypatch):
    # Review round 2 (#147087): the fork joins parent._active_children BEFORE its first
    # run_conversation opens the turn-facade count. A parent eviction landing in that join
    # window must defer (pending), not shut the engine the about-to-start request uses; the
    # fork's own request end closes it, exactly once.
    import agent.background_review as br

    engine = _OrderingEngine()
    fork = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    fork._session_messages = []
    seen = {}

    def _stub_run_conversation(*args, **kwargs):
        seen["inflight"] = int(getattr(fork, "_engine_inflight", 0))
        return {}

    fork.run_conversation = _stub_run_conversation
    parent = _bare_agent(context_compressor=_OrderingEngine(), _context_engine_owned=True)

    real_track = br._track_review_fork

    def _track_then_evict(agent_arg, child, *, register):
        real_track(agent_arg, child, register=register)
        if register:
            # The window the reviewer flagged: joined the table, first request not open yet.
            agent_arg._close_active_children(soft=True)

    monkeypatch.setattr(br, "_track_review_fork", _track_then_evict)
    monkeypatch.setattr(br, "build_cache_parity_fork", lambda *a, **k: (fork, None, False))
    monkeypatch.setattr(br, "_review_tool_whitelist", lambda *a, **k: (set(), set()))
    monkeypatch.setattr(br, "_snapshot_review_usage", lambda ag: {})
    monkeypatch.setattr(br, "_record_review_usage_to_parent", lambda *a, **k: None)
    monkeypatch.setattr(br, "finish_background_review_run", lambda *a, **k: None)
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "set_thread_tool_whitelist", lambda *a, **k: None)
    monkeypatch.setattr(plugins, "clear_thread_tool_whitelist", lambda *a, **k: None)

    st = br._ReviewForkState()
    br._run_review_fork(parent, [], "prompt", None, None, st)

    assert seen["inflight"] >= 1, "join-window begin missing: the request opened with a zero count"
    assert engine.calls == ["on_session_end", "shutdown"], (
        "the join-window eviction must defer, then close once at the fork's own end")


def test_compress_mutation_defers_soft_release(monkeypatch):
    # Finding 2's other path: gateway/session_mutation_compress.prepare_compress summarizes in a
    # worker thread against the session's RESIDENT agent, so that mutation region must be bracketed
    # by engine_request(resident). Drive the real coroutine: the fake compressor plays the eviction
    # (release_clients) from inside compressor.compress and reports the resident's in-flight count.
    import asyncio
    from types import SimpleNamespace

    from gateway import session_policy
    from agent import context_compressor
    from gateway.session_mutation_compress import prepare_compress

    engine = _OrderingEngine()
    resident = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    seen = {}

    class _Summarizer:
        """Stands in for the out-of-process ContextCompressor; mid-compress, the MM thread evicts."""

        compression_count = 0

        def __init__(self, model, **kwargs):
            self.model = model

        def compress(self, messages, **kwargs):
            seen["inflight"] = getattr(resident, "_engine_inflight", 0)
            resident.release_clients()
            seen["calls_during_compress"] = list(engine.calls)
            return messages[:-1] + [{"role": "assistant", "content": "s", "_compressed_summary": "s"}]

    class _Policy:
        safe_mode = True

        def config(self, authority):
            return {"compression": {}}

    @contextmanager
    def _scope(policy, authority=None):
        yield

    monkeypatch.setattr(session_policy, "restore_policy", lambda receipt: _Policy())
    monkeypatch.setattr(session_policy, "policy_scope", _scope)
    monkeypatch.setattr(context_compressor, "ContextCompressor", _Summarizer)
    authority = SimpleNamespace(
        runner=SimpleNamespace(
            _resolve_session_agent_runtime=lambda **kw: ("m-1", {"api_mode": "openai"}),
            _cached_agent_for=lambda route: resident,
        ),
        db=SimpleNamespace(get_messages_as_conversation=lambda sid: [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]),
    )
    live = SimpleNamespace(source="web", route="route-1")
    prepared = {"snapshot": {"target": "sid-1", "receipt": {"policy": {}}}}
    result = asyncio.run(prepare_compress(authority, live, {}, prepared))

    assert result["messages"], "the committed mutation must carry the compressed history"
    assert seen["inflight"] == 1, "the compress mutation region must hold an engine request"
    assert seen["calls_during_compress"] == [], "eviction inside compress must not shut the engine down"
    assert engine.calls == ["on_session_end", "shutdown"], "the last user (compress exit) plays the pending release once"
    assert getattr(resident, "_engine_inflight", 0) == 0

    # Cold session (no resident agent): nothing to count, the immediate soft release is unchanged.
    cold_engine = _OrderingEngine()
    cold = _bare_agent(context_compressor=cold_engine, _context_engine_owned=True)
    with nullcontext():
        cold.release_clients()
    assert cold_engine.calls == ["on_session_end", "shutdown"]


class _RecordingEngine(_OrderingEngine):
    """Also records what each on_session_end carried, for the round-2 delivery pins."""

    def __init__(self):
        super().__init__()
        self.session_end_args = []

    def on_session_end(self, session_id, messages):
        self.session_end_args.append((str(session_id), len(messages or [])))
        super().on_session_end(session_id, messages)


def test_soft_released_cli_agent_still_gets_session_end_with_transcript():
    # Round 2 item 1 (the maintainer's probe, made a test): a CLI session runs some turns, the
    # user /model-switches — _retire_agent soft-releases the IDLE agent, so nothing is in flight
    # and the soft release owns the shutdown — and the process-exit finalizer then skips the
    # notify via the soft_release claim. Before this fix the engine got shutdown() with no
    # on_session_end AT ALL: an LCM-style store saves the last reply only at session end, so the
    # session's tail was silently dropped. The soft release now delivers the end, with the live
    # transcript, under its own claim and before shutdown(); the finalizer stays skipped.
    from hermes_cli.cli_agent_setup_mixin import _retire_agent
    from hermes_cli.cli_shutdown import _shutdown_agent_memory_provider

    engine = _RecordingEngine()
    agent = _bare_agent(
        context_compressor=engine, _context_engine_owned=True,
        _session_messages=[{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "yo"}])
    cli = SimpleNamespace(agent=agent)

    _retire_agent(cli)  # the REAL retire site: cli.agent = None after release_clients()
    assert cli.agent is None
    assert engine.calls == ["on_session_end", "shutdown"], (
        "the soft-released engine must receive on_session_end BEFORE shutdown, never nothing")
    assert engine.session_end_args == [("sid", 2)], (
        "the delivered session end must carry this session's live transcript, not an empty list")

    _shutdown_agent_memory_provider(agent)  # the REAL CLI exit finalizer
    assert engine.calls.count("on_session_end") == 1 and engine.calls.count("shutdown") == 1, (
        "the skipped finalizer must neither re-notify (claim soft_release) nor re-shut (once-flag)")


def test_hard_finalizer_takes_the_engine_claim_under_the_lock():
    # Round 2 item 2: the hard finalizer must TAKE the claim, not read it bare. A bare read can
    # see None one heartbeat before a racing release_clients() claims "soft_release" and shuts
    # the engine, landing on_session_end after the engine already stopped ingesting (the
    # reviewer's gateway-stop interleave). Observable state, no timing needed: with no prior
    # claim, the finalizer must leave the claim held as "session_end".
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True, _session_messages=[])
    agent.shutdown_memory_provider([])
    assert engine.calls == ["on_session_end", "shutdown"]
    assert getattr(agent, "_context_engine_teardown_claim", None) == "session_end", (
        "shutdown_memory_and_context read the claim instead of taking it via "
        "claim_engine_session_end() under the lock")


def test_real_turn_facade_wraps_the_turn_in_an_engine_request(monkeypatch):
    # Round 2 test gap: every other pin opens its requests BY HAND, so replacing
    # `with engine_request(self):` at agent/turn_facade.py with `if True:` left them all green —
    # findings 1+2 were unpinned for actual turns. Drive the REAL facade run_conversation with
    # only the inner conversation loop stubbed: the stub snapshots the count mid-turn and fires
    # the eviction the memory-manager thread would fire; the facade's own finally must play it.
    import agent.background_review as br
    import agent.conversation_loop as loop_mod
    import agent.relay_cwd as cwd_mod
    import agent.relay_runtime as rr
    import agent.turn_facade_lease as lease_mod
    from hermes_cli.observability import relay_shared_metrics as rsm

    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True, _session_messages=[])
    seen = {}

    def _stub_loop(*args, **kwargs):
        seen["inflight"] = int(getattr(agent, "_engine_inflight", 0))
        agent.release_clients()  # the eviction shape, mid-turn
        seen["mid"] = list(engine.calls)
        return {"final_response": "ok", "messages": [], "api_calls": 1, "completed": True}

    monkeypatch.setattr(loop_mod, "run_conversation", _stub_loop)
    monkeypatch.setattr(br, "cancel_background_review_for_live_turn", lambda *a, **k: None)
    monkeypatch.setattr(
        lease_mod, "admit_durable_turn_lease",
        lambda *a, **k: SimpleNamespace(early_result=None, lease=None, conversation_history=[]))
    monkeypatch.setattr(cwd_mod, "resolve_relay_scope_cwds", lambda *a, **k: (None, None))
    monkeypatch.setattr(rr.SESSION_COORDINATOR, "acquire_conversation", lambda *a, **k: None)
    monkeypatch.setattr(rr.SESSION_COORDINATOR, "begin_turn",
                        lambda *a, **k: SimpleNamespace(relay_enabled=False))
    monkeypatch.setattr(rr.SESSION_COORDINATOR, "finish_logical_calls", lambda *a, **k: None)
    monkeypatch.setattr(rr.SESSION_COORDINATOR, "end_turn", lambda *a, **k: None)
    monkeypatch.setattr(rr.SESSION_COORDINATOR, "release_conversation", lambda *a, **k: None)
    monkeypatch.setattr(rsm, "start_task_run", lambda *a, **k: None)
    monkeypatch.setattr(rsm, "finish_task_run", lambda *a, **k: None)

    result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert seen["inflight"] >= 1, (
        "the facade ran the turn with no engine request: the with-statement is the pin")
    assert seen["mid"] == [], "a mid-turn eviction must stay pending while the facade holds the turn"
    assert engine.calls == ["on_session_end", "shutdown"], (
        "the facade's finally must play the deferred release exactly once")
    assert getattr(agent, "_engine_inflight", 0) == 0
