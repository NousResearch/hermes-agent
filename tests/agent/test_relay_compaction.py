"""Compaction marks land on the owning session's Relay scope stack.

A mark goes under the session's live turn when there is one, else under the session scope (gateway
hygiene runs outside any turn). Relay resets LLM-history freshness only for the agent scope that owns a
``compaction`` mark, so a mark on the wrong stack would leave the next LLM start projected.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from agent import compaction_events, relay_compaction, relay_runtime
from agent.relay_compaction import COMPACTION_DATA_SCHEMA, emit_compaction_mark
from agent.relay_runtime import RelayRuntime, RelaySessionCoordinator


class _Handle:
    def __init__(self, name: str) -> None:
        self.name = name


class _Scope:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def push(self, name: str, _scope_type: Any, **_kwargs: Any) -> _Handle:
        return _Handle(name)

    def pop(self, _handle: _Handle, **_kwargs: Any) -> None:
        return None

    def event(self, name: str, **kwargs: Any) -> None:
        self.events.append({"name": name, **kwargs})


class _Plugin:
    def report(self) -> None:
        return None


class _Relay:
    class ScopeType:
        Function = "function"
        Agent = "agent"

    def __init__(self) -> None:
        self.scope = _Scope()
        self.plugin = _Plugin()

    def get_scope_stack(self) -> None:
        return None


class _Registry:
    def __init__(self, host: Any) -> None:
        self.host = host

    def for_profile(self, _profile_key: str | None = None, *, create: bool = True) -> Any:
        return self.host


@pytest.fixture
def relay(monkeypatch):
    """A Relay host for the active profile, reachable through the host registry."""
    fake = _Relay()
    runtime = RelayRuntime(relay=fake, profile_key=relay_runtime.current_profile_key())
    monkeypatch.setattr(relay_runtime, "HOST_REGISTRY", _Registry(runtime))
    coordinator = RelaySessionCoordinator(registry=_Registry(runtime))
    coordinator._prepare_session = lambda _host, _context: None
    yield fake, runtime, coordinator
    runtime.shutdown()


def _committed(session_id: str) -> dict[str, Any]:
    return compaction_events.attempt_payload(
        {"session_id": session_id, "commit_status": "committed", "route": "hermes", "trigger_source": "pre_api"}
    )


def test_mark_inside_a_turn_is_parented_to_that_turn(relay):
    fake, runtime, coordinator = relay
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="s1", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task")
    try:
        payload = _committed("s1")
        assert emit_compaction_mark("s1", compaction_events.event_name(payload), payload) is True
    finally:
        coordinator.end_turn(turn, outcome="success")

    [event] = fake.scope.events
    assert event["name"] == "compaction"
    assert event["handle"] is turn.handle
    assert event["data"] == payload
    assert event["data_schema"] == COMPACTION_DATA_SCHEMA
    assert event["metadata"][relay_runtime.RUNTIME_INSTANCE_KEY] == runtime.runtime_id


def test_mark_outside_a_turn_uses_the_session_scope(relay):
    fake, runtime, coordinator = relay
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="hygiene", platform="telegram")
    payload = compaction_events.attempt_payload({"session_id": "hygiene", "commit_status": "blocked"})

    assert emit_compaction_mark("hygiene", compaction_events.event_name(payload), payload) is True

    [event] = fake.scope.events
    assert (event["name"], event["handle"]) == ("compaction.attempt", lease.session.handle)


def test_mark_for_another_session_does_not_land_on_the_current_turn(relay):
    fake, runtime, coordinator = relay
    other = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="other", platform="cli")
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="s1", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task")
    try:
        emit_compaction_mark("other", "compaction", _committed("other"))
    finally:
        coordinator.end_turn(turn, outcome="success")

    [event] = fake.scope.events
    assert event["handle"] is other.session.handle


def test_no_relay_session_skips_quietly(relay):
    fake, _runtime, _coordinator = relay
    assert emit_compaction_mark("never-opened", "compaction", _committed("never-opened")) is False
    assert emit_compaction_mark("", "compaction", _committed("")) is False
    assert fake.scope.events == []


def test_no_relay_host_skips_quietly(monkeypatch):
    monkeypatch.setattr(relay_runtime, "HOST_REGISTRY", _Registry(None))
    assert emit_compaction_mark("s1", "compaction", _committed("s1")) is False


def test_stalled_relay_event_does_not_hold_the_caller(monkeypatch, relay):
    fake, runtime, coordinator = relay
    monkeypatch.setattr(relay_compaction, "_stalled", False)
    coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="s1", platform="cli")
    release, finished = threading.Event(), threading.Event()

    def stalled_event(_name, **_kwargs):
        release.wait()

    monkeypatch.setattr(fake.scope, "event", stalled_event)
    monkeypatch.setattr(relay_runtime, "SCOPE_OP_TIMEOUT", 2.0)
    result = []

    def publish():
        try:
            emit_compaction_mark("s1", "compaction", _committed("s1"))
        except Exception as exc:
            result.append(exc)
        finally:
            finished.set()

    publisher = threading.Thread(target=publish, daemon=True)
    try:
        publisher.start()
        assert finished.wait(timeout=4)
        assert len(result) == 1
        assert isinstance(result[0], TimeoutError)
        assert "Relay scope operation exceeded" in str(result[0])
        # One stall trips the breaker: the next compaction returns at once instead of waiting again.
        started = time.monotonic()
        assert emit_compaction_mark("s1", "compaction", _committed("s1")) is False
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        publisher.join(timeout=2)


def test_a_turn_relay_does_not_instrument_records_no_mark(relay):
    """A second concurrent turn on a live session (a background-review fork, or a concurrent gateway turn) runs
    with Relay instrumentation off. Its compaction must not mark the live session or reset its freshness."""
    fake, runtime, coordinator = relay
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="s1", platform="cli")
    live = coordinator.begin_turn(lease, turn_id="live", task_id="task")
    fork = coordinator.begin_turn(lease, turn_id="fork", task_id="task")
    try:
        assert fork.relay_enabled is False
        assert emit_compaction_mark("s1", "compaction", _committed("s1")) is False
    finally:
        coordinator.end_turn(fork, outcome="success")
        coordinator.end_turn(live, outcome="success")

    assert fake.scope.events == []


def test_a_turn_fallback_lands_only_on_the_turn_the_agent_started_in(relay):
    """An unscoped child session falls back to the live turn only when the agent's turn started in that
    turn's session; a turn of any other session never receives the mark."""
    fake, runtime, coordinator = relay
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="s1", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task")
    try:
        assert emit_compaction_mark("child", "compaction", _committed("child"), turn_session_id="other") is False
        assert emit_compaction_mark("child", "compaction", _committed("child"), turn_session_id="s1") is True
    finally:
        coordinator.end_turn(turn, outcome="success")

    [event] = fake.scope.events
    assert (event["handle"], event["data"]["session_id"]) == (turn.handle, "child")


def _transcript(start: int) -> list[dict[str, Any]]:
    filler = " ".join(["context"] * 200)
    return [
        message
        for idx in range(start, start + 10)
        for message in (
            {"role": "user", "content": f"user message {idx} {filler}"},
            {"role": "assistant", "content": f"assistant reply {idx} {filler}"},
        )
    ]


def test_every_rotation_in_one_turn_is_marked_on_that_turn(relay, tmp_path, monkeypatch):
    """With ``compression.in_place: false`` each commit moves the agent to a child session, and the child has
    no Relay scope while the turn that rotated into it is still running. A second rotation in that turn (a
    multi-pass preflight) must still be marked on the live turn, not dropped."""
    from agent.agent_runtime_helpers import note_turn_persisted, note_turn_start
    from hermes_state import SessionDB
    from run_agent import AIAgent

    fake, runtime, coordinator = relay
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
        session_db=SessionDB(db_path=tmp_path / "state.db"), session_id="rotating-session",
        skip_context_files=True, skip_memory=True,
    )
    agent.compression_in_place = False
    agent.context_compressor.tail_token_budget = 10
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="rotating-session", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task")
    note_turn_start(agent, "t1")
    try:
        with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
            first, _ = agent._compress_context(
                [{"role": "system", "content": "sys"}, *_transcript(0)], "sys", approx_tokens=80_000,
                trigger="turn_start_threshold",
            )
            child = agent.session_id
            agent._compress_context(first + _transcript(10), "sys", approx_tokens=80_000, trigger="turn_start_threshold")
    finally:
        note_turn_persisted(agent)
        coordinator.end_turn(turn, outcome="success")
        agent.close()

    assert "rotating-session" != child != agent.session_id
    marks = [event for event in fake.scope.events if event["name"].startswith("compaction")]
    assert [(mark["name"], mark["data"]["session_id"]) for mark in marks] == [
        ("compaction", "rotating-session"), ("compaction", child),
    ]
    assert all(mark["handle"] is turn.handle for mark in marks)
    assert all(mark["data"]["split_status"] == "rotated_committed" for mark in marks)


def test_a_post_turn_micro_compaction_in_a_rotated_child_is_marked_on_the_turn(relay, tmp_path, monkeypatch):
    """A rotating commit moves the agent to a child session with no Relay scope. The micro-compaction pass the
    finalizer then runs in that child, before the turn ends, must be marked on the live turn like the
    rotation itself, not dropped."""
    import logging

    from agent.agent_runtime_helpers import note_turn_persisted, note_turn_start
    from agent.turn_finalizer import _micro_compact_after_turn
    from hermes_state import SessionDB
    from run_agent import AIAgent

    fake, runtime, coordinator = relay
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
        session_db=SessionDB(db_path=tmp_path / "state.db"), session_id="rotating-session",
        skip_context_files=True, skip_memory=True,
    )
    agent.compression_in_place = False
    compressor = agent.context_compressor
    compressor.tail_token_budget = 10
    lease = coordinator.acquire_conversation(profile_key=runtime.profile_key, session_id="rotating-session", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task")
    note_turn_start(agent, "t1")
    try:
        with patch.object(compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
            compacted, _ = agent._compress_context(
                [{"role": "system", "content": "sys"}, *_transcript(0)], "sys", approx_tokens=80_000,
                trigger="turn_start_threshold",
            )
        child = agent.session_id
        compressor._micro_compact_enabled = True
        with patch.object(compressor, "_micro_summarize_one", return_value="ROLLING SUMMARY"):
            _micro_compact_after_turn(
                agent, compacted + _transcript(10), "done", logging.getLogger(__name__), "task",
            )
    finally:
        note_turn_persisted(agent)
        coordinator.end_turn(turn, outcome="success")
        agent.close()

    assert child != "rotating-session"
    micro = [event for event in fake.scope.events if event["data"]["kind"] == "micro_summarize"]
    assert [(mark["data"]["session_id"], mark["handle"]) for mark in micro] == [(child, turn.handle)]
