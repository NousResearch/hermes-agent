"""Compaction marks land on the owning session's Relay scope stack.

A mark goes under the session's live turn when there is one, else under the session scope (gateway
hygiene runs outside any turn). Relay resets LLM-history freshness only for the agent scope that owns a
``compaction`` mark, so a mark on the wrong stack would leave the next LLM start projected.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent import compaction_events, relay_runtime
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
