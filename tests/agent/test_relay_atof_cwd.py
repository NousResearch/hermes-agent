"""ATOF workspace cwd on hermes.session / hermes.turn scope input.

Relay exports scope ``input`` as ATOF ``data``. These tests record the
``input`` kwarg on ``scope.push`` — the existing session-segment fake does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent import relay_runtime
from agent.relay_runtime import RelayRuntime, RelaySessionCoordinator
from agent import turn_facade


class _ScopeHandle:
    def __init__(self, name: str, seq: int) -> None:
        self.name = name
        self.seq = seq


class _FakeScopeModule:
    def __init__(self) -> None:
        self._seq = 0
        self.pushes: list[dict[str, Any]] = []
        self.pops: list[_ScopeHandle] = []

    def push(self, name: str, scope_type: Any, **kwargs: Any) -> _ScopeHandle:
        self._seq += 1
        self.pushes.append(
            {
                "name": name,
                "metadata": dict(kwargs.get("metadata") or {}),
                "parent": kwargs.get("handle"),
                "input": dict(kwargs.get("input") or {}),
                "seq": self._seq,
            }
        )
        return _ScopeHandle(name, self._seq)

    def pop(self, handle: _ScopeHandle, **kwargs: Any) -> None:
        self.pops.append(handle)

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeSubscribers:
    def flush(self) -> None:
        return None


class _FakeScopeType:
    Function = "function"
    Agent = "agent"


class _FakeRelay:
    def __init__(self) -> None:
        self.scope = _FakeScopeModule()
        self.subscribers = _FakeSubscribers()
        self.ScopeType = _FakeScopeType()

    def get_scope_stack(self) -> None:
        return None


_LIVE: list[tuple[RelayRuntime, _FakeRelay]] = []


def _make_runtime(fake: _FakeRelay) -> RelayRuntime:
    runtime = RelayRuntime(relay=fake, profile_key="/tmp/test-profile")
    _LIVE.append((runtime, fake))
    return runtime


@pytest.fixture(autouse=True)
def _teardown_runtimes():
    yield
    for runtime, _fake in _LIVE:
        runtime.shutdown()
    _LIVE.clear()


@pytest.fixture(autouse=True)
def _fast_scope_timeout(monkeypatch):
    monkeypatch.setattr(relay_runtime, "_SCOPE_OP_TIMEOUT", 1.0)


@pytest.fixture()
def coordinator() -> RelaySessionCoordinator:
    return RelaySessionCoordinator()


def _acquire(coordinator, runtime, session_id="sess-1", **kwargs):
    class _Registry:
        def for_profile(self, key):
            return runtime

    coordinator.registry = _Registry()
    return coordinator.acquire_conversation(
        profile_key=runtime.profile_key,
        session_id=session_id,
        platform="test",
        **kwargs,
    )


def _session_pushes(fake):
    return [p for p in fake.scope.pushes if p["name"] == relay_runtime.SESSION_SCOPE]


def _turn_pushes(fake):
    return [p for p in fake.scope.pushes if p["name"] == relay_runtime.TURN_SCOPE]


class TestSessionScopeInput:
    def test_acquire_with_cwd_sets_session_input(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime, cwd="/workspace/proj")
        assert _session_pushes(fake)[-1]["input"] == {"cwd": "/workspace/proj"}

    def test_acquire_without_cwd_omits_key(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime)
        assert "cwd" not in _session_pushes(fake)[-1]["input"]

    def test_acquire_empty_cwd_omits_key(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime, cwd="")
        assert "cwd" not in _session_pushes(fake)[-1]["input"]

    def test_remote_origin_included_with_cwd(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime, cwd="/remote/proj", cwd_origin="docker")
        assert _session_pushes(fake)[-1]["input"] == {
            "cwd": "/remote/proj",
            "cwd_origin": "docker",
        }

    def test_unknown_origin_omitted(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime, cwd="/workspace/proj", cwd_origin="local")
        assert _session_pushes(fake)[-1]["input"] == {"cwd": "/workspace/proj"}

    def test_initializer_sees_cwd_before_scope_open(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        seen: list[dict[str, Any]] = []

        def probe(_host, context):
            seen.append({"cwd": context.get("cwd"), "pushes": len(fake.scope.pushes)})

        coordinator.register_session_initializer("cwd-probe", probe)
        _acquire(coordinator, runtime, cwd="/workspace/proj")
        assert seen == [{"cwd": "/workspace/proj", "pushes": 0}]


class TestTurnScopeInput:
    def test_begin_turn_with_cwd_sets_turn_input(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime, cwd="/workspace/proj")
        coordinator.begin_turn(
            lease, turn_id="t1", task_id="task1", cwd="/workspace/task"
        )
        assert _turn_pushes(fake)[-1]["input"] == {"cwd": "/workspace/task"}
        assert _session_pushes(fake)[-1]["input"] == {"cwd": "/workspace/proj"}

    def test_begin_turn_without_cwd_uses_stored_session_cwd(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime, cwd="/workspace/proj")
        coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        assert _turn_pushes(fake)[-1]["input"] == {"cwd": "/workspace/proj"}

    def test_begin_turn_without_any_cwd_omits_key(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)
        coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        assert "cwd" not in _turn_pushes(fake)[-1]["input"]


class TestRotationPreservesCwd:
    def test_rotate_session_scope_reemits_stored_cwd(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime, cwd="/workspace/proj")
        assert len(_session_pushes(fake)) == 1
        runtime.rotate_session_scope(lease.session, reason="compaction")
        sessions = _session_pushes(fake)
        assert len(sessions) == 2
        assert sessions[-1]["input"] == {"cwd": "/workspace/proj"}

    def test_rotate_preserves_remote_origin(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime, cwd="/remote/proj", cwd_origin="docker")
        runtime.rotate_session_scope(lease.session, reason="max_turns")
        assert _session_pushes(fake)[-1]["input"] == {
            "cwd": "/remote/proj",
            "cwd_origin": "docker",
        }


class TestTurnFacadeResolver:
    def test_prefers_task_session_cwd(self, monkeypatch):
        monkeypatch.setattr(
            "tools.terminal_tool.get_session_cwd", lambda key: "/workspace/task"
        )

        class Agent:
            session_cwd = "/workspace/session"

        cwd, origin = turn_facade._relay_atof_cwd(Agent(), "task-1")
        assert cwd == "/workspace/task"
        assert origin is None

    def test_falls_back_to_agent_session_cwd(self, monkeypatch):
        monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda key: None)

        class Agent:
            session_cwd = "/workspace/session"

        cwd, origin = turn_facade._relay_atof_cwd(Agent(), "task-1")
        assert cwd == "/workspace/session"
        assert origin is None

    def test_omits_when_unconfigured_without_process_fallback(self, monkeypatch):
        monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda key: None)
        monkeypatch.setattr(
            "agent.runtime_cwd.resolve_context_cwd", lambda: None
        )
        called: list[str] = []
        monkeypatch.setattr(
            "agent.runtime_cwd.resolve_agent_cwd",
            lambda: called.append("resolve_agent_cwd") or Path("/tmp/invented"),
        )
        monkeypatch.setattr(
            "os.getcwd",
            lambda: called.append("getcwd") or "/tmp/invented",
        )

        class Agent:
            session_cwd = None

        cwd, origin = turn_facade._relay_atof_cwd(Agent(), "task-1")
        assert cwd is None
        assert origin is None
        assert called == []

    def test_resolver_error_does_not_raise(self, monkeypatch):
        def boom(_key):
            raise RuntimeError("cwd lookup failed")

        monkeypatch.setattr("tools.terminal_tool.get_session_cwd", boom)

        class Agent:
            session_cwd = "/workspace/session"

        cwd, origin = turn_facade._relay_atof_cwd(Agent(), "task-1")
        assert cwd == "/workspace/session"
        assert origin is None

    def test_remote_origin_from_agent_backend(self, monkeypatch):
        monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda key: None)

        class Agent:
            session_cwd = "/remote/proj"
            terminal_backend = "docker"

        cwd, origin = turn_facade._relay_atof_cwd(Agent(), "task-1")
        assert cwd == "/remote/proj"
        assert origin == "docker"
