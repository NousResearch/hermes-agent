"""Session-span segmentation for continuous sessions.

Continuous gateway sessions keep the Relay session scope open indefinitely;
close-driven export means the session root span (and out-of-turn marks) are
unexported until /new or idle-end, and a crash loses the whole segment.

Segmentation closes the current session scope at a TURN BOUNDARY and pushes
a fresh one, chaining segments via metadata:

  gateway.telemetry.session_segments.on_compaction  (default False)
  gateway.telemetry.session_segments.max_turns      (default 0 = unlimited)

Both defaults off => behavior identical to today (no rotation, ever).
Rotation never happens mid-turn: compaction only sets rotate_pending,
consumed at the next begin_turn before the turn scope pushes.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from agent import relay_runtime
from agent.relay_runtime import (
    RelayRuntime,
    RelaySessionCoordinator,
)


class _ScopeHandle:
    def __init__(self, name: str, seq: int) -> None:
        self.name = name
        self.seq = seq


class _FakeScopeModule:
    def __init__(self, wedge_pop: threading.Event | None = None) -> None:
        self._wedge = wedge_pop
        self._seq = 0
        self.pushes: list[dict[str, Any]] = []  # {name, metadata, handle}
        self.pops: list[_ScopeHandle] = []

    def push(self, name: str, scope_type: Any, **kwargs: Any) -> _ScopeHandle:
        self._seq += 1
        self.pushes.append(
            {
                "name": name,
                "metadata": dict(kwargs.get("metadata") or {}),
                "parent": kwargs.get("handle"),
                "seq": self._seq,
            }
        )
        return _ScopeHandle(name, self._seq)

    def pop(self, handle: _ScopeHandle, **kwargs: Any) -> None:
        if self._wedge is not None:
            self._wedge.wait()
        self.pops.append(handle)

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeSubscribers:
    def __init__(self) -> None:
        self.flushed = 0

    def flush(self) -> None:
        self.flushed += 1


class _FakeScopeType:
    Function = "function"
    Agent = "agent"


class _FakeRelay:
    def __init__(self, wedge_pop: threading.Event | None = None) -> None:
        self.scope = _FakeScopeModule(wedge_pop)
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
    """Unwedge and drain every runtime so exit paths never replay wedged ops."""
    yield
    for runtime, fake in _LIVE:
        if fake.scope._wedge is not None:
            fake.scope._wedge.set()
        runtime.shutdown()
    _LIVE.clear()


@pytest.fixture(autouse=True)
def _fast_scope_timeout(monkeypatch):
    monkeypatch.setattr(relay_runtime, "_SCOPE_OP_TIMEOUT", 1.0)


@pytest.fixture(autouse=True)
def _default_config(monkeypatch):
    """No config on disk by default; tests override _segments_config directly."""
    monkeypatch.setattr(
        "gateway.run._load_gateway_config", lambda: {}, raising=False
    )
    relay_runtime._reset_segments_config_for_tests()


def _set_segments(monkeypatch, *, on_compaction=False, max_turns=0):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "gateway": {
                "telemetry": {
                    "session_segments": {
                        "on_compaction": on_compaction,
                        "max_turns": max_turns,
                    }
                }
            }
        },
        raising=False,
    )
    relay_runtime._reset_segments_config_for_tests()


@pytest.fixture()
def coordinator() -> RelaySessionCoordinator:
    return RelaySessionCoordinator()


def _acquire(coordinator, runtime, session_id="sess-1"):
    class _Registry:
        def for_profile(self, key):
            return runtime

    coordinator.registry = _Registry()
    coordinator._prepare_session = lambda host, ctx: None
    return coordinator.acquire_conversation(
        profile_key=runtime.profile_key,
        session_id=session_id,
        platform="test",
    )


def _session_pushes(fake):
    return [p for p in fake.scope.pushes if p["name"] == relay_runtime.SESSION_SCOPE]


def _run_turn(coordinator, lease, turn_id):
    turn = coordinator.begin_turn(lease, turn_id=turn_id, task_id=f"task-{turn_id}")
    coordinator.end_turn(turn, outcome="success")
    return turn


class TestDefaultsNeverRotate:
    def test_no_rotation_across_many_turns_and_compactions(self, coordinator):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)
        assert lease.session is not None

        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="sess-1"
        )
        for i in range(5):
            _run_turn(coordinator, lease, f"t{i}")

        assert len(_session_pushes(fake)) == 1, (
            "defaults off must never rotate the session scope — "
            "today's behavior is the contract"
        )


class TestCompactionRotation:
    def test_compaction_rotates_at_next_begin_turn_not_immediately(
        self, coordinator, monkeypatch
    ):
        _set_segments(monkeypatch, on_compaction=True)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)
        original_handle = lease.session.handle

        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="sess-1"
        )
        # No rotation yet — compaction only flags; scope stack untouched.
        assert len(_session_pushes(fake)) == 1
        assert not fake.scope.pops

        turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        sessions = _session_pushes(fake)
        assert len(sessions) == 2, "rotation must happen at the next begin_turn"
        # Old session scope was popped before the new push.
        assert any(p.seq == 1 for p in fake.scope.pops), "old segment scope popped"
        assert lease.session.handle is not original_handle
        # The turn scope parents to the NEW segment handle.
        turn_push = [p for p in fake.scope.pushes if p["name"] == relay_runtime.TURN_SCOPE][-1]
        assert turn_push["parent"] is lease.session.handle
        coordinator.end_turn(turn, outcome="success")

    def test_segment_metadata_on_rotated_scope(self, coordinator, monkeypatch):
        _set_segments(monkeypatch, on_compaction=True)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)

        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="sess-1"
        )
        _run_turn(coordinator, lease, "t1")

        new_seg = _session_pushes(fake)[-1]["metadata"]
        assert new_seg.get("hermes.session.segment") == 1
        assert new_seg.get("hermes.session.segment_reason") == "compaction"

    def test_unknown_session_compaction_is_noop(self, coordinator, monkeypatch):
        _set_segments(monkeypatch, on_compaction=True)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime)
        # Must not raise, must not rotate anything.
        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="never-seen"
        )
        assert len(_session_pushes(fake)) == 1

    def test_rotating_compaction_closes_old_session_scope(
        self, coordinator, monkeypatch
    ):
        """Legacy compaction rotates to a child session id: the OLD session's
        scope must close (export) instead of orphaning unexported forever."""
        _set_segments(monkeypatch, on_compaction=True)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime, session_id="parent-1")
        assert not fake.scope.pops

        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key,
            session_id="child-1",
            old_session_id="parent-1",
        )
        assert len(fake.scope.pops) == 1, (
            "rotating compaction must close the old session scope"
        )
        assert fake.subscribers.flushed >= 1

    def test_rotating_compaction_noop_when_disabled(self, coordinator, monkeypatch):
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        _acquire(coordinator, runtime, session_id="parent-1")
        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key,
            session_id="child-1",
            old_session_id="parent-1",
        )
        assert not fake.scope.pops, "defaults off: rotating compaction is a no-op"


class TestMaxTurnsRotation:
    def test_rotates_after_cap(self, coordinator, monkeypatch):
        _set_segments(monkeypatch, max_turns=2)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)

        for i in range(5):
            _run_turn(coordinator, lease, f"t{i}")

        # turns 0,1 in segment 0; rotation before turn 2; turns 2,3 in
        # segment 1; rotation before turn 4.
        sessions = _session_pushes(fake)
        assert len(sessions) == 3, "cap of 2 over 5 turns => 2 rotations"
        assert sessions[-1]["metadata"].get("hermes.session.segment_reason") == "max_turns"

    def test_zero_cap_means_unlimited(self, coordinator, monkeypatch):
        _set_segments(monkeypatch, max_turns=0)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)
        for i in range(4):
            _run_turn(coordinator, lease, f"t{i}")
        assert len(_session_pushes(fake)) == 1


class TestRotationSafety:
    def test_never_rotates_mid_turn(self, coordinator, monkeypatch):
        _set_segments(monkeypatch, on_compaction=True)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)

        turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        # Compaction lands while the turn is LIVE.
        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="sess-1"
        )
        assert len(_session_pushes(fake)) == 1, "no rotation while a turn is live"
        coordinator.end_turn(turn, outcome="success")
        assert len(_session_pushes(fake)) == 1, "end_turn does not rotate either"

        # The NEXT turn consumes the pending rotation.
        turn2 = coordinator.begin_turn(lease, turn_id="t2", task_id="task2")
        assert len(_session_pushes(fake)) == 2
        coordinator.end_turn(turn2, outcome="success")

    def test_wedged_rotation_is_bounded_and_agent_continues(
        self, coordinator, monkeypatch
    ):
        _set_segments(monkeypatch, on_compaction=True)
        wedge = threading.Event()  # never set until teardown
        fake = _FakeRelay(wedge_pop=wedge)
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)
        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="sess-1"
        )

        result: list[Any] = []

        def _begin():
            result.append(
                coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
            )

        worker = threading.Thread(target=_begin, daemon=True)
        worker.start()
        worker.join(5.0)
        assert not worker.is_alive(), (
            "begin_turn must return even when the rotation pop wedges — "
            "a wedged pipeline costs one segment span, never the agent"
        )
        turn = result[0]
        coordinator.end_turn(turn, outcome="success")

    def test_subagent_children_parent_to_new_segment_after_rotation(
        self, coordinator, monkeypatch
    ):
        _set_segments(monkeypatch, on_compaction=True)
        fake = _FakeRelay()
        runtime = _make_runtime(fake)
        lease = _acquire(coordinator, runtime)
        coordinator.notify_session_compacted(
            profile_key=runtime.profile_key, session_id="sess-1"
        )
        _run_turn(coordinator, lease, "t1")  # consumes rotation
        new_handle = lease.session.handle

        child = runtime.register_subagent(
            {"parent_session_id": "sess-1", "child_session_id": "child-1"}
        )
        assert child is not None
        child_push = [
            p
            for p in fake.scope.pushes
            if p["name"] == relay_runtime.SESSION_SCOPE
            and p["parent"] is not None
        ][-1]
        assert child_push["parent"] is new_handle, (
            "post-rotation children must parent to the new segment handle"
        )
