"""Bounded native scope operations in the Relay session coordinator.

The NeMo Relay native binding's ``scope.pop``/``scope.push`` are synchronous
and unbounded ("returns after the scope is closed successfully").  When the
native pipeline cannot make progress — proven live 2026-08-10 in the
delegation topology, where child sessions register scopes under the parent's
handle on a shared runtime — the coordinator's turn/session finalization
blocks forever inside ``run_conversation``.  Children finish their turns but
never return; delegation batches die on the stall watchdog.

Contract under test: observability must never block the product.  Scope
lifecycle operations that gate turn/session completion are bounded; on
breach the existing per-site exception handling degrades gracefully (warn,
retain diagnostics, continue) and the agent lives.  A lost span is always
the right trade against a dead agent.

These tests use a fake relay whose ``pop`` blocks on a never-set Event —
the minimal stand-in for a wedged native pipeline.  They assert the
END-TO-END contract (the coordinator method RETURNS) rather than any
helper's internal shape.
"""

from __future__ import annotations

import threading
import time
import types
from typing import Any

import pytest

from agent import relay_runtime
from agent.relay_runtime import (
    RelayRuntime,
    RelaySessionCoordinator,
)


# ---------------------------------------------------------------------------
# Fake relay: minimal surface the coordinator touches, with a wedgeable pop.
# ---------------------------------------------------------------------------


class _ScopeHandle:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeScopeModule:
    """Stands in for ``nemo_relay.scope`` with a controllable pop."""

    def __init__(self, wedge_event: threading.Event | None = None) -> None:
        self._wedge = wedge_event
        self.pushed: list[str] = []
        self.popped: list[str] = []

    def push(self, name: str, scope_type: Any, **kwargs: Any) -> _ScopeHandle:
        self.pushed.append(name)
        return _ScopeHandle(name)

    def pop(self, handle: _ScopeHandle, **kwargs: Any) -> None:
        if self._wedge is not None:
            # Simulates the wedged native pipeline: blocks until the event
            # is set — which the tests never do.
            self._wedge.wait()
        self.popped.append(handle.name)

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeSubscribers:
    def __init__(self, wedge_event: threading.Event | None = None) -> None:
        self._wedge = wedge_event
        self.flushed = 0

    def flush(self) -> None:
        if self._wedge is not None:
            self._wedge.wait()
        self.flushed += 1


class _FakeScopeType:
    Function = "function"
    Agent = "agent"


class _FakeRelay:
    def __init__(
        self,
        *,
        wedge_pop: threading.Event | None = None,
        wedge_flush: threading.Event | None = None,
    ) -> None:
        self.scope = _FakeScopeModule(wedge_pop)
        self.subscribers = _FakeSubscribers(wedge_flush)
        self.ScopeType = _FakeScopeType()

    def get_scope_stack(self) -> None:
        return None


def _make_runtime(fake_relay: _FakeRelay) -> RelayRuntime:
    """Build a RelayRuntime around the fake relay without native imports."""
    runtime = RelayRuntime(relay=fake_relay, profile_key="/tmp/test-profile")
    _LIVE_FAKES.append((runtime, fake_relay))
    return runtime


_LIVE_FAKES: list[tuple[RelayRuntime, _FakeRelay]] = []


@pytest.fixture(autouse=True)
def _release_wedges_after_test():
    """Unwedge every fake and drain runtimes at teardown.

    The wedge tests deliberately park shared-executor daemon workers on
    Event.wait() forever.  Without this teardown those workers — and the
    sessions still registered on each runtime's atexit shutdown hook —
    outlive the test session, and the interpreter's exit path re-runs the
    wedged pops (bounded, but 10s each): the CI per-file runner then hits
    its 300s file timeout AFTER '6 passed' (2026-08-12 CI hang).  Setting
    the events lets abandoned workers finish; shutdown() then drains fast
    and unregisters the atexit hook.
    """
    yield
    for runtime, fake in _LIVE_FAKES:
        for wedge in (fake.scope._wedge, fake.subscribers._wedge):
            if wedge is not None:
                wedge.set()
        runtime.shutdown()
    _LIVE_FAKES.clear()


def _run_with_join(fn, timeout: float = 5.0) -> tuple[bool, list[Any]]:
    """Run ``fn`` on a thread; return (returned_within_timeout, result)."""
    result: list[Any] = []

    def _target() -> None:
        result.append(fn())

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    return (not t.is_alive(), result)


@pytest.fixture()
def coordinator() -> RelaySessionCoordinator:
    return RelaySessionCoordinator()


@pytest.fixture(autouse=True)
def _fast_scope_timeout(monkeypatch):
    """Shrink the scope-op bound so wedge tests run in seconds.

    The production constant is generous (healthy ops are microseconds);
    tests only need 'bounded', not the specific bound.
    """
    monkeypatch.setattr(relay_runtime, "_SCOPE_OP_TIMEOUT", 1.0)


def _acquire(coordinator, runtime, session_id="sess-1", monkeypatch=None):
    """Acquire a conversation lease against the fake runtime."""

    class _Registry:
        def for_profile(self, key):
            return runtime

    coordinator.registry = _Registry()
    # _prepare_session invokes plugin initializers; make it inert for the
    # scope-op tests (plugin behavior is covered in tests/plugins/).
    coordinator._prepare_session = lambda host, ctx: None
    return coordinator.acquire_conversation(
        profile_key=runtime.profile_key,
        session_id=session_id,
        platform="test",
    )


# ---------------------------------------------------------------------------
# RED tests: today these HANG (the fake pop blocks forever) and the join
# times out.  Post-fix the coordinator bounds the native call and returns.
# ---------------------------------------------------------------------------


class TestBoundedScopeFinalization:
    def test_end_turn_returns_when_native_pop_wedges(self, coordinator):
        wedge = threading.Event()  # never set
        runtime = _make_runtime(_FakeRelay(wedge_pop=wedge))
        lease = _acquire(coordinator, runtime)
        assert lease.session is not None, "fake session must initialize"
        turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        assert turn.handle is not None, "turn scope must push on fake relay"

        returned, _ = _run_with_join(
            lambda: coordinator.end_turn(turn, outcome="success")
        )
        assert returned, (
            "end_turn must return even when the native scope.pop wedges — "
            "observability must never block turn completion"
        )

    def test_close_session_returns_when_native_pop_wedges(self, coordinator):
        wedge = threading.Event()
        runtime = _make_runtime(_FakeRelay(wedge_pop=wedge))
        lease = _acquire(coordinator, runtime)
        assert lease.session is not None

        returned, _ = _run_with_join(
            lambda: runtime.close_session({"session_id": "sess-1"})
        )
        assert returned, (
            "close_session must return even when the native scope.pop wedges"
        )

    def test_close_session_returns_when_subscriber_flush_wedges(
        self, coordinator
    ):
        wedge = threading.Event()
        runtime = _make_runtime(_FakeRelay(wedge_flush=wedge))
        lease = _acquire(coordinator, runtime)
        assert lease.session is not None

        returned, _ = _run_with_join(
            lambda: runtime.close_session({"session_id": "sess-1"})
        )
        assert returned, (
            "close_session must return even when subscribers.flush wedges"
        )

    def test_finish_logical_calls_returns_and_retains_prefix(
        self, coordinator
    ):
        wedge = threading.Event()
        runtime = _make_runtime(_FakeRelay(wedge_pop=wedge))
        lease = _acquire(coordinator, runtime)
        turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        # Register two logical LLM scopes the way relay_llm does.
        h1, h2 = _ScopeHandle("llm-1"), _ScopeHandle("llm-2")
        with turn.logical_llm_lock:
            turn.logical_llm_calls["req-1"] = h1
            turn.logical_llm_calls["req-2"] = h2

        returned, _ = _run_with_join(
            lambda: coordinator.finish_logical_calls(turn, outcome="success")
        )
        assert returned, (
            "finish_logical_calls must return even when native pops wedge"
        )
        # Existing diagnostic contract: the unclosed prefix is retained.
        with turn.logical_llm_lock:
            assert turn.logical_llm_calls, (
                "wedged logical scopes must be retained for diagnostics, "
                "not silently dropped"
            )


class TestHealthyPathUnchanged:
    """The bound must be invisible when the native pipeline is healthy."""

    def test_full_turn_lifecycle_healthy(self, coordinator):
        runtime = _make_runtime(_FakeRelay())  # no wedges
        lease = _acquire(coordinator, runtime)
        turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        assert turn.handle is not None

        coordinator.finish_logical_calls(turn, outcome="success")
        coordinator.end_turn(turn, outcome="success")
        coordinator.release_conversation(lease)
        runtime.close_session({"session_id": "sess-1"})

        fake = runtime.relay
        # Turn scope and session scope both pushed and popped exactly once.
        assert fake.scope.pushed.count(relay_runtime.TURN_SCOPE) == 1
        assert relay_runtime.TURN_SCOPE in fake.scope.popped
        assert fake.subscribers.flushed >= 1

    def test_healthy_pop_result_propagates_synchronously(self, coordinator):
        """A healthy pop completes and is observed before end_turn returns."""
        runtime = _make_runtime(_FakeRelay())
        lease = _acquire(coordinator, runtime)
        turn = coordinator.begin_turn(lease, turn_id="t1", task_id="task1")
        coordinator.end_turn(turn, outcome="success")
        assert relay_runtime.TURN_SCOPE in runtime.relay.scope.popped, (
            "healthy-path pop must complete before end_turn returns "
            "(no fire-and-forget on the default lane)"
        )
