"""Deterministic, hermetic contract tests for the public subagent lifecycle API.

This module is the canonical contract oracle for ``agent/subagent_lifecycle.py``.
It drives the service through the *observed* state graph using only the public
methods (``launch`` / ``status`` / ``wait`` / ``cancel`` / ``result`` /
``reconnect``) plus faked child construction and execution — no live provider,
no wall-clock sleeps, no subprocess, and no credential-shaped env vars.

Every transition asserted here is one the production service actually
performs.  The enum's ``STARTING`` state is deliberately *not* treated as
observable: the service never enters it, so the contract documents the real
edge (``PENDING`` -> ``RUNNING``) and asserts that no state is skipped.
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.subagent_lifecycle import (
    PUBLIC_CONTRACT_VERSION,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    bind_subagent_parent,
    get_active_subagent_parent,
)

# ── Canonical observed transition matrix ────────────────────────────────────
# (from_state, event, to_state).  Events are the observable API actions that
# drive the state machine — they are NOT a public transition RPC; the tests
# drive them through the real service and observe the resulting state.
CANONICAL_TRANSITIONS = [
    ("PENDING", "start", "RUNNING"),
    ("RUNNING", "succeed", "SUCCEEDED"),
    ("RUNNING", "fail", "FAILED"),
    ("RUNNING", "interrupt", "INTERRUPTED"),
    ("RUNNING", "cancel", "CANCEL_REQUESTED"),
    ("CANCEL_REQUESTED", "cancel-complete", "CANCELLED"),
]

# Transitions the service must reject.  ``STARTING`` appears here only to
# document that it is never a valid source — it is not observable.
CANONICAL_INVALID_TRANSITIONS = [
    ("PENDING", "succeed"),
    ("STARTING", "succeed"),
    ("SUCCEEDED", "run"),
    ("FAILED", "cancel"),
    ("INTERRUPTED", "cancel"),
    ("CANCELLED", "cancel"),
]


class _ControlledExecutor:
    """Runs submitted tasks in a single daemon thread, one at a time.

    ``submit`` queues a task; ``start_next`` runs the next queued task to
    completion in a daemon thread; ``join`` waits for it.  This lets tests
    observe intermediate states (``PENDING``, ``RUNNING``) without real
    wall-clock timing or a thread pool.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[object, tuple, Future]] = []
        self._thread: threading.Thread | None = None

    def submit(self, fn, *args):
        future = Future()
        self._pending.append((fn, args, future))
        return future

    def start_next(self) -> None:
        fn, args, future = self._pending.pop(0)

        def invoke() -> None:
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # pragma: no cover - defensive
                future.set_exception(exc)

        self._thread = threading.Thread(target=invoke, daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5)
            assert not self._thread.is_alive()


class _FakeChild:
    """Minimal child double: identity fields plus the hard-interrupt ABI."""

    def __init__(self, subagent_id: str) -> None:
        self._subagent_id = subagent_id
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "contract-provider"
        self.model = "contract-model"
        self.interrupted = False

    def hard_interrupt(self, _reason, *, tool_reason=None) -> None:
        del tool_reason
        self.interrupted = True


@pytest.fixture
def contract():
    """A hermetic lifecycle service with controlled child construction.

    Yields a namespace with:

    * ``service`` — the ``SubagentLifecycleService`` under test.
    * ``executor`` — the controlled executor driving child runs.
    * ``launch(goal)`` — launch a child with the given goal, returning
      ``(handle, running, release)`` where ``running`` is set once the child's
      run function starts and ``release`` unblocks the child.
    """
    parent = SimpleNamespace(
        session_id="parent-contract", enabled_toolsets=["file"]
    )
    executor = _ControlledExecutor()
    gates: dict[str, tuple[threading.Event, threading.Event]] = {}
    counter = iter(range(1, 1_000))

    def build(**_kwargs):
        return _FakeChild(f"sa-{next(counter)}")

    def run(_index, goal, child, _parent):
        running, release = gates[goal]
        running.set()
        if goal == "success":
            release.wait(5)
            return {"status": "completed", "summary": "ok", "api_calls": 1}
        if goal == "failure":
            release.wait(5)
            return {"status": "failed", "error": "boom", "api_calls": 1}
        if goal == "interrupted":
            release.wait(5)
            return {"status": "interrupted", "api_calls": 0}
        if goal == "interrupt":
            # Cooperative cancellation: the child honours a requested
            # interrupt and reports "interrupted" only after cancel() flips
            # child.interrupted.  release.wait() (no polling) keeps it
            # deterministic.
            release.wait(5)
            return {
                "status": "interrupted" if child.interrupted else "completed",
                "api_calls": 0,
            }
        if goal == "block":
            release.wait(5)
            return {"status": "completed", "summary": "ok", "api_calls": 1}
        release.wait(5)
        return {"status": "completed", "summary": "ok", "api_calls": 1}

    with patch("agent.subagent_lifecycle._EXECUTOR", executor), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    ), patch("tools.delegate_tool._run_child_lifecycle", run):
        service = SubagentLifecycleService(lambda: parent)

        def launch(goal: str):
            running, release = threading.Event(), threading.Event()
            gates[goal] = (running, release)
            handle = service.launch(SubagentLaunchRequest(goal=goal))
            assert service.status(handle).state is SubagentState.PENDING
            return handle, running, release

        yield SimpleNamespace(
            service=service, executor=executor, launch=launch
        )


# ── Transition matrix ───────────────────────────────────────────────────────

def test_pending_to_running(contract):
    """PENDING + start -> RUNNING: the child starts when the executor runs it."""
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    executor.start_next()
    assert running.wait(5)
    assert service.status(handle).state is SubagentState.RUNNING
    release.set()
    executor.join()


def test_running_to_succeeded(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    executor.start_next()
    assert running.wait(5)
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.SUCCEEDED
    assert service.result(handle).terminal_state is SubagentState.SUCCEEDED
    assert service.result(handle).error_classification is None


def test_running_to_failed(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("failure")
    executor.start_next()
    assert running.wait(5)
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.FAILED
    assert service.result(handle).terminal_state is SubagentState.FAILED
    assert service.result(handle).error_classification == "FAILED"


def test_running_to_interrupted(contract):
    """RUNNING + interrupt -> INTERRUPTED: an external stop, not a cancel."""
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("interrupted")
    executor.start_next()
    assert running.wait(5)
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.INTERRUPTED
    assert service.result(handle).terminal_state is SubagentState.INTERRUPTED


def test_running_to_cancel_requested(contract):
    """RUNNING + cancel -> CANCEL_REQUESTED: cancellation is cooperative."""
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("interrupt")
    executor.start_next()
    assert running.wait(5)
    result = service.cancel(handle, reason="fixture cancellation")
    assert result.accepted is True
    assert result.state is SubagentState.CANCEL_REQUESTED
    assert service.status(handle).state is SubagentState.CANCEL_REQUESTED
    release.set()
    executor.join()


def test_cancel_requested_to_cancelled(contract):
    """CANCEL_REQUESTED + cancel-complete -> CANCELLED: the child honours it."""
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("interrupt")
    executor.start_next()
    assert running.wait(5)
    service.cancel(handle, reason="fixture cancellation")
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.CANCELLED
    assert service.result(handle).terminal_state is SubagentState.CANCELLED


def test_canonical_matrix_is_well_formed():
    """The documented transition matrix references real states with no dupes."""
    states = {state.name for state in SubagentState}
    seen = set()
    for from_state, event, to_state in CANONICAL_TRANSITIONS:
        assert from_state in states, f"unknown source state: {from_state}"
        assert to_state in states, f"unknown target state: {to_state}"
        key = (from_state, event)
        assert key not in seen, f"duplicate edge: {key}"
        seen.add(key)
    for from_state, event in CANONICAL_INVALID_TRANSITIONS:
        assert from_state in states, f"unknown source state: {from_state}"


def test_canonical_transition_matrix_is_observed(contract):
    """Every documented transition is reachable through the public API."""
    service, executor, launch = contract.service, contract.executor, contract.launch

    # PENDING -> RUNNING
    handle, running, release = launch("success")
    executor.start_next()
    assert running.wait(5)
    assert service.status(handle).state is SubagentState.RUNNING

    # RUNNING -> SUCCEEDED
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.SUCCEEDED

    # RUNNING -> FAILED
    handle, running, release = launch("failure")
    executor.start_next()
    assert running.wait(5)
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.FAILED

    # RUNNING -> INTERRUPTED
    handle, running, release = launch("interrupted")
    executor.start_next()
    assert running.wait(5)
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.INTERRUPTED

    # RUNNING -> CANCEL_REQUESTED -> CANCELLED
    handle, running, release = launch("interrupt")
    executor.start_next()
    assert running.wait(5)
    assert service.cancel(handle, reason="x").state is SubagentState.CANCEL_REQUESTED
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.CANCELLED


def test_invalid_transitions_are_rejected(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch

    # PENDING + succeed: a launched child cannot skip to a terminal state.
    handle, running, release = launch("success")
    assert service.status(handle).state is SubagentState.PENDING
    assert service.result(handle).error_classification == "NOT_READY"

    # STARTING + succeed: the service never enters the unsupported STARTING
    # state; it goes straight from PENDING to RUNNING.
    executor.start_next()
    assert running.wait(5)
    assert service.status(handle).state is SubagentState.RUNNING
    release.set()
    executor.join()

    # SUCCEEDED + cancel / FAILED + cancel / INTERRUPTED + cancel: terminal
    # handles cannot be cancelled.
    for goal, terminal in [
        ("success", SubagentState.SUCCEEDED),
        ("failure", SubagentState.FAILED),
        ("interrupted", SubagentState.INTERRUPTED),
    ]:
        handle, running, release = launch(goal)
        executor.start_next()
        assert running.wait(5)
        release.set()
        executor.join()
        assert service.status(handle).state is terminal
        result = service.cancel(handle, reason="too late")
        assert result.accepted is False
        assert result.already_terminal is True
        assert result.state is terminal

    # CANCELLED + cancel: a cancelled handle is already terminal.
    handle, running, release = launch("interrupt")
    executor.start_next()
    assert running.wait(5)
    service.cancel(handle, reason="cancel")
    release.set()
    executor.join()
    assert service.status(handle).state is SubagentState.CANCELLED
    result = service.cancel(handle, reason="too late")
    assert result.accepted is False
    assert result.already_terminal is True
    assert result.state is SubagentState.CANCELLED


# ── Observable API outcomes ─────────────────────────────────────────────────

def test_launch_validation(contract):
    service = contract.service
    bad_requests = [
        SubagentLaunchRequest(goal=""),
        SubagentLaunchRequest(goal="   "),
        SubagentLaunchRequest(goal="x", role="invalid"),
        SubagentLaunchRequest(goal="x", timeout_seconds=1),
        SubagentLaunchRequest(goal="x", working_directory="/tmp"),
        SubagentLaunchRequest(goal="x", blocked_tools=("shell",)),
        SubagentLaunchRequest(goal="x", allowed_toolsets=("nope",)),
        SubagentLaunchRequest(goal="x", allowed_toolsets=("web",)),
        SubagentLaunchRequest(goal="x", metadata={"bad": object()}),
        SubagentLaunchRequest(goal="x" * 20_000),
        SubagentLaunchRequest(goal="x", context=123),
    ]
    for request in bad_requests:
        with pytest.raises(SubagentLifecycleError):
            service.launch(request)


def test_valid_launch_returns_conformant_handle(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    try:
        assert handle.contract_version == PUBLIC_CONTRACT_VERSION
        assert handle.subagent_id.startswith("sa-")
        assert handle.parent_session_id == "parent-contract"
        assert handle.role == "leaf"
        assert handle.depth == 1
        assert handle.provider == "contract-provider"
        assert handle.model == "contract-model"
        assert isinstance(handle.capability, str) and handle.capability
        assert handle.to_dict()["subagent_id"] == handle.subagent_id
        # A round-tripped handle resolves to the same record.
        round_tripped = type(handle).from_dict(handle.to_dict())
        assert service.status(round_tripped).state is SubagentState.PENDING
    finally:
        release.set()
        executor.join()


def test_forged_handle_is_unknown(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    try:
        forged = dataclasses.replace(handle, capability="forged")
        assert service.status(forged).state is SubagentState.UNKNOWN
        assert service.result(forged).error_classification == "UNKNOWN_HANDLE"
        assert service.cancel(forged, reason="x").unknown_handle is True
        assert service.reconnect(forged).connected is False
    finally:
        release.set()
        executor.join()


def test_cross_parent_isolation(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    try:
        other_parent = SimpleNamespace(
            session_id="other-parent", enabled_toolsets=["file"]
        )
        other_service = SubagentLifecycleService(lambda: other_parent)
        assert other_service.status(handle).state is SubagentState.UNKNOWN
        assert other_service.reconnect(handle).connected is False
    finally:
        release.set()
        executor.join()


def test_result_before_ready(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    try:
        result = service.result(handle)
        assert result.terminal_state is SubagentState.PENDING
        assert result.ready is False
        assert result.error_classification == "NOT_READY"
    finally:
        release.set()
        executor.join()


def test_wait_timeout_keeps_child_running(contract):
    """The caller's wait timeout fires while the child stays RUNNING."""
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("block")
    executor.start_next()
    assert running.wait(5)
    timed = service.wait(handle, timeout_seconds=0.001)
    assert timed.timed_out is True
    assert timed.completed is False
    assert timed.state is SubagentState.RUNNING
    # After the caller's timeout, the fixture deterministically drives the
    # child to a terminal state.
    release.set()
    executor.join()
    assert service.wait(handle).state is SubagentState.SUCCEEDED


def test_reconnect_reports_state(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    handle, running, release = launch("success")
    assert service.reconnect(handle).connected is True
    executor.start_next()
    assert running.wait(5)
    assert service.reconnect(handle).state is SubagentState.RUNNING
    release.set()
    executor.join()
    assert service.reconnect(handle).state is SubagentState.SUCCEEDED


def test_duplicate_correlation_is_rejected(contract):
    service = contract.service
    service.launch(SubagentLaunchRequest(goal="success", correlation_id="corr-1"))
    with pytest.raises(SubagentLifecycleError):
        service.launch(
            SubagentLaunchRequest(goal="success", correlation_id="corr-1")
        )


def test_outcomes_are_deterministic(contract):
    service, executor, launch = contract.service, contract.executor, contract.launch
    outcomes = []
    for _ in range(2):
        handle, running, release = launch("success")
        executor.start_next()
        assert running.wait(5)
        release.set()
        executor.join()
        outcomes.append(
            (
                service.result(handle).terminal_state,
                service.result(handle).error_classification,
            )
        )
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == (SubagentState.SUCCEEDED, None)


def test_parent_binding_is_context_safe():
    assert get_active_subagent_parent() is None
    parent = object()
    with bind_subagent_parent(parent):
        assert get_active_subagent_parent() is parent
    assert get_active_subagent_parent() is None