"""Regression tests for #103301: a timed-out tool call must stop its delegated children.

The sequential and concurrent timeout paths only raised the worker thread's interrupt
bit (``tools.interrupt`` is per-thread); a ``delegate_task`` child runs on its own
daemon worker and only observes its own interrupt state, so after the parent reported
"timed out" the child kept running — issuing tool calls and git commits — with no stop
signal. ``agent.interrupt()`` already propagates to ``_active_children`` (see
``interrupt_control``); these tests pin the same propagation on both timeout paths.
"""

import concurrent.futures
import threading
import time

import pytest

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ConcurrentBatch,
    _ManagedToolResult,
    _ParsedCall,
    _ToolTimeoutResult,
    _interrupt_active_children,
    _run_sequential_tool_execution_middleware,
)


class _FakeChild:
    """Delegate-child double recording the stop signals it received."""

    def __init__(self) -> None:
        self.hard_interrupt_calls = []
        self._interrupt_requested = False

    def hard_interrupt(self, message=None, **kwargs):
        self.hard_interrupt_calls.append(message)
        self._interrupt_requested = True


class _LegacyChild:
    """Third-party/legacy child exposing only ``interrupt(message=None)``."""

    def __init__(self) -> None:
        self.interrupt_calls = []

    def interrupt(self, message=None, **kwargs):
        self.interrupt_calls.append(message)


class _BareChild:
    """Child double with no interrupt ABI at all (fallback flag path)."""

    def __init__(self) -> None:
        self._interrupt_requested = False


class _FakeAgent:
    def __init__(self, children=None):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._active_children = list(children or [])
        self._active_children_lock = threading.Lock()
        self._interrupt_requested = False
        self.activity = []

    def _touch_activity(self, msg):
        self.activity.append(msg)


@pytest.fixture()
def fake_agent():
    return _FakeAgent()


@pytest.fixture(autouse=True)
def _quiet_emits(monkeypatch):
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda agent, **kw: None,
    )


def test_sequential_timeout_stops_delegate_children(monkeypatch, fake_agent):
    """The sequential deadline path must propagate the stop to active children."""

    child = _FakeChild()
    fake_agent._active_children.append(child)

    def _blocking_middleware(agent_arg, **kwargs):
        time.sleep(30)  # non-cooperative: never returns on its own
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _blocking_middleware)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="delegate_task",
        function_args={"goal": "x"},
        effective_task_id="t",
        tool_call_id="call_1",
        execute=lambda a: "unused",
    )

    assert isinstance(managed.result, _ToolTimeoutResult)
    assert "timed out after 0.2s" in str(managed.result)
    # The child saw the stop signal even though the tool itself never cooperated.
    assert child.hard_interrupt_calls == [None]
    assert child._interrupt_requested is True


def test_concurrent_batch_timeout_stops_delegate_children(fake_agent):
    """The concurrent batch deadline path must propagate the stop to active children."""

    child = _FakeChild()
    fake_agent._active_children.append(child)

    parsed = _ParsedCall(
        tool_call=None, name="delegate_task", args={}, middleware_trace=[],
        parse_error=None, scope_block=None,
    )
    batch = _ConcurrentBatch(fake_agent, [], "t", [parsed], 0.1)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(lambda: time.sleep(30))
        # Deadline already elapsed: the wait loop must take the timed_out branch.
        abandoned = batch.await_completion({future}, {future: 0}, time.monotonic() - 1.0)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert abandoned is True
    assert batch.timed_out_indices == {0}
    assert child.hard_interrupt_calls == [None]


def test_helper_covers_all_child_abis():
    """hard_interrupt children get the new ABI; legacy children get interrupt();
    ABI-less children still get the cooperative flag."""

    agent = _FakeAgent(children=[])
    hard, legacy, bare = _FakeChild(), _LegacyChild(), _BareChild()
    agent._active_children = [hard, legacy, bare]

    _interrupt_active_children(agent)

    assert hard.hard_interrupt_calls == [None]
    assert legacy.interrupt_calls == [None]
    assert bare._interrupt_requested is True


def test_helper_safe_without_children():
    """Agents without the delegation registry (or with no active children) are a no-op."""

    _interrupt_active_children(_FakeAgent(children=[]))

    class _Minimal:
        pass

    _interrupt_active_children(_Minimal())  # no _active_children attribute at all
