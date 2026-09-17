"""Sequential tool execution must abandon the wait when the user interrupts.

Regression tests for the "interrupt doesn't end a running tool" class:
the sequential executor path previously ran the tool inline (when the
deadline was disabled) or waited in 5s slices without checking
``agent._interrupt_requested`` — a non-cooperative tool (e.g. a blocking
FAL ``handler.get()``) held the whole turn hostage until it returned.

Now the wait loop polls the interrupt flag every
``_SEQUENTIAL_INTERRUPT_POLL_SECONDS`` and, after a 3s cooperative grace,
synthesizes a cancelled tool result and abandons the worker.
"""

import threading
import time

import pytest

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolCancelledResult,
    _run_sequential_tool_execution_middleware,
)
from tests.agent._liveness import wait_event

# The reported tool runtime (the non-cooperative tool in this file blocks this
# long): a ceiling tied to *that* is the invariant under test — "abandons well
# before the tool's own runtime" — instead of a free-floating wall-clock number
# that quietly encodes how fast the host was.
_TOOL_RUNTIME_S = 30.0


class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False
        self.activity = []

    def _touch_activity(self, msg):
        self.activity.append(msg)


@pytest.fixture()
def fake_agent():
    return _FakeAgent()


@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    # Keep the test fast: short poll slice, no config lookups.
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    emitted = []
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda agent, **kw: emitted.append(kw),
    )
    yield emitted


def test_interrupt_abandons_noncooperative_tool(monkeypatch, fake_agent, _fast_polls):
    """The wait is abandoned without waiting for the tool to finish.

    The invariant is an ordering, not a duration: the executor returns a
    cancelled result *while the tool is still running*. The tool blocks until
    this test releases it (its own ``_TOOL_RUNTIME_S`` is only a failsafe, so a
    regression that blocks inline returns at the tool's runtime instead of
    hanging the file).
    """

    started = threading.Event()
    tool_returned = threading.Event()
    tool_released = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        started.set()
        # Non-cooperative: never checks is_interrupted(), and cannot finish
        # until the test lets it. The bound mirrors the reported tool runtime —
        # "abandons well before the tool's own 30s" is the claim under test.
        tool_released.wait(timeout=_TOOL_RUNTIME_S)
        tool_returned.set()
        return _ManagedToolResult(
            result="late result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )

    def _interrupt_soon():
        wait_event(started, "the tool never started")
        time.sleep(0.1)
        fake_agent._interrupt_requested = True

    threading.Thread(target=_interrupt_soon, daemon=True).start()

    try:
        t0 = time.monotonic()
        managed = _run_sequential_tool_execution_middleware(
            fake_agent,
            function_name="image_generate",
            function_args={"prompt": "x"},
            effective_task_id="t",
            tool_call_id="call_1",
            execute=lambda a: "unused",
        )
        elapsed = time.monotonic() - t0

        assert isinstance(managed.result, _ToolCancelledResult)
        assert "cancelled" in str(managed.result)
        # The tool is still blocked, so the executor cannot have waited for it:
        # this is the barrier, and it holds on an idle box and a loaded one alike.
        assert not tool_returned.is_set(), (
            "the executor waited for the non-cooperative tool to finish instead "
            "of abandoning the wait"
        )
        assert elapsed < _TOOL_RUNTIME_S, (
            f"abandon took {elapsed:.1f}s — not well before the tool's own "
            f"{_TOOL_RUNTIME_S:.0f}s runtime"
        )
        # The executor emitted the terminal post_tool_call itself.
        assert any(kw.get("status") == "cancelled" for kw in _fast_polls)
    finally:
        tool_released.set()


def test_interrupt_prefers_real_result_from_cooperative_tool(
    monkeypatch, fake_agent, _fast_polls
):
    """A tool that finishes within the grace window returns its real result."""

    def _fake_middleware(agent_arg, **kwargs):
        # Cooperative-ish: returns quickly once running (well inside grace).
        time.sleep(0.3)
        return _ManagedToolResult(
            result="real result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )
    fake_agent._interrupt_requested = True  # interrupted before first poll

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="web_search",
        function_args={},
        effective_task_id="t",
        tool_call_id="call_2",
        execute=lambda a: "unused",
    )

    assert managed.result == "real result"
    assert not isinstance(managed.result, _ToolCancelledResult)


def test_no_deadline_still_runs_on_worker(monkeypatch, fake_agent):
    """timeout disabled (None) must not fall back to inline blocking."""

    seen_thread = []

    def _fake_middleware(agent_arg, **kwargs):
        seen_thread.append(threading.current_thread().ident)
        return _ManagedToolResult(
            result="ok", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="read_file",
        function_args={},
        effective_task_id="t",
        tool_call_id="call_3",
        execute=lambda a: "unused",
    )

    assert managed.result == "ok"
    assert seen_thread and seen_thread[0] != threading.current_thread().ident


def test_never_parallel_tools_stay_inline(monkeypatch, fake_agent):
    """clarify (interactive) keeps the inline path — it owns its own wait."""

    seen_thread = []

    def _fake_middleware(agent_arg, **kwargs):
        seen_thread.append(threading.current_thread().ident)
        return _ManagedToolResult(
            result="ok", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="clarify",
        function_args={},
        effective_task_id="t",
        tool_call_id="call_4",
        execute=lambda a: "unused",
    )

    assert managed.result == "ok"
    assert seen_thread and seen_thread[0] == threading.current_thread().ident
