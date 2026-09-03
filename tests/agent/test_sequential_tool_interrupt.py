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
    """A blocking tool is abandoned within ~poll+grace once interrupted."""

    started = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        started.set()
        time.sleep(30)  # non-cooperative: never checks is_interrupted()
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
        started.wait(5)
        time.sleep(0.1)
        fake_agent._interrupt_requested = True

    threading.Thread(target=_interrupt_soon, daemon=True).start()

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
    # poll (0.05s) + interrupt delay (0.1s) + grace (3s) + slack — nowhere
    # near the 30s tool runtime.
    assert elapsed < 10.0
    # The executor emitted the terminal post_tool_call itself.
    assert any(kw.get("status") == "cancelled" for kw in _fast_polls)


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
