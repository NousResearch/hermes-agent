"""Sequential tool execution must degrade cleanly when interpreter finalization refuses the submit.

Regression tests for the "cannot schedule new futures after interpreter shutdown" traceback storm:
during process teardown mid-turn (fleet restarts, cron/gateway shutdown), a turn's sequential tool
call reaches ``DaemonThreadPoolExecutor.submit`` AFTER CPython's module-global shutdown flag is set.
The RuntimeError propagated out of ``_run_sequential_tool_execution_middleware``, so every affected
session logged ``handle_function_call raised for <tool>: cannot schedule new futures after interpreter
shutdown`` with a full traceback per attempt (measured on the reporting host: 83 tracebacks through
this one site in one week, several tool calls per teardown session).

The concurrent batch path already synthesizes per-slot error results for unsubmitted work from the
same signal (``_ConcurrentBatch.submit_all`` via the shared ``tools.interpreter_shutdown`` predicate —
the #55924/#58720 bug class). The sequential path now does the same: no worker was started, so it
returns the sibling ``_ToolShutdownResult`` marker and emits the terminal post_tool_call exactly once.
"""

import threading

import pytest

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolShutdownResult,
    _run_sequential_tool_execution_middleware,
)
from tools.daemon_pool import DaemonThreadPoolExecutor


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


@pytest.fixture()
def emitted_posts(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda agent, **kw: emitted.append(kw),
    )
    # Keep the path fast: no config lookups for the deadline.
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: None)
    return emitted


def _submit_raises(exc):
    def _fake_submit(self, fn, /, *args, **kwargs):
        raise exc

    return _fake_submit


@pytest.mark.parametrize(
    "msg",
    [
        "cannot schedule new futures after interpreter shutdown",
        "cannot schedule new futures after shutdown",
    ],
)
def test_shutdown_submit_synthesizes_marker_result(monkeypatch, fake_agent, emitted_posts, msg):
    """Both CPython shutdown-message variants yield one clean synthesized result instead of a raise."""
    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", _submit_raises(RuntimeError(msg)))

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="read_file",
        function_args={"path": "x"},
        effective_task_id="t",
        tool_call_id="call_sd",
        execute=lambda a: "unused",
    )

    assert isinstance(managed, _ManagedToolResult)
    assert isinstance(managed.result, _ToolShutdownResult)
    assert str(managed.result) == (
        "Error executing tool 'read_file': Python interpreter is shutting down; tool was not started"
    )
    # The terminal post_tool_call fired exactly once, carrying the shutdown outcome.
    assert len(emitted_posts) == 1
    assert emitted_posts[0]["status"] == "error"
    assert emitted_posts[0]["error_type"] == "interpreter_shutdown"
    assert emitted_posts[0]["tool_call_id"] == "call_sd"


def test_non_shutdown_submit_runtime_error_still_propagates(monkeypatch, fake_agent, emitted_posts):
    """Only the shutdown signal is converted; a genuine broken-pool RuntimeError keeps raising."""
    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", _submit_raises(RuntimeError("broken pool")))

    with pytest.raises(RuntimeError, match="broken pool"):
        _run_sequential_tool_execution_middleware(
            fake_agent,
            function_name="read_file",
            function_args={},
            effective_task_id="t",
            tool_call_id="call_boom",
            execute=lambda a: "unused",
        )

    assert emitted_posts == []
