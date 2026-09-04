"""Sequential tool worker must shut down its executor before failing open.

Regression test for the t_5c83b6fe PR-102111 review finding: when the
sequential tool worker's ``DaemonThreadPoolExecutor.submit()`` raises a
thread-start-exhaustion error, ``_run_sequential_tool_execution_middleware``
fell back to running the tool inline without ever calling
``executor.shutdown()`` on the executor it had just created — leaking
executor-internal resources (queue, work-item registration) even though no
worker thread was ever started.
"""

import threading

import pytest

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _run_sequential_tool_execution_middleware,
)
from tools.daemon_pool import DaemonThreadPoolExecutor


class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False


@pytest.fixture()
def fake_agent():
    return _FakeAgent()


class _ExhaustedExecutor(DaemonThreadPoolExecutor):
    """Fails every submit() like a process out of OS threads, tracking shutdown()."""

    shutdown_calls: list = []

    def submit(self, fn, /, *args, **kwargs):
        raise RuntimeError("can't start new thread")

    def shutdown(self, *args, **kwargs):
        _ExhaustedExecutor.shutdown_calls.append((args, kwargs))
        return super().shutdown(*args, **kwargs)


def test_thread_exhaustion_shuts_down_executor_before_inline_fallback(
    monkeypatch, fake_agent
):
    _ExhaustedExecutor.shutdown_calls = []
    monkeypatch.setattr(
        "tools.daemon_pool.DaemonThreadPoolExecutor", _ExhaustedExecutor
    )
    monkeypatch.setattr(
        tool_executor,
        "_run_agent_tool_execution_middleware",
        lambda agent_arg, **kwargs: _ManagedToolResult(
            result="inline result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        ),
    )

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="image_generate",
        function_args={"prompt": "x"},
        effective_task_id="t",
        tool_call_id="call_1",
        execute=lambda a: "unused",
    )

    assert managed.result == "inline result"
    assert _ExhaustedExecutor.shutdown_calls, (
        "executor.shutdown() was never called after the thread-exhausted "
        "submit() fell back to running the tool inline"
    )
