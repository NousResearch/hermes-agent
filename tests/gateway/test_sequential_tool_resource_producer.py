"""Actual sequential producer lifecycle with a fake external middleware leaf."""
import threading
import sqlite3
from types import SimpleNamespace

import pytest

from tools.daemon_pool import DaemonThreadPoolExecutor

from agent.chat_execution_context import bind_chat_execution
from agent.tool_executor import _ToolTimeoutResult, _run_sequential_tool_execution_middleware
from gateway.chat_execution_recovery import ChatExecutionStore


def test_timed_out_sequential_worker_stays_started_until_physical_exit(tmp_path, monkeypatch):
    store = ChatExecutionStore(tmp_path)
    assert store.start("key", 1, "session")
    assert store.worker_started("key", 1)
    entered = threading.Event()
    release = threading.Event()

    futures = []
    original_submit = DaemonThreadPoolExecutor.submit
    def submit(executor, fn, *args, **kwargs):
        future = original_submit(executor, fn, *args, **kwargs)
        futures.append(future)
        return future

    def middleware(agent, **kwargs):
        entered.set()
        release.wait()
        return "finished"

    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit)
    monkeypatch.setattr("agent.tool_executor._run_agent_tool_execution_middleware", middleware)
    monkeypatch.setattr("agent.tool_executor._resolve_sequential_tool_timeout", lambda: 0.05)
    agent = SimpleNamespace(_interrupt_requested=False, _tool_interrupt_reason=None, _tool_worker_threads_lock=threading.Lock(), _tool_worker_threads=set())
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            result = _run_sequential_tool_execution_middleware(agent, function_name="tool", function_args={}, effective_task_id="task", tool_call_id="call", execute=lambda: None)
        assert entered.wait(5)
        assert isinstance(result.result, _ToolTimeoutResult)
        row, = store.inspect_resources("key")
        assert (row["kind"], row["state"]) == ("sequential_tool_thread", "started")
    finally:
        release.set()
        for future in futures:
            if not future.cancelled():
                future.result(timeout=5)
    assert store.inspect_resources("key")[0]["state"] == "finished"


@pytest.mark.parametrize("phase", ["reserve", "submit", "start", "finish", "normal"])
def test_sequential_producer_storage_boundaries(tmp_path, monkeypatch, phase):
    from tests.gateway.test_chat_execution_tool_dispatch import started
    store = started(tmp_path)
    bodies, submits = [], []
    def middleware(*a, **k):
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        assert row["kind"] == "sequential_tool_thread"
        assert row["state"] == "started"
        bodies.append("body")
        return "ok"
    monkeypatch.setattr("agent.tool_executor._run_agent_tool_execution_middleware", middleware)
    methods = {"reserve": "reserve_concurrent_tool_resource", "start": "start_concurrent_tool_resource",
               "finish": "finish_concurrent_tool_resource"}
    def broken(*a, **k):
        raise sqlite3.OperationalError("injected lifecycle persistence failure")
    if phase in methods:
        monkeypatch.setattr(ChatExecutionStore, methods[phase], broken)
    original_submit = DaemonThreadPoolExecutor.submit
    def submit(executor, fn, *args, **kwargs):
        assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "reserved"
        submits.append("submit")
        if phase == "submit":
            raise RuntimeError("injected submit failure")
        return original_submit(executor, fn, *args, **kwargs)
    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit)
    agent = SimpleNamespace(_interrupt_requested=False, _tool_worker_threads_lock=threading.Lock(), _tool_worker_threads=set())
    with bind_chat_execution(tmp_path, "key", 1):
        args = dict(function_name="tool", function_args={}, effective_task_id="task", tool_call_id="boundary-call", execute=lambda: None)
        if phase in {"reserve", "start", "submit"}:
            with pytest.raises((sqlite3.OperationalError, RuntimeError)):
                _run_sequential_tool_execution_middleware(agent, **args)
        else:
            assert _run_sequential_tool_execution_middleware(agent, **args) == "ok"
    assert bodies == (["body"] if phase in {"finish", "normal"} else [])
    assert submits == ([] if phase == "reserve" else ["submit"])
    assert [row["state"] for row in store.inspect_resources("key")] == {
        "reserve": [], "submit": ["reserved"], "start": ["reserved"],
        "finish": ["started"], "normal": ["finished"]}[phase]


def test_distinct_sequential_workers_do_not_alias_repeated_provider_call_ids(tmp_path, monkeypatch):
    from tests.gateway.test_chat_execution_tool_dispatch import started
    store = started(tmp_path)
    monkeypatch.setattr("agent.tool_executor._run_agent_tool_execution_middleware", lambda *a, **k: "ok")
    agent = SimpleNamespace(_interrupt_requested=False, _tool_worker_threads_lock=threading.Lock(), _tool_worker_threads=set())
    with bind_chat_execution(tmp_path, "key", 1):
        for _ in range(2):
            assert _run_sequential_tool_execution_middleware(
                agent, function_name="tool", function_args={}, effective_task_id="task",
                tool_call_id="same-provider-id", execute=lambda: None) == "ok"
    rows = store.inspect_resources("key")
    assert len(rows) == 2
    assert len({row["resource_id"] for row in rows}) == 2
    assert all(row["state"] == "finished" for row in rows)


def _wait_for_state(store, key, state):
    import time
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        rows = store.inspect_resources(key)
        if rows and rows[0]["state"] == state:
            return True
        time.sleep(0.01)
    return False
