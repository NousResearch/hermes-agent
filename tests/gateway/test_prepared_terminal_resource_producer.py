"""Prepared-terminal physical targets, using real pools and fake middleware."""
from dataclasses import replace
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from agent.chat_execution_context import bind_chat_execution
from agent.terminal_approval_batch import _TerminalBatch
from agent.tool_executor import _ToolCallRef
from gateway.chat_execution_recovery import ChatExecutionStore
from tests.gateway.test_chat_execution_tool_dispatch import started
from tests.gateway.test_concurrent_tool_resource_producer import QueuedExecutor
from tools.daemon_pool import DaemonThreadPoolExecutor


def prepared_batch():
    agent = SimpleNamespace(_interrupt_requested=False,
                            _tool_worker_threads_lock=threading.Lock(),
                            _tool_worker_threads=set(), _touch_activity=lambda *a: None)
    ref = _ToolCallRef("terminal", {"command": "unused fake transport"}, "task", "prepared-call", [])
    pc = SimpleNamespace(scope_block=None, ref=lambda task: ref)
    return _TerminalBatch(agent, [], "task", [pc])


def patch_external_middleware(monkeypatch, body):
    import agent.tool_executor as te
    monkeypatch.setattr(te, "_resolve_sequential_dispatch", lambda *a: SimpleNamespace(execute=lambda args: None))
    monkeypatch.setattr(te, "_run_agent_tool_execution_middleware", lambda *a, **k: body())


def test_prepared_close_leaves_real_thread_started_until_actual_exit(tmp_path, monkeypatch):
    store = started(tmp_path)
    entered, release = threading.Event(), threading.Event()
    batch = prepared_batch()
    def body():
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        assert row["state"] == "started"
        assert row["kind"] == "prepared_terminal_tool_thread"
        batch.slots[0].ready.set()
        entered.set()
        release.wait()  # The test coordinator releases it in finally; no synthetic early exit.
    patch_external_middleware(monkeypatch, body)
    original_submit = batch.executor.submit
    def submit(fn, *args):
        assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "reserved"
        return original_submit(fn, *args)
    monkeypatch.setattr(batch.executor, "submit", submit)
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            batch.start()
        assert entered.wait(5)
        batch.close()
        assert store.worker_finished("key", 1)
        assert store.begin_recovery("key", 1, "fence")
        assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "started"
    finally:
        release.set()
        batch.executor.shutdown(wait=True)
    batch.slots[0].future.result(timeout=5)
    assert store.inspect_resources("key")[0]["state"] == "finished"
    assert store.inspect("key", 1)["effects"] == "unknown"


def test_prepared_submission_captures_handle_before_mutable_slot_changes(tmp_path, monkeypatch):
    store = started(tmp_path)
    batch = prepared_batch()
    batch.executor.shutdown(wait=True)
    queued = QueuedExecutor()
    batch.executor = queued
    calls = []
    patch_external_middleware(monkeypatch, lambda: calls.append("original-worker"))
    import agent.tool_executor as te
    monkeypatch.setattr(te, "_resolve_sequential_tool_timeout", lambda: 0.0)
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(TimeoutError):
            batch.start()
    original = batch.slots[0].resource
    batch.slots[0].resource = replace(original, home=str(tmp_path / "other"))
    fn, args = queued.calls[0]
    with DaemonThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(fn, *args).result(timeout=5)
    assert calls == ["original-worker"]
    assert store.inspect_resources("key")[0]["state"] == "finished"
    assert not (tmp_path / "other" / "chat-execution.sqlite3").exists()


@pytest.mark.parametrize("phase", ["reserve", "start", "finish"])
def test_prepared_storage_failures_never_fabricate_finish(tmp_path, monkeypatch, phase):
    store = started(tmp_path)
    batch = prepared_batch()
    calls, submits = [], []
    patch_external_middleware(monkeypatch, lambda: calls.append("body"))
    methods = {"reserve": "reserve_concurrent_tool_resource", "start": "start_concurrent_tool_resource",
               "finish": "finish_concurrent_tool_resource"}
    def broken(*a, **k):
        raise sqlite3.OperationalError("injected lifecycle write failure")
    monkeypatch.setattr(ChatExecutionStore, methods[phase], broken)
    original_submit = batch.executor.submit
    def submit(fn, *args):
        submits.append("submit")
        return original_submit(fn, *args)
    monkeypatch.setattr(batch.executor, "submit", submit)
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            if phase == "reserve":
                with pytest.raises(sqlite3.OperationalError):
                    batch.start()
            else:
                batch.start()
        if phase == "start":
            with pytest.raises(sqlite3.OperationalError):
                batch.slots[0].future.result(timeout=5)
        elif phase == "finish":
            batch.slots[0].future.result(timeout=5)
    finally:
        batch.close()
        batch.executor.shutdown(wait=True)
    assert calls == (["body"] if phase == "finish" else [])
    assert submits == ([] if phase == "reserve" else ["submit"])
    rows = store.inspect_resources("key")
    assert [row["state"] for row in rows] == {"reserve": [], "start": ["reserved"], "finish": ["started"]}[phase]


@pytest.mark.parametrize("mode", ["cancel", "fence", "submit_failure"])
def test_prepared_queued_or_failed_submission_never_fabricates_finish(tmp_path, monkeypatch, mode):
    store = started(tmp_path)
    batch = prepared_batch()
    batch.executor.shutdown(wait=True)
    queued = QueuedExecutor()
    batch.executor = queued
    bodies = []
    patch_external_middleware(monkeypatch, lambda: bodies.append("body"))
    import agent.tool_executor as te
    monkeypatch.setattr(te, "_resolve_sequential_tool_timeout", lambda: 0.0)
    if mode == "submit_failure":
        def fail(*a):
            raise RuntimeError("injected submit failure")
        queued.submit = fail
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises((TimeoutError, RuntimeError)):
            batch.start()
    if mode == "cancel":
        assert queued.future.cancel()
    if mode == "fence":
        assert store.begin_recovery("key", 1, "fence")
        fn, args = queued.calls[0]
        with DaemonThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="admission refused"):
                pool.submit(fn, *args).result(timeout=5)
        assert batch.slots[0].ready.is_set()
    assert bodies == []
    assert store.inspect_resources("key")[0]["state"] == "reserved"


def test_prepared_physical_slots_do_not_alias_repeated_provider_call_ids(tmp_path, monkeypatch):
    from agent.terminal_approval_batch import _TerminalSlot
    store = started(tmp_path)
    batch = prepared_batch()
    batch.slots.append(_TerminalSlot(batch, batch.slots[0].parsed, 1))
    calls = []
    patch_external_middleware(monkeypatch, lambda: calls.append("body"))
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            batch.start()
        for slot in batch.slots:
            slot.future.result(timeout=5)
    finally:
        batch.close()
        batch.executor.shutdown(wait=True)
    rows = store.inspect_resources("key")
    assert calls == ["body", "body"]
    assert len({row["resource_id"] for row in rows}) == 2
    assert all(row["state"] == "finished" for row in rows)


@pytest.mark.parametrize("kind", ["sequential_tool_thread", "prepared_terminal_tool_thread"])
def test_resource_kind_mismatch_refuses_start_and_finish(tmp_path, kind):
    store = started(tmp_path)
    assert store.reserve_concurrent_tool_resource("key", 1, "resource", kind)
    assert not store.start_concurrent_tool_resource("key", 1, "resource", "session", "concurrent_tool_thread")
    assert store.start_concurrent_tool_resource("key", 1, "resource", "session", kind)
    assert not store.finish_concurrent_tool_resource("key", 1, "resource", "session", "concurrent_tool_thread")
    assert store.finish_concurrent_tool_resource("key", 1, "resource", "session", kind)
