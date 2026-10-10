"""Actual concurrent submit/run_worker lifetimes, with fake external tool bodies."""
import asyncio
from concurrent.futures import Future
from contextvars import ContextVar
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from agent.chat_execution_context import bind_chat_execution
from agent.tool_executor import _ConcurrentBatch
from gateway.chat_execution_recovery import ChatExecutionStore
from tests.gateway.test_chat_execution_tool_dispatch import started
from tools.daemon_pool import DaemonThreadPoolExecutor


def batch_for(body):
    agent = SimpleNamespace(_interrupt_requested=False,
                            _tool_worker_threads_lock=threading.Lock(),
                            _tool_worker_threads=set(), _touch_activity=lambda *args: None)
    pc = SimpleNamespace(name="probe", parse_error=None, scope_block=None,
                         ref=lambda task: SimpleNamespace(name="probe"))
    batch = _ConcurrentBatch(agent, [], "task", [pc], None)
    batch._dispatch_worker = lambda *args: body()
    return batch


class QueuedExecutor:
    def __init__(self):
        self.calls = []
        self.future = Future()
    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return self.future


def test_real_submit_marks_reserved_before_pool_and_started_before_tool(tmp_path):
    store = started(tmp_path)
    entered, release = threading.Event(), threading.Event()
    propagated = ContextVar("resource_probe", default="absent")
    seen = []
    def body():
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        seen.append((row["state"], propagated.get()))
        entered.set()
        assert release.wait(5)
    batch = batch_for(body)
    pool = DaemonThreadPoolExecutor(max_workers=1)
    class SubmitProbe:
        def submit(self, fn, *args):
            assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "reserved"
            return pool.submit(fn, *args)
    token = propagated.set("copied")
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            futures, indices = batch.submit_all(SubmitProbe(), [0])
        assert entered.wait(5)
        assert seen == [("started", "copied")]
        assert futures[0].cancel() is False
        pool.shutdown(wait=False, cancel_futures=True)
        assert store.worker_finished("key", 1)
        assert store.begin_recovery("key", 1, "fence")
        assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "started"
        assert store.recover("key", 1, "fence", lambda proof: pytest.fail("transfer invoked"))["status"] == "blocked"
    finally:
        release.set()
        propagated.reset(token)
        pool.shutdown(wait=True)
    futures[0].result(timeout=5)
    row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
    assert row["state"] == "finished"
    assert row["session_id"] == "session"
    assert store.inspect("key", 1)["effects"] == "unknown"


@pytest.mark.parametrize("mode", ["cancel", "submit_failure"])
def test_unstarted_submissions_stay_reserved(tmp_path, mode):
    started(tmp_path)
    bodies = []
    batch = batch_for(lambda: bodies.append("effect"))
    executor = QueuedExecutor()
    if mode == "submit_failure":
        def fail(*args):
            raise RuntimeError("injected submission failure")
        executor.submit = fail
        with bind_chat_execution(tmp_path, "key", 1):
            with pytest.raises(RuntimeError, match="submission failure"):
                batch.submit_all(executor, [0])
    else:
        with bind_chat_execution(tmp_path, "key", 1):
            futures, indices = batch.submit_all(executor, [0])
        assert futures[0].cancel()
    assert bodies == []
    assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "reserved"


def test_reservation_database_failure_submits_zero_workers(tmp_path, monkeypatch):
    started(tmp_path)
    batch = batch_for(lambda: pytest.fail("tool executed"))
    executor = QueuedExecutor()
    def broken(self):
        raise sqlite3.OperationalError("database unavailable")
    monkeypatch.setattr(ChatExecutionStore, "_db", broken)
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(sqlite3.OperationalError):
            batch.submit_all(executor, [0])
    assert executor.calls == []


@pytest.mark.parametrize("phase", ["start", "finish"])
def test_physical_persistence_failure_retains_unresolved_resource(tmp_path, monkeypatch, phase):
    store = started(tmp_path)
    bodies = []
    batch = batch_for(lambda: bodies.append("body"))
    method = {"start": "start_concurrent_tool_resource", "finish": "finish_concurrent_tool_resource"}[phase]
    def broken(*args):
        raise sqlite3.OperationalError("injected lifecycle write failure")
    monkeypatch.setattr(ChatExecutionStore, method, broken)
    with DaemonThreadPoolExecutor(max_workers=1) as pool:
        with bind_chat_execution(tmp_path, "key", 1):
            futures, indices = batch.submit_all(pool, [0])
        if phase == "start":
            with pytest.raises(sqlite3.OperationalError):
                futures[0].result(timeout=5)
        else:
            futures[0].result(timeout=5)
    assert bodies == ([] if phase == "start" else ["body"])
    assert batch.agent._tool_worker_threads == set()
    assert store.inspect_resources("key")[0]["state"] == ("reserved" if phase == "start" else "started")


@pytest.mark.parametrize("reason", ["fence", "successor"])
def test_queued_worker_refuses_stale_admission_before_registration(tmp_path, reason):
    store = started(tmp_path)
    bodies = []
    batch = batch_for(lambda: bodies.append("body"))
    executor = QueuedExecutor()
    with bind_chat_execution(tmp_path, "key", 1):
        batch.submit_all(executor, [0])
    if reason == "fence":
        assert store.begin_recovery("key", 1, "fence")
    else:
        assert store.reserve("key", 1, 1) == 2
    fn, args = executor.calls[0]
    with DaemonThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(fn, *args).result(timeout=5)
    assert bodies == []
    assert batch.agent._tool_worker_threads == set()
    assert store.inspect_resources("key")[0]["state"] == "reserved"


@pytest.mark.parametrize("field", ["home", "generation", "resource_id", "session_id"])
def test_mismatched_immutable_handle_executes_zero_body(tmp_path, field):
    from dataclasses import replace
    store = started(tmp_path)
    bodies = []
    batch = batch_for(lambda: bodies.append("body"))
    executor = QueuedExecutor()
    with bind_chat_execution(tmp_path, "key", 1):
        batch.submit_all(executor, [0])
    fn, args = executor.calls[0]
    replacements = {"home": str(tmp_path / "other"), "generation": 2,
                    "resource_id": "different", "session_id": "different"}
    handle = replace(args[-1], **{field: replacements[field]})
    with DaemonThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(fn, *args[:-1], handle).result(timeout=5)
    assert bodies == []
    assert store.inspect_resources("key")[0]["state"] == "reserved"


@pytest.mark.parametrize("reason", ["fence", "generation", "completed", "unbound", "missing"])
def test_refused_resource_reservation_submits_zero_workers(tmp_path, reason):
    store = ChatExecutionStore(tmp_path)
    if reason == "unbound":
        assert store.reserve("key", 0, 0) == 1
    elif reason != "missing":
        store = started(tmp_path)
        actions = {"fence": lambda: store.begin_recovery("key", 1, "fence"),
                   "generation": lambda: store.reserve("key", 1, 1),
                   "completed": lambda: store.worker_finished("key", 1)}
        assert actions[reason]()
    batch = batch_for(lambda: pytest.fail("tool executed"))
    executor = QueuedExecutor()
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(RuntimeError, match="resource reservation refused"):
            batch.submit_all(executor, [0])
    assert executor.calls == []
    assert store.inspect_resources("key") == []


def test_pending_resource_cannot_be_hidden_by_execution_only_proof(tmp_path):
    store = started(tmp_path)
    batch = batch_for(lambda: pytest.fail("tool executed"))
    executor = QueuedExecutor()
    with bind_chat_execution(tmp_path, "key", 1):
        batch.submit_all(executor, [0])
    # Adversarial scratch fixture, not an actual effect-free production proof.
    with store._db() as db:
        db.execute("UPDATE executions SET worker='completed', effects='none', tool_observed=0 WHERE session_key='key'")
    assert store.inspect("key", 1)["effects"] == "unknown"
    result = store.recover("key", 1, "fence", lambda proof: pytest.fail("transfer invoked"))
    assert result["status"] == "blocked"


def test_legacy_submit_still_executes_without_journal_context():
    bodies = []
    batch = batch_for(lambda: bodies.append("legacy"))
    with DaemonThreadPoolExecutor(max_workers=1) as pool:
        futures, indices = batch.submit_all(pool, [0])
        futures[0].result(timeout=5)
    assert bodies == ["legacy"]


@pytest.mark.asyncio
async def test_parent_physical_completion_does_not_settle_blocked_tool_thread(tmp_path, monkeypatch):
    from tests.gateway.test_chat_execution_worker import reserved_worker
    runner, state, ctx, store = reserved_worker(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    tool_futures = []
    def body():
        entered.set()
        assert release.wait(5)
    batch = batch_for(body)
    pool = DaemonThreadPoolExecutor(max_workers=1)
    def parent():
        futures, indices = batch.submit_all(pool, [0])
        tool_futures.extend(futures)
        assert entered.wait(5)
        pool.shutdown(wait=False, cancel_futures=True)
    worker = runner._run_agent_start_turn_worker(ctx, parent)
    try:
        await worker.executor_task
        assert await asyncio.to_thread(worker.worker_done.wait, 5)
        assert store.inspect(ctx.session_key, 1)["worker"] == "completed"
        assert store.inspect_resources(ctx.session_key)[0]["state"] == "started"
    finally:
        release.set()
        await asyncio.to_thread(pool.shutdown, wait=True)
    tool_futures[0].result(timeout=5)
    assert store.inspect_resources(ctx.session_key)[0]["state"] == "finished"
    assert store.inspect(ctx.session_key, 1)["effects"] == "unknown"
