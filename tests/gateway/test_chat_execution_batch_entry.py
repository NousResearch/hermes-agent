"""Public batch-entry observation precedes tool-associated callbacks.

Boundary probes stop before real tool execution. They do not prove child lifetimes.
"""
from types import SimpleNamespace
import sqlite3

import pytest

from agent.chat_execution_context import bind_chat_execution, observe_tool_before_execution
from gateway.chat_execution_recovery import ChatExecutionStore
from tests.gateway.test_chat_execution_tool_dispatch import started, observation


class BoundaryReached(Exception):
    pass


def entry_probe(monkeypatch, mode, boundary):
    import agent.tool_executor as executor
    import agent.terminal_approval_batch as approval
    targets = {
        "sequential": (approval, "terminal_approval_runs", executor.execute_tool_calls_sequential),
        "concurrent": (executor, "_budget_for_agent", executor.execute_tool_calls_concurrent),
    }
    module, name, entry = targets[mode]
    def probe(*args, **kwargs):
        boundary()
        raise BoundaryReached()
    monkeypatch.setattr(module, name, probe)
    return lambda calls: entry(SimpleNamespace(), SimpleNamespace(tool_calls=calls), [], "task", finalize=False)


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_batch_observation_is_committed_before_approval_or_concurrent_boundary(tmp_path, monkeypatch, mode):
    started(tmp_path)
    calls = []
    def boundary():
        assert observation(tmp_path) == (1, "unknown")
        calls.append("boundary")
    execute = entry_probe(monkeypatch, mode, boundary)
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(BoundaryReached):
            execute([SimpleNamespace()])
    assert calls == ["boundary"]


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
@pytest.mark.parametrize("reason", ["missing", "unbound", "generation", "completed", "database"])
def test_refusal_prevents_any_batch_entry_callback(tmp_path, monkeypatch, mode, reason):
    store = ChatExecutionStore(tmp_path)
    if reason == "unbound":
        assert store.reserve("key", 0, 0) == 1
    elif reason != "missing":
        store = started(tmp_path)
        if reason == "completed":
            assert store.worker_finished("key", 1)
    callbacks = []
    execute = entry_probe(monkeypatch, mode, lambda: callbacks.append("effect"))
    if reason == "database":
        def broken(self):
            raise sqlite3.OperationalError("injected database unavailable")
        monkeypatch.setattr(ChatExecutionStore, "_db", broken)
    with bind_chat_execution(tmp_path, "key", 2 if reason == "generation" else 1):
        with pytest.raises((RuntimeError, sqlite3.OperationalError)):
            execute([SimpleNamespace()])
    assert callbacks == []


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_empty_batch_does_not_mark_execution(tmp_path, monkeypatch, mode):
    started(tmp_path)
    execute = entry_probe(monkeypatch, mode, lambda: None)
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(BoundaryReached):
            execute([])
    assert observation(tmp_path) == (0, "unknown")


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_legacy_batch_context_preserves_entry_behavior(monkeypatch, mode):
    callbacks = []
    execute = entry_probe(monkeypatch, mode, lambda: callbacks.append("legacy"))
    assert observe_tool_before_execution() is False
    with pytest.raises(BoundaryReached):
        execute([SimpleNamespace()])
    assert callbacks == ["legacy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
async def test_physical_worker_binds_identity_before_public_batch_entry(tmp_path, monkeypatch, mode):
    import asyncio
    from tests.gateway.test_chat_execution_worker import reserved_worker
    runner, state, ctx, store = reserved_worker(tmp_path, monkeypatch)
    callbacks = []
    def boundary():
        assert observation(tmp_path, ctx.session_key) == (1, "unknown")
        callbacks.append("worker-boundary")
    execute = entry_probe(monkeypatch, mode, boundary)
    worker = runner._run_agent_start_turn_worker(ctx, lambda: execute([SimpleNamespace()]))
    with pytest.raises(BoundaryReached):
        await worker.executor_task
    assert await asyncio.to_thread(worker.worker_done.wait, 5)
    assert callbacks == ["worker-boundary"]
    assert store.inspect(ctx.session_key, 1)["worker"] == "completed"
    assert observe_tool_before_execution() is False
