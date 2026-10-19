"""Actual sequential dispatch boundaries, with real durable journal storage."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
import sqlite3

import pytest

from agent.chat_execution_context import bind_chat_execution, observe_tool_before_execution
from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef
from gateway.chat_execution_recovery import ChatExecutionStore


def started(home, key="key"):
    store = ChatExecutionStore(home)
    assert store.reserve(key, 0, 0) == 1
    assert store.bind_worker_started(key, 1, "session")
    return store


def observation(home, key="key"):
    with ChatExecutionStore(home)._db() as db:
        row = db.execute("SELECT tool_observed, effects FROM executions WHERE session_key=?", (key,)).fetchone()
    return tuple(row) if row else None


def dispatch(monkeypatch, branch, leaf):
    import agent.tool_executor as executor
    import model_tools
    agent = SimpleNamespace(
        _context_engine_tool_names=set(), _memory_manager=None, quiet_mode=False,
        session_id="session", _current_turn_id="", _current_api_request_id="",
        valid_tool_names=[], enabled_toolsets=None, disabled_toolsets=None,
    )
    monkeypatch.setattr(executor, "_start_quiet_tool_spinner", lambda *a, **k: None)
    name = "delegate_task" if branch == "delegate" else "probe_tool"
    def inline():
        monkeypatch.setitem(executor.INLINE_TOOL_EXECUTORS, name, lambda *a: leaf())
    def delegate():
        agent._dispatch_delegate_task = lambda args: leaf()
    def context():
        agent._context_engine_tool_names = {name}
        agent.context_compressor = SimpleNamespace(handle_tool_call=lambda *a, **k: leaf())
    def memory():
        agent._memory_manager = SimpleNamespace(has_tool=lambda n: n == name, handle_tool_call=lambda *a, **k: leaf())
    def registry():
        monkeypatch.setattr(model_tools, "handle_function_call", lambda *a, **k: leaf())
    {"inline": inline, "delegate": delegate, "context": context,
     "memory": memory, "registry": registry}[branch]()
    return _resolve_sequential_dispatch(agent, _ToolCallRef(name=name, args={}, task_id="task", call_id="call", trace=[]), [])


@pytest.mark.parametrize("branch", ["inline", "delegate", "context", "memory", "registry"])
def test_committed_observation_precedes_each_actual_sequential_leaf(tmp_path, monkeypatch, branch):
    started(tmp_path)
    calls = []
    def leaf():
        assert observation(tmp_path) == (1, "unknown")
        calls.append(branch)
        return "ok"
    resolved = dispatch(monkeypatch, branch, leaf)
    with bind_chat_execution(tmp_path, "key", 1):
        assert resolved.execute({}) == "ok"
    assert calls == [branch]


@pytest.mark.parametrize("reason", ["missing", "unbound", "generation", "completed", "database"])
@pytest.mark.parametrize("branch", ["inline", "delegate", "context", "memory", "registry"])
def test_refused_durable_observation_executes_zero_leaf(tmp_path, monkeypatch, reason, branch):
    store = ChatExecutionStore(tmp_path)
    if reason == "unbound":
        assert store.reserve("key", 0, 0) == 1
    elif reason != "missing":
        store = started(tmp_path)
        if reason == "completed":
            assert store.worker_finished("key", 1)
    calls = []
    resolved = dispatch(monkeypatch, branch, lambda: calls.append("effect"))
    if reason == "database":
        def broken(self):
            raise sqlite3.OperationalError("injected database unavailable")
        monkeypatch.setattr(ChatExecutionStore, "_db", broken)
    with bind_chat_execution(tmp_path, "key", 2 if reason == "generation" else 1):
        with pytest.raises((RuntimeError, sqlite3.OperationalError)):
            resolved.execute({})
    assert calls == []


@pytest.mark.parametrize("branch", ["inline", "delegate", "context", "memory", "registry"])
def test_legacy_dispatch_without_binding_preserves_normal_execution(monkeypatch, branch):
    calls = []
    resolved = dispatch(monkeypatch, branch, lambda: calls.append("legacy") or "ok")
    assert resolved.execute({}) == "ok"
    assert calls == ["legacy"]
    assert observe_tool_before_execution() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["inline", "delegate", "context", "memory", "registry"])
async def test_physical_worker_binds_real_sequential_dispatch(tmp_path, monkeypatch, branch):
    import asyncio
    from tests.gateway.test_chat_execution_worker import reserved_worker
    runner, _state, ctx, store = reserved_worker(tmp_path, monkeypatch)
    calls = []
    def leaf():
        assert observation(tmp_path, ctx.session_key) == (1, "unknown")
        calls.append(branch)
        return "ok"
    resolved = dispatch(monkeypatch, branch, leaf)
    worker = runner._run_agent_start_turn_worker(ctx, lambda: resolved.execute({}))
    assert await worker.executor_task == "ok"
    assert await asyncio.to_thread(worker.worker_done.wait, 5)
    assert calls == [branch]
    assert store.inspect(ctx.session_key, 1)["worker"] == "completed"
    assert observe_tool_before_execution() is False


def test_copied_context_threads_keep_exact_home_and_reset_after_exception(tmp_path):
    homes = [tmp_path / "one", tmp_path / "two"]
    for home in homes:
        started(home)
    contexts = []
    for home in homes:
        with bind_chat_execution(home, "key", 1):
            contexts.append(copy_context())
    def run(context, home):
        def body():
            assert observe_tool_before_execution() is True
            assert observation(home) == (1, "unknown")
        context.run(body)
        assert observe_tool_before_execution() is False
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, context, home) for context, home in zip(contexts, homes)]
        for future in futures:
            future.result(timeout=5)
    with pytest.raises(ValueError):
        with bind_chat_execution(homes[0], "key", 1):
            raise ValueError("body failed")
    assert observe_tool_before_execution() is False
    for home in homes:
        assert observation(home) == (1, "unknown")
