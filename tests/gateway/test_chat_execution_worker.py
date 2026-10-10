"""Durable observations come from the physical executor, not its asyncio wrapper."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.chat_execution_recovery import ChatExecutionStore
from tests.gateway.test_chat_run_stop import _running_turn


def test_sequential_dispatch_marks_started_gateway_tool_before_leaf(tmp_path):
    from agent.chat_execution_context import bind_chat_execution
    from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef

    store = ChatExecutionStore(tmp_path)
    assert store.reserve("key", 0, 0) == 1
    assert store.bind_worker_started("key", 1, "session")
    agent = SimpleNamespace(
        _context_engine_tool_names=set(), _memory_manager=None, quiet_mode=False,
        session_id="session", _current_turn_id="", _current_api_request_id="",
        valid_tool_names=[], enabled_toolsets=None, disabled_toolsets=None,
    )
    observed = []

    def leaf(*args, **kwargs):
        from gateway.chat_execution_recovery import ChatExecutionStore
        observed.append(ChatExecutionStore(tmp_path).inspect("key", 1)["effects"])
        with ChatExecutionStore(tmp_path)._db() as db:
            observed.append(db.execute("SELECT tool_observed FROM executions WHERE session_key='key'").fetchone()[0])
        return "ok"

    ref = _ToolCallRef(name="fake_registry", args={}, task_id="task", call_id="call", trace=[])
    with bind_chat_execution(tmp_path, "key", 1):
        dispatch = _resolve_sequential_dispatch(agent, ref, [])
        import model_tools
        old = model_tools.handle_function_call
        model_tools.handle_function_call = leaf
        try:
            dispatch.execute({})
        finally:
            model_tools.handle_function_call = old
    assert observed == ["unknown", 1]
    row = store.inspect("key", 1)
    assert row["effects"] == "unknown"
    with store._db() as db:
        assert db.execute("SELECT tool_observed FROM executions WHERE session_key='key'").fetchone()[0] == 1


def test_completed_worker_context_cannot_admit_leaf_tool(tmp_path):
    from agent.chat_execution_context import bind_chat_execution, observe_tool_before_execution

    store = ChatExecutionStore(tmp_path)
    assert store.reserve("completed", 0, 0) == 1
    assert store.bind_worker_started("completed", 1, "session")
    assert store.worker_finished("completed", 1)
    with bind_chat_execution(tmp_path, "completed", 1):
        with pytest.raises(RuntimeError, match="Durable tool observation refused"):
            observe_tool_before_execution()
    with store._db() as db:
        row = db.execute("SELECT tool_observed, effects FROM executions WHERE session_key='completed'").fetchone()
    assert tuple(row) == (0, "unknown")


def reserved_worker(home, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    runner, state, source, key = _running_turn(home, generation=1)
    store = ChatExecutionStore(home)
    assert store.reserve(key, 0, 0) == 1
    ctx = SimpleNamespace(agent_holder=[None], session_key=key, run_generation=1,
                          session_id="physical-session", source=source,
                          chat_execution_reservation=(str(home), key, 1))
    return runner, state, ctx, store


@pytest.mark.asyncio
async def test_physical_completion_survives_wrapper_cancellation_and_restart(tmp_path, monkeypatch):
    runner, state, ctx, store = reserved_worker(tmp_path, monkeypatch)
    key = ctx.session_key
    started, release = threading.Event(), threading.Event()

    observed_after_fence = []

    def body():
        started.set()
        assert release.wait(10)
        from agent.chat_execution_context import observe_tool_before_execution
        observed_after_fence.append(observe_tool_before_execution())

    worker = runner._run_agent_start_turn_worker(ctx, body)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        proof = store.inspect(ctx.session_key, 1)
        assert proof["session_id"] == "physical-session"
        assert proof["worker"] == "pending"
        assert proof["effects"] == "unknown"
        worker.executor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker.executor_task
        assert not worker.worker_done.is_set()
        assert store.inspect(ctx.session_key, 1)["worker"] == "pending"
        # Stop fences the old worker; its completion must still be recorded.
        assert store.begin_recovery(ctx.session_key, 1, "stop-fence")
        state.persistent.run_generation = 2
        # Mutating shared context after launch cannot redirect its durable completion.
        ctx.session_key = "successor"
        ctx.run_generation = 2
        ctx.chat_execution_reservation = (str(tmp_path / "other"), "successor", 2)
    finally:
        release.set()
        assert await asyncio.to_thread(worker.worker_done.wait, 5)
    assert observed_after_fence == [True]
    proof = ChatExecutionStore(tmp_path).inspect(key, 1)
    assert proof["worker"] == "completed"
    assert proof["effects"] == "unknown"
    with ChatExecutionStore(tmp_path)._db() as db:
        assert db.execute("SELECT tool_observed FROM executions WHERE session_key=?", (key,)).fetchone()[0] == 1
    calls = []
    blocked = ChatExecutionStore(tmp_path).recover(key, 1, "stop-fence", lambda proof: calls.append(proof))
    assert blocked["status"] == "blocked"
    assert calls == []
    assert not (tmp_path / "other" / "chat-execution.sqlite3").exists()


@pytest.mark.asyncio
async def test_actual_inbound_context_carries_reservation_to_physical_worker(tmp_path, monkeypatch):
    from tests.gateway.test_chat_execution_admission import inbound_fixture
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    runner, event, key, slot = inbound_fixture(tmp_path)
    seen = []

    async def agent_leaf(event, source, session_key, generation):
        fields = {name: None for name in runner._DISPLAY_TO_TURN_CTX}
        fields.update(resolve_display_setting=lambda *args: False, user_config={}, needs_progress_queue=False,
                      platform_key="telegram")
        ctx, _, _ = runner._run_agent_build_turn_context(
            SimpleNamespace(**fields), object, message=event.text, source=source,
            session_key=session_key, run_generation=generation, session_id="physical-session")
        assert ctx.chat_execution_reservation == (str(tmp_path.resolve()), key, generation)
        worker = runner._run_agent_start_turn_worker(ctx, lambda: seen.append("body"))
        await worker.executor_task
        return "executed"

    runner._handle_message_with_agent = agent_leaf
    assert await runner._handle_message(event) == "executed"
    assert seen == ["body"]
    slot.release.assert_called_once_with()
    proof = ChatExecutionStore(tmp_path).inspect(key, 1)
    assert proof["session_id"] == "physical-session"
    assert proof["worker"] == "completed"
    assert proof["effects"] == "unknown"


@pytest.mark.asyncio
async def test_failed_completion_write_stays_pending(tmp_path, monkeypatch):
    runner, _, ctx, store = reserved_worker(tmp_path, monkeypatch)
    saved = tmp_path / "saved.sqlite3"

    def body():
        store.path.rename(saved)
        store.path.mkdir()

    worker = runner._run_agent_start_turn_worker(ctx, body)
    await worker.executor_task
    assert worker.worker_done.is_set()
    store.path.rmdir()
    saved.rename(store.path)
    proof = ChatExecutionStore(tmp_path).inspect(ctx.session_key, 1)
    assert proof["worker"] == "pending"
    assert proof["effects"] == "unknown"


@pytest.mark.asyncio
async def test_physical_exception_records_completion(tmp_path, monkeypatch):
    runner, _, ctx, store = reserved_worker(tmp_path, monkeypatch)

    def body():
        raise ValueError("physical failure")

    worker = runner._run_agent_start_turn_worker(ctx, body)
    with pytest.raises(ValueError, match="physical failure"):
        await worker.executor_task
    assert worker.worker_done.is_set()
    assert store.inspect(ctx.session_key, 1)["worker"] == "completed"
    assert store.inspect(ctx.session_key, 1)["effects"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["database", "stale", "wrong-profile", "duplicate"])
async def test_worker_start_refusal_never_executes_body(tmp_path, monkeypatch, failure):
    runner, _, ctx, store = reserved_worker(tmp_path, monkeypatch)
    calls = []
    if failure == "database":
        store.path.unlink()
        store.path.mkdir()
    elif failure == "stale":
        assert store.reserve(ctx.session_key, 1, 0) == 2
    elif failure == "wrong-profile":
        ctx.chat_execution_reservation = (str(tmp_path / "wrong"), ctx.session_key, 1)
    else:
        assert store.bind_worker_started(ctx.session_key, 1, ctx.session_id)
    worker = runner._run_agent_start_turn_worker(ctx, lambda: calls.append("executed"))
    with pytest.raises(Exception):
        await worker.executor_task
    assert calls == []
    assert worker.worker_done.is_set()
