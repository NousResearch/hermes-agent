"""Queued physical targets retain logical chain guards and immutable durable identities."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.chat_execution_followup import reserve_followup_execution
from gateway.chat_execution_recovery import ChatExecutionStore
from tests.gateway.test_chat_execution_worker import reserved_worker


async def finished_parent(home, monkeypatch):
    runner, state, ctx, store = reserved_worker(home, monkeypatch)
    monkeypatch.setattr(runner, "_resolve_profile_home_for_source", lambda source: home)
    worker = runner._run_agent_start_turn_worker(ctx, lambda: None)
    await worker.executor_task
    ctx._interrupt_depth = 0
    ctx.history = []
    ctx.context_prompt = ""
    ctx.channel_prompt = None
    ctx._status_thread_metadata = None
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._delivery_adapter_for = lambda source: None
    return runner, state, ctx, store


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["stop", "fence", "successor", "database", "unfinished", "wrong_session"])
async def test_actual_followup_reservation_refusal_executes_zero_body(tmp_path, monkeypatch, refusal):
    runner, _state, ctx, store = await finished_parent(tmp_path, monkeypatch)
    key = ctx.session_key
    def fail_database(self, *args, **kwargs):
        raise OSError("journal unavailable")
    actions = {
        "stop": lambda: runner._invalidate_session_run_generation(key),
        "fence": lambda: store.begin_recovery(key, 1, "recovery"),
        "successor": lambda: store.reserve(key, 1, 1),
        "database": lambda: monkeypatch.setattr(ChatExecutionStore, "reserve", fail_database),
        "unfinished": lambda: None,
        "wrong_session": lambda: setattr(ctx, "session_id", "other-session"),
    }
    actions[refusal]()
    if refusal == "unfinished":
        with store._db() as db:
            db.execute("UPDATE executions SET worker='pending' WHERE session_key=?", (key,))
    body = []
    async def agent(**kwargs):
        body.append(kwargs)
        return {}
    runner._run_agent = agent
    with pytest.raises((RuntimeError, OSError)):
        await runner._run_agent_queued_followup(ctx, None, "next", None, "", {"interrupted": True}, None)
    assert body == []
    assert store.inspect(key, 1)["effects"] == "unknown"
    expected_latest = 2 if refusal == "successor" else 1
    assert store.latest_generation(key) == expected_latest


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["stop", "fence", "duplicate", "session", "routing", "physical", "database"])
async def test_queued_physical_start_refusal_executes_zero_body(tmp_path, monkeypatch, refusal):
    runner, _state, ctx, store = await finished_parent(tmp_path, monkeypatch)
    handle = reserve_followup_execution(runner, ctx, ctx.source, ctx.session_key)
    next_ctx = SimpleNamespace(**vars(ctx))
    next_ctx.chat_execution_reservation = handle
    def fail_start(*args, **kwargs):
        raise OSError("start write failed")
    def fence_row():
        # An unbound execution cannot request recovery; exercise the durable start guard directly.
        with store._db() as db:
            db.execute("INSERT INTO session_fences VALUES (?, ?, 'fence', 'fenced', NULL)",
                       (ctx.session_key, handle.generation))
    actions = {
        "stop": lambda: runner._invalidate_session_run_generation(ctx.session_key),
        "fence": fence_row,
        "duplicate": lambda: store.bind_worker_started(ctx.session_key, handle.generation, ctx.session_id),
        "session": lambda: setattr(next_ctx, "session_id", "wrong-session"),
        "routing": lambda: setattr(next_ctx, "chat_execution_reservation", replace(handle, run_generation=2)),
        "physical": lambda: setattr(next_ctx, "chat_execution_reservation", replace(handle, generation=1)),
        "database": lambda: monkeypatch.setattr(ChatExecutionStore, "bind_worker_started", fail_start),
    }
    actions[refusal]()
    body = []
    worker = runner._run_agent_start_turn_worker(next_ctx, lambda: body.append("ran"))
    with pytest.raises((RuntimeError, OSError)):
        await worker.executor_task
    assert worker.worker_done.is_set() and body == []
    assert store.inspect(ctx.session_key, handle.generation)["effects"] == "unknown"


@pytest.mark.asyncio
async def test_followup_identity_snapshot_and_physical_completion(tmp_path, monkeypatch):
    from agent.chat_execution_context import observe_tool_before_execution
    runner, state, ctx, store = await finished_parent(tmp_path, monkeypatch)
    handle = reserve_followup_execution(runner, ctx, ctx.source, ctx.session_key)
    next_ctx = SimpleNamespace(**vars(ctx))
    next_ctx.chat_execution_reservation = handle
    seen = []
    def body():
        with store._db() as db:
            seen.append(tuple(db.execute(
                "SELECT worker,worker_started,effects FROM executions WHERE session_key=? AND generation=?",
                (handle.session_key, handle.generation),
            ).fetchone()))
        assert observe_tool_before_execution() is True
    worker = runner._run_agent_start_turn_worker(next_ctx, body)
    # The scheduled target captured its identity before any mutable slot/context replacement.
    next_ctx.chat_execution_reservation = None
    next_ctx.session_id = "mutated"
    await worker.executor_task
    assert seen == [("pending", 1, "unknown")]
    assert state.persistent.run_generation == 1
    assert store.inspect(handle.session_key, 1)["worker"] == "completed"
    assert store.inspect(handle.session_key, handle.generation)["worker"] == "completed"
    assert not store.bind_worker_started(handle.session_key, handle.generation, handle.session_id)
    assert store.recover(handle.session_key, handle.generation, "unused", lambda proof: None)["status"] == "blocked"


@pytest.mark.asyncio
async def test_followup_reservation_requires_latest_predecessor_once(tmp_path, monkeypatch):
    runner, _, ctx, store = await finished_parent(tmp_path, monkeypatch)
    handle = reserve_followup_execution(runner, ctx, ctx.source, ctx.session_key)
    assert handle.generation == 2
    assert store.inspect(ctx.session_key, 2)["session_id"] is None
    assert store.inspect(ctx.session_key, 2)["worker"] == "unknown"
    with pytest.raises(RuntimeError, match="reservation refused"):
        reserve_followup_execution(runner, ctx, ctx.source, ctx.session_key)
    assert store.latest_generation(ctx.session_key) == 2


def test_legacy_followup_does_not_manufacture_reservation():
    ctx = SimpleNamespace(chat_execution_reservation=None)
    assert reserve_followup_execution(None, ctx, None, None) is None
