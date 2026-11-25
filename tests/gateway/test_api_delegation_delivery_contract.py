"""A detached API result must have an addressable consumer and one durable row."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import clear_session_vars
from gateway.wake import persist_delegation_delivery
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_session
from tools.delegate_tool_dispatch import _resolve_async_wake_sid


@pytest.mark.asyncio
async def test_detached_dispatch_requires_a_declared_consumer(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_HISTORY_DELIVERY", "1")
    for capability in (None, "", "1"):
        kw = dict(chat_id="api-parent", session_id="api-parent")
        import inspect
        if capability is not None and "session_history_delivery" in inspect.signature(APIServerAdapter._bind_api_server_session).parameters:
            kw["session_history_delivery"] = capability
        tokens = APIServerAdapter._bind_api_server_session(**kw)
        try:
            args = ["api-parent"]
            if len(inspect.signature(_resolve_async_wake_sid).parameters) > 1:
                from gateway.session_context import session_history_delivery_supported
                args.append(session_history_delivery_supported())
            target = _resolve_async_wake_sid(*args)
            assert target == ("api-parent" if capability == "1" else None)
        finally:
            clear_session_vars(tokens)
    from evals.api_delegation_http_probe import probe
    result = await probe()
    for request in result["requests"]:
        assert request["status"] == 200
        runtime = request["runtime"]
        assert runtime["target"] == ("child" if request["explicit"] else None)
        if request["explicit"]:
            assert runtime["session_id"] == "child"
            assert request["header"] == "parent"
    assert result["unsolicited_calls"] == 0
    assert result["durable_child_rows"] == 1
    assert sum(m["content"] == "DELIVERY_RESULT" for m in result["resumed_history"]) == 1
    runs = {r["name"]: r for r in result["runs"]}
    assert all(r["status"] == 202 for r in runs.values())
    for name in ("caller_history", "response_chain"):
        assert runs[name]["runtime"]["target"] is None
        assert runs[name]["runtime"]["history"] == [{"role": "user", "content": "caller snapshot"}]
    assert runs["session"]["runtime"]["target"] == "child"
    assert runs["session"]["runtime"]["history"] == result["resumed_history"]
    assert runs["declared_key"]["runtime"]["target"] == "declared"
    assert runs["declared_key"]["runtime"]["history"][0]["content"] == "DECLARED_HISTORY"


@pytest.mark.asyncio
async def test_delivery_replay_is_atomic_across_continuation_and_busy_turn(tmp_path):
    from persistence_assembly import open_persistence, profile_scope
    stores = [open_persistence(profile_scope(home=tmp_path)) for _ in range(2)]
    db, peer = stores
    try:
        seed_session(db, "parent", "api_server")
        seed_session(db, "other", "api_server")
        adapters = [SimpleNamespace(_ensure_store=lambda: db, _store_beside=lambda _db: stores[0]),
                    SimpleNamespace(_ensure_store=lambda: peer, _store_beside=lambda _db: stores[1])]
        evt = {"type": "async_delegation", "delegation_id": "unique-unit"}
        async def send(adapter, event=evt):
            await persist_delegation_delivery(adapter, text="RESULT", session_id="parent", evt=event)
        await asyncio.gather(*(send(a) for a in adapters))
        assert len(transcript_rows(db, "parent")) == 1
        db.sessions.end("parent", "compression")
        seed_session(db, "child", "api_server", parent_session_id="parent")
        await send(adapters[0])
        assert transcript_rows(db, "child") == []  # old event was already recorded in the lineage
        from persistence import TurnLeaseLost
        assert db.coordination.try_acquire_turn_lease(session_id="child", holder="client-turn")
        later = {**evt, "delegation_id": "later-unit"}
        try:
            with pytest.raises(TurnLeaseLost):
                await send(adapters[0], later)
            assert transcript_rows(db, "child") == []
        finally:
            db.coordination.release_turn_lease(session_id="child", holder="client-turn")
        await send(adapters[0], later)
        assert len(transcript_rows(db, "child")) == 1
        notice = {**later, "task_failure_notice": True, "results": [{"task_index": 0, "status": "failed"}]}
        await send(adapters[0], notice)
        await send(adapters[1], notice)
        assert len(transcript_rows(db, "child")) == 2  # interim notice cannot consume the final's identity
        assert transcript_rows(db, "other") == []
    finally:
        for store in stores:
            store.close()
