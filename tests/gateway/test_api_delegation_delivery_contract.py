"""A detached API result must have an addressable consumer and one durable row."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import clear_session_vars
from gateway.wake import persist_delegation_delivery
from hermes_state import SessionDB
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
    snapshot = [{"role": "user", "content": "caller snapshot"}]
    for name in ("caller_history_no_session", "response_chain"):
        assert runs[name]["runtime"]["target"] is None
        assert runs[name]["runtime"]["history"] == snapshot
    # Caller history + session id consumes the delivery row once, alternation preserved.
    assert runs["caller_history"]["runtime"]["target"] == "child"
    assert [{k: m[k] for k in ("role", "content")}
            for m in runs["caller_history"]["runtime"]["history"]] == [
        {"role": "user", "content": "caller snapshot\n\nDELIVERY_RESULT"}]
    assert runs["caller_history_again"]["runtime"]["target"] == "child"
    assert runs["caller_history_again"]["runtime"]["history"] == snapshot
    assert runs["session"]["runtime"]["target"] == "child"
    assert runs["session"]["runtime"]["history"] == result["resumed_history"]
    assert runs["declared_key"]["runtime"]["target"] == "declared"
    assert runs["declared_key"]["runtime"]["history"][0]["content"] == "DECLARED_HISTORY"


@pytest.mark.asyncio
async def test_delivery_replay_is_atomic_across_continuation_and_busy_turn(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    peer = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="api_server")
        db.create_session("other", source="api_server")
        adapters = [SimpleNamespace(_ensure_session_db=lambda: db), SimpleNamespace(_ensure_session_db=lambda: peer)]
        evt = {"type": "async_delegation", "delegation_id": "unique-unit"}
        async def send(adapter, event=evt):
            await persist_delegation_delivery(adapter, text="RESULT", session_id="parent", evt=event)
        await asyncio.gather(*(send(a) for a in adapters))
        assert len(db.get_messages("parent")) == 1
        db.end_session("parent", "compression")
        db.create_session("child", source="api_server", parent_session_id="parent")
        await send(adapters[0])
        assert db.get_messages("child") == []  # old event was already recorded in the lineage
        from hermes_state_errors import SessionTurnLeaseLostError
        assert db.acquire_session_turn_lease("child", "client-turn", wait_seconds=0)
        later = {**evt, "delegation_id": "later-unit"}
        try:
            with pytest.raises(SessionTurnLeaseLostError):
                await send(adapters[0], later)
            assert db.get_messages("child") == []
        finally:
            db.release_session_turn_lease("child", "client-turn")
        await send(adapters[0], later)
        assert len(db.get_messages("child")) == 1
        notice = {**later, "task_failure_notice": True, "results": [{"task_index": 0, "status": "failed"}]}
        await send(adapters[0], notice)
        await send(adapters[1], notice)
        assert len(db.get_messages("child")) == 2  # interim notice cannot consume the final's identity
        assert db.get_messages("other") == []
    finally:
        peer.close()
        db.close()


def _fold_adapter(db):
    async def ensure():
        return db
    return SimpleNamespace(_ensure_session_db_async=ensure)


def _launch(history, session_id="s1"):
    from gateway.platforms.api_server_runs import _RunLaunch
    return _RunLaunch(None, "run", None, session_id, None, False, "next", history, True,
                      agent_kwargs={}, request_profile=None, browser_control_principal=None,
                      browser_control_transport_family=None, fold_deliveries=True)


@pytest.mark.asyncio
async def test_caller_history_folds_each_delivery_exactly_once(tmp_path):
    from gateway.platforms.api_server_runs import _fold_caller_history_deliveries
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="api_server")
        db.append_message("s1", "user", "go")
        db.append_message("s1", "assistant", "dispatched")
        adapter = SimpleNamespace(_ensure_session_db=lambda: db)
        await persist_delegation_delivery(adapter, text="RESULT-A", session_id="s1",
                                          evt={"type": "async_delegation", "delegation_id": "a"})
        await persist_delegation_delivery(adapter, text="RESULT-B", session_id="s1",
                                          evt={"type": "async_delegation", "delegation_id": "b"})
        caller = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "dispatched"},
                  {"role": "user", "content": "RESULT-B"}]  # caller already stored B
        run = _launch(list(caller))
        await _fold_caller_history_deliveries(_fold_adapter(db), run)
        history = run.conversation_history
        assert sum("RESULT-A" in m["content"] for m in history) == 1
        assert sum("RESULT-B" in m["content"] for m in history) == 1
        assert all(a["role"] != b["role"] for a, b in zip(history, history[1:]))
        again = _launch(list(caller))
        await _fold_caller_history_deliveries(_fold_adapter(db), again)
        assert again.conversation_history == caller  # both rows already consumed
        # Rows stay in the transcript for pollers and session-history continuations.
        contents = [m["content"] for m in db.get_messages_as_conversation("s1")]
        assert contents.count("RESULT-A") == 1 and contents.count("RESULT-B") == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_caller_history_fold_and_reservation_deliver_single_copy(tmp_path):
    from gateway.platforms.api_server_runs import _fold_caller_history_deliveries
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="api_server")
        adapter = SimpleNamespace(_ensure_session_db=lambda: db)
        await persist_delegation_delivery(adapter, text="RESULT-R", session_id="s1",
                                          evt={"type": "async_delegation", "delegation_id": "r"})
        lease = db.reserve_caller_history_deliveries("s1", "webui", ttl_seconds=60)
        caller = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"}]
        run = _launch(list(caller))
        await _fold_caller_history_deliveries(_fold_adapter(db), run)
        assert run.conversation_history == caller  # reserved for the wake turn, not folded
        assert db.commit_caller_history_deliveries(lease[0]["reservation_token"], "webui") == 1
        await persist_delegation_delivery(adapter, text="RESULT-F", session_id="s1",
                                          evt={"type": "async_delegation", "delegation_id": "f"})
        run = _launch(list(caller))
        await _fold_caller_history_deliveries(_fold_adapter(db), run)
        assert sum("RESULT-F" in m["content"] for m in run.conversation_history) == 1
        assert db.reserve_caller_history_deliveries("s1", "webui", ttl_seconds=60) == []
    finally:
        db.close()
