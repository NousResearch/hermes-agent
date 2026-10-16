"""A deleted session's admission tombstones age out after the retry horizon; the receipts and
pointers that name them do not. None of them may turn into ``storage_unavailable``: a Bot receipt
re-read at every owner start would otherwise abort the gateway on every boot."""
import asyncio
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import hermes_state_runtime as rt
from gateway.session_results import retain_result
from hermes_state import SessionDB
from tools.bot_live_delivery import _locked, _read, _write

KEY = "a" * 32


def _delete(db, epoch, session_id, *, days_ago):
    snap = db.get_session(session_id)
    at = time.time() - days_ago * 86400
    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(time, "time", lambda: at)
        rt.mutate_runtime_session(db, epoch=epoch, principal_id="human", session_id=session_id,
                                  request_id="del", operation="delete", payload={},
                                  expected_revision=snap["runtime_revision"],
                                  expected_generation=snap["runtime_generation"])


def _receipt_then_deleted(tmp_path, *, status, days_ago, **fields):
    db = SessionDB(tmp_path / "state.db")
    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    db.create_session("botchat", source="cli")
    admission = rt.admit_session_input(db, epoch=epoch, principal_id="owner", session_id="botchat",
                                       request_id="bot:" + KEY, payload={"text": "ping"})
    row = rt.claim_session_input(db, epoch=epoch, session_id="botchat")
    retain_result(db, epoch=epoch, row=row, result={"result": {"final_response": "pong"}, "usage": {}})
    record = dict(status=status, delivery_id=KEY, profile_home=str(tmp_path), session_id="botchat",
                  admission_id=admission["admission_id"], principal_id="owner", message="ping")
    if status == "settled":
        record["reply"] = "pong"
    record.update(fields)
    with _locked(tmp_path) as root:
        _write(root / f"{KEY}.json", record)
    _delete(db, epoch, "botchat", days_ago=days_ago)
    db.prune_sessions(older_than_days=90)
    return db


def _receipt(tmp_path):
    with _locked(tmp_path) as root:
        return _read(root / f"{KEY}.json")


@pytest.mark.parametrize("days_ago", [6, 8])
def test_a_settled_receipt_keeps_its_answer_after_its_chat_is_deleted(tmp_path, days_ago):
    from gateway.session_bot import recover_bot_deliveries
    db = _receipt_then_deleted(tmp_path, status="settled", days_ago=days_ago)
    try:
        asyncio.run(recover_bot_deliveries(SimpleNamespace(db=db, waiters={})))
        record = _receipt(tmp_path)
        assert (record["status"], record["reply"]) == ("settled", "pong")
    finally:
        db.close()


def test_a_failed_receipt_keeps_its_error_after_its_chat_expired(tmp_path):
    from gateway.session_bot import recover_bot_deliveries
    db = _receipt_then_deleted(tmp_path, status="failed", days_ago=8,
                               error="HTTP 401 invalid api key", reason="auth")
    try:
        asyncio.run(recover_bot_deliveries(SimpleNamespace(db=db, waiters={})))
        record = _receipt(tmp_path)
        assert (record["status"], record["error"], record["reason"]) == ("failed", "HTTP 401 invalid api key", "auth")
    finally:
        db.close()


@pytest.mark.parametrize("status", ["queued", "claimed", "canonical"])
def test_a_pending_receipt_behind_an_expired_deleted_chat_becomes_ambiguous(tmp_path, status):
    from gateway.session_bot import recover_bot_deliveries
    db = _receipt_then_deleted(tmp_path, status=status, days_ago=8)
    try:
        asyncio.run(recover_bot_deliveries(SimpleNamespace(db=db, waiters={})))
        assert _receipt(tmp_path)["status"] == "ambiguous"
    finally:
        db.close()


def test_the_runtime_api_starts_after_a_deleted_bot_chat_expired(tmp_path, monkeypatch):
    import gateway.run_api
    from gateway.run_runtime import start_gateway_runtime_api
    db = _receipt_then_deleted(tmp_path, status="queued", days_ago=8)
    try:
        async def fake_api(runner):
            return SimpleNamespace(api_origin="http://127.0.0.1:1")
        monkeypatch.setattr(gateway.run_api, "start_gateway_api", fake_api)
        authority = SimpleNamespace(db=db, waiters={}, runner=None, profile_id=str(tmp_path))
        runner = SimpleNamespace(session_authority=authority, session_runtime_descriptor={})
        asyncio.run(start_gateway_runtime_api(runner))
        assert runner.session_runtime_descriptor["api_origin"] == "http://127.0.0.1:1"
        assert _receipt(tmp_path)["status"] == "ambiguous"  # recovery ran and answered it
    finally:
        db.close()


@pytest.mark.parametrize("days_ago, expected, text", [(6, 200, "answer"), (8, 409, "admission_conflict")])
@pytest.mark.asyncio
async def test_a_responses_key_of_a_deleted_session_never_reruns_and_never_503s(
        api, owner, days_ago, expected, text):
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()["result"] = {"final_response": "answer", "messages": []}
        return "answer"

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post("/v1/responses", api._handle_responses)
    body, headers = {"input": "once-only"}, {"Idempotency-Key": "aged-key"}
    async with TestClient(TestServer(app)) as client:
        assert (await client.post("/v1/responses", json=body, headers=headers)).status == 200
        store = api._current_response_store()
        # The replay body was evicted; the compact key -> admission pointer stays.
        store._conn.execute("DELETE FROM responses WHERE response_id LIKE 'idem:%'")
        store._conn.commit()
        sid = owner.db._read_one("SELECT * FROM session_admissions")["target_session_id"]
        at = time.time() - days_ago * 86400
        with pytest.MonkeyPatch.context() as clock:
            clock.setattr(time, "time", lambda: at)
            assert owner.db.delete_session(sid)
        owner.db.prune_sessions(older_than_days=90)
        retry = await client.post("/v1/responses", json=body, headers=headers)
        assert retry.status == expected and text in await retry.text(), await retry.text()
    assert calls == ["once-only"]
