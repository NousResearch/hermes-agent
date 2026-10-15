"""REST callers, not a client-supplied reason, establish explicit deletion intent."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from hermes_cli.web_routers import sessions
from hermes_state import SessionDB


def test_bulk_rest_delete_prepares_committed_identity(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "profile" / "state.db") as db:
        monkeypatch.setattr(sessions, "_with_db", lambda profile, fn, **kwargs: fn(db))
        monkeypatch.setattr(sessions, "_session_files_dir", lambda profile: tmp_path / "profile" / "sessions")
        monkeypatch.setattr(sessions, "destructive_profile", lambda profile, operation: profile)
        db.create_session("bulk", source="example", thread_id="thread-a")
        assert asyncio.run(sessions.bulk_delete_sessions_endpoint(sessions.BulkDeleteSessions(
            ids=["bulk", "absent"], profile="sample")))["deleted"] == 1
        receipts = db.list_session_deletion_receipts()
        assert len(receipts) == 1
        assert receipts[0]["deletion"]["surface"] == "dashboard_rest"
        assert receipts[0]["deletion"]["identities"][0]["id"] == "bulk"


@pytest.mark.asyncio
@pytest.mark.parametrize("live_owner", [False, True])
async def test_single_rest_delete_receipt_is_atomic_and_exact_retry_safe(tmp_path, monkeypatch, live_owner):
    from gateway.session_authority import LiveSession, SessionAuthority
    from hermes_cli import web_server
    from hermes_state_runtime import begin_runtime_epoch

    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "receipt-test-token")
    monkeypatch.setattr("hermes_state._default_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    app = FastAPI()
    app.state.auth_required = False
    app.include_router(sessions.manage_router)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("single", source="example", thread_id="thread-a")
        if live_owner:
            epoch = begin_runtime_epoch(db, instance_id="owner")
            authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                         instance_id="owner", db=db, epoch=epoch)
            authority.sessions["single"] = LiveSession(None, "route")
            app.state.session_authority = authority
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost",
                                     headers={"Authorization": "Bearer receipt-test-token"}) as client:
            query = {"request_id": "delete", "expected_revision": 0, "expected_generation": 0}
            refused = await client.delete("/api/sessions/single", params=query | {"expected_revision": 99})
            assert refused.status_code == 409
            assert db.list_session_deletion_receipts() == []
            db._execute_write(lambda conn: conn.execute(
                "CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta "
                "WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
            failed = await client.delete("/api/sessions/single", params=query)
            assert failed.status_code == 503, failed.text
            assert db.get_session("single") is not None
            assert db.list_session_deletion_receipts() == []
            db._execute_write(lambda conn: conn.execute("DROP TRIGGER fail_receipt"))
            deleted = await client.delete("/api/sessions/single", params=query)
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["deleted_ids"] == ["single"]
            receipts = db.list_session_deletion_receipts()
            assert len(receipts) == 1
            deletion = receipts[0]["deletion"]
            assert deletion["surface"] == "dashboard_rest"
            assert deletion["profile_home"] == str(tmp_path.resolve())
            assert deletion["identities"][0]["thread_id"] == "thread-a"
            assert (await client.delete("/api/sessions/single", params=query)).json() == deleted.json()
            assert (await client.delete("/api/sessions/single", params=query | {"request_id": "absent"})).json()["already_absent"]
            assert db.list_session_deletion_receipts() == receipts
        with SessionDB(tmp_path / "other" / "state.db") as other:
            assert other.list_session_deletion_receipts() == []


def test_empty_sweep_rest_is_not_individually_selected_user_delete(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        monkeypatch.setattr(sessions, "_with_db", lambda profile, fn, **kwargs: fn(db))
        monkeypatch.setattr(sessions, "_session_files_dir", lambda profile: tmp_path / "sessions")
        monkeypatch.setattr(sessions, "destructive_profile", lambda profile, operation: profile)
        db.create_session("empty", source="example")
        db.end_session("empty", "done")
        assert asyncio.run(sessions.delete_empty_sessions_endpoint("sample"))["deleted"] == 1
        assert db.list_session_deletion_receipts() == []
