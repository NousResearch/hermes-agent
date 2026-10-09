"""REST callers, not a client-supplied reason, establish explicit deletion intent."""

import asyncio

from hermes_cli.web_routers import sessions
from hermes_state import SessionDB


def test_single_and_bulk_rest_deletes_prepare_committed_identity(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "profile" / "state.db") as db:
        monkeypatch.setattr(sessions, "_with_db", lambda profile, fn, **kwargs: fn(db))
        monkeypatch.setattr(sessions, "_session_files_dir", lambda profile: tmp_path / "profile" / "sessions")
        monkeypatch.setattr(sessions, "destructive_profile", lambda profile, operation: profile)
        for sid in ("single", "bulk"):
            db.create_session(sid, source="example", thread_id="thread-a")
        assert asyncio.run(sessions.delete_session_endpoint("single", profile="sample"))["ok"]
        assert asyncio.run(sessions.bulk_delete_sessions_endpoint(sessions.BulkDeleteSessions(
            ids=["bulk", "absent"], profile="sample")))["deleted"] == 1
        receipts = db.list_session_deletion_receipts()
        assert len(receipts) == 2
        assert all(row["deletion"]["surface"] == "dashboard_rest" for row in receipts)
        assert [row["deletion"]["identities"][0]["id"] for row in receipts] == ["single", "bulk"]
        assert asyncio.run(sessions.delete_session_endpoint("single", profile="sample"))["already_absent"]
        assert len(db.list_session_deletion_receipts()) == 2


def test_empty_sweep_rest_is_not_individually_selected_user_delete(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        monkeypatch.setattr(sessions, "_with_db", lambda profile, fn, **kwargs: fn(db))
        monkeypatch.setattr(sessions, "_session_files_dir", lambda profile: tmp_path / "sessions")
        monkeypatch.setattr(sessions, "destructive_profile", lambda profile, operation: profile)
        db.create_session("empty", source="example")
        db.end_session("empty", "done")
        assert asyncio.run(sessions.delete_empty_sessions_endpoint("sample"))["deleted"] == 1
        assert db.list_session_deletion_receipts() == []
