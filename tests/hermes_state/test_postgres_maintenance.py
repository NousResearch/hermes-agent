"""PostgreSQL retention keeps SQLite's recovery window and coordinates remote clients."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import pytest

from hermes_state import SessionDB


def test_retention_preserves_pinned_and_newly_closed_sessions_without_local_files(postgres_db, postgres_home):
    db = postgres_db
    old = time.time() - 120 * 86400
    for session_id in ("expired", "pinned", "orphan"):
        db.create_session(session_id, "cron")
        db.append_message(session_id, "user", session_id, timestamp=old)
        db._write_sql("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?", (old, old, session_id))
    db.end_session("expired", "done")
    db.end_session("pinned", "done")
    db.set_session_pinned("pinned", True)
    before = {path.relative_to(postgres_home) for path in postgres_home.rglob("*")}

    result = db.maybe_auto_prune_and_vacuum(retention_days=90)

    assert result == {"skipped": False, "pruned": 1, "closed": 1, "vacuumed": False}
    assert db.get_session("expired") is None
    assert db.get_session("pinned") is not None
    assert db.get_session("orphan")["end_reason"] == "startup_orphan_reap"
    assert db.get_messages("orphan")[0]["content"] == "orphan"
    assert db.maybe_auto_prune_and_vacuum(retention_days=90)["skipped"] is True
    assert db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0)["pruned"] == 0
    assert db.logical_size_bytes() > 0
    assert db.vacuum() == 0
    assert {path.relative_to(postgres_home) for path in postgres_home.rglob("*")} == before


def test_maintenance_excludes_another_handle_and_read_only_never_writes(postgres_db, postgres_home, monkeypatch):
    db = postgres_db
    entered = Event()
    release = Event()
    prune = db.prune_sessions

    def held_prune(*args, **kwargs):
        entered.set()
        assert release.wait(10), "maintenance caller was not released"
        return prune(*args, **kwargs)

    monkeypatch.setattr(db, "prune_sessions", held_prune)
    with SessionDB(postgres_home / "state.db") as other:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(db.maybe_auto_prune_and_vacuum)
            try:
                assert entered.wait(10), "maintenance caller did not acquire its lock"
                assert other.maybe_auto_prune_and_vacuum()["skipped"] is True
            finally:
                release.set()
            assert first.result(timeout=10) == {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False}

    timestamp = db.get_meta("last_auto_prune")
    with SessionDB(postgres_home / "state.db", read_only=True) as reader:
        assert reader.maybe_auto_prune_and_vacuum(min_interval_hours=0)["skipped"] is True
        assert reader.vacuum() == 0
        assert reader.rebuild_fts() == 0
        assert reader.optimize_fts_storage() == {"ok": False, "reason": "read_only"}
        assert reader.get_meta("last_auto_prune") == timestamp


def test_marker_cleanup_requires_explicit_backup_and_preserves_other_content(postgres_db, postgres_home):
    db = postgres_db
    assert db.purge_stale_tool_call_markers()["rows_affected"] == 0
    db.create_session("markers", "cli")
    tool_calls = [{"id": "call", "function": {"name": "memory", "arguments": "{}"}}]
    marker = db.append_message("markers", "assistant", "[memory]", tool_calls=tool_calls)
    prose = db.append_message("markers", "assistant", "[memory] details", tool_calls=tool_calls)
    before = {path.relative_to(postgres_home) for path in postgres_home.rglob("*")}

    with SessionDB(postgres_home / "state.db", read_only=True) as reader:
        preview = reader.purge_stale_tool_call_markers(dry_run=True)
        assert preview["dry_run"] is True
        assert preview["row_ids"] == [marker]
        assert preview["backup_path"] is None
    with pytest.raises(ValueError, match="pg_dump backup.*backup=False"):
        db.purge_stale_tool_call_markers()
    assert db.get_messages("markers")[0]["content"] == "[memory]"

    result = db.purge_stale_tool_call_markers(backup=False)

    assert result["rows_affected"] == 1
    messages = {row["id"]: row for row in db.get_messages("markers")}
    assert messages[marker]["content"] == ""
    assert messages[prose]["content"] == "[memory] details"
    assert messages[marker]["tool_calls"] == tool_calls
    assert db.purge_stale_tool_call_markers()["rows_affected"] == 0
    assert {path.relative_to(postgres_home) for path in postgres_home.rglob("*")} == before
