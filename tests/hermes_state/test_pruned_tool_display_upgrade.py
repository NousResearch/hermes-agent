"""The tool display layout upgrade is atomic and leaves settled history reads bounded."""

import sqlite3
from contextlib import contextmanager

import pytest

from hermes_state import SessionDB
from hermes_state_timeline import get_session_timeline
from agent.message_metadata import MESSAGE_UID


@pytest.mark.parametrize("blocked_key", ["tool_display_identity_version", "tool_display_identity_cursor"])
def test_failed_tool_display_upgrade_rolls_back_index_invalidation(tmp_path, monkeypatch, blocked_key):
    import hermes_state_schema

    if blocked_key.endswith("cursor"):
        monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_CHUNK", 1)
        monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_BUDGET_S", 0.0)
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", source="test")
    db.append_message("chat", "tool", "complete result", tool_call_id="call")
    db.append_message("chat", "tool", "second result", tool_call_id="call-2")
    with db._read_ctx() as conn:
        before = [tuple(row) for row in conn.execute("SELECT * FROM messages")]
    db._execute_write(lambda conn: conn.execute(
        "DELETE FROM state_meta WHERE key = 'tool_display_identity_version'"))
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_tool_display_marker BEFORE INSERT ON state_meta "
        f"WHEN new.key = '{blocked_key}' BEGIN "
        "SELECT RAISE(ABORT, 'blocked marker'); END"))
    db.close()

    with pytest.raises(sqlite3.IntegrityError, match="blocked marker"):
        SessionDB(db.db_path)
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT * FROM messages").fetchall() == before
        assert conn.execute("SELECT value FROM state_meta WHERE key = 'tool_display_identity_version'").fetchone() is None
        assert conn.execute("SELECT value FROM state_meta WHERE key = 'tool_display_identity_cursor'").fetchone() is None
        conn.execute("DROP TRIGGER reject_tool_display_marker")
    monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_CHUNK", 2000)
    monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_BUDGET_S", 1.0)
    with SessionDB(db.db_path) as upgraded:
        assert upgraded.get_meta("tool_display_identity_version") == "1"
        assert [m["content"] for m in upgraded.get_messages("chat", include_compacted=True)] == [
            "complete result", "second result"]


@pytest.mark.parametrize("layout_complete", [False, True])
def test_uid_and_tool_display_upgrades_converge_across_bounded_reopens(tmp_path, monkeypatch, layout_complete):
    import hermes_state_schema

    monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_CHUNK", 2)
    monkeypatch.setattr(hermes_state_schema, "_MESSAGE_UID_BACKFILL_BUDGET_S", 0.0)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", source="test")
        db.append_message("chat", "user", "start", timestamp=100.0)
        db.append_message("chat", "assistant", "working", timestamp=101.0)
        db.append_messages_batch("chat", [
            {"role": "tool", "content": "result", "tool_call_id": "call", "timestamp": 102.0}
            for _ in range(4)
        ])
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET message_uid = NULL"))
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET display_order = 3 WHERE role = 'tool'"))
        db._execute_write(lambda conn: conn.execute(
            "DELETE FROM state_meta WHERE key = 'message_uid_backfill'"))
        if not layout_complete:
            db._execute_write(lambda conn: conn.execute(
                "DELETE FROM state_meta WHERE key = 'tool_display_identity_version'"))
    for opened in range(3):
        with SessionDB(db.db_path) as handle:
            visible = handle.get_messages("chat", include_compacted=True)
            resumed = handle.get_resume_conversations("chat")[1]
            assert [m.get(MESSAGE_UID) for m in visible if m["role"] == "tool"] == [
                m.get(MESSAGE_UID) for m in resumed if m["role"] == "tool"]
            with SessionDB(db.db_path, read_only=True) as reader:
                assert reader.get_messages("chat", include_compacted=True) == visible
            if not layout_complete and opened < 2:
                assert handle.get_meta("tool_display_identity_version") is None
            if opened == 2:
                assert handle.get_meta("tool_display_identity_version") == "1"
                assert len([m for m in visible if m["role"] == "tool"]) == 4


def test_settled_display_pages_do_not_scan_tool_history_or_reinvalidate_on_reopen(tmp_path, monkeypatch):
    counts = []
    for size in (100, 10000):
        with SessionDB(tmp_path / f"state-{size}.db") as db:
            db.create_session("chat", source="test")
            db.append_message("chat", "user", "start", timestamp=100.0)
            db.append_messages_batch("chat", [
                {"role": "tool", "content": "result", "tool_call_id": f"call-{i}", "timestamp": 101.0 + i}
                for i in range(size)
            ])
            db.get_messages("chat", include_compacted=True, limit=1)
        with SessionDB(db.db_path) as reopened:
            assert reopened._read_one(
                "SELECT 1 FROM messages WHERE display_identity IS NULL OR display_order IS NULL LIMIT 1") is None
            original_read = reopened._read_ctx
            steps = []

            @contextmanager
            def measured():
                with original_read() as conn:
                    conn.set_progress_handler(lambda: steps.append(1) or 0, 1)
                    try:
                        yield conn
                    finally:
                        conn.set_progress_handler(None, 0)

            monkeypatch.setattr(reopened, "_read_ctx", measured)
            assert reopened.get_messages("chat", include_compacted=True, limit=1)[0]["content"] == "start"
            counts.append(len(steps))
            monkeypatch.setattr(reopened, "_read_ctx", original_read)
            reopened._execute_write(lambda writer: writer.execute(
                "DELETE FROM state_meta WHERE key = 'tool_display_identity_version'"))

            def authorize(action, table, column, *_):
                if action == sqlite3.SQLITE_READ and table == "messages" and column == "tool_calls":
                    return sqlite3.SQLITE_DENY
                if action in (sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            @contextmanager
            def guarded():
                with original_read() as conn:
                    conn.set_authorizer(authorize)
                    try:
                        yield conn
                    finally:
                        conn.set_authorizer(None)

            monkeypatch.setattr(reopened, "_read_ctx", guarded)
            assert get_session_timeline(reopened, "chat")["entries"][0]["preview"] == "start"
            monkeypatch.setattr(reopened, "_read_ctx", original_read)
    assert counts[1] <= counts[0] * 2
