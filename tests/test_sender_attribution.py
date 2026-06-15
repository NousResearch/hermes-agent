"""Tests for per-message sender attribution (F-003 multi-participant channels).

User messages record WHICH device they were typed on (``messages.sender_device``,
schema v17) so a session shared across devices reads like a group chat. The
agent's own rows stay NULL — the assistant is not a "sender".
"""

import pytest
from unittest.mock import patch

from hermes_state import SCHEMA_VERSION, SessionDB


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def _new_session(db, session_id="sess-attr-1"):
    db.create_session(session_id=session_id, source="tui")
    return session_id


class TestSenderDeviceColumn:
    def test_schema_version_is_17_or_later(self):
        assert SCHEMA_VERSION >= 17

    def test_user_message_auto_stamps_local_device(self, db):
        sid = _new_session(db)
        with patch("hermes_state.get_device_name", return_value="ko-mac"):
            db.append_message(sid, "user", "hello from my laptop")
        msg = db.get_messages(sid)[0]
        assert msg["sender_device"] == "ko-mac"

    def test_explicit_sender_wins_over_auto_stamp(self, db):
        sid = _new_session(db)
        with patch("hermes_state.get_device_name", return_value="ko-mac"):
            db.append_message(sid, "user", "hi", sender_device="omar-iphone")
        msg = db.get_messages(sid)[0]
        assert msg["sender_device"] == "omar-iphone"

    def test_assistant_and_tool_rows_stay_null(self, db):
        sid = _new_session(db)
        with patch("hermes_state.get_device_name", return_value="ko-mac"):
            db.append_message(sid, "assistant", "hello back")
            db.append_message(sid, "tool", "result", tool_name="terminal",
                              tool_call_id="tc-1")
        msgs = db.get_messages(sid)
        assert all(m["sender_device"] is None for m in msgs)

    def test_device_resolution_failure_degrades_to_null(self, db):
        sid = _new_session(db)
        with patch("hermes_state.get_device_name", side_effect=RuntimeError("no net")):
            db.append_message(sid, "user", "still works")
        msg = db.get_messages(sid)[0]
        assert msg["sender_device"] is None

    def test_migration_adds_column_to_v16_database(self, tmp_path):
        # Simulate a pre-v17 database: create fresh, drop the column marker by
        # rebuilding messages without sender_device, then reopen to migrate.
        db_path = tmp_path / "old_state.db"
        seeded = SessionDB(db_path=db_path)
        seeded.create_session(session_id="old-sess", source="cli")
        conn = seeded._conn
        with seeded._lock:
            conn.executescript(
                """
                CREATE TABLE messages_old AS SELECT id, session_id, role,
                    content, timestamp FROM messages;
                """
            )
            conn.execute("UPDATE schema_version SET version = 16")
            conn.commit()
        seeded.close()

        migrated = SessionDB(db_path=db_path)
        try:
            cols = {
                row[1]
                for row in migrated._conn.execute("PRAGMA table_info(messages)")
            }
            assert "sender_device" in cols
            version = migrated._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0]
            assert version >= 17
        finally:
            migrated.close()


class TestSenderDeviceSurvivesRewrites:
    def test_replace_messages_preserves_attribution(self, db):
        sid = _new_session(db)
        with patch("hermes_state.get_device_name", return_value="ko-mac"):
            db.append_message(sid, "user", "first", sender_device="omar-iphone")
            db.append_message(sid, "assistant", "reply")
        rewritten = db.get_messages(sid)
        # Simulate a /compress//retry transcript rewrite round-trip.
        db.replace_messages(sid, rewritten)
        msgs = db.get_messages(sid)
        assert msgs[0]["sender_device"] == "omar-iphone"
        assert msgs[1]["sender_device"] is None
