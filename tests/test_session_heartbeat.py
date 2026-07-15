"""Tests for the session activity heartbeat (hermes_state + web_server)."""

import time

import pytest

from hermes_state import SessionDB
from hermes_cli import web_server


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestTouchSessionHeartbeat:
    def test_touch_session_heartbeat_sets_column(self, db):
        db.create_session("s1", "cli")
        assert db.get_session("s1")["last_heartbeat"] is None

        before = time.time()
        db.touch_session_heartbeat("s1")
        after = time.time()

        hb = db.get_session("s1")["last_heartbeat"]
        assert before <= hb <= after

    def test_touch_session_heartbeat_unknown_id_is_noop(self, db):
        db.touch_session_heartbeat("does-not-exist")  # must not raise

    def test_touch_session_heartbeat_by_key_resolves_newest_row(self, db):
        # Two rows sharing a session_key: only the newest (highest started_at)
        # should be stamped, matching list_gateway_sessions's dedupe rule.
        db.create_session("gw-old", "telegram", session_key="agent:main:telegram:dm:c1")
        db._conn.execute(
            "UPDATE sessions SET started_at = started_at - 100 WHERE id = 'gw-old'"
        )
        db._conn.commit()
        db.create_session("gw-new", "telegram", session_key="agent:main:telegram:dm:c1")

        db.touch_session_heartbeat_by_key("agent:main:telegram:dm:c1")

        assert db.get_session("gw-new")["last_heartbeat"] is not None
        assert db.get_session("gw-old")["last_heartbeat"] is None

    def test_touch_session_heartbeat_by_key_unknown_key_is_noop(self, db):
        db.touch_session_heartbeat_by_key("no-such-key")  # must not raise


class TestListSessionsRichExposesHeartbeat:
    def test_last_heartbeat_present_and_null_by_default(self, db):
        db.create_session("s1", "cli")
        rows = db.list_sessions_rich()
        assert rows[0]["last_heartbeat"] is None

    def test_last_heartbeat_reflects_touch(self, db):
        db.create_session("s1", "cli")
        db.touch_session_heartbeat("s1")
        rows = db.list_sessions_rich()
        assert rows[0]["last_heartbeat"] is not None


class TestSessionActivityFlags:
    """Exercises web_server._session_activity_flags, the shared expression
    used by every /api/sessions-style is_active/bg_active computation."""

    def test_stale_message_but_recent_heartbeat_is_active_and_bg_active(self):
        now = time.time()
        s = {
            "ended_at": None,
            "last_active": now - 600,
            "last_heartbeat": now - 10,
        }
        is_active, bg_active = web_server._session_activity_flags(s, now)
        assert is_active is True
        assert bg_active is True

    def test_no_recent_message_or_heartbeat_is_inactive(self):
        now = time.time()
        s = {
            "ended_at": None,
            "last_active": now - 600,
            "last_heartbeat": now - 600,
        }
        is_active, bg_active = web_server._session_activity_flags(s, now)
        assert is_active is False
        assert bg_active is False

    def test_recent_message_without_heartbeat_is_active_not_bg_active(self):
        now = time.time()
        s = {
            "ended_at": None,
            "last_active": now - 5,
            "last_heartbeat": None,
        }
        is_active, bg_active = web_server._session_activity_flags(s, now)
        assert is_active is True
        assert bg_active is False

    def test_ended_session_is_never_active(self):
        now = time.time()
        s = {
            "ended_at": now - 1,
            "last_active": now - 1,
            "last_heartbeat": now - 1,
        }
        is_active, bg_active = web_server._session_activity_flags(s, now)
        assert is_active is False
        assert bg_active is False
