"""Tests for #65194: startup-time sweep of orphaned TUI/subagent sessions.

The TUI gateway reaps disconnected websocket sessions with an in-process
``threading.Timer`` grace timer.  A gateway restart destroys the timer, so
the session row stays ``ended_at IS NULL`` forever — nothing re-checks stale
rows on the next boot.  ``SessionDB.sweep_orphaned_sessions()`` is the
DB-level startup sweep that closes such rows with a distinct
``end_reason='startup_orphan_reap'``.

Staleness requires BOTH ``started_at`` and the newest ``messages.timestamp``
to be older than the cutoff:

* message-recency alone would sweep a freshly created compression/branch
  child that carries old *copied* message timestamps;
* ``started_at`` alone would sweep a long-lived session that is still
  actively producing messages.
"""

import time

import pytest

from hermes_state import SessionDB

IDLE_S = 6 * 3600  # mirror the TUI gateway's default session TTL


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _backdate_session(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (ts, session_id)
    )
    db._conn.commit()


def _set_message_timestamps(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?", (ts, session_id)
    )
    db._conn.commit()


def _make_session(
    db: SessionDB,
    session_id: str,
    *,
    source: str,
    started_at: float,
    message_at: float = None,
) -> None:
    db.create_session(session_id, source=source)
    if message_at is not None:
        db.append_message(session_id, role="user", content="hello")
        _set_message_timestamps(db, session_id, message_at)
    _backdate_session(db, session_id, started_at)


class TestSweepOrphanedSessions:
    def test_stale_tui_session_swept(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 1

        row = db.get_session("stale-tui")
        assert row["ended_at"] is not None
        assert row["end_reason"] == "startup_orphan_reap"

    def test_stale_subagent_session_swept(self, db):
        stale = time.time() - 8 * 3600
        _make_session(
            db, "stale-sub", source="subagent", started_at=stale, message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 1
        assert db.get_session("stale-sub")["end_reason"] == "startup_orphan_reap"

    def test_recent_message_spares_old_session(self, db):
        """A long-lived session that is still talking is NOT an orphan."""
        stale = time.time() - 48 * 3600
        _make_session(
            db, "active", source="tui", started_at=stale, message_at=time.time()
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 0
        assert db.get_session("active")["ended_at"] is None

    def test_fresh_session_with_old_copied_messages_spared(self, db):
        """Compression/branch children copy history — old message timestamps
        on a just-created row must not get it swept."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "fresh-child", source="tui", started_at=time.time(), message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 0
        assert db.get_session("fresh-child")["ended_at"] is None

    def test_gateway_owned_source_not_swept(self, db):
        """telegram/discord/... rows belong to the messaging gateway (#60609)."""
        stale = time.time() - 8 * 3600
        _make_session(
            db, "tg-sess", source="telegram", started_at=stale, message_at=stale
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 0
        assert db.get_session("tg-sess")["ended_at"] is None

    def test_already_ended_session_untouched(self, db):
        """First end_reason wins — the sweep never rewrites history."""
        stale = time.time() - 8 * 3600
        _make_session(db, "done", source="tui", started_at=stale, message_at=stale)
        db.end_session("done", "user_exit")

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 0
        assert db.get_session("done")["end_reason"] == "user_exit"

    def test_stale_empty_session_swept_fresh_spared(self, db):
        """Rows without messages fall back to started_at staleness."""
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-empty", source="tui", started_at=stale)
        _make_session(db, "fresh-empty", source="tui", started_at=time.time())

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 1
        assert db.get_session("stale-empty")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("fresh-empty")["ended_at"] is None

    def test_custom_sources_respected(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "stale-cli", source="cli", started_at=stale, message_at=stale)
        _make_session(db, "stale-tui", source="tui", started_at=stale, message_at=stale)

        assert (
            db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S, sources=("cli",)) == 1
        )
        assert db.get_session("stale-cli")["end_reason"] == "startup_orphan_reap"
        assert db.get_session("stale-tui")["ended_at"] is None

    def test_returns_zero_on_empty_db(self, db):
        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == 0
