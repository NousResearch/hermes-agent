"""Regression tests for the orphaned transcript spool defect.

``messages.session_id REFERENCES sessions(id)`` (hermes_state_common.py) with foreign keys on,
so appending a spooled transcript to a DELETED session raises sqlite3.IntegrityError.
``recover_pending_to_db``'s except branch deliberately never unlinks, so the same failure
recurred on every subsequent gateway boot, permanently.

The fix checks session existence before appending and drops a spool whose parent session is
gone for good - but only after a grace window, because a resume can recreate the same session
id moments after its spool is written and a first-pass miss must not destroy that message.
"""
import json
import os
import sqlite3
import time

import pytest

from gateway import shutdown_flush as sf


def _make_db(with_session, session_id="live-sess"):
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL "
        "REFERENCES sessions(id), content TEXT)"
    )
    if with_session:
        conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    conn.commit()
    return conn


class _DB:
    """SessionDB stand-in that enforces the real foreign key."""

    def __init__(self, conn):
        self._conn = conn

    def append_message(self, session_id=None, message=None, **_kw):
        self._conn.execute(
            "INSERT INTO messages (session_id, content) VALUES (?, ?)",
            (session_id, (message or {}).get("content", "")),
        )
        self._conn.commit()

    def get_session(self, session_id):
        row = self._conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
        return {"id": session_id} if row else None


def _write_spool(tmp_path, session_id, age_seconds=0.0):
    payload = {
        "reason": sf.TRANSCRIPT_CAP_DROP_REASON,
        "data": {"session_id": session_id, "message": {"role": "user", "content": "hi"}},
    }
    p = tmp_path / f"{int(time.time() * 1000)}.json"
    p.write_text(json.dumps(payload))
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(p, (old, old))
    return p


@pytest.fixture
def flush_dir(tmp_path, monkeypatch):
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setattr(sf, "_get_flush_dir", lambda: d)
    return d


def test_live_session_spool_is_replayed_and_removed(flush_dir):
    conn = _make_db(with_session=True)
    spool = _write_spool(flush_dir, "live-sess")
    assert sf.recover_pending_to_db(session_db=_DB(conn)) == 1
    assert not spool.exists()


def test_orphan_beyond_grace_is_dropped_not_retried_forever(flush_dir):
    conn = _make_db(with_session=False)
    spool = _write_spool(flush_dir, "deleted-sess", age_seconds=sf._ORPHAN_SPOOL_GRACE_S + 60)
    sf.recover_pending_to_db(session_db=_DB(conn))
    assert not spool.exists(), "orphaned spool must be dropped, else it fails on every boot"
    # The decisive assertion: a second boot must not reproduce the failure.
    sf.recover_pending_to_db(session_db=_DB(conn))


def test_orphan_inside_grace_is_preserved(flush_dir):
    conn = _make_db(with_session=False)
    spool = _write_spool(flush_dir, "deleted-sess")
    sf.recover_pending_to_db(session_db=_DB(conn))
    assert spool.exists(), "a fresh spool may belong to a session about to be recreated"