"""Tests for SessionDB.active_session_count() O(1) SQL method.

Covers:
  - Returns 0 on empty database
  - Counts only open sessions with recent activity
  - Excludes ended sessions
  - Excludes idle sessions beyond idle_secs threshold
"""

import time
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    db = SessionDB(db_path=tmp_path / "test_state.db")
    yield db
    db.close()


def _create_session(db, ended=False, with_message=False, msg_age=0):
    sid = str(uuid.uuid4())
    now = time.time()
    with db._lock:
        db._conn.execute(
            "INSERT INTO sessions (id, started_at, ended_at, source) VALUES (?, ?, ?, ?)",
            (sid, now - 100, now if ended else None, "cli"),
        )
        if with_message:
            db._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (sid, "user", "hello", now - msg_age),
            )
        db._conn.commit()
    return sid


def test_active_session_count_empty(db):
    assert db.active_session_count() == 0


def test_active_session_count_open_with_recent_message(db):
    _create_session(db, ended=False, with_message=True, msg_age=60)
    assert db.active_session_count(idle_secs=300) == 1


def test_active_session_count_excludes_ended(db):
    _create_session(db, ended=True, with_message=True, msg_age=60)
    assert db.active_session_count(idle_secs=300) == 0


def test_active_session_count_excludes_idle(db):
    _create_session(db, ended=False, with_message=True, msg_age=600)
    assert db.active_session_count(idle_secs=300) == 0


def test_active_session_count_mixed(db):
    _create_session(db, ended=False, with_message=True, msg_age=60)   # active
    _create_session(db, ended=True, with_message=True, msg_age=60)    # ended
    _create_session(db, ended=False, with_message=True, msg_age=600)  # idle
    _create_session(db, ended=False, with_message=False)              # no msgs, started recently-ish
    count = db.active_session_count(idle_secs=300)
    assert count == 2  # active + no-msgs-but-open-recently
