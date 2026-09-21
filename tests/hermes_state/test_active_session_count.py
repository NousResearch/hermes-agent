"""``count_active_sessions`` answers in SQL what a page of rows cannot.

``/api/status`` counted active sessions by listing 50 rich rows and filtering in Python.
``list_sessions_rich`` orders by ``started_at``, so a long-running session that is still
active but not among the newest 50 was missed outright — an undercount, not a truncation.
"""

import time

import pytest

from hermes_state import SessionDB

WINDOW = 300


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _mk(db, sid, *, started, active=None, ended=None, hidden=0, archived=0):
    db.create_session(sid, source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=?, hidden=?, archived=? WHERE id=?",
        (started, started if active is None else active, ended, hidden, archived, sid))
    db._conn.commit()


def _old_way(db, now):
    """The implementation this replaced: a page of 50, filtered in Python."""
    rows = db.list_sessions_rich(limit=50, compact_rows=True)
    return sum(1 for s in rows if s.get("ended_at") is None
               and (now - s.get("last_active", s.get("started_at", 0))) < WINDOW)


def test_counts_an_active_session_older_than_the_50_row_page(db):
    """The regression: 60 newer idle sessions push the active one off the page."""
    now = time.time()
    _mk(db, "long-runner", started=now - 86400, active=now - 10)     # active, oldest by started_at
    for i in range(60):
        _mk(db, f"idle-{i}", started=now - 100 - i, active=now - 4000)  # newer, idle

    assert db.count_active_sessions(WINDOW, now=now) == 1
    assert _old_way(db, now) == 0, "sanity: the old page-of-50 approach misses it"


def test_matches_the_listing_when_everything_fits_on_one_page(db):
    now = time.time()
    _mk(db, "a", started=now - 50, active=now - 5)
    _mk(db, "b", started=now - 60, active=now - 6)
    _mk(db, "idle", started=now - 70, active=now - 9999)

    assert db.count_active_sessions(WINDOW, now=now) == 2
    assert _old_way(db, now) == 2


def test_excludes_ended_hidden_and_archived_like_the_listing(db):
    """Filters must mirror the default list or the number contradicts the UI.
    Bot Mode marks its sessions hidden, so those must not inflate the count."""
    now = time.time()
    _mk(db, "live", started=now - 10, active=now - 1)
    _mk(db, "ended", started=now - 10, active=now - 1, ended=now - 1)
    _mk(db, "hidden", started=now - 10, active=now - 1, hidden=1)
    _mk(db, "archived", started=now - 10, active=now - 1, archived=1)

    assert db.count_active_sessions(WINDOW, now=now) == 1


def test_idle_window_is_honoured(db):
    now = time.time()
    _mk(db, "just-inside", started=now - 1000, active=now - (WINDOW - 5))
    _mk(db, "just-outside", started=now - 1000, active=now - (WINDOW + 5))

    assert db.count_active_sessions(WINDOW, now=now) == 1
