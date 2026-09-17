"""Desktop cron delivery: one session per invocation, dated unique titles.

Invariants pinned here:
- Each invocation gets its OWN ``cron_desktop`` session (the old stable per-job
  session id made one thread grow forever, so every reply paid for days of
  accumulated briefing output plus the tool trace that produced it).
- The session title carries no "Cron Delivery: " prefix and carries the run
  stamp, because ``sessions.title`` is UNIQUE — two runs must never fight over
  one bare name.
"""

from datetime import datetime

import pytest

from cron import desktop_delivery
from cron.desktop_delivery import _deliver_to_desktop_session
from hermes_state import SessionDB

JOB = {"id": "54405811a560", "name": "morning-briefing"}


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _freeze(monkeypatch, *stamps: datetime):
    """Drive ``_hermes_now`` through the given instants, one per delivery."""
    calls = iter(stamps)
    monkeypatch.setattr(desktop_delivery, "_hermes_now", lambda: next(calls))


def _rows(db):
    return db.list_sessions_rich(source="cron_desktop", min_message_count=1, compact_rows=True)


def test_each_invocation_is_its_own_session(db, monkeypatch):
    day_one = datetime(2026, 9, 16, 9, 5, 12)
    day_two = datetime(2026, 9, 17, 9, 4, 41)

    _freeze(monkeypatch, day_one, day_two)
    first = day_one.strftime("%Y%m%d_%H%M%S")
    second = day_two.strftime("%Y%m%d_%H%M%S")

    assert _deliver_to_desktop_session(JOB, "Tuesday brief", db) is None
    assert _deliver_to_desktop_session(JOB, "Wednesday brief", db) is None

    sessions = {row["id"]: row for row in _rows(db)}
    assert set(sessions) == {
        f"cron_delivery_{JOB['id']}_{first}",
        f"cron_delivery_{JOB['id']}_{second}",
    }
    # Yesterday's session keeps exactly its own delivery — nothing accumulates.
    yesterday = db.get_messages(sessions[f"cron_delivery_{JOB['id']}_{first}"]["id"])
    assert [m["content"] for m in yesterday] == ["Tuesday brief"]
    # The sidebar lists both, so neither run has to be opened through the other.
    assert sorted(row["title"] for row in _rows(db)) == [
        "morning-briefing · Sep 16 09:05",
        "morning-briefing · Sep 17 09:04",
    ]


def test_name_hint_titles_the_session_without_the_cron_delivery_prefix(db, monkeypatch):
    _freeze(monkeypatch, datetime(2026, 9, 17, 9, 5, 3))

    assert _deliver_to_desktop_session(JOB, "brief", db, session_name_hint="Daily Brief") is None

    (row,) = _rows(db)
    assert row["title"] == "Daily Brief · Sep 17 09:05"


def test_a_second_delivery_in_the_same_minute_stays_titled_and_unique(db, monkeypatch):
    """Titles are unique-indexed; a collision takes the lineage fallback, never a blank row."""
    _freeze(
        monkeypatch,
        datetime(2026, 9, 17, 9, 5, 3),
        datetime(2026, 9, 17, 9, 5, 47),
    )

    assert _deliver_to_desktop_session(JOB, "brief", db) is None
    # A failure notice for the same run lands seconds later, on its own session.
    assert _deliver_to_desktop_session(JOB, "run failed", db) is None

    assert sorted(row["title"] for row in _rows(db)) == [
        "morning-briefing · Sep 17 09:05",
        "morning-briefing · Sep 17 09:05 #2",
    ]


def test_delivery_opens_its_own_session_db_when_none_is_passed(tmp_path, monkeypatch):
    """The real path (``session_db=None``): delivery runs after the agent's DB closed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _freeze(monkeypatch, datetime(2026, 9, 17, 9, 5, 3))

    assert _deliver_to_desktop_session(JOB, "brief") is None

    # Same resolution the delivery used (no explicit path), so this asserts
    # against the store the owned handle actually wrote to rather than assuming
    # it is tmp_path/state.db.
    session_db = SessionDB()
    try:
        (row,) = _rows(session_db)
        assert row["id"] == f"cron_delivery_{JOB['id']}_20260917_090503"
        assert row["title"] == "morning-briefing · Sep 17 09:05"
    finally:
        session_db.close()


def test_bare_target_carries_no_name_hint(monkeypatch):
    """``desktop-session`` must not title a session with the raw job id."""
    from cron import scheduler_delivery

    bare = scheduler_delivery._resolve_single_delivery_target(JOB, "desktop-session")
    assert bare is not None and bare["chat_id"] == ""
    named = scheduler_delivery._resolve_single_delivery_target(JOB, "desktop-session:Daily Brief")
    assert named is not None and named["chat_id"] == "Daily Brief"
