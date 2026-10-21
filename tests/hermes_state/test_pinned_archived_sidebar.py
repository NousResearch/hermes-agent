"""Pinned rows must stay reachable from the sidebar session query.

The sidebar calls ``list_sessions_rich`` with ``include_archived=False`` and
``include_pinned=True``. A pin is an explicit keep: the back-fill must not
inherit the archived constraint, and the stale-session sweep must not archive
a pinned row.
"""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _sidebar_ids(db, **extra):
    """Same flags as the desktop sidebar slice."""
    return [
        row["id"]
        for row in db.list_sessions_rich(
            limit=20,
            offset=0,
            min_message_count=1,
            include_archived=False,
            archived_only=False,
            order_by_last_active=True,
            compact_rows=True,
            include_pinned=True,
            **extra,
        )
    ]


def _seed(db, sid, *, source="cli", pinned=False, archived=False):
    db.create_session(session_id=sid, source=source)
    db.append_message(session_id=sid, role="user", content=f"msg {sid}")
    if pinned:
        assert db.set_session_pinned(sid, True)
    if archived:
        assert db.set_session_archived(sid, True)


def test_sidebar_query_returns_pinned_archived_session(db):
    """A pinned row stamped archived is absent from the page and must back-fill."""
    _seed(db, "live")
    _seed(db, "pin-archived", pinned=True, archived=True)
    _seed(db, "archived-only", archived=True)

    page = [
        row["id"]
        for row in db.list_sessions_rich(
            limit=20,
            offset=0,
            min_message_count=1,
            include_archived=False,
            order_by_last_active=True,
            compact_rows=True,
            include_pinned=False,
        )
    ]
    assert "pin-archived" not in page
    assert "archived-only" not in page

    ids = _sidebar_ids(db)
    assert "live" in ids
    assert "archived-only" not in ids
    assert "pin-archived" in ids
    row = db.get_session("pin-archived")
    assert row["pinned"] == 1
    assert row["archived"] == 1


def test_pinned_backfill_still_obeys_other_filters(db):
    """Dropping the archived constraint must not drop source exclusion."""
    _seed(db, "cron-pin", source="cron", pinned=True, archived=True)
    assert "cron-pin" not in _sidebar_ids(db, exclude_sources=["cron"])


def test_stale_sweep_does_not_archive_pinned_session(db):
    """The idle sweep retires an unpinned stale row and leaves a pin active."""
    db.create_session(session_id="keep", source="cli")
    db.append_message(session_id="keep", role="user", content="old pin")
    db.set_session_pinned("keep", True)
    db.create_session(session_id="drop", source="cli")
    db.append_message(session_id="drop", role="user", content="old")
    old = time.time() - 10 * 86400
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id IN (?, ?)", (old, "keep", "drop"))
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id IN (?, ?)",
        (old, "keep", "drop"),
    )
    db._conn.commit()

    assert db.archive_stale_sessions(3) == 1
    assert db.get_session("keep")["archived"] == 0
    assert db.get_session("keep")["pinned"] == 1
    assert db.get_session("drop")["archived"] == 1
