"""Pinned rows must stay reachable from the sidebar session query on a SQLite backend store.

The sidebar calls ``list_sessions_rich`` with ``include_archived=False`` and
``include_pinned=True``. A pin is an explicit keep: the back-fill must not
inherit the archived constraint, and the stale-session sweep
(``session_retention.run_auto_archive``) must not archive a pinned row.
"""
import time

import pytest

from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, get_session, list_sessions_rich, set_session_archived, set_session_pinned,
)
from session_retention import run_auto_archive


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _sidebar_ids(db, **extra):
    """Same flags as the desktop sidebar slice."""
    return [
        row["id"]
        for row in list_sessions_rich(
            db,
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
    create_session(db, session_id=sid, source=source)
    append_message(db, session_id=sid, role="user", content=f"msg {sid}")
    if pinned:
        assert set_session_pinned(db, sid, True)
    if archived:
        assert set_session_archived(db, sid, True)[0]


def test_sidebar_query_returns_pinned_archived_session(db):
    """A pinned row stamped archived is absent from the page and must back-fill."""
    _seed(db, "live")
    _seed(db, "pin-archived", pinned=True, archived=True)
    _seed(db, "archived-only", archived=True)

    page = [
        row["id"]
        for row in list_sessions_rich(
            db,
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
    row = get_session(db, "pin-archived")
    assert row["pinned"] == 1
    assert row["archived"] == 1


def test_pinned_backfill_still_obeys_other_filters(db):
    """Dropping the archived constraint must not drop source exclusion."""
    _seed(db, "cron-pin", source="cron", pinned=True, archived=True)
    assert "cron-pin" not in _sidebar_ids(db, exclude_sources=["cron"])


def test_stale_sweep_does_not_archive_pinned_session(db, store):
    """The idle sweep retires an unpinned stale row and leaves a pin active."""
    create_session(db, session_id="keep", source="cli")
    append_message(db, session_id="keep", role="user", content="old pin")
    set_session_pinned(db, "keep", True)
    create_session(db, session_id="drop", source="cli")
    append_message(db, session_id="drop", role="user", content="old")
    old = time.time() - 10 * 86400
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id IN (?, ?)", (old, "keep", "drop"))
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id IN (?, ?)",
        (old, "keep", "drop"),
    )
    db._conn.commit()

    assert run_auto_archive(store, idle_days=3, min_interval_hours=0)["archived"] == 1
    assert get_session(db, "keep")["archived"] == 0
    assert get_session(db, "keep")["pinned"] == 1
    assert get_session(db, "drop")["archived"] == 1
