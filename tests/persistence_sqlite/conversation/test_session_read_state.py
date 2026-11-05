import time

import pytest

from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, list_sessions_rich, set_session_read


@pytest.fixture
def db(backend, scope_a):
    return backend.open(scope_a).holder()


def _last_read(db, sid):
    row = db._read_one(
        "SELECT last_read_at FROM sessions WHERE id = ?", (sid,)
    )
    return row["last_read_at"] if row is not None else None


def _row(db, sid):
    rows = list_sessions_rich(db, include_archived=True)
    return next(s for s in rows if s["id"] == sid)


def test_untracked_sessions_are_read(db):
    """NULL watermark = never tracked = read, so shipping the column doesn't
    badge a user's entire pre-feature history at once."""
    create_session(db, session_id="s1", source="cli")
    append_message(db, session_id="s1", role="user", content="hi")

    assert _last_read(db, "s1") is None
    assert _row(db, "s1")["unread"] is False


def test_mark_read_then_new_activity_flips_back_to_unread(db):
    create_session(db, session_id="s1", source="cli")
    append_message(db, session_id="s1", role="user", content="hi")

    assert set_session_read(db, "s1") is True
    assert _row(db, "s1")["unread"] is False

    # New activity postdating the watermark makes it unread again without
    # any write on the message path.
    time.sleep(0.01)  # health: allow HX013 -- the next message's timestamp must postdate the read watermark
    append_message(db, session_id="s1", role="assistant", content="reply")
    assert _row(db, "s1")["unread"] is True


def test_mark_unread_explicitly(db):
    create_session(db, session_id="s1", source="cli")
    append_message(db, session_id="s1", role="user", content="hi")
    set_session_read(db, "s1")

    assert set_session_read(db, "s1", read=False) is True
    assert _last_read(db, "s1") == 0.0
    assert _row(db, "s1")["unread"] is True


def test_missing_session_returns_false(db):
    assert set_session_read(db, "nope") is False


def _compression_pair(db):
    base = time.time() - 100
    create_session(db, "root", source="cli")
    create_session(db, "tip", source="cli", parent_session_id="root")
    db._write_sql(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._write_sql(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )


def test_reading_compression_tip_stamps_whole_lineage(db):
    _compression_pair(db)

    assert set_session_read(db, "tip") is True

    root_read = _last_read(db, "root")
    assert root_read is not None and root_read > 0
    assert root_read == _last_read(db, "tip")

    # The projected conversation row (root surfaced as tip) derives read.
    rows = list_sessions_rich(db, order_by_last_active=True)
    assert [s["id"] for s in rows] == ["tip"]
    assert rows[0]["unread"] is False


def test_marking_root_unread_marks_projected_conversation(db):
    _compression_pair(db)
    set_session_read(db, "tip")

    assert set_session_read(db, "root", read=False) is True

    rows = list_sessions_rich(db, order_by_last_active=True)
    assert [s["id"] for s in rows] == ["tip"]
    assert rows[0]["unread"] is True
