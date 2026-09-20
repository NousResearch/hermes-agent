import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _compression_pair(db: SessionDB):
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def test_archiving_compression_tip_archives_projected_root(db):
    _compression_pair(db)

    assert db.set_session_archived("tip", True) is True

    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["tip"]


def test_unarchiving_compression_tip_unarchives_projected_root(db):
    _compression_pair(db)
    db.set_session_archived("tip", True)

    assert db.set_session_archived("tip", False) is True

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_archived_only_view_includes_hidden_archived_sessions(db):
    """The archived-only view is the recovery surface: a session that is both
    archived and hidden (Bot Mode marks its sessions hidden) must appear
    there, otherwise it is unreachable from every UI list (#90946)."""
    db.create_session("plain", source="cli")
    db.create_session("both", source="cli")
    assert db.set_session_hidden("both", True) is True
    assert db.set_session_archived("both", True) is True

    # Default list: hidden rows stay excluded (unchanged behaviour)...
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["plain"]
    # ...and the archived-only view must surface the archived+hidden row.
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["both"]
def test_unarchive_sessions_restores_archived_rows_and_spares_live_ones(db):
    """``unarchive_sessions`` inverts ``archive_sessions`` for every archived row (lineage
    included), leaves live rows alone, and is idempotent."""
    _compression_pair(db)
    db.create_session("live", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'live'", (time.time(),)
    )
    db._conn.commit()

    assert db.archive_sessions(source="cli") == 1  # the ended root; the open continuation is spared
    assert db.get_session("live")["archived"] == 0

    assert db.unarchive_sessions() == 2  # root + tip
    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert sorted(s["id"] for s in db.list_sessions_rich(order_by_last_active=True)) == [
        "live", "tip",
    ]

    assert db.unarchive_sessions() == 0  # only archived rows are candidates, so this is a no-op


def test_unarchive_reaches_an_open_session_that_the_prune_path_cannot_see(db):
    """The prune/archive candidate path is ended-only by design, but ``archive_stale_sessions``
    may retire an open session. Unarchive must reach those rows, or auto-archive can hide a
    session with no way back."""
    db.create_session("open", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = NULL WHERE id = 'open'",
        (time.time() - 40 * 86400,),
    )
    db._conn.commit()

    assert db.archive_stale_sessions(30) == 1
    assert db.get_session("open")["archived"] == 1
    assert db.list_prune_candidates(archived=True) == []  # ended-only: blind to this row
    assert [s["id"] for s in db.list_archived_candidates()] == ["open"]

    assert db.unarchive_sessions() == 1
    assert db.get_session("open")["archived"] == 0


def test_unarchive_restores_a_pinned_row_that_archive_would_spare(db):
    """A pin is a durable *keep* flag, not a lock: it must never be able to block recovery."""
    db.create_session("kept", source="cli")
    db.set_session_pinned("kept", True)
    db.set_session_archived("kept", True)

    assert [s["id"] for s in db.list_archived_candidates()] == ["kept"]
    assert db.unarchive_sessions() == 1
    assert db.get_session("kept")["archived"] == 0
