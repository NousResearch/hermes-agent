"""Automatic lineage archives must spare the open live tip (#115489).

Incident shape: a long compression lineage whose old root matched a bulk
``archive_sessions --older-than`` sweep; ``set_session_archived`` fanned out
over the whole compression lineage and flipped the OPEN, recently-active,
lease-holding live tip to ``archived=1``, hiding the live chat from the
Desktop sidebar (``archived=exclude`` default) while messages kept flowing.

Deliberate single-session archives keep the full lineage fan-out (an
archived ancestor would otherwise resurrect the chat on refresh); only the
automatic paths spare the live tip.
"""
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


def _backdate(db, sid, days):
    stale = time.time() - days * 86400
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?",
        (stale, stale, sid),
    )
    db._conn.commit()


def _end_as_compression(db, sid, days_ago):
    base = time.time() - days_ago * 86400
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' WHERE id = ?",
        (base, base + 10, sid),
    )
    db._conn.commit()


def _live_lineage(db):
    """root(compression, stale) -> mid(compression, stale) -> tip(open, live)."""
    db.create_session("root", source="feishu")
    db.create_session("mid", source="feishu", parent_session_id="root")
    db.create_session("tip", source="feishu", parent_session_id="mid")
    _end_as_compression(db, "root", 40)
    _end_as_compression(db, "mid", 35)
    db.append_message(session_id="tip", role="user", content="live dm")
    assert db.acquire_session_turn_lease("tip", "gateway-test", wait_seconds=0)


def test_bulk_archive_spares_open_live_tip(db):
    _live_lineage(db)

    assert db.archive_sessions(30) == 2
    assert db.get_session("root")["archived"] == 1
    assert db.get_session("mid")["archived"] == 1
    tip = db.get_session("tip")
    assert tip["archived"] == 0
    assert tip["end_reason"] is None


def test_bulk_archive_still_hides_fully_ended_chain(db):
    _live_lineage(db)
    db.release_session_turn_lease("tip", "gateway-test")
    db.end_session("tip", "user_closed")
    _backdate(db, "tip", 35)
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?",
        (time.time() - 35 * 86400, "tip"),
    )
    db._conn.commit()

    assert db.archive_sessions(30) == 3
    assert db.get_session("root")["archived"] == 1
    assert db.get_session("mid")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1


def test_sweep_still_archives_matched_stale_open_tip_and_ended_chain(db):
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    _end_as_compression(db, "root", 40)
    _backdate(db, "tip", 10)

    assert db.archive_stale_sessions(3) == 1
    assert db.get_session("tip")["archived"] == 1
    assert db.get_session("root")["archived"] == 1


def test_deliberate_archive_keeps_full_lineage_fanout(db):
    _live_lineage(db)

    assert db.set_session_archived("root", True) is True
    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
