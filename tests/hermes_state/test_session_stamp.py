"""The per-session stamp label: normalization contract + lineage propagation.

A stamp is ONE short free-text label per session (Merged, WIP, Review, …) kept
server-side so every client agrees. It rides the compression lineage exactly
like pinned/archived — a root would otherwise resurrect a stale label on
refresh, because list readers project a root to its live tip.
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


def _compression_pair(db: SessionDB):
    """A compression parent (root) plus its live tip child — the shape a stamp must span."""
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression',"
        " message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def test_stamp_set_on_tip_reaches_the_whole_compression_lineage(db):
    """Writing the tip must stamp the root too: the Desktop projects roots onto the tip,
    so a tip-only write leaves the root free to resurrect its old (absent) label."""
    _compression_pair(db)

    assert db.set_session_stamp("tip", "Merged") is True

    assert db.get_session("root")["stamp"] == "Merged"
    assert db.get_session("tip")["stamp"] == "Merged"
    # The compact (desktop list) projection carries the column too.
    assert db.list_sessions_rich(compact_rows=True)[0]["stamp"] == "Merged"


def test_clearing_a_stamp_writes_sql_null_across_the_lineage(db):
    """Clearing stores NULL (no stamp), never '' — an empty string would read as
    'stamped with nothing' to every client."""
    _compression_pair(db)
    db.set_session_stamp("root", "WIP")

    assert db.set_session_stamp("tip", "   ") is True

    for sid in ("root", "tip"):
        assert db.get_session(sid)["stamp"] is None
        raw = db._conn.execute(
            "SELECT stamp FROM sessions WHERE id = ?", (sid,)).fetchone()[0]
        assert raw is None


def test_stamp_normalization_contract(db):
    """Strip + collapse, blank/None clears, and refuse (ValueError) control characters or
    text past the cap instead of silently truncating."""
    db.create_session("norm", source="cli")

    db.set_session_stamp("norm", "  On   Hold  ")
    assert db.get_session_stamp("norm") == "On Hold"
    assert db.set_session_stamp("norm", "") is True
    assert db.get_session_stamp("norm") is None
    assert db.set_session_stamp("norm", None) is True
    assert db.get_session_stamp("norm") is None

    db.set_session_stamp("norm", "x" * SessionDB.MAX_STAMP_LENGTH)
    assert db.get_session_stamp("norm") == "x" * SessionDB.MAX_STAMP_LENGTH

    with pytest.raises(ValueError):
        db.set_session_stamp("norm", "x" * (SessionDB.MAX_STAMP_LENGTH + 1))
    for bad in ("WIP\nHold", "WIP\tHold"):
        with pytest.raises(ValueError):
            db.set_session_stamp("norm", bad)
    # The refusals left the previously stored label untouched.
    assert db.get_session_stamp("norm") == "x" * SessionDB.MAX_STAMP_LENGTH
