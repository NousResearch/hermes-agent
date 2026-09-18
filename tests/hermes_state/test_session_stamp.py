"""The per-session stamp label: normalization contract + lineage propagation.

A stamp is ONE short free-text label per session (Merged, WIP, Review, …) kept
server-side so every client agrees. It rides the compression lineage exactly
like pinned/archived — a root would otherwise resurrect a stale label on
refresh, because list readers project a root to its live tip.
"""

import sqlite3
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


# ── The list (up to three labels on one session) ────────────────────────────

def test_stamp_list_keeps_order_dedupes_case_insensitively_and_refuses_a_fourth(db):
    """Order = the order added, dedupe is case-insensitive onto the FIRST spelling, and a
    fourth distinct label is REFUSED — the list is what the user built, so silently dropping
    the tail would hide a real refusal."""
    db.create_session("multi", source="cli")

    assert db.set_session_stamps("multi", ["WIP", " Review ", "wip", "Hold"]) is True

    assert db.get_session_stamps("multi") == ["WIP", "Review", "Hold"]
    # The compact (desktop list) projection decodes the stored payload into labels.
    assert db.list_sessions_rich(compact_rows=True)[0]["stamps"] == ["WIP", "Review", "Hold"]
    # ...and the singular column mirrors the FIRST label, so a one-label reader stays coherent.
    assert db.get_session_stamp("multi") == "WIP"
    assert db.get_session("multi")["stamp"] == "WIP"

    with pytest.raises(ValueError):
        db.set_session_stamps("multi", ["a", "b", "c", "d"])
    # The refusal left the stored list untouched.
    assert db.get_session_stamps("multi") == ["WIP", "Review", "Hold"]


def test_clearing_a_stamp_list_writes_sql_null_to_both_columns(db):
    """[] clears, and it clears the singular mirror too — otherwise the CLI would keep reading
    a label the list no longer carries."""
    db.create_session("clear", source="cli")
    db.set_session_stamps("clear", ["WIP", "Hold"])

    assert db.set_session_stamps("clear", []) is True

    assert db.get_session_stamps("clear") == []
    assert db.get_session_stamp("clear") is None
    raw = db._conn.execute(
        "SELECT stamp, stamps FROM sessions WHERE id = ?", ("clear",)).fetchone()
    assert (raw[0], raw[1]) == (None, None)


def test_stamp_list_spans_the_whole_compression_lineage(db):
    """Same lineage rule as the single label: a tip-only write lets the root resurrect a stale
    list on the next refresh."""
    _compression_pair(db)

    assert db.set_session_stamps("tip", ["WIP", "Hold"]) is True

    for sid in ("root", "tip"):
        assert db.get_session_stamps(sid) == ["WIP", "Hold"]
        assert db.get_session(sid)["stamp"] == "WIP"
        assert db.list_sessions_rich(compact_rows=True)
    assert all(row["stamps"] == ["WIP", "Hold"]
               for row in db.list_sessions_rich(compact_rows=True))


def test_the_single_label_door_replaces_the_list(db):
    """`hermes sessions stamp` and any older client speak one label; writing one must not leave
    an orphaned second label behind on the session."""
    db.create_session("single", source="cli")
    db.set_session_stamps("single", ["WIP", "Hold"])

    assert db.set_session_stamp("single", "Review") is True

    assert db.get_session_stamps("single") == ["Review"]
    assert db.get_session_stamps("single") != ["Review", "Hold"]


def test_a_row_stamped_before_the_list_column_reads_as_its_one_label(db):
    """Read-compat: a session written while `stamp` was the only column must still read as
    stamped — on the single-session getter AND on every list row a client renders."""
    db.create_session("legacy", source="cli")
    db._conn.execute("UPDATE sessions SET stamp = 'Merged' WHERE id = 'legacy'")
    db._conn.commit()

    assert db.get_session_stamps("legacy") == ["Merged"]
    assert db.get_session("legacy")["stamps"] == ["Merged"]
    assert db.list_sessions_rich(compact_rows=True)[0]["stamps"] == ["Merged"]


def test_an_existing_store_gains_the_column_without_a_migration(tmp_path):
    """A store that predates ``stamps`` must gain it on the next OPEN — the declarative column
    reconciler IS the migration, which is why this feature ships no migration file.

    The store is built by the real code and then has the column dropped underneath it, so the
    only thing the reopen can use to put it back is ``_reconcile_columns()``."""
    path = tmp_path / "pre-stamps.db"
    db = SessionDB(path)
    db.create_session("old", source="cli")
    db.set_session_stamp("old", "Merged")
    db.close()

    con = sqlite3.connect(path)
    con.execute("ALTER TABLE sessions DROP COLUMN stamps")
    con.commit()
    assert "stamps" not in {row[1] for row in con.execute("PRAGMA table_info(sessions)")}
    con.close()

    db = SessionDB(path)
    try:
        assert "stamps" in {row[1] for row in db._conn.execute("PRAGMA table_info(sessions)")}
        # The label the row already carried still reads as a stamp, and the list writes.
        assert db.get_session_stamps("old") == ["Merged"]
        assert db.set_session_stamps("old", ["Merged", "Hold"]) is True
        assert db.get_session_stamps("old") == ["Merged", "Hold"]
    finally:
        db.close()


def test_a_malformed_stored_payload_reads_as_unstamped_rather_than_raising(db):
    """The decoder is total: bad hand-written SQL must not break a list render."""
    db.create_session("junk", source="cli")
    for junk in ("not json", '{"a":1}', "[1,2]", '"WIP"'):
        db._conn.execute("UPDATE sessions SET stamps = ? WHERE id = 'junk'", (junk,))
        db._conn.commit()

        assert db.get_session_stamps("junk") == []
        assert db.get_session("junk")["stamps"] == []
