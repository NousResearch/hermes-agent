"""A failed state.db schema repair must leave the database byte-unchanged.

Reported incident: the automatic repair path deleted a user's transcripts and
then reported that it had failed.

``_repair_state_db_schema_locked`` ran its strategies on the live file, and
Strategy 2 ends in ``VACUUM``::

    PRAGMA writable_schema=ON
    DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'
    PRAGMA writable_schema=OFF
    VACUUM

VACUUM does not preserve what it cannot parse — it rebuilds the file from the
schema SQLite can still read. When the damage IS in the schema b-tree (page
1's child pointers aimed at data pages, which is exactly the ``malformed
database schema ()`` class this function exists to handle), the rebuild drops
every table hanging off the unreadable part. Measured on the reporting
install: ``state.db`` went from 3048 pages / 29 sessions / 2537 messages to
113 pages, in place.

The probe that follows then correctly reported the file was *still* malformed,
so the function returned ``repaired=False`` with "manual restore from backup
may be required" — after the only live copy had already been gutted.
Destroying the data and reporting the repair failed are not mutually exclusive
outcomes, and nothing in the code treated them as a contradiction.

The pre-repair backup (#69603) is a forensic artefact, not a recovery path:
nothing reads it back. So the invariant under test is the stronger one — a
repair that does not succeed must not change the file at all — plus the
structural assertion that makes it hold: the strategies never receive the live
database.

Fix under test: every strategy runs on a ``<db>.repair-scratch`` copy and is
promoted over the original with ``os.replace`` only once the result is proven
to open cleanly.

Mutation-checked: pointing ``_run_repair_strategies`` back at ``db_path``
instead of the scratch copy fails
``test_failed_repair_leaves_the_original_byte_identical`` and
``test_strategies_never_receive_the_live_database``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path

import pytest

import hermes_state
from hermes_state import repair_state_db_schema

PAGE_SIZE = 4096


def _write_populated_db(path: Path, *, sessions: int = 3, messages: int = 25) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA page_size={PAGE_SIZE}")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany(
        "INSERT INTO sessions (name) VALUES (?)",
        [(f"session-{i}",) for i in range(sessions)],
    )
    conn.executemany(
        "INSERT INTO messages (body) VALUES (?)",
        [(f"message body {i}" * 20,) for i in range(messages)],
    )
    conn.commit()
    conn.close()


def _break_the_schema_btree(path: Path) -> None:
    """Aim page 1's rightmost child at a data page.

    This is the shape of the reported corruption: ``sqlite_master``'s b-tree
    resolves to pages holding table content, so SQLite reports "malformed
    database schema ()" — the parentheses empty because the bogus row's name
    is not text.
    """
    data = bytearray(path.read_bytes())
    page_count = struct.unpack(">I", data[28:32])[0]
    assert page_count >= 3, "fixture needs a multi-page database"
    # Byte 100 is page 1's b-tree header; offset 108 is the rightmost pointer
    # on an interior page. Force page 1 to be interior and point it at the
    # last page, which holds table data rather than schema records.
    data[100] = 0x05
    struct.pack_into(">H", data, 103, 1)  # one cell
    struct.pack_into(">I", data, 108, page_count)  # rightmost -> data page
    struct.pack_into(">H", data, 112, PAGE_SIZE - 6)  # cell pointer
    struct.pack_into(">I", data, PAGE_SIZE - 6, page_count)
    path.write_bytes(bytes(data))


@pytest.fixture
def corrupt_db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    _write_populated_db(path)
    _break_the_schema_btree(path)
    with pytest.raises(sqlite3.DatabaseError):
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("SELECT * FROM sessions").fetchall()
        finally:
            conn.close()
    return path


# ---------------------------------------------------------------------------
# The structural guarantee
# ---------------------------------------------------------------------------


def test_strategies_never_receive_the_live_database(corrupt_db, monkeypatch):
    """Every strategy mutates its argument in place, so the property that
    makes them safe is simply that the argument is never the real file."""
    seen: list[Path] = []
    real = hermes_state._run_repair_strategies

    def spy(path, report):
        seen.append(path)
        return real(path, report)

    monkeypatch.setattr(hermes_state, "_run_repair_strategies", spy)
    repair_state_db_schema(corrupt_db)

    assert seen, "the repair path did not run at all"
    for path in seen:
        assert path != corrupt_db
        assert path.name.endswith(".repair-scratch")


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


def test_failed_repair_leaves_the_original_byte_identical(corrupt_db):
    before = hashlib.sha256(corrupt_db.read_bytes()).hexdigest()

    report = repair_state_db_schema(corrupt_db)

    after = hashlib.sha256(corrupt_db.read_bytes()).hexdigest()
    assert not report.get("repaired"), (
        "fixture precondition: this corruption is not automatically repairable"
    )
    assert before == after, (
        "a repair that FAILED rewrote the database anyway — this is the "
        "reported data loss: 29 sessions / 2537 messages became 113 pages "
        "while the function reported 'manual restore may be required'"
    )


def test_failed_repair_leaves_no_scratch_file_behind(corrupt_db):
    """A half-repaired file beside the DB is a trap for the next probe."""
    repair_state_db_schema(corrupt_db)
    leftovers = sorted(p.name for p in corrupt_db.parent.glob("*repair-scratch*"))
    assert leftovers == []


def test_failed_repair_still_takes_the_forensic_backup(corrupt_db):
    """Non-destructive repair does not make the #69603 backup redundant."""
    report = repair_state_db_schema(corrupt_db)
    assert report["backup_path"], "the pre-repair forensic copy is still required"
    assert Path(report["backup_path"]).exists()


# ---------------------------------------------------------------------------
# ...and a repair that DOES succeed must still land on the original path
# ---------------------------------------------------------------------------


def test_successful_repair_is_promoted_over_the_original(tmp_path, monkeypatch):
    """The scratch copy is a staging area, not a detour: a strategy that
    heals the copy must leave the healed bytes at ``db_path``."""
    db = tmp_path / "state.db"
    _write_populated_db(db)
    # Force the "already healthy" short-circuit off so the staging path runs,
    # and have the strategy pass mark a repair after writing a marker row.
    monkeypatch.setattr(
        hermes_state, "_db_opens_cleanly", lambda path: "forced-unhealthy"
    )

    def fake_strategies(scratch_path, report):
        conn = sqlite3.connect(str(scratch_path))
        conn.execute("INSERT INTO sessions (name) VALUES ('healed-on-scratch')")
        conn.commit()
        conn.close()
        report["repaired"] = True
        report["strategy"] = "test_strategy"
        return report

    monkeypatch.setattr(hermes_state, "_run_repair_strategies", fake_strategies)

    report = repair_state_db_schema(db)
    assert report["repaired"] is True

    conn = sqlite3.connect(str(db))
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sessions")]
    finally:
        conn.close()
    assert "healed-on-scratch" in names, (
        "the repaired copy was never promoted over the original"
    )
    assert not list(db.parent.glob("*repair-scratch*"))
