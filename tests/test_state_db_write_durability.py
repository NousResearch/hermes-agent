"""Regression: state.db repair-path writes must be durable on macOS.

Incident (2026-08-19, recurrence of 2026-08-18/19): `state.db` was recovered
clean at 01:02, tore again in the pages holding rows written 02:18-02:22, and
the damage went undetected until 13:36 when a write finally landed on a
damaged page (`append_message failed: constraint failed`). `PRAGMA
integrity_check` on the file reported the torn-b-tree signature:

    Tree 5 page 47256 cell 423..429: 2nd reference to page ...
    Tree 5 page 60788 cell 4: Rowid 34637 out of order
    Page 50549..52587: never used

The defect: hermes_state already knows macOS `fsync()` does not guarantee
write ordering, and mitigates it with `synchronous=FULL` +
`checkpoint_fullfsync=1` (see `_enforce_macos_synchronous_full`, whose
docstring names this exact failure: "a WAL checkpoint race with process
termination ... can leave the main DB with half-written btree pages").
Those pragmas are per-connection and were applied only via
`apply_wal_with_fallback()`. The repair path opened `state.db` with a bare
`sqlite3.connect()` five times and then ran REINDEX, VACUUM and
`writable_schema` surgery through it — the operations that rewrite nearly
every page of the file — with no barrier at all.

(The proactive `verify_state_db_integrity()` gate the original PR #90747 also
carried is deferred to the follow-up that wires it into gateway startup —
PR #91754 — since it ships as dead code without that caller. This file covers
only the repair-connection durability half.)
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import hermes_state
from hermes_state import (
    _connect_repair_durable,
    repair_state_db_schema,
)


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("INSERT INTO messages (body) VALUES ('seed')")
    conn.commit()
    conn.close()
    return db


# ── Repair-path write durability ────────────────────────────────────────


def test_connect_repair_durable_sets_macos_barriers(tmp_path: Path) -> None:
    """The repair connection must carry both macOS durability barriers."""
    db = _make_db(tmp_path)
    conn = _connect_repair_durable(db)
    try:
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
        checkpoint_fullfsync = conn.execute(
            "PRAGMA checkpoint_fullfsync"
        ).fetchone()[0]
    finally:
        conn.close()

    if sys.platform == "darwin":
        # SQLite: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA. NORMAL is what tore the
        # b-tree pages; FULL is what _enforce_macos_synchronous_full sets.
        assert synchronous == 2, (
            f"repair connection opened with synchronous={synchronous}; on "
            "Darwin this lets REINDEX/VACUUM leave half-written b-tree pages"
        )
        assert checkpoint_fullfsync == 1, (
            "repair connection has no F_FULLFSYNC barrier at checkpoint "
            "boundaries; macOS fsync() does not flush the drive cache"
        )
    else:
        # Elsewhere the helper is a plain connect — no behaviour change.
        assert synchronous in (0, 1, 2, 3)


def test_connect_repair_durable_is_autocommit(tmp_path: Path) -> None:
    """Must preserve isolation_level=None — repair runs DDL and VACUUM."""
    db = _make_db(tmp_path)
    conn = _connect_repair_durable(db)
    try:
        assert conn.isolation_level is None
        # VACUUM is only legal outside an implicit transaction.
        conn.execute("VACUUM")
    finally:
        conn.close()


def test_repair_path_has_no_bare_connects() -> None:
    """No repair/probe site may bypass the durability helper.

    Source-level guard: the bare form is exactly what regressed, and a unit
    test on the helper alone would not notice a sixth site being added.
    """
    source = Path(hermes_state.__file__).read_text()
    pattern = r"^\s*conn = sqlite3\.connect\(str\(db_path\), isolation_level=None\)"

    # The one legitimate bare connect is inside the helper itself; everything
    # after that definition must go through it.
    helper = source.index("def _connect_repair_durable(")
    body_end = source.index("\ndef ", helper + 1)
    inside_helper = re.findall(pattern, source[helper:body_end], flags=re.MULTILINE)
    assert len(inside_helper) == 1, (
        "_connect_repair_durable no longer opens the connection itself"
    )

    elsewhere = re.findall(
        pattern, source[:helper] + source[body_end:], flags=re.MULTILINE
    )
    assert elsewhere == [], (
        f"{len(elsewhere)} repair-path connection(s) still bypass "
        "_connect_repair_durable() and write state.db without the macOS "
        "fsync barriers"
    )


def test_repair_still_works_through_durable_connection(tmp_path: Path) -> None:
    """Routing every strategy through the helper must not break the path.

    The helper is entered once per strategy, so a plumbing fault (recursion,
    a leaked connection, a refused pragma) surfaces as an exception rather
    than a report. Whether this fixture's minimal schema is *repairable* is
    beside the point — the assertion is that the path runs to completion.
    """
    db = _make_db(tmp_path)
    report = repair_state_db_schema(db, backup=False)
    assert isinstance(report, dict)
    assert set(report) >= {"repaired", "strategy", "backup_path"}
    # The file must still open afterwards — repair may fail, but it must not
    # leave the database less usable than it found it.
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()
