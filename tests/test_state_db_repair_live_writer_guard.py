"""Regression: the state.db repair path must be bounded and must never run
surgery against a database another connection is still writing.

Incident (2026-08-18/19): FTS5 shadow-table corruption escalated into b-tree
page damage across `system_prompts`, `session_model_usage` and the `sessions`
index. Two defects in this module turned a contained, rebuildable FTS fault
into unrecoverable data loss (292 `delivery_obligations` rows):

1. `_db_fingerprint` keyed the persistent attempt ledger on ``size:mtime_ns``,
   documented as "stable for a file nothing can successfully write to". That
   premise is false: on FTS corruption hermes_state deliberately keeps
   "canonical writes enabled with FTS detached", so the gateway kept writing
   and mtime churned. Every repair pass re-keyed the ledger and reset the
   counter to 1 — three real passes (00:14, 00:35, 00:45) all recorded
   ``failed_attempts: 1``, so `_MAX_PERSISTENT_REPAIR_ATTEMPTS` could never
   be reached and the damaging surgery could retry forever.

2. `repair_state_db_schema` ran its REINDEX/FTS-rebuild strategies while other
   connections still held the database open. The caller closes only its own
   `self._conn`; the incident process held seven descriptors on state.db.
   Rewriting b-tree pages under concurrent writers is what spread the damage
   out of the FTS shadow tables and into the canonical tables.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from hermes_state import (
    SessionDB,
    _MAX_PERSISTENT_REPAIR_ATTEMPTS,
    _db_fingerprint,
    _persistent_repair_attempts_exhausted,
    _record_repair_outcome,
    repair_state_db_schema,
)


def _make_wal_db(tmp_path: Path) -> Path:
    """A state.db the repair path will actually work on.

    Built through the real ``SessionDB`` rather than a hand-rolled two-table
    schema. The repair path probes the canonical schema as it goes —
    ``_db_opens_cleanly`` runs ``SELECT COUNT(*) FROM sessions`` and a
    rolled-back ``messages`` write — so a toy schema aborted every repair
    ("no such table: sessions", then "table sessions has no column named id")
    long before reaching the guards these tests exist to cover. The
    assertions below were passing over a code path that never ran.
    """
    db = tmp_path / "state.db"
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    handle.append_message(sid, role="user", content="seed")
    handle.close()
    return db


def _write_once(db: Path) -> None:
    """Simulate the gateway's ongoing canonical writes (FTS detached)."""
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    handle.append_message(sid, role="user", content="canonical write")
    handle.close()


# ---------------------------------------------------------------------------
# Defect 1: the ledger fingerprint must survive ongoing writes
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_while_the_gateway_keeps_writing(tmp_path):
    """Identity must track the FILE, not its mtime/contents.

    A corrupt state.db still accepts canonical writes, so a mtime- or
    content-derived fingerprint changes constantly and silently re-keys the
    attempt ledger.
    """
    db = _make_wal_db(tmp_path)
    before = _db_fingerprint(db)

    time.sleep(0.01)
    _write_once(db)

    assert _db_fingerprint(db) == before


def test_repair_budget_is_exhausted_despite_ongoing_writes(tmp_path):
    """Three failed passes must exhaust the budget even with writes between.

    This is the exact incident shape: three real repair attempts, each
    separated by gateway writes, all recorded ``failed_attempts: 1``.
    """
    db = _make_wal_db(tmp_path)

    for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        _record_repair_outcome(db, repaired=False)
        time.sleep(0.01)
        _write_once(db)

    assert _persistent_repair_attempts_exhausted(db) is True


def test_successful_repair_still_clears_the_budget(tmp_path):
    """A healed database must not inherit a spent budget."""
    db = _make_wal_db(tmp_path)

    for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        _record_repair_outcome(db, repaired=False)
    assert _persistent_repair_attempts_exhausted(db) is True

    _record_repair_outcome(db, repaired=True)

    assert _persistent_repair_attempts_exhausted(db) is False


# ---------------------------------------------------------------------------
# Defect 2: repair must refuse to operate under a live writer
# ---------------------------------------------------------------------------


def test_repair_refuses_while_another_connection_holds_the_db(tmp_path):
    """Surgery under concurrent writers is what spread the corruption."""
    db = _make_wal_db(tmp_path)

    holder = sqlite3.connect(str(db))
    holder.execute("SELECT count(*) FROM messages").fetchone()
    try:
        report = repair_state_db_schema(db, backup=False)
    finally:
        holder.close()

    assert report["repaired"] is False
    assert "live writer" in (report["error"] or "").lower()


def test_repair_proceeds_once_the_database_is_quiescent(tmp_path):
    """The guard must not deadlock repair on an exclusively-held file."""
    db = _make_wal_db(tmp_path)

    report = repair_state_db_schema(db, backup=False)

    assert "live writer" not in (report["error"] or "").lower()
