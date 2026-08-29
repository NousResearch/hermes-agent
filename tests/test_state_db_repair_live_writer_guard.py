"""Regression: the state.db repair path must never run surgery against a
database another connection is still writing.

Incident (2026-08-18/19): FTS5 shadow-table corruption escalated into b-tree
page damage across `system_prompts`, `session_model_usage` and the `sessions`
index. `repair_state_db_schema` ran its REINDEX/FTS-rebuild strategies while
other connections still held the database open. The caller closes only its own
`self._conn`; the incident process held seven descriptors on state.db.
Rewriting b-tree pages under concurrent writers is what spread the damage out
of the FTS shadow tables and into the canonical tables.

(The companion repair-attempt-ledger fingerprint fix — keying the budget on
something stable across ongoing writes so the cap can actually be reached — is
tracked separately in the fingerprint/repair-loop salvage PR #88425, which
preserves @jirathip-k's #88224 diagnosis and credit. This file covers only the
live-writer guard.)
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from hermes_state import (
    SessionDB,
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


# ---------------------------------------------------------------------------
# Repair must refuse to operate under a live writer
# ---------------------------------------------------------------------------


@pytest.mark.requires_wal
def test_repair_refuses_while_another_connection_holds_the_db(tmp_path):
    """Surgery under concurrent writers is what spread the corruption.

    Gated on ``requires_wal``: ``_live_writer_holds_db`` detects an
    out-of-process holder via ``PRAGMA locking_mode=EXCLUSIVE`` + a
    ``BEGIN IMMEDIATE`` that a concurrent connection makes fail with
    SQLITE_BUSY through the WAL index. On SQLite builds carrying the
    WAL-reset bug (and on NFS/SMB) Hermes deliberately runs ``state.db`` in
    ``journal_mode=DELETE``, where a held reader takes only a SHARED lock and
    ``BEGIN IMMEDIATE`` can still acquire RESERVED — so the probe cannot see
    the holder and the guard fails open. In DELETE mode repair is instead
    serialised only by the cross-process repairer lock (see
    ``_live_writer_holds_db``'s docstring). The conftest auto-skips this test
    where WAL is unusable rather than assert a guarantee the runtime doesn't
    make there.
    """
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
