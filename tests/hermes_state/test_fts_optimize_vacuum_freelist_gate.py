"""The FTS storage optimize must not VACUUM a dense database for nothing.

``optimize_fts_storage`` finishes its chunked, duty-cycled rebuild/teardown and
then calls ``VACUUM`` unconditionally. ``VACUUM`` is a single exclusive
transaction: unlike the rebuild phases it cannot be chunked, yields no lock
between pages, and blocks every other writer for its whole duration. On a dense
multi-GB ``state.db`` that is a measured ~44s stall which reclaims almost
nothing -- the freelist is a rounding error, so the rewrite copies the entire
file to hand back a few MB.

The main auto-maintenance path already refuses that trade via
``AUTO_VACUUM_MIN_FREELIST_RATIO`` (#54189). These tests pin the same gate on
the FTS path, and pin that skipping stays honest: a skipped VACUUM reports
``vacuumed is False``, never ``True``.

Databases are real SQLite files under tmp_path.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO


def _make_state_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    for i in range(40):
        handle.append_message(sid, role="user", content=f"needle-{i} " + "payload " * 40)
    handle.close()
    return db


def _freelist_ratio(db: Path) -> float:
    conn = sqlite3.connect(str(db))
    try:
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        free = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    finally:
        conn.close()
    return free / pages if pages else 0.0


def test_dense_db_skips_the_vacuum(tmp_path):
    """A freshly written DB has a near-empty freelist: no VACUUM, no 44s stall."""
    db = _make_state_db(tmp_path)
    assert _freelist_ratio(db) < AUTO_VACUUM_MIN_FREELIST_RATIO

    handle = SessionDB(db_path=db)
    vacuums = []
    original = handle._optimize_vacuum

    def _spy():
        vacuums.append(True)
        return original()

    handle._optimize_vacuum = _spy
    try:
        result = handle.optimize_fts_storage()
    finally:
        handle.close()

    assert result["ok"] is True
    assert vacuums == [], "VACUUM ran on a database with nothing to reclaim"


def test_skipped_vacuum_is_reported_honestly(tmp_path):
    """Skipping must not be dressed up as success: ``vacuumed`` is False, not True.

    The maintenance conformance suite treats ``vacuumed is False`` as 'no work
    done'. A skip that reported True would claim a reclaim that never happened.
    """
    db = _make_state_db(tmp_path)
    handle = SessionDB(db_path=db)
    try:
        result = handle.optimize_fts_storage()
    finally:
        handle.close()

    assert result["ok"] is True
    assert result["vacuumed"] is False


def test_sparse_db_still_vacuums(tmp_path):
    """The gate must not disable VACUUM outright -- real reclaim still runs.

    Deleting most of a populated DB pushes the freelist above the ratio, which
    is exactly the case where a full rewrite pays for itself.
    """
    db = tmp_path / "state.db"
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    for i in range(400):
        handle.append_message(sid, role="user", content=f"needle-{i} " + "payload " * 200)
    handle.close()

    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DELETE FROM messages WHERE id % 10 != 0")
        conn.commit()
    finally:
        conn.close()

    ratio = _freelist_ratio(db)
    if ratio <= AUTO_VACUUM_MIN_FREELIST_RATIO:
        pytest.skip(f"could not build a sparse enough DB (freelist ratio {ratio:.3f})")

    handle = SessionDB(db_path=db)
    vacuums = []
    original = handle._optimize_vacuum

    def _spy():
        vacuums.append(True)
        return original()

    handle._optimize_vacuum = _spy
    try:
        result = handle.optimize_fts_storage()
    finally:
        handle.close()

    assert result["ok"] is True
    assert vacuums == [True], "VACUUM was skipped on a database with real reclaimable space"


def test_explicit_vacuum_false_still_wins(tmp_path):
    """``vacuum=False`` callers keep their existing contract (``vacuumed is None``)."""
    db = _make_state_db(tmp_path)
    handle = SessionDB(db_path=db)
    try:
        result = handle.optimize_fts_storage(vacuum=False)
    finally:
        handle.close()

    assert result["ok"] is True
    assert result["vacuumed"] is None
