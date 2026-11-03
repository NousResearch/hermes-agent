"""Segment-dependent FTS5 trigram corruption (#100227).

A stale ``messages_fts_trigram_idx`` row sitting at the segid FTS5 allocates next makes every
committed append fail with ``IntegrityError: constraint failed`` while ``PRAGMA integrity_check``,
the FTS5 ``integrity-check`` command and ``MATCH`` all report healthy. The write probe must
report it (and the repair must heal it) without any mocked detector.
"""
import sqlite3
import time
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_repair import _db_opens_cleanly, repair_state_db_schema


def _build_db_with_trigram(db_path: Path) -> str:
    db = SessionDB(db_path=db_path)
    if not db._trigram_available:
        db.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    for i in range(60):
        db.append_message(sid, role="user", content=f"quick brown fox {i} lorem ipsum dolor {i * 7}")
    db.close()
    return sid


def _plant_stale_trigram_segment(db_path: Path) -> None:
    """Leave an index row at the next free segid: the shape an aborted segment write leaves behind."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    used = {r[0] for r in conn.execute("SELECT segid FROM messages_fts_trigram_idx")}
    stale = next(s for s in range(1, 1 << 20) if s not in used)
    conn.execute("INSERT INTO messages_fts_trigram_idx(segid, term, pgno) VALUES (?, X'', 2)", (stale,))
    conn.close()


def _real_append_fails(db_path: Path, sid: str) -> bool:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                     (sid, "user", "zebra yak xylophone wombat", time.time()))
        return False
    except sqlite3.IntegrityError:
        return True
    finally:
        conn.close()


def test_write_probe_reports_segment_collision_that_integrity_check_misses(tmp_path):
    db_path = tmp_path / "state.db"
    sid = _build_db_with_trigram(db_path)
    _plant_stale_trigram_segment(db_path)

    assert sqlite3.connect(str(db_path)).execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert _real_append_fails(db_path, sid), "fixture must break real appends"

    reason = _db_opens_cleanly(db_path)
    assert reason is not None and "constraint failed" in reason
    # The probe rolls back: it must not have added rows or moved the FTS state.
    assert sqlite3.connect(str(db_path)).execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_repair_heals_segment_collision_and_restores_appends(tmp_path):
    db_path = tmp_path / "state.db"
    sid = _build_db_with_trigram(db_path)
    _plant_stale_trigram_segment(db_path)

    report = repair_state_db_schema(db_path, backup=False)
    assert report.get("repaired"), report
    assert _db_opens_cleanly(db_path) is None
    assert not _real_append_fails(db_path, sid)
    with SessionDB(db_path=db_path) as db:
        assert db._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)).fetchone()[0] == 61
