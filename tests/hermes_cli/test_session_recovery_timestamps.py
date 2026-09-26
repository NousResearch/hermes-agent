"""Invariant: session recovery never ships an out-of-window timestamp cell (#91536).

A damaged page can decode as a valid-looking garbage double (``5.49e+246``, ``1e-310`` magnitudes
are uninitialised-memory patterns) that the copy writes verbatim. Both recovery lanes pass through
``_finalize_derived_metadata``, so the repair lives there.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_cli.session_recovery import recover_session_database
from hermes_cli.timefmt import coerce_epoch
from hermes_state import SessionDB


def _damaged_source(path: Path) -> list[float]:
    db = SessionDB(db_path=path)
    try:
        db.create_session("s", "cli")
        for text in ("one", "two", "three"):
            db.append_message("s", "user", text)
        stamps = [row["timestamp"] for row in db.get_messages("s")]
    finally:
        db.close()
    raw = sqlite3.connect(str(path))
    try:
        middle = raw.execute("SELECT id FROM messages WHERE session_id = 's' ORDER BY id LIMIT 1 OFFSET 1").fetchone()[0]
        raw.execute("UPDATE messages SET timestamp = 5.4905047707024164e+246 WHERE id = ?", (middle,))
        raw.execute("UPDATE sessions SET started_at = 'garbage', ended_at = 1e300, last_activity_at = -5 WHERE id = 's'")
        raw.commit()
    finally:
        raw.close()
    return stamps


def test_recovery_repairs_out_of_window_timestamps(tmp_path):
    source, output = tmp_path / "state.db", tmp_path / "recovered.db"
    stamps = _damaged_source(source)

    report = recover_session_database(source, output)

    assert report["verification"]["healthy"], report["verification"]["errors"]
    conn = sqlite3.connect(str(output))
    try:
        messages = [r[0] for r in conn.execute("SELECT timestamp FROM messages WHERE session_id = 's' ORDER BY id")]
        started, ended, active = conn.execute(
            "SELECT started_at, ended_at, last_activity_at FROM sessions WHERE id = 's'").fetchone()
    finally:
        conn.close()
    # Every cell is a trusted epoch again, message order is preserved, and the good cells are untouched.
    assert all(coerce_epoch(ts) is not None for ts in messages)
    assert messages[0] == stamps[0] and messages[2] == stamps[2] and messages == sorted(messages)
    assert started == min(messages) and ended is None and active is None
