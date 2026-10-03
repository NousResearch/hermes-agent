"""rewind_to_message must not bind one placeholder per rewound row (#126759).

The old UPDATE ... WHERE id IN (<one placeholder per id>) exceeded SQLite's
SQLITE_MAX_VARIABLE_NUMBER ceiling on long transcripts, so /undo failed with
OperationalError before the rewind could persist. The fix deactivates via the
same range predicate the SELECT already uses.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    state = SessionDB(db_path=tmp_path / "state.db")
    yield state
    state.close()


def test_rewind_survives_transcript_longer_than_variable_ceiling(db):
    sid = "rewind-ceiling"
    db.create_session(sid, source="tui")
    # Rewind target is the FIRST user row: everything after it (33,001 rows)
    # gets deactivated, so the old per-id IN (...) bind would have needed
    # > SQLITE_MAX_VARIABLE_NUMBER (32766 on SQLite >= 3.32) placeholders and
    # raised "too many SQLITE_MAX_VARIABLE_NUMBER".
    target_id = db.append_message(sid, "user", "rewind to me")
    bulk = 33_000
    db._conn.execute("BEGIN")  # autocommit connection: one explicit txn, not 33k fsyncs
    try:
        db._conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [(sid, "assistant", f"m{i}", 1_700_000_000.0 + i) for i in range(bulk)],
        )
    finally:
        db._conn.execute("COMMIT")
    tail_id = db.append_message(sid, "assistant", "tail")

    result = db.rewind_to_message(sid, target_id)

    assert result["rewound_count"] == bulk + 2
    active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1",
        (sid,),
    ).fetchone()[0]
    assert active == 0
    assert result["new_head_id"] is None
