"""Bounded, read-only admission history for reconnecting session observers."""
from hermes_state_runtime import _row


def admission_cursor(db, session_id: str) -> int:
    with db._read_ctx() as conn:
        row = conn.execute('SELECT MAX(seq) FROM session_admissions WHERE target_session_id=?',
                           (session_id,)).fetchone()
        return row[0] or 0


def admission_history(db, session_id: str, after_sequence: int, limit: int) -> list[dict]:
    with db._read_ctx() as conn:
        return [_row(row) for row in conn.execute(
            'SELECT * FROM session_admissions WHERE target_session_id=? AND seq>? ORDER BY seq LIMIT ?',
            (session_id, after_sequence, limit))]
