"""Regression tests for #106271: compression cooldown rollback must tolerate a
missing session row (retired/expired mid-attempt) instead of crashing the turn
dispatcher with ``RuntimeError("compression cooldown rollback session missing: ...")``.

The top of ``restore_compression_failure_cooldown_row`` already early-returns when
the cooldown row is absent; the inner ``_do`` must extend the same tolerance to the
case where the session row disappears between the snapshot capture and the
compensating UPDATE (``rowcount == 0``). Genuine write failures and verification
mismatches must still propagate so cancellation cannot masquerade as a
mutation-free restore.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _record_cooldown(
    db: SessionDB, session_id: str, error: str = "summary-failed"
) -> float:
    deadline = time.time() + 120.0
    db.record_compression_failure_cooldown(session_id, deadline, error)
    return deadline


def test_cooldown_rollback_no_op_when_session_row_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Session retired mid-attempt: the UPDATE matches 0 rows (rowcount == 0).
    ``restore_compression_failure_cooldown_row`` must no-op + warn instead of raising
    RuntimeError (which would crash the turn dispatcher)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "ROLLED_BACK_MISSING_SESSION"
    db.create_session(session_id, source="test")
    deadline = _record_cooldown(db, session_id)
    snapshot = db.get_compression_failure_cooldown_row(session_id)
    assert snapshot["session_exists"] is True

    # Simulate the session row being retired/expired mid-attempt.
    db._execute_write(
        lambda conn: conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    )
    assert (
        db.get_compression_failure_cooldown_row(session_id)["session_exists"] is False
    )

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        # Must NOT raise - the session is gone, there is no cooldown left to restore.
        db.restore_compression_failure_cooldown_row(
            session_id,
            {
                "session_exists": True,
                "cooldown_until": deadline,
                "error": "summary-failed",
            },
        )
    # The no-op path emits a warning naming the missing session.
    assert any(session_id in record.message for record in caplog.records), [
        record.message for record in caplog.records
    ]


def test_cooldown_rollback_still_raises_on_verification_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the session row IS updated (rowcount == 1) but the read-back disagrees
    with the snapshot, the verification RuntimeError must still propagate - the
    no-op tolerance is only for a missing session, not a wrong-value restore."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "VERIFICATION_MISMATCH"
    db.create_session(session_id, source="test")
    deadline = _record_cooldown(db, session_id, error="first-error")

    # Corrupt the read-back so the restored value disagrees with the snapshot.
    real_get = db.get_compression_failure_cooldown_row

    def _lying_get(session_id_arg: str) -> dict:
        row = real_get(session_id_arg)
        return {
            "session_exists": row["session_exists"],
            "cooldown_until": row["cooldown_until"],
            "error": "DIFFERENT",
        }

    monkeypatch.setattr(db, "get_compression_failure_cooldown_row", _lying_get)

    with pytest.raises(RuntimeError, match="verification failed"):
        db.restore_compression_failure_cooldown_row(
            session_id,
            {
                "session_exists": True,
                "cooldown_until": deadline,
                "error": "first-error",
            },
        )


def test_cooldown_rollback_still_propagates_sqlite_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Genuine sqlite write failures (locked/corrupt/IOERR) must still propagate -
    the no-op tolerance is only for rowcount == 0, not for a failed write."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "WRITE_FAILURE_STILL_RAISES"
    db.create_session(session_id, source="test")

    def _write_fails(_callback) -> None:
        raise sqlite3.OperationalError("forced write failure")

    monkeypatch.setattr(db, "_execute_write", _write_fails)

    with pytest.raises(sqlite3.OperationalError, match="forced write failure"):
        db.restore_compression_failure_cooldown_row(
            session_id,
            {
                "session_exists": True,
                "cooldown_until": time.time() + 10.0,
                "error": "x",
            },
        )


def test_cooldown_rollback_restores_value_when_session_present(tmp_path: Path) -> None:
    """Happy path: session still exists (rowcount == 1), the cooldown is restored
    exactly and verification passes - unchanged from pre-fix behaviour."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HAPPY_PATH_RESTORE"
    db.create_session(session_id, source="test")
    deadline = _record_cooldown(db, session_id, error="to-restore")

    # Clear the cooldown (NULLs the columns; the session row stays) then restore the
    # exact snapshot - the UPDATE matches 1 row (rowcount == 1), verification passes.
    db.clear_compression_failure_cooldown(session_id)
    cleared = db.get_compression_failure_cooldown_row(session_id)
    assert cleared["session_exists"] is True
    assert cleared["cooldown_until"] is None

    db.restore_compression_failure_cooldown_row(
        session_id,
        {"session_exists": True, "cooldown_until": deadline, "error": "to-restore"},
    )
    row = db.get_compression_failure_cooldown_row(session_id)
    assert row["session_exists"] is True
    assert row["cooldown_until"] == deadline
    assert row["error"] == "to-restore"
