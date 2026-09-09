"""Regression tests for #106271: cooldown rollback must tolerate a vanished session row.

Issue: a bot-mode turn died with a turn-dispatcher exception during the compression
summary phase. ``restore_compression_failure_cooldown_row`` snapshots the cooldown row,
the summary attempt fails, and the rollback runs ``UPDATE sessions ... WHERE id = ?`` —
raising ``RuntimeError`` when ``rowcount != 1``. When the session row is retired/expired
mid-attempt (e.g. by the maintenance sweep) that routine race turned into a dispatcher
crash, even though the missing row means there is no cooldown left to restore.

Fix: treat ``rowcount == 0`` as a tolerated no-op (warn + return, skipping the read-back
verification), mirroring the existing early-return for snapshots taken while the session
already did not exist. A session row retired between the compensating UPDATE and the
read-back verification is the same lifecycle outcome and is tolerated the same way,
while a mismatch on a surviving row still raises.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from hermes_state import SessionDB


def _retire_session_row(db: SessionDB, session_id: str) -> None:
    """Delete the sessions row the way the maintenance sweep does mid-attempt."""
    def _delete(conn):
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    db._execute_write(_delete)


def test_rollback_tolerates_session_row_retired_mid_attempt(tmp_path: Path, caplog) -> None:
    db = SessionDB(tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_SESSION_GONE"
    db.create_session(session_id, source="test")
    snapshot = {
        "session_exists": True,
        "cooldown_until": time.time() + 120.0,
        "error": "attempt-failed",
    }
    db.restore_compression_failure_cooldown_row(session_id, snapshot)
    assert db.get_compression_failure_cooldown_row(session_id)["session_exists"] is True

    _retire_session_row(db, session_id)

    # Before the fix this raised RuntimeError("compression cooldown rollback session
    # missing: ...") and crashed the turn dispatcher.
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        db.restore_compression_failure_cooldown_row(session_id, snapshot)
    assert any("session missing" in rec.message for rec in caplog.records)
    assert db.get_compression_failure_cooldown_row(session_id)["session_exists"] is False


def test_rollback_still_restores_exact_row_when_session_exists(tmp_path: Path) -> None:
    """The tolerated-missing path must not loosen the exact restore semantics."""
    db = SessionDB(tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_SESSION_PRESENT"
    db.create_session(session_id, source="test")

    db.restore_compression_failure_cooldown_row(
        session_id,
        {"session_exists": True, "cooldown_until": 1234.5, "error": "must-restore"},
    )
    row = db.get_compression_failure_cooldown_row(session_id)
    assert row["session_exists"] is True
    assert row["cooldown_until"] == 1234.5
    assert row["error"] == "must-restore"


def test_rollback_tolerates_session_row_deleted_between_update_and_readback(
        tmp_path: Path, caplog, monkeypatch) -> None:
    """The session may also be retired after the compensating UPDATE commits but before
    the read-back verification: the same lifecycle race as the pre-update window, so it
    must not crash the turn dispatcher either."""
    db = SessionDB(tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_DELETED_AFTER_UPDATE"
    db.create_session(session_id, source="test")
    original_execute_write = db._execute_write

    def _delete_session(conn):
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def _execute_write_then_retire(fn):
        ok = original_execute_write(fn)
        # The maintenance sweep retires the session between the compensating UPDATE
        # and the read-back verification (bypassing the patched entry point itself).
        original_execute_write(_delete_session)
        return ok

    monkeypatch.setattr(db, "_execute_write", _execute_write_then_retire)
    snapshot = {
        "session_exists": True,
        "cooldown_until": time.time() + 120.0,
        "error": "attempt-failed",
    }
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        db.restore_compression_failure_cooldown_row(session_id, snapshot)
    # Before the fix this raised RuntimeError("compression cooldown rollback
    # verification failed: ...") because the read-back saw the session row gone.
    assert any("session missing" in rec.message for rec in caplog.records)
    assert db.get_compression_failure_cooldown_row(session_id)["session_exists"] is False


def test_rollback_still_raises_on_mismatch_for_surviving_session(tmp_path: Path, monkeypatch) -> None:
    """The post-update tolerated-missing path must not mask a real verification failure
    on a session row that still exists."""
    db = SessionDB(tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_SURVIVING_MISMATCH"
    db.create_session(session_id, source="test")

    monkeypatch.setattr(
        db, "get_compression_failure_cooldown_row",
        lambda _session_id: {"session_exists": True, "cooldown_until": 9999.0, "error": "other-writer"})
    try:
        db.restore_compression_failure_cooldown_row(
            session_id,
            {"session_exists": True, "cooldown_until": 1234.5, "error": "attempt-failed"},
        )
    except RuntimeError as exc:
        assert "compression cooldown rollback verification failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a surviving-row mismatch")
