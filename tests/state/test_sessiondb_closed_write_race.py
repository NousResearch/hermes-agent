"""SessionDB writes after close() must raise SessionDBClosedError, not AttributeError.

When the gateway tears down (shutdown/restart) while a session-hygiene
compression worker thread is still in flight, the worker can reach
``_execute_write`` after ``close()`` has already nulled ``self._conn``.
Without a guard, the raw ``AttributeError: 'NoneType' object has no
attribute 'execute'`` propagates through an orphaned executor future whose
exception is never retrieved — producing a noisy ``ERROR asyncio: Future
exception was never retrieved`` in the logs.

The cooldown-restore path (``restore_compression_failure_cooldown_row``) and
the best-effort record/clear helpers must treat a closed SessionDB as a
moot write, not a crash.
"""
from pathlib import Path

import pytest

from hermes_state import SessionDB, SessionDBClosedError


def _db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


class TestExecuteWriteAfterClose:
    def test_raises_sessiondb_closed_error_not_attribute_error(self, tmp_path):
        """_execute_write must raise SessionDBClosedError after close()."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        db.close()

        def _do(conn):
            conn.execute("UPDATE sessions SET compression_failure_cooldown_until = 1.0 WHERE id = 's1'")

        with pytest.raises(SessionDBClosedError):
            db._execute_write(_do)

    def test_error_is_runtime_error_subclass(self):
        """SessionDBClosedError must be catchable as RuntimeError."""
        assert issubclass(SessionDBClosedError, RuntimeError)


class TestCooldownHelpersAfterClose:
    def test_restore_cooldown_row_is_moot_after_close(self, tmp_path):
        """restore_compression_failure_cooldown_row must not raise after close."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        db.close()

        # Should return silently — the session is being torn down anyway.
        db.restore_compression_failure_cooldown_row(
            "s1",
            {"session_exists": True, "cooldown_until": 123.0, "error": "boom"},
        )

    def test_record_cooldown_is_moot_after_close(self, tmp_path):
        """record_compression_failure_cooldown must not raise after close."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        db.close()

        # Should log a warning, not raise.
        db.record_compression_failure_cooldown("s1", 999.0, "err")

    def test_clear_cooldown_is_moot_after_close(self, tmp_path):
        """clear_compression_failure_cooldown must not raise after close."""
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        db.close()

        # Should log a warning, not raise.
        db.clear_compression_failure_cooldown("s1")
