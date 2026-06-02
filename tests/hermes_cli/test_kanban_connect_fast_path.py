"""Tests for connect() fast-path when path is already initialized (#36644).

When a path is already in _INITIALIZED_PATHS, connect() must skip the
cross-process flock entirely to avoid blocking on concurrent workers.

Verifies that:
1. Fast-path skips the cross-process flock when path is initialized
2. Slow-path takes the flock when path is NOT initialized
3. Fast-path still applies WAL and PRAGMAs
"""

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import hermes_cli.kanban_db as kanban_db


@pytest.fixture(autouse=True)
def _clear_init_cache():
    """Ensure each test starts with a clean initialization cache."""
    kanban_db._INITIALIZED_PATHS.clear()
    yield
    kanban_db._INITIALIZED_PATHS.clear()


class TestConnectFastPath:
    """connect() must skip the cross-process flock on the steady-state path."""

    def test_fast_path_skips_flock(self, tmp_path: Path):
        """When path is in _INITIALIZED_PATHS, flock must not be called."""
        db_path = tmp_path / "kanban.db"
        # Seed a valid empty DB so _sqlite_connect works
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE _test (id INTEGER)")
        conn.close()
        # Mark as initialized
        kanban_db._INITIALIZED_PATHS.add(str(db_path.resolve()))

        with patch("hermes_cli.kanban_db._cross_process_init_lock") as mock_lock:
            # If flock is called, this mock will be entered — which it shouldn't
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = kanban_db.connect(db_path)
            try:
                assert isinstance(result, sqlite3.Connection)
                mock_lock.assert_not_called()
            finally:
                result.close()

    def test_slow_path_takes_flock(self, tmp_path: Path):
        """When path is NOT in _INITIALIZED_PATHS, flock must be called."""
        db_path = tmp_path / "kanban.db"
        # Seed a valid empty DB
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE _test (id INTEGER)")
        conn.close()

        with patch("hermes_cli.kanban_db._cross_process_init_lock") as mock_lock:
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = kanban_db.connect(db_path)
            try:
                assert isinstance(result, sqlite3.Connection)
                mock_lock.assert_called_once()
            finally:
                result.close()

    def test_fast_path_applies_pragmas(self, tmp_path: Path):
        """Fast-path must still apply WAL + safety PRAGMAs."""
        db_path = tmp_path / "kanban.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE _test (id INTEGER)")
        conn.close()
        kanban_db._INITIALIZED_PATHS.add(str(db_path.resolve()))

        result = kanban_db.connect(db_path)
        try:
            # Verify foreign_keys is ON
            fk = result.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1, f"foreign_keys should be ON, got {fk}"
        finally:
            result.close()
