"""Tests for database: config PRAGMA application on state.db (#57820 / #21807)."""

import sqlite3
import uuid
from unittest.mock import patch

import pytest

from hermes_state import (
    DatabasePragmaConfig,
    SessionDB,
    apply_database_pragmas,
    load_database_pragma_config,
)


def _db_cfg(**kwargs) -> DatabasePragmaConfig:
    defaults = {
        "journal_mode": None,
        "wal_autocheckpoint": None,
        "journal_size_limit": None,
    }
    defaults.update(kwargs)
    return DatabasePragmaConfig(**defaults)


class TestLoadDatabasePragmaConfig:
    def test_defaults_when_section_missing(self, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", "/nonexistent/hermes_test_empty")
        with patch("hermes_cli.config.load_config", return_value={}):
            cfg = load_database_pragma_config()
        assert cfg.journal_mode is None
        assert cfg.wal_autocheckpoint is None
        assert cfg.journal_size_limit is None

    def test_normalizes_journal_mode(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"database": {"journal_mode": "DELETE"}},
        ):
            cfg = load_database_pragma_config()
        assert cfg.journal_mode == "delete"

    def test_invalid_journal_mode_raises(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"database": {"journal_mode": "memory"}},
        ):
            with pytest.raises(ValueError, match="journal_mode"):
                load_database_pragma_config()


class TestSessionDBJournalConfig:
    def test_default_opens_wal_on_posix(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows forces DELETE journal mode")
        db_path = tmp_path / "state.db"
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(),
        ):
            db = SessionDB(db_path=db_path)
        try:
            mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            db.close()

    def test_config_delete_opens_delete_mode(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows always uses DELETE")
        db_path = tmp_path / "state.db"
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(journal_mode="delete"),
        ):
            db = SessionDB(db_path=db_path)
        try:
            mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "delete"
            wal, shm = db_path.with_name(db_path.name + "-wal"), db_path.with_name(
                db_path.name + "-shm"
            )
            assert not wal.exists()
            assert not shm.exists()
        finally:
            db.close()

    def test_config_wal_explicit(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows forces DELETE")
        db_path = tmp_path / "state.db"
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(journal_mode="wal"),
        ):
            db = SessionDB(db_path=db_path)
        try:
            mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            db.close()

    def test_wal_autocheckpoint_from_config(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows forces DELETE")
        db_path = tmp_path / "state.db"
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(wal_autocheckpoint=200),
        ):
            db = SessionDB(db_path=db_path)
        try:
            val = db._conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            assert val == 200
        finally:
            db.close()

    def test_journal_size_limit_from_config(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows forces DELETE")
        db_path = tmp_path / "state.db"
        limit = 10_485_760
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(journal_size_limit=limit),
        ):
            db = SessionDB(db_path=db_path)
        try:
            val = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
            assert val == limit
        finally:
            db.close()

    def test_pragmas_skipped_in_delete_mode(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows always DELETE")
        db_path = tmp_path / "state.db"
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(
                journal_mode="delete",
                wal_autocheckpoint=1,
                journal_size_limit=999,
            ),
        ):
            db = SessionDB(db_path=db_path)
        try:
            # SQLite default wal_autocheckpoint is 1000 when unset by us
            val = db._conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            assert val == 1000
            lim = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
            assert lim == -1
        finally:
            db.close()

    def test_wal_to_delete_migration_preserves_data(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows always DELETE")
        db_path = tmp_path / "state.db"
        sid = str(uuid.uuid4())
        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(),
        ):
            db = SessionDB(db_path=db_path)
            db.create_session(sid, source="cli")
            db.append_message(sid, role="user", content="migration test payload")
            db.close()

        assert (tmp_path / "state.db-wal").exists() or db_path.stat().st_size > 4096

        with patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(journal_mode="delete"),
        ):
            db2 = SessionDB(db_path=db_path)
        try:
            mode = db2._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "delete"
            count = db2._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE content LIKE ?",
                ("migration test%",),
            ).fetchone()[0]
            assert count == 1
            assert not (tmp_path / "state.db-wal").exists()
        finally:
            db2.close()

    def test_windows_forces_delete_even_when_config_wal(self, tmp_path):
        with patch("sys.platform", "win32"):
            db_path = tmp_path / "state.db"
            with patch(
                "hermes_state.load_database_pragma_config",
                return_value=_db_cfg(journal_mode="wal"),
            ):
                db = SessionDB(db_path=db_path)
            try:
                mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
                assert mode.lower() == "delete"
            finally:
                db.close()

    def test_macos_checkpoint_fullfsync_when_wal(self, tmp_path):
        if __import__("sys").platform == "win32":
            pytest.skip("Windows forces DELETE")
        db_path = tmp_path / "state.db"
        with patch("sys.platform", "darwin"), patch(
            "hermes_state.load_database_pragma_config",
            return_value=_db_cfg(),
        ):
            db = SessionDB(db_path=db_path)
        try:
            val = db._conn.execute("PRAGMA checkpoint_fullfsync").fetchone()[0]
            assert val == 1
        finally:
            db.close()


class TestApplyDatabasePragmasUnit:
    def test_applies_wal_pragmas(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "u.db"))
        cfg = _db_cfg(wal_autocheckpoint=42, journal_size_limit=8192)
        apply_database_pragmas(conn, cfg, journal_mode="wal")
        assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 42
        assert conn.execute("PRAGMA journal_size_limit").fetchone()[0] == 8192
        conn.close()

    def test_skips_wal_autocheckpoint_in_delete_mode(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "d.db"))
        conn.execute("PRAGMA journal_mode=DELETE")
        cfg = _db_cfg(wal_autocheckpoint=1)
        apply_database_pragmas(conn, cfg, journal_mode="delete")
        assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1000
        conn.close()
