"""Tests for shared file permission hardening helpers."""

import logging
import os
import stat
from pathlib import Path

import pytest

from hermes_constants import secure_dir, secure_file


class TestSecureFile:
    def test_sets_0600_on_existing_file(self, tmp_path):
        target = tmp_path / "state.db"
        target.write_text("data")
        os.chmod(target, 0o644)

        secure_file(target)

        assert stat.S_IMODE(os.stat(target).st_mode) == 0o600

    def test_noop_on_nonexistent_file(self):
        secure_file(Path("/nonexistent/path/file.db"))

    @pytest.mark.skipif(os.name != "posix", reason="chmod test")
    def test_idempotent(self, tmp_path):
        target = tmp_path / "state.db"
        target.write_text("data")

        secure_file(target)
        secure_file(target)

        assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


class TestSecureDir:
    def test_sets_0700_on_existing_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        target = tmp_path / "logs"
        target.mkdir()
        os.chmod(target, 0o755)

        secure_dir(target)

        assert stat.S_IMODE(os.stat(target).st_mode) == 0o700

    def test_honors_hermes_home_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME_MODE", "0750")
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        target = tmp_path / "logs"
        target.mkdir()
        os.chmod(target, 0o755)

        secure_dir(target)

        assert stat.S_IMODE(os.stat(target).st_mode) == 0o750

    def test_managed_mode_keeps_group_access(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_MANAGED", "true")
        target = tmp_path / "logs"
        target.mkdir()
        os.chmod(target, 0o755)

        secure_dir(target)

        assert stat.S_IMODE(os.stat(target).st_mode) == 0o2770

    def test_noop_on_nonexistent_dir(self):
        secure_dir(Path("/nonexistent/path"))


class TestStateDBPermissions:
    def test_sessiondb_creates_0600_state_db(self, tmp_path):
        from hermes_state import SessionDB

        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600
        finally:
            db.close()

    def test_sessiondb_parent_dir_not_changed(self, tmp_path):
        from hermes_state import SessionDB

        db_path = tmp_path / "subdir" / "state.db"
        db_path.parent.mkdir()
        parent_mode_before = stat.S_IMODE(os.stat(db_path.parent).st_mode)

        db = SessionDB(db_path=db_path)
        try:
            assert stat.S_IMODE(os.stat(db_path.parent).st_mode) == parent_mode_before
        finally:
            db.close()


class TestLogFilePermissions:
    def test_setup_logging_creates_private_log_dir_and_files(self, tmp_path, monkeypatch):
        import hermes_cli.config as config
        import hermes_logging

        monkeypatch.setattr(config, "is_managed", lambda: False)
        monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        root = logging.getLogger()
        before_handlers = set(root.handlers)
        before_initialized = hermes_logging._logging_initialized

        log_dir = hermes_logging.setup_logging(
            hermes_home=tmp_path,
            mode="gateway",
            force=True,
        )
        try:
            assert stat.S_IMODE(os.stat(log_dir).st_mode) == 0o700
            for name in ("agent.log", "errors.log", "gateway.log"):
                path = log_dir / name
                assert path.exists()
                assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        finally:
            for handler in list(root.handlers):
                if handler not in before_handlers:
                    root.removeHandler(handler)
                    handler.close()
            hermes_logging._logging_initialized = before_initialized

    def test_setup_logging_keeps_managed_group_access(self, tmp_path, monkeypatch):
        import hermes_cli.config as config
        import hermes_logging

        monkeypatch.setattr(config, "is_managed", lambda: True)
        monkeypatch.setenv("HERMES_MANAGED", "true")
        root = logging.getLogger()
        before_handlers = set(root.handlers)
        before_initialized = hermes_logging._logging_initialized

        log_dir = hermes_logging.setup_logging(
            hermes_home=tmp_path,
            mode="gateway",
            force=True,
        )
        try:
            assert stat.S_IMODE(os.stat(log_dir).st_mode) == 0o2770
            for name in ("agent.log", "errors.log", "gateway.log"):
                path = log_dir / name
                assert path.exists()
                assert stat.S_IMODE(os.stat(path).st_mode) == 0o660
        finally:
            for handler in list(root.handlers):
                if handler not in before_handlers:
                    root.removeHandler(handler)
                    handler.close()
            hermes_logging._logging_initialized = before_initialized
