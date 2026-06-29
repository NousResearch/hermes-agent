"""Tests for the interprocess config write lock and related helpers."""

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.config import (
    config_read_lock,
    config_write_lock,
    get_config_lock_path,
    get_config_path,
    get_config_seal_path,
    restore_config,
    save_config,
    seal_config,
    verify_config_integrity,
)


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


class TestConfigLockPaths:
    def test_lock_path_adjacent_to_config(self, isolated_hermes_home):
        assert get_config_lock_path() == isolated_hermes_home / "config.yaml.lock"

    def test_lock_path_same_dir_as_config(self, isolated_hermes_home):
        assert get_config_lock_path().parent == get_config_path().parent


class TestConfigWriteLock:
    def test_exclusive_lock_context_manager(self, isolated_hermes_home):
        """Lock is acquired and released without error."""
        with config_write_lock():
            assert get_config_lock_path().exists()

    def test_lock_file_created(self, isolated_hermes_home):
        with config_write_lock():
            pass
        assert get_config_lock_path().exists()


class TestConfigReadLock:
    def test_read_lock_context_manager(self, isolated_hermes_home):
        """Shared lock acquires and releases without error."""
        with config_read_lock():
            pass

    def test_multiple_read_locks_coexist(self, isolated_hermes_home):
        """Two threads can hold read locks simultaneously."""
        results = []

        def reader():
            with config_read_lock():
                time.sleep(0.05)
                results.append("read")

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)
        assert results.count("read") == 2


class TestVerifyConfigIntegrity:
    def test_no_config_returns_ok(self, isolated_hermes_home):
        ok, cur, sealed = verify_config_integrity()
        assert ok is True
        assert cur == ""
        assert sealed == ""

    def test_no_seal_returns_ok(self, isolated_hermes_home):
        config = {"model": {"default": "test-model"}}
        save_config(config)
        get_config_seal_path().unlink()
        ok, cur, sealed = verify_config_integrity()
        assert ok is True
        assert sealed == ""

    def test_matching_seal_returns_ok(self, isolated_hermes_home):
        save_config({"model": {"default": "test-model"}})
        ok, cur, sealed = verify_config_integrity()
        assert ok is True
        assert cur == sealed

    def test_tampered_config_detected(self, isolated_hermes_home):
        save_config({"model": {"default": "test-model"}})
        # Tamper directly, bypassing save_config
        get_config_path().write_text("model:\n  default: evil-model\n")
        ok, cur, sealed = verify_config_integrity()
        assert ok is False
        assert cur != sealed

    def test_locked_verify_works(self, isolated_hermes_home):
        save_config({"model": {"default": "test-model"}})
        ok, cur, sealed = verify_config_integrity(locked=True)
        assert ok is True

    def test_save_config_updates_seal(self, isolated_hermes_home):
        save_config({"model": {"default": "first"}})
        ok1, d1, _ = verify_config_integrity()
        save_config({"model": {"default": "second"}})
        ok2, d2, _ = verify_config_integrity()
        assert ok1 is True
        assert ok2 is True
        assert d1 != d2


class TestRestoreConfig:
    def test_restore_from_dict(self, isolated_hermes_home):
        save_config({"model": {"default": "original"}})
        backup = restore_config({"model": {"default": "restored"}}, reason="test")
        assert backup.exists()
        ok, _, _ = verify_config_integrity()
        assert ok is True

    def test_restore_creates_backup(self, isolated_hermes_home):
        save_config({"model": {"default": "original"}})
        backup = restore_config({"model": {"default": "new"}})
        assert backup.name.startswith("config.yaml.pre-restore-")
        assert backup.exists()

    def test_restore_from_yaml_string(self, isolated_hermes_home):
        save_config({"model": {"default": "original"}})
        yaml_str = "model:\n  default: from-yaml\n"
        backup = restore_config(yaml_str, reason="yaml-test")
        ok, _, _ = verify_config_integrity()
        assert ok is True

    def test_restore_with_reason_in_backup_name(self, isolated_hermes_home):
        save_config({"model": {"default": "original"}})
        backup = restore_config({"model": {"default": "new"}}, reason="watchdog")
        assert "watchdog" in backup.name

    def test_seal_updated_after_restore(self, isolated_hermes_home):
        save_config({"model": {"default": "original"}})
        # Tamper to break the seal
        get_config_path().write_text("model:\n  default: tampered\n")
        # Restore fixes it
        restore_config({"model": {"default": "restored"}})
        ok, _, _ = verify_config_integrity()
        assert ok is True
