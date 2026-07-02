"""Regression tests for #56402: gateway-locks must use a writable state dir."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway import status


class TestResolveStateHomeForLocks:
    def test_uses_home_derived_state_when_writable(self, tmp_path, monkeypatch):
        home_state = tmp_path / "home" / ".local" / "state"
        with patch.object(status, "_home_derived_state_dir", return_value=home_state), patch.object(
            status, "_state_dir_is_writable", return_value=True
        ):
            resolved = status._resolve_state_home_for_locks()
        assert resolved == home_state

    def test_falls_back_to_hermes_home_when_home_state_not_writable(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / "hermes_data"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)

        unwritable = tmp_path / "root_home"
        unwritable.mkdir()
        monkeypatch.setenv("HOME", str(unwritable))

        with patch.object(status, "_state_dir_is_writable", side_effect=[False, True]):
            resolved = status._resolve_state_home_for_locks()

        assert resolved == hermes_home / ".local" / "state"

    def test_get_lock_dir_uses_hermes_home_fallback(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_data"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)

        with patch.object(
            status,
            "_resolve_state_home_for_locks",
            return_value=hermes_home / ".local" / "state",
        ):
            lock_dir = status._get_lock_dir()

        assert lock_dir == hermes_home / ".local" / "state" / "hermes" / "gateway-locks"


class TestAcquireScopedLockWritableFallback:
    def test_acquire_lock_under_hermes_home_fallback(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_data"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)

        with patch.object(
            status,
            "_resolve_state_home_for_locks",
            return_value=hermes_home / ".local" / "state",
        ):
            acquired, existing = status.acquire_scoped_lock(
                "telegram-bot-token",
                "secret-token",
                metadata={"platform": "telegram"},
            )

        assert acquired is True
        assert existing is None
        lock_files = list((hermes_home / ".local" / "state" / "hermes" / "gateway-locks").glob("*.lock"))
        assert len(lock_files) == 1
        payload = json.loads(lock_files[0].read_text(encoding="utf-8"))
        assert payload["scope"] == "telegram-bot-token"
        assert payload["pid"] == os.getpid()
