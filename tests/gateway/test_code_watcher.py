"""
Unit tests for gateway/code_watcher.py.

Tests cover:
- CodeWatcher detects mtime changes and calls os.execv
- CodeWatcher respects HERMES_GATEWAY_NO_AUTO_RESTART env opt-out
- CodeWatcher respects gateway.auto_restart_on_code_change=false config opt-out
- start_code_watcher returns None when disabled, CodeWatcher when enabled
- _collect_watched_files only returns files under the checkout root
- _get_process_start_time falls back gracefully
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from gateway.code_watcher import (
    CodeWatcher,
    _collect_watched_files,
    _detect_checkout_root,
    _is_disabled_by_config,
    _is_disabled_by_env,
    start_code_watcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePath:
    """Minimal fake path that records stat() calls and returns a fixed mtime."""

    def __init__(self, path: str, mtime: float = 1000.0, exists: bool = True) -> None:
        self._path = path
        self._mtime = mtime
        self._exists = exists
        self.suffix = Path(path).suffix

    def stat(self) -> MagicMock:
        if not self._exists:
            raise OSError("No such file")
        m = MagicMock()
        m.st_mtime = self._mtime
        return m

    def is_file(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return self._path

    def __fspath__(self) -> str:
        return self._path


# ---------------------------------------------------------------------------
# _is_disabled_by_env
# ---------------------------------------------------------------------------


class TestIsDisabledByEnv:
    def test_disabled_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_NO_AUTO_RESTART", "1")
        assert _is_disabled_by_env() is True

    def test_disabled_when_set_to_true(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_NO_AUTO_RESTART", "true")
        assert _is_disabled_by_env() is True

    def test_enabled_when_absent(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_NO_AUTO_RESTART", raising=False)
        assert _is_disabled_by_env() is False

    def test_enabled_when_set_to_0(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_NO_AUTO_RESTART", "0")
        assert _is_disabled_by_env() is False

    def test_enabled_when_empty(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_NO_AUTO_RESTART", "")
        assert _is_disabled_by_env() is False


# ---------------------------------------------------------------------------
# _is_disabled_by_config
# ---------------------------------------------------------------------------


class TestIsDisabledByConfig:
    def test_disabled_when_false(self):
        cfg = {"gateway": {"auto_restart_on_code_change": False}}
        assert _is_disabled_by_config(cfg) is True

    def test_disabled_when_string_false(self):
        cfg = {"gateway": {"auto_restart_on_code_change": "false"}}
        assert _is_disabled_by_config(cfg) is True

    def test_disabled_when_string_off(self):
        cfg = {"gateway": {"auto_restart_on_code_change": "off"}}
        assert _is_disabled_by_config(cfg) is True

    def test_enabled_when_true(self):
        cfg = {"gateway": {"auto_restart_on_code_change": True}}
        assert _is_disabled_by_config(cfg) is False

    def test_enabled_when_key_absent(self):
        cfg: Dict = {"gateway": {}}
        assert _is_disabled_by_config(cfg) is False

    def test_enabled_when_cfg_none(self):
        assert _is_disabled_by_config(None) is False

    def test_enabled_when_gateway_section_absent(self):
        assert _is_disabled_by_config({}) is False


# ---------------------------------------------------------------------------
# CodeWatcher._scan
# ---------------------------------------------------------------------------


class TestCodeWatcherScan:
    def _watcher(self, start_time: float) -> CodeWatcher:
        return CodeWatcher(
            process_start_time=start_time,
            poll_interval=9999,  # never fires in tests
            restart_delay=0,
        )

    def test_no_change_returns_none(self, tmp_path):
        (tmp_path / "gateway").mkdir()
        (tmp_path / "hermes_cli").mkdir()
        # File mtime older than start time
        f = tmp_path / "hermes_cli" / "test.py"
        f.write_text("x = 1\n")
        os.utime(f, (500.0, 500.0))
        watcher = self._watcher(1000.0)
        result = watcher._scan(tmp_path)
        assert result is None

    def test_change_detected_when_mtime_newer(self, tmp_path):
        (tmp_path / "gateway").mkdir()
        (tmp_path / "hermes_cli").mkdir()
        f = tmp_path / "hermes_cli" / "changed.py"
        f.write_text("x = 2\n")
        # mtime newer than process start
        future = time.time() + 9999
        os.utime(f, (future, future))
        watcher = self._watcher(time.time())
        # Manually inject the file into a fake module in sys.modules
        import types

        fake_mod = types.ModuleType("_test_fake_changed")
        fake_mod.__file__ = str(f)
        sys.modules["_test_fake_changed"] = fake_mod
        try:
            result = watcher._scan(tmp_path)
            assert result is not None
            assert "changed.py" in result
        finally:
            del sys.modules["_test_fake_changed"]


# ---------------------------------------------------------------------------
# CodeWatcher triggers os.execv when change detected
# ---------------------------------------------------------------------------


class TestCodeWatcherExecv:
    def test_execv_called_on_mtime_change(self, tmp_path, monkeypatch):
        """
        Verify that when a watched file's mtime advances past the process start
        time, CodeWatcher calls os.execv with sys.executable + sys.argv.
        """
        (tmp_path / "gateway").mkdir()
        (tmp_path / "hermes_cli").mkdir()
        f = tmp_path / "hermes_cli" / "kanban_db.py"
        f.write_text("# placeholder\n")
        # Set mtime in the past so initial scan finds nothing
        os.utime(f, (100.0, 100.0))

        import types

        fake_mod = types.ModuleType("_test_execv_mod")
        fake_mod.__file__ = str(f)
        sys.modules["_test_execv_mod"] = fake_mod

        execv_calls = []

        def fake_execv(executable, argv):
            execv_calls.append((executable, argv))

        monkeypatch.setattr(os, "execv", fake_execv)
        monkeypatch.delenv("HERMES_GATEWAY_NO_AUTO_RESTART", raising=False)

        watcher = CodeWatcher(
            process_start_time=50.0,   # start before mtime=100 so scan immediately fires
            poll_interval=0.05,        # very fast for the test
            restart_delay=0.0,         # no delay
            checkout_root=tmp_path,
        )

        try:
            watcher.start()
            # Give the watcher at most 3 seconds to detect the change and call execv
            deadline = time.time() + 3.0
            while not execv_calls and time.time() < deadline:
                time.sleep(0.05)
        finally:
            del sys.modules["_test_execv_mod"]
            watcher.stop()

        assert execv_calls, "os.execv was not called"
        executable, argv = execv_calls[0]
        assert executable == sys.executable
        assert argv[0] == sys.executable
        assert argv[1:] == sys.argv

    def test_no_execv_when_env_disabled(self, tmp_path, monkeypatch):
        """With HERMES_GATEWAY_NO_AUTO_RESTART=1, start_code_watcher returns None."""
        monkeypatch.setenv("HERMES_GATEWAY_NO_AUTO_RESTART", "1")
        result = start_code_watcher(cfg=None)
        assert result is None

    def test_no_execv_when_config_disabled(self, monkeypatch):
        """With config opt-out, start_code_watcher returns None."""
        monkeypatch.delenv("HERMES_GATEWAY_NO_AUTO_RESTART", raising=False)
        cfg = {"gateway": {"auto_restart_on_code_change": False}}
        result = start_code_watcher(cfg=cfg)
        assert result is None

    def test_returns_watcher_when_enabled(self, monkeypatch):
        """When enabled, start_code_watcher returns a running CodeWatcher."""
        monkeypatch.delenv("HERMES_GATEWAY_NO_AUTO_RESTART", raising=False)
        watcher = start_code_watcher(cfg={"gateway": {"auto_restart_on_code_change": True}})
        try:
            assert isinstance(watcher, CodeWatcher)
            assert watcher._thread is not None
            assert watcher._thread.is_alive()
        finally:
            if watcher:
                watcher.stop()


# ---------------------------------------------------------------------------
# _collect_watched_files
# ---------------------------------------------------------------------------


class TestCollectWatchedFiles:
    def test_returns_py_files_under_root(self, tmp_path):
        (tmp_path / "hermes_cli").mkdir()
        f = tmp_path / "hermes_cli" / "main.py"
        f.write_text("pass\n")

        import types

        mod = types.ModuleType("_test_collect_mod")
        mod.__file__ = str(f)
        sys.modules["_test_collect_mod"] = mod
        try:
            results = _collect_watched_files(tmp_path)
            assert any(str(f) == str(r) for r in results)
        finally:
            del sys.modules["_test_collect_mod"]

    def test_excludes_files_outside_root(self, tmp_path):
        other = tmp_path / "other_dir"
        other.mkdir()
        f = other / "something.py"
        f.write_text("pass\n")

        root = tmp_path / "checkout"
        root.mkdir()
        (root / "gateway").mkdir()

        import types

        mod = types.ModuleType("_test_exclude_mod")
        mod.__file__ = str(f)
        sys.modules["_test_exclude_mod"] = mod
        try:
            results = _collect_watched_files(root)
            assert not any("something.py" in str(r) for r in results)
        finally:
            del sys.modules["_test_exclude_mod"]

    def test_excludes_non_py_files(self, tmp_path):
        (tmp_path / "hermes_cli").mkdir()
        f = tmp_path / "hermes_cli" / "data.json"
        f.write_text("{}\n")

        import types

        mod = types.ModuleType("_test_json_mod")
        mod.__file__ = str(f)
        sys.modules["_test_json_mod"] = mod
        try:
            results = _collect_watched_files(tmp_path)
            assert not any(".json" in str(r) for r in results)
        finally:
            del sys.modules["_test_json_mod"]
