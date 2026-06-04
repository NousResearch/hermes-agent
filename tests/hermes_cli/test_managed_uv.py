"""Tests for hermes_cli.managed_uv — one path, no guessing."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executable(path: Path) -> None:
    """Create a minimal fake uv binary at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho uv 0.1.2\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


# ---------------------------------------------------------------------------
# managed_uv_path
# ---------------------------------------------------------------------------

class TestManagedUvPath:
    def test_posix(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"):
            from hermes_cli.managed_uv import managed_uv_path
            assert managed_uv_path() == tmp_path / "bin" / "uv"

    def test_windows(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"):
            from hermes_cli.managed_uv import managed_uv_path
            assert managed_uv_path() == tmp_path / "bin" / "uv.exe"


# ---------------------------------------------------------------------------
# resolve_uv
# ---------------------------------------------------------------------------

class TestResolveUv:
    def test_missing_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            assert resolve_uv() is None

    def test_existing_executable(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(tmp_path / "bin" / "uv")

    def test_non_executable_file_returns_none(self, tmp_path):
        uv = tmp_path / "bin" / "uv"
        uv.parent.mkdir(parents=True)
        uv.write_text("not a binary")
        # Ensure no execute bit
        uv.chmod(0o644)
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            assert resolve_uv() is None


# ---------------------------------------------------------------------------
# ensure_uv
# ---------------------------------------------------------------------------

class TestEnsureUv:
    def test_already_installed_no_bootstrap(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import ensure_uv
            path, fresh = ensure_uv()
            assert path == str(tmp_path / "bin" / "uv")
            assert fresh is False

    def test_installs_if_missing_sets_bootstrap_flag(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv") as mock_install:
            # Simulate the installer creating the binary
            def fake_install(target):
                _make_executable(target)
            mock_install.side_effect = fake_install

            from hermes_cli.managed_uv import ensure_uv
            path, fresh = ensure_uv()
            assert path == str(tmp_path / "bin" / "uv")
            assert fresh is True
            mock_install.assert_called_once()

    def test_install_failure_returns_none_false(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")):
            from hermes_cli.managed_uv import ensure_uv
            path, fresh = ensure_uv()
            assert path is None
            assert fresh is False


# ---------------------------------------------------------------------------
# rebuild_venv
# ---------------------------------------------------------------------------

class TestRebuildVenv:
    def test_removes_old_venv_and_creates_new(self, tmp_path):
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "old_file").write_text("stale")

        uv_bin = str(tmp_path / "bin" / "uv")

        def fake_run(cmd, **kwargs):
            m = MagicMock(returncode=0)
            if "venv" in cmd:
                # Simulate uv creating the venv dir
                venv_dir.mkdir(exist_ok=True)
                bin_dir = venv_dir / "bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                (bin_dir / "python").write_text("#!/bin/sh\necho Python 3.11.0")
            elif "--version" in cmd:
                m.stdout = "Python 3.11.0"
            return m

        with patch("hermes_cli.managed_uv.subprocess.run", side_effect=fake_run), \
             patch("hermes_cli.managed_uv.shutil.rmtree") as mock_rmtree:
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is True
            # rmtree is called without ignore_errors; failures are caught explicitly
            mock_rmtree.assert_called_once_with(venv_dir)

    def test_uv_venv_called_with_clear_flag(self, tmp_path):
        """uv venv must always receive --clear so locked Windows dirs are handled atomically."""
        venv_dir = tmp_path / "venv"
        uv_bin = str(tmp_path / "bin" / "uv")

        def fake_run(cmd, **kwargs):
            m = MagicMock(returncode=0)
            if "venv" in cmd:
                venv_dir.mkdir(exist_ok=True)
                bin_dir = venv_dir / "bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                (bin_dir / "python").write_text("fake python")
            elif "--version" in cmd:
                m.stdout = "Python 3.11.0"
            return m

        with patch("hermes_cli.managed_uv.subprocess.run", side_effect=fake_run) as mock_run, \
             patch("hermes_cli.managed_uv.shutil.rmtree"):
            from hermes_cli.managed_uv import rebuild_venv
            rebuild_venv(uv_bin, venv_dir)

        venv_call = next(c for c in mock_run.call_args_list if "venv" in c[0][0])
        assert "--clear" in venv_call[0][0], "uv venv must be called with --clear"

    def test_rmtree_failure_on_windows_is_warned_not_raised(self, tmp_path):
        """On Windows, locked files cause rmtree to fail; this must be logged/warned, not raised."""
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        uv_bin = str(tmp_path / "bin" / "uv")

        def fake_run(cmd, **kwargs):
            m = MagicMock(returncode=0)
            if "venv" in cmd:
                venv_dir.mkdir(exist_ok=True)
                bin_dir = venv_dir / "bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                (bin_dir / "python").write_text("fake python")
            elif "--version" in cmd:
                m.stdout = "Python 3.11.0"
            return m

        lock_error = PermissionError("file in use by another process")
        with patch("hermes_cli.managed_uv.subprocess.run", side_effect=fake_run), \
             patch("hermes_cli.managed_uv.shutil.rmtree", side_effect=lock_error), \
             patch("hermes_cli.managed_uv.logger") as mock_logger:
            from hermes_cli.managed_uv import rebuild_venv
            # Must not raise even though rmtree failed
            result = rebuild_venv(uv_bin, venv_dir)

        assert result is True, "rebuild_venv should succeed when rmtree fails but uv --clear succeeds"
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "--clear" in warning_msg or "clear" in warning_msg.lower()

    def test_rebuild_failure_returns_false(self, tmp_path):
        venv_dir = tmp_path / "venv"
        uv_bin = str(tmp_path / "bin" / "uv")

        with patch("hermes_cli.managed_uv.subprocess.run") as mock_run, \
             patch("hermes_cli.managed_uv.shutil.rmtree"):
            mock_run.return_value = MagicMock(returncode=1, stderr="nope")
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is False


# ---------------------------------------------------------------------------
# update_managed_uv
# ---------------------------------------------------------------------------

class TestUpdateManagedUv:
    def test_no_uv_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import update_managed_uv
            assert update_managed_uv() is None

    def test_self_update_success(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.subprocess.run") as mock_run:
            # uv self update succeeds
            mock_run.return_value = MagicMock(returncode=0, stdout="uv 0.2.0")
            from hermes_cli.managed_uv import update_managed_uv
            result = update_managed_uv()
            assert result == str(tmp_path / "bin" / "uv")
            # First call is self update, second is --version
            assert mock_run.call_count == 2
            assert mock_run.call_args_list[0][0][0] == [str(tmp_path / "bin" / "uv"), "self", "update"]

    def test_self_update_failure_non_fatal(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="nope")
            from hermes_cli.managed_uv import update_managed_uv
            result = update_managed_uv()
            # Still returns the path — failure is non-fatal
            assert result == str(tmp_path / "bin" / "uv")


# ---------------------------------------------------------------------------
# _install_uv internals
# ---------------------------------------------------------------------------

class TestInstallUvInternals:
    def test_posix_sets_uv_unmanaged_install(self, tmp_path):
        target = tmp_path / "bin" / "uv"
        with patch("hermes_cli.managed_uv._install_uv_posix") as mock_posix:
            from hermes_cli.managed_uv import _install_uv
            _install_uv(target)
            mock_posix.assert_called_once()
            call_env = mock_posix.call_args[0][0]
            assert call_env["UV_UNMANAGED_INSTALL"] == str(tmp_path / "bin")

    def test_windows_sets_uv_install_dir(self, tmp_path):
        target = tmp_path / "bin" / "uv.exe"
        with patch("hermes_cli.managed_uv.platform.system", return_value="Windows"), \
             patch("hermes_cli.managed_uv._install_uv_windows") as mock_windows:
            from hermes_cli.managed_uv import _install_uv
            _install_uv(target)
            mock_windows.assert_called_once()
            call_env = mock_windows.call_args[0][0]
            assert call_env["UV_INSTALL_DIR"] == str(tmp_path / "bin")
