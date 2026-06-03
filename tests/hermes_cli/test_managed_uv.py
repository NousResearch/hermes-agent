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
        seen_cmds = []

        def fake_run(cmd, **kwargs):
            seen_cmds.append(cmd)
            m = MagicMock(returncode=0)
            if cmd[1] == "venv":
                # Simulate uv creating a fresh venv dir (uv removes the old one
                # too, but here it has already been moved aside to <name>.old).
                venv_dir.mkdir(exist_ok=True)
                for sub in ("bin", "Scripts"):
                    bin_dir = venv_dir / sub
                    bin_dir.mkdir(parents=True, exist_ok=True)
                    (bin_dir / "python").write_text("#!/bin/sh\necho Python 3.11.0")
            elif "--version" in cmd:
                m.stdout = "Python 3.11.0"
            return m

        with patch("hermes_cli.managed_uv.subprocess.run", side_effect=fake_run):
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is True
            # Fresh venv exists, old contents gone, backup cleaned up.
            assert venv_dir.exists()
            assert not (venv_dir / "old_file").exists()
            assert not (tmp_path / "venv.old").exists()
            # uv venv was invoked with --clear so a leftover partial dir can't
            # wedge the rebuild.
            venv_cmd = next(c for c in seen_cmds if c[1] == "venv")
            assert "--clear" in venv_cmd

    def test_aborts_without_deleting_when_venv_in_use(self, tmp_path):
        """Regression: a venv in use (Windows file lock) must be left intact.

        Previously rebuild_venv ran ``shutil.rmtree(ignore_errors=True)`` which
        deleted what it could (site-packages, certifi) and left a gutted venv,
        bricking the install.  Now the move-aside fails fast and the existing
        venv is untouched, with no ``uv venv`` attempted.
        """
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        marker = venv_dir / "site-packages-marker"
        marker.write_text("do not delete me")
        uv_bin = str(tmp_path / "bin" / "uv")

        with patch("hermes_cli.managed_uv.os.replace", side_effect=OSError("in use")), \
             patch("hermes_cli.managed_uv.subprocess.run") as mock_run:
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is False
            # No rebuild attempted, existing venv fully intact.
            mock_run.assert_not_called()
            assert marker.exists()
            assert marker.read_text() == "do not delete me"

    def test_restores_old_venv_when_rebuild_fails(self, tmp_path):
        """If ``uv venv`` fails, the moved-aside venv is restored, not lost."""
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "marker").write_text("original")
        uv_bin = str(tmp_path / "bin" / "uv")

        with patch(
            "hermes_cli.managed_uv.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="boom"),
        ):
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is False
            # Old venv restored; no orphaned backup left behind.
            assert (venv_dir / "marker").exists()
            assert (venv_dir / "marker").read_text() == "original"
            assert not (tmp_path / "venv.old").exists()

    def test_rebuild_failure_returns_false(self, tmp_path):
        venv_dir = tmp_path / "venv"
        uv_bin = str(tmp_path / "bin" / "uv")

        with patch("hermes_cli.managed_uv.subprocess.run") as mock_run, \
             patch("hermes_cli.managed_uv.shutil.rmtree"):
            mock_run.return_value = MagicMock(returncode=1, stderr="nope")
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is False

    def test_rebuild_success_without_python_returns_false(self, tmp_path):
        """uv can exit 0 yet leave no interpreter; that must not count as success
        (guard adapted from #38511)."""
        venv_dir = tmp_path / "venv"
        uv_bin = str(tmp_path / "bin" / "uv")

        with patch("hermes_cli.managed_uv.subprocess.run") as mock_run, \
             patch("hermes_cli.managed_uv.shutil.rmtree"):
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            from hermes_cli.managed_uv import rebuild_venv
            result = rebuild_venv(uv_bin, venv_dir)
            assert result is False
            # Returned before the `python --version` probe ran.
            assert mock_run.call_count == 1


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
