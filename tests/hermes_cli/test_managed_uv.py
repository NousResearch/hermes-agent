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
            path = ensure_uv()
            assert path == str(tmp_path / "bin" / "uv")

    def test_installs_if_missing(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv") as mock_install:
            # Simulate the installer creating the binary
            def fake_install(target):
                _make_executable(target)
            mock_install.side_effect = fake_install

            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            assert path == str(tmp_path / "bin" / "uv")
            mock_install.assert_called_once()

    def test_install_failure_returns_falsy(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")):
            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            # Failure is a falsy sentinel (not None) so legacy 2-target call
            # sites can still unpack it without raising — see
            # TestEnsureUvUpdateBoundary for why.
            assert not path


class TestEnsureUvUpdateBoundary:
    """``ensure_uv()`` must answer to both the single-value and the legacy
    ``(path, fresh_bootstrap)`` call conventions.

    ``hermes update`` runs the call site from the old, already-imported
    ``hermes_cli.main`` against the freshly pulled ``managed_uv``. A release
    parked on a ``(path, fresh)`` tuple runs ``uv_bin, fresh = ensure_uv()``
    against the single-value module; the path is an iterable ``str`` so the
    2-target unpack walked its characters and raised
    ``ValueError: too many values to unpack (expected 2)`` (root cause behind
    PR #39763), or ``TypeError`` on the ``None`` failure path. The result must
    therefore be usable as a bare path *and* unpackable as a 2-tuple, in both
    the success and failure cases.
    """

    def test_success_usable_as_single_value(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin = ensure_uv()
            assert uv_bin == str(tmp_path / "bin" / "uv")
            assert bool(uv_bin) is True

    def test_success_unpacks_as_legacy_two_tuple(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin, fresh = ensure_uv()  # old: uv_bin, fresh_bootstrap = ensure_uv()
            assert uv_bin == str(tmp_path / "bin" / "uv")
            assert fresh is False

    def test_failure_unpacks_without_raising(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin, fresh = ensure_uv()
            assert uv_bin is None
            assert fresh is False


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
