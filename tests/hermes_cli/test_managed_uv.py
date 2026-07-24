"""Tests for hermes_cli.managed_uv — fallback resolver chain."""

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
# resolve_uv — fallback chain
# ---------------------------------------------------------------------------

class TestResolveUv:
    def test_managed_path_preferred(self, tmp_path):
        """Managed binary is returned when it exists and is executable."""
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(tmp_path / "bin" / "uv")

    def test_missing_managed_returns_none(self, tmp_path):
        """Returns None when no binary exists anywhere."""
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import resolve_uv
            assert resolve_uv() is None

    def test_non_executable_managed_file_returns_none(self, tmp_path):
        """Non-executable managed file is skipped."""
        uv = tmp_path / "bin" / "uv"
        uv.parent.mkdir(parents=True)
        uv.write_text("not a binary")
        uv.chmod(0o644)
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import resolve_uv
            assert resolve_uv() is None

    def test_falls_back_to_local_bin(self, tmp_path, monkeypatch):
        """Falls back to ~/.local/bin/uv when managed copy is missing."""
        home = tmp_path / "home"
        _make_executable(home / ".local" / "bin" / "uv")
        monkeypatch.setenv("HOME", str(home))
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(home / ".local" / "bin" / "uv")

    def test_falls_back_to_cargo_bin(self, tmp_path, monkeypatch):
        """Falls back to ~/.cargo/bin/uv when earlier paths are missing."""
        home = tmp_path / "home"
        _make_executable(home / ".cargo" / "bin" / "uv")
        monkeypatch.setenv("HOME", str(home))
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(home / ".cargo" / "bin" / "uv")

    def test_falls_back_to_path(self, tmp_path):
        """Falls back to shutil.which when no known locations have uv."""
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv.shutil.which", return_value="/usr/bin/uv"):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == "/usr/bin/uv"

    def test_preference_order_local_before_cargo(self, tmp_path, monkeypatch):
        """~/.local/bin is preferred over ~/.cargo/bin."""
        home = tmp_path / "home"
        _make_executable(home / ".local" / "bin" / "uv")
        _make_executable(home / ".cargo" / "bin" / "uv")
        monkeypatch.setenv("HOME", str(home))
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(home / ".local" / "bin" / "uv")

    def test_windows_fallback(self, tmp_path, monkeypatch):
        """Windows uses .exe suffix in fallback paths."""
        home = tmp_path / "home"
        _make_executable(home / ".local" / "bin" / "uv.exe")
        monkeypatch.setenv("HOME", str(home))
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(home / ".local" / "bin" / "uv.exe")

    def test_oserror_on_candidate_is_skipped(self, tmp_path, monkeypatch):
        """OSError during is_file() check on a candidate is silently skipped."""
        home = tmp_path / "home"
        _make_executable(home / ".cargo" / "bin" / "uv")
        monkeypatch.setenv("HOME", str(home))

        real_is_file = Path.is_file

        def patched_is_file(self):
            if ".local" in str(self):
                raise OSError("permission denied")
            return real_is_file(self)

        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None), \
             patch.object(Path, "is_file", patched_is_file):
            from hermes_cli.managed_uv import resolve_uv
            result = resolve_uv()
            assert result == str(home / ".cargo" / "bin" / "uv")


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
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv._install_uv") as mock_install, \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
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
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            # Failure is a falsy sentinel (not None) so legacy 2-target call
            # sites can still unpack it without raising — see
            # TestEnsureUvUpdateBoundary for why.
            assert not path

    def test_uses_fallback_instead_of_installing(self, tmp_path, monkeypatch):
        """When managed copy is missing but fallback exists, no install occurs."""
        home = tmp_path / "home"
        _make_executable(home / ".local" / "bin" / "uv")
        monkeypatch.setenv("HOME", str(home))
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._install_uv") as mock_install, \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import ensure_uv
            path = ensure_uv()
            assert path == str(home / ".local" / "bin" / "uv")
            mock_install.assert_not_called()


class TestEnsureUvUpdateBoundary:
    """``ensure_uv()`` must answer to both the single-value and the legacy
    ``(path, fresh_bootstrap)`` call conventions — **on POSIX**.

    ``hermes update`` runs the call site from the old, already-imported
    ``hermes_cli.main`` against the freshly pulled ``managed_uv``. A release
    parked on a ``(path, fresh)`` tuple runs ``uv_bin, fresh = ensure_uv()``
    against the single-value module; the path is an iterable ``str`` so the
    2-target unpack walked its characters and raised
    ``ValueError: too many values to unpack (expected 2)`` (root cause behind
    PR #39763), or ``TypeError`` on the ``None`` failure path. On POSIX the
    result must therefore be usable as a bare path *and* unpackable as a
    2-tuple, in both the success and failure cases.

    The dual contract is intentionally **not** offered on Windows — see
    ``TestEnsureUvWindowsSafe`` for why — so these tests pin ``platform.system``
    to a POSIX value.
    """

    def test_success_usable_as_single_value(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin = ensure_uv()
            assert uv_bin == str(tmp_path / "bin" / "uv")
            assert bool(uv_bin) is True

    def test_success_unpacks_as_legacy_two_tuple(self, tmp_path):
        _make_executable(tmp_path / "bin" / "uv")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin, fresh = ensure_uv()  # old: uv_bin, fresh_bootstrap = ensure_uv()
            assert uv_bin == str(tmp_path / "bin" / "uv")
            assert fresh is False

    def test_failure_unpacks_without_raising(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Linux"), \
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import ensure_uv
            uv_bin, fresh = ensure_uv()
            assert uv_bin is None
            assert fresh is False


class TestEnsureUvWindowsSafe:
    """On Windows ``ensure_uv()`` must return a plain ``str``/``None``.

    ``subprocess`` on Windows serializes argv through
    ``subprocess.list2cmdline``, which iterates every entry *as a string*
    (``for c in arg``). The dependency installer feeds uv straight into the
    command list (``[uv_bin, "pip", "install", ...]``). A ``str`` subclass
    whose ``__iter__`` yields ``(path, fresh_bootstrap)`` instead of characters
    therefore injects the bool into the command line and crashes the install
    with ``TypeError: sequence item 1: expected str instance, bool found``
    (a real field report on a 10-commits-behind Windows install). A single
    return value cannot serve both the legacy 2-tuple unpack and Windows
    char-iteration — both use the iterator protocol — so Windows opts out of
    the wrapper entirely.
    """

    def test_uvresult_would_break_windows_list2cmdline(self):
        # Canary: this is *why* the wrapper is gated off Windows. If a future
        # change makes _UvResult char-iterable (and thus list2cmdline-safe),
        # the gate may be revisited.
        import subprocess
        from hermes_cli.managed_uv import _UvResult
        with pytest.raises(TypeError):
            subprocess.list2cmdline([_UvResult("C:\\hermes\\uv.exe"), "pip"])

    def test_windows_returns_plain_str_safe_for_subprocess(self, tmp_path):
        import subprocess
        # On (mocked) Windows the managed binary is uv.exe.
        _make_executable(tmp_path / "bin" / "uv.exe")
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"):
            from hermes_cli.managed_uv import _UvResult, ensure_uv
            uv_bin = ensure_uv()
            assert type(uv_bin) is str and not isinstance(uv_bin, _UvResult)
            # The exact operation that crashed in the field must now succeed.
            cmdline = subprocess.list2cmdline([uv_bin, "pip", "install", "-e", "."])
            assert "pip" in cmdline and "install" in cmdline

    def test_windows_failure_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv.platform.system", return_value="Windows"), \
             patch("hermes_cli.managed_uv._install_uv", side_effect=RuntimeError("network down")), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
            from hermes_cli.managed_uv import ensure_uv
            assert ensure_uv() is None


# ---------------------------------------------------------------------------
# update_managed_uv
# ---------------------------------------------------------------------------

class TestUpdateManagedUv:
    def test_no_uv_returns_none(self, tmp_path):
        with patch("hermes_cli.managed_uv.get_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.managed_uv._candidate_paths", return_value=[tmp_path / "bin" / "uv"]), \
             patch("hermes_cli.managed_uv.shutil.which", return_value=None):
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
