"""Tests for ``install_cua_driver`` upgrade semantics.

The cua-driver upstream installer always pulls the latest release tag, so
re-running it is the canonical upgrade path. ``install_cua_driver(upgrade=True)``
must:

* Be supported-platform-only — no-op silently elsewhere so ``hermes update``
  can call it unconditionally without warning unsupported-platform users.
* Re-run the installer even when the binary is already on PATH (this is the
  fix for the "we only pulled cua-driver once on enable" complaint).
* Preserve original ``upgrade=False`` behaviour for the toolset-enable flow:
  skip if installed, install otherwise, warn on unsupported platforms.

The pre-install arch probe that used to live alongside this function was
deleted (see top-of-file comment in tools_config.py) — the upstream
installer has CUA_DRIVER_RS_BAKED_VERSION baked in by CD and errors
cleanly on missing-arch assets, and the upgrade path uses
``cua_driver_update_check()`` (which shells `cua-driver check-update
--json` against the already-installed binary).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class TestInstallCuaDriverUpgrade:
    def test_upgrade_on_unsupported_platform_is_silent_noop(self):
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="FreeBSD"):
            assert tools_config.install_cua_driver(upgrade=True) is False
            warn.assert_not_called()

    def test_non_upgrade_on_unsupported_platform_warns(self):
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="FreeBSD"):
            assert tools_config.install_cua_driver(upgrade=False) is False
            warn.assert_called()

    def test_upgrade_on_macos_with_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()
            kwargs = runner.call_args.kwargs
            assert kwargs.get("verbose") is False

    def test_upgrade_on_macos_without_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()

    def test_quiet_refresh_prints_single_contextual_progress_line(self):
        import subprocess
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        with patch("platform.system", return_value="Linux"), \
             patch(
                 "subprocess.run",
                 return_value=MagicMock(returncode=0, stderr=""),
             ), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch.object(
                 tools_config.shutil,
                 "which",
                 return_value="/usr/local/bin/cua-driver",
             ), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_info") as info:
            assert tools_config._run_cua_driver_installer(
                label="Refreshing",
                verbose=False,
            ) is True

        info.assert_called_once_with(
            "→ Refreshing cua-driver (Computer Use)..."
        )

    def test_quiet_refresh_can_suppress_progress_line(self):
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        with patch("platform.system", return_value="Linux"), \
             patch(
                 "subprocess.run",
                 return_value=MagicMock(returncode=0, stderr=""),
             ), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch.object(
                 tools_config.shutil,
                 "which",
                 return_value="/usr/local/bin/cua-driver",
             ), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_info") as info:
            assert tools_config._run_cua_driver_installer(
                label="Refreshing",
                verbose=False,
                show_progress=False,
            ) is True

        info.assert_not_called()

    def test_upgrade_can_suppress_installer_progress(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(
                 tools_config.shutil,
                 "which",
                 side_effect=lambda name: (
                     f"/usr/local/bin/{name}"
                     if name in {"cua-driver", "curl"}
                     else None
                 ),
             ), \
             patch.object(
                 tools_config,
                 "_run_cua_driver_installer",
                 return_value=True,
             ) as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(
                upgrade=True,
                show_installer_progress=False,
            ) is True

        assert runner.call_args.kwargs["show_progress"] is False

    def test_upgrade_on_macos_non_writable_applications_skips_refresh(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=False), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner, \
             patch.object(tools_config, "_print_info") as info:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_not_called()
            assert any(
                "/Applications is not writable" in call.args[0]
                for call in info.call_args_list
            )

    def test_fresh_install_on_macos_non_writable_applications_skips_install(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=False), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner, \
             patch.object(tools_config, "_print_info") as info:
            assert tools_config.install_cua_driver(upgrade=False) is False
            runner.assert_not_called()
            assert any(
                "/Applications is not writable" in call.args[0]
                for call in info.call_args_list
            )

    def test_non_upgrade_on_macos_with_binary_skips_install(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_not_called()

    def test_non_upgrade_on_macos_without_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_called_once()


class TestRequireConfirmedUpdate:
    """`hermes update` passes require_confirmed_update=True: the full
    upstream installer (multi-minute, output captured, plus install.ps1's
    600s lock window on Windows) may only run when the driver's native
    ``check-update`` verb positively confirms a newer release. An
    indeterminate check (old driver, offline, GitHub rate-limited, probe
    timeout) keeps the installed version and returns fast.

    Explicit `hermes computer-use install --upgrade` keeps the old
    fall-through (require_confirmed_update=False): a force-refresh should
    still reinstall when the check can't answer.
    """

    def _install(self, system, check_state, require_confirmed):
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        exe = "cua-driver" + (".exe" if system == "Windows" else "")
        with patch("platform.system", return_value=system), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/x/" + n
                          if n in {"cua-driver", "curl", "powershell"} else None), \
             patch.object(tools_config, "_resolved_cua_driver_cmd",
                          return_value="/x/" + exe), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=True), \
             patch("tools.computer_use.cua_backend.cua_driver_update_check",
                   return_value=check_state), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run",
                   return_value=MagicMock(stdout="cua-driver 0.5.0", returncode=0)), \
             patch.object(tools_config, "_print_success"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info") as info:
            ok = tools_config.install_cua_driver(
                upgrade=True, require_confirmed_update=require_confirmed
            )
        return ok, runner, info

    def test_indeterminate_check_keeps_installed_version(self):
        ok, runner, info = self._install("Windows", None, require_confirmed=True)
        assert ok is True
        runner.assert_not_called()
        assert any(
            "keeping the installed version" in call.args[0]
            for call in info.call_args_list
        )

    def test_indeterminate_check_points_at_force_path(self):
        ok, runner, info = self._install("Darwin", None, require_confirmed=True)
        assert ok is True
        runner.assert_not_called()
        assert any(
            "computer-use install --upgrade" in call.args[0]
            for call in info.call_args_list
        )

    def test_confirmed_update_still_runs_installer(self):
        state = {"current_version": "0.5.0", "latest_version": "0.6.0",
                 "update_available": True}
        ok, runner, _ = self._install("Windows", state, require_confirmed=True)
        assert ok is True
        runner.assert_called_once()

    def test_up_to_date_short_circuits(self):
        state = {"current_version": "0.6.0", "latest_version": "0.6.0",
                 "update_available": False}
        ok, runner, _ = self._install("Windows", state, require_confirmed=True)
        assert ok is True
        runner.assert_not_called()

    def test_explicit_upgrade_still_falls_through_on_indeterminate(self):
        # `hermes computer-use install --upgrade` (default flag): the old
        # behaviour — indeterminate check re-runs the installer.
        ok, runner, _ = self._install("Darwin", None, require_confirmed=False)
        assert ok is True
        runner.assert_called_once()


class TestUpdateCheckTimeoutDefaults:
    """cua_driver_update_check: platform-sensitive default timeout.

    8s is fine on POSIX but too tight for Windows first-spawn (Defender /
    SmartScreen scanning), and a false timeout is what used to trigger the
    full reinstall fall-through during `hermes update`.
    """

    def _captured_timeout(self, platform_name):
        from unittest.mock import MagicMock
        from tools.computer_use import cua_backend

        captured = {}

        def fake_run(cmd, **kw):
            captured["timeout"] = kw.get("timeout")
            m = MagicMock()
            m.stdout = '{"update_available": false, "current_version": "1.0"}'
            return m

        with patch("tools.computer_use.cua_backend.resolve_cua_driver_cmd",
                   return_value="/x/cua-driver"), \
             patch("tools.computer_use.cua_backend.sys.platform", platform_name), \
             patch("tools.computer_use.cua_backend.subprocess.run",
                   side_effect=fake_run):
            cua_backend.cua_driver_update_check()
        return captured.get("timeout")

    def test_windows_default_is_generous(self):
        assert self._captured_timeout("win32") == 25.0

    def test_posix_default_unchanged(self):
        assert self._captured_timeout("linux") == 8.0

    def test_explicit_timeout_wins(self):
        from unittest.mock import MagicMock
        from tools.computer_use import cua_backend

        captured = {}

        def fake_run(cmd, **kw):
            captured["timeout"] = kw.get("timeout")
            m = MagicMock()
            m.stdout = "{}"
            return m

        with patch("tools.computer_use.cua_backend.resolve_cua_driver_cmd",
                   return_value="/x/cua-driver"), \
             patch("tools.computer_use.cua_backend.subprocess.run",
                   side_effect=fake_run):
            cua_backend.cua_driver_update_check(timeout=3.0)
        assert captured.get("timeout") == 3.0


class TestArchProbeRemoval:
    """Regression tests for the deletion of `_check_cua_driver_asset_for_arch`.

    The old probe queried ``/releases/latest`` on trycua/cua and inspected
    asset names. That was wrong in two ways:

    1. cua-driver-rs releases are marked **prerelease** on every cut, so
       ``/releases/latest`` returns the Python ``cua-agent`` / ``cua-computer``
       package instead — a release with zero binary assets. The probe then
       reported "no asset for $arch" on Linux x86_64, Windows, macOS Intel,
       Linux arm64 — every non-Apple-Silicon host.
    2. Even with the right endpoint, it duplicated tag-resolution the upstream
       installer already does correctly via ``CUA_DRIVER_RS_BAKED_VERSION``
       (auto-baked by CD on every release).

    The fix: stop probing. Trust the upstream installer for fresh installs
    (it has the baked version + correct API fallback) and the
    ``cua-driver check-update --json`` MCP-binary native command for the
    upgrade path.
    """

    def test_probe_function_is_gone(self):
        from hermes_cli import tools_config
        assert not hasattr(tools_config, "_check_cua_driver_asset_for_arch")
        assert not hasattr(tools_config, "_latest_cua_driver_rs_release")

    def test_fresh_install_does_not_call_github_api(self):
        """Pre-install no longer probes the GitHub API — the upstream
        ``install.sh`` resolves the tag from its baked CUA_DRIVER_RS_BAKED_VERSION
        line. install.sh errors cleanly when the arch has no asset, so the
        probe was duplicate gatekeeping.
        """
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch("urllib.request.urlopen") as urlopen, \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_called_once()
            urlopen.assert_not_called()

    def test_upgrade_with_binary_does_not_call_github_api_directly(self):
        """The upgrade path no longer hits GitHub from Python — it delegates
        to the upstream ``install.sh`` (which has the baked release tag and
        the proper API fallback). When cua-driver is already installed,
        ``cua_driver_update_check()`` (added in a separate change) further
        short-circuits the network re-install via the binary's native
        ``check-update --json`` verb.
        """
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in ("cua-driver", "curl") else None), \
             patch("urllib.request.urlopen") as urlopen, \
             patch("subprocess.run"), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()
            # Probe deleted — no direct GitHub API call from Python.
            urlopen.assert_not_called()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX installer uses the .install.lock.d directory protocol",
)
class TestPosixStaleInstallLockClear:
    """_clear_stale_cua_install_lock: pre-clears the upstream installer's
    concurrent-install lock only when the holder is provably dead (or the
    lock is old and pid-less). Issue #58762."""

    def _make_lock(self, tmp_path, pid=None):
        import os
        home = tmp_path / ".cua-driver"
        lock = home / "packages" / ".install.lock.d"
        lock.mkdir(parents=True)
        if pid is not None:
            (lock / "info").write_text(f"pid={pid}\n")
        os.environ["CUA_DRIVER_RS_HOME"] = str(home)
        return lock

    def teardown_method(self):
        import os
        os.environ.pop("CUA_DRIVER_RS_HOME", None)

    def test_dead_holder_lock_is_cleared(self, tmp_path):
        from hermes_cli import tools_config

        dead_pid = 4194000  # above default pid_max on most systems
        lock = self._make_lock(tmp_path, pid=dead_pid)
        with patch.object(tools_config, "_print_info"):
            tools_config._clear_stale_cua_install_lock()
        assert not lock.exists()

    def test_live_holder_lock_is_kept(self, tmp_path):
        import os
        from hermes_cli import tools_config

        lock = self._make_lock(tmp_path, pid=os.getpid())
        tools_config._clear_stale_cua_install_lock()
        assert lock.exists()

    def test_pidless_fresh_lock_is_kept(self, tmp_path):
        from hermes_cli import tools_config

        lock = self._make_lock(tmp_path, pid=None)
        tools_config._clear_stale_cua_install_lock()
        assert lock.exists()

    def test_pidless_old_lock_is_cleared(self, tmp_path):
        import os
        import time
        from hermes_cli import tools_config

        lock = self._make_lock(tmp_path, pid=None)
        old = time.time() - (tools_config._CUA_LOCK_STALE_AFTER + 60)
        os.utime(lock, (old, old))
        with patch.object(tools_config, "_print_info"):
            tools_config._clear_stale_cua_install_lock()
        assert not lock.exists()

    def test_no_lock_is_noop(self, tmp_path):
        import os
        os.environ["CUA_DRIVER_RS_HOME"] = str(tmp_path / ".cua-driver")
        from hermes_cli import tools_config
        tools_config._clear_stale_cua_install_lock()  # must not raise


class TestWindowsStaleInstallLockClearDispatch:
    def test_windows_branch_uses_file_lock_probe(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.sys, "platform", "win32"), \
             patch.object(
                 tools_config, "_clear_stale_windows_cua_install_lock"
             ) as clear_windows:
            tools_config._clear_stale_cua_install_lock()

        clear_windows.assert_called_once_with()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="requires native Win32 FileShare semantics",
)
class TestWindowsStaleInstallLockClear:
    def _make_lock(self, tmp_path):
        import os

        home = tmp_path / ".cua-driver"
        home.mkdir()
        lock = home / "install.lock"
        lock.write_text("pid=stale\n", encoding="utf-8")
        os.environ["CUA_DRIVER_RS_HOME"] = str(home)
        return lock

    def teardown_method(self):
        import os

        os.environ.pop("CUA_DRIVER_RS_HOME", None)

    def test_unlocked_lock_file_is_cleared(self, tmp_path):
        from hermes_cli import tools_config

        lock = self._make_lock(tmp_path)
        with patch.object(tools_config, "_print_info"):
            tools_config._clear_stale_cua_install_lock()

        assert not lock.exists()

    def test_lock_held_with_file_share_none_is_kept(self, tmp_path):
        import ctypes
        from ctypes import wintypes
        from hermes_cli import tools_config

        lock = self._make_lock(tmp_path)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = create_file(
            str(lock),
            0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
            0,  # FileShare::None, matching install.ps1
            None,
            3,  # OPEN_EXISTING
            0x00000080,  # FILE_ATTRIBUTE_NORMAL
            None,
        )
        assert handle != wintypes.HANDLE(-1).value

        try:
            tools_config._clear_stale_cua_install_lock()
            assert lock.exists()
        finally:
            assert close_handle(handle)


class TestInstallerTimeoutKillsProcessGroup:
    """On timeout the whole installer process group must be killed, so the
    `curl | bash` grandchildren can't survive holding the install lock."""

    def test_timeout_kills_process_group_and_returns_false(self, tmp_path):
        import os
        import signal
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        killed = {}
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)

        fake_proc = MagicMock()
        fake_proc.pid = 12345
        # First communicate() raises TimeoutExpired, second (post-kill) returns.
        fake_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="x", timeout=1),
            ("", None),
        ]

        def fake_killpg(pgid, sig):
            killed["pgid"] = pgid
            killed["sig"] = sig

        with patch("platform.system", return_value="Linux"), \
             patch.object(signal, "SIGKILL", sigkill, create=True), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch.object(
                 tools_config.os, "getpgid", return_value=99999, create=True
             ), \
             patch.object(
                 tools_config.os, "killpg", side_effect=fake_killpg, create=True
             ), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            ok = tools_config._run_cua_driver_installer(label="Refreshing", verbose=False)

        assert ok is False
        assert killed.get("pgid") == 99999
        assert killed.get("sig") == sigkill
        # Post-kill reap happened.
        assert fake_proc.communicate.call_count == 2

    def test_timeout_ceiling_exceeds_upstream_lock_window(self):
        from hermes_cli import tools_config
        # The upstream installer waits up to 600s before reclaiming a stale
        # lock; our ceiling must give that window room to complete.
        assert tools_config._CUA_INSTALLER_TIMEOUT > tools_config._CUA_LOCK_STALE_AFTER

    def test_installer_runs_in_new_session_on_posix(self, tmp_path):
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        captured = {}
        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 1
        fake_proc.communicate.return_value = ("", None)

        def fake_popen(*args, **kwargs):
            captured.update(kwargs)
            return fake_proc

        with patch("platform.system", return_value="Linux"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            tools_config._run_cua_driver_installer(label="Refreshing", verbose=False)

        assert captured.get("start_new_session") is True

    def test_windows_timeout_kills_descendants_and_parent(self):
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        child = MagicMock()
        parent = MagicMock()
        parent.children.return_value = [child]

        fake_proc = MagicMock()
        fake_proc.pid = 12345
        fake_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="powershell", timeout=1),
            ("", None),
        ]

        with patch("platform.system", return_value="Windows"), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch("psutil.Process", return_value=parent), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            ok = tools_config._run_cua_driver_installer(
                label="Refreshing", verbose=False
            )

        assert ok is False
        parent.children.assert_called_once_with(recursive=True)
        child.kill.assert_called_once_with()
        parent.kill.assert_called_once_with()
        fake_proc.kill.assert_not_called()
        assert fake_proc.communicate.call_count == 2

    def test_windows_tree_enumeration_failure_falls_back_to_direct_kill(self):
        import psutil
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        parent = MagicMock()
        parent.children.side_effect = psutil.AccessDenied(pid=12345)

        fake_proc = MagicMock()
        fake_proc.pid = 12345
        fake_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="powershell", timeout=1),
            ("", None),
        ]

        with patch("platform.system", return_value="Windows"), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch("psutil.Process", return_value=parent), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            ok = tools_config._run_cua_driver_installer(
                label="Refreshing", verbose=False
            )

        assert ok is False
        fake_proc.kill.assert_called_once_with()
        assert fake_proc.communicate.call_count == 2


class TestInstallerNoShell:
    """The POSIX installer path must not use shell=True or command
    substitution: the script is downloaded to a mkstemp file and exec'd
    as a plain argv list (salvage of #34974's intent, without the fixed
    /tmp path TOCTOU that PR introduced)."""

    def _run(self, download_rc=0):
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        calls = []
        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        def fake_run(cmd, **kw):
            calls.append(("run", cmd, kw))
            m = MagicMock()
            m.returncode = download_rc
            m.stderr = "curl: (6) could not resolve" if download_rc else ""
            return m

        def fake_popen(cmd, **kw):
            calls.append(("popen", cmd, kw))
            return fake_proc

        with patch("platform.system", return_value="Linux"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(tools_config.shutil, "which", return_value="/usr/local/bin/cua-driver"), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"), \
             patch.object(tools_config, "_print_success"):
            ok = tools_config._run_cua_driver_installer(label="Refreshing", verbose=False)
        return ok, calls

    def test_posix_path_downloads_then_execs_argv_list(self):
        ok, calls = self._run()
        assert ok is True
        run_calls = [c for c in calls if c[0] == "run"]
        popen_calls = [c for c in calls if c[0] == "popen"]
        assert len(run_calls) == 1 and len(popen_calls) == 1
        # Download: plain argv curl, no shell.
        dl_cmd = run_calls[0][1]
        assert isinstance(dl_cmd, list) and dl_cmd[0] == "curl"
        # Exec: argv list ["/bin/bash", <mkstemp path>], shell=False.
        exec_cmd, exec_kw = popen_calls[0][1], popen_calls[0][2]
        assert isinstance(exec_cmd, list) and exec_cmd[0] == "/bin/bash"
        assert "cua-driver-install-" in exec_cmd[1]
        assert exec_kw.get("shell") is False

    def test_download_failure_returns_false_without_exec(self):
        ok, calls = self._run(download_rc=6)
        assert ok is False
        assert not [c for c in calls if c[0] == "popen"]

    def test_temp_script_removed_after_run(self, tmp_path):
        import os
        captured = {}
        import subprocess
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        def fake_run(cmd, **kw):
            m = MagicMock(); m.returncode = 0; m.stderr = ""
            return m

        def fake_popen(cmd, **kw):
            captured["script"] = cmd[1]
            return fake_proc

        with patch("platform.system", return_value="Linux"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(tools_config.shutil, "which", return_value="/usr/local/bin/cua-driver"), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"), \
             patch.object(tools_config, "_print_success"):
            tools_config._run_cua_driver_installer(label="Refreshing", verbose=False)

        assert "script" in captured
        assert not os.path.exists(captured["script"])


class TestConfirmedVersionPinning:
    """When check-update confirms a newer release, the installer run must be
    pinned to that exact version via CUA_DRIVER_RS_VERSION.

    The upstream installer scripts on `main` carry a baked version that
    Release Please bumps in the release PR *before* the release assets are
    published. An unpinned install inside that window 404s (observed
    2026-07-29: baked 0.14.0 vs latest published release 0.13.1). Pinning to
    check-update's `latest_version` — which comes from the Releases API and
    therefore has published assets — sidesteps the race.
    """

    def _install(self, check_state):
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        with patch("platform.system", return_value="Windows"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/x/" + n
                          if n in {"cua-driver", "curl", "powershell"} else None), \
             patch.object(tools_config, "_resolved_cua_driver_cmd",
                          return_value="/x/cua-driver.exe"), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=True), \
             patch("tools.computer_use.cua_backend.cua_driver_update_check",
                   return_value=check_state), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run",
                   return_value=MagicMock(stdout="cua-driver 0.5.0", returncode=0)), \
             patch.object(tools_config, "_print_success"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            ok = tools_config.install_cua_driver(
                upgrade=True, require_confirmed_update=True
            )
        return ok, runner

    def test_confirmed_update_pins_latest_version(self):
        state = {"current_version": "0.12.6", "latest_version": "0.13.1",
                 "update_available": True}
        ok, runner = self._install(state)
        assert ok is True
        assert runner.call_args.kwargs.get("pin_version") == "0.13.1"

    def test_v_prefixed_latest_version_is_normalized(self):
        state = {"current_version": "0.12.6", "latest_version": "v0.13.1",
                 "update_available": True}
        ok, runner = self._install(state)
        assert ok is True
        assert runner.call_args.kwargs.get("pin_version") == "0.13.1"

    def test_malformed_latest_version_falls_back_unpinned(self):
        state = {"current_version": "0.12.6", "latest_version": "not a version",
                 "update_available": True}
        ok, runner = self._install(state)
        assert ok is True
        assert runner.call_args.kwargs.get("pin_version") is None

    def test_missing_latest_version_falls_back_unpinned(self):
        state = {"current_version": "0.12.6", "update_available": True}
        ok, runner = self._install(state)
        assert ok is True
        assert runner.call_args.kwargs.get("pin_version") is None


class TestRunInstallerPinEnv:
    """_run_cua_driver_installer(pin_version=...) exports CUA_DRIVER_RS_VERSION
    into the installer child env; unpinned runs leave it untouched."""

    def _run(self, pin_version):
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        captured = {}
        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 1
        fake_proc.communicate.return_value = ("", None)

        def fake_popen(cmd, **kw):
            captured["env"] = kw.get("env")
            return fake_proc

        def fake_run(cmd, **kw):
            m = MagicMock(); m.returncode = 0; m.stderr = ""
            return m

        with patch("platform.system", return_value="Linux"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(tools_config, "_cua_driver_env",
                          return_value={"PATH": "/usr/bin"}), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            tools_config._run_cua_driver_installer(
                label="Refreshing", verbose=False, pin_version=pin_version
            )
        return captured.get("env") or {}

    def test_pin_version_exported_to_installer_env(self):
        env = self._run("0.13.1")
        assert env.get("CUA_DRIVER_RS_VERSION") == "0.13.1"

    def test_no_pin_leaves_env_untouched(self):
        env = self._run(None)
        assert "CUA_DRIVER_RS_VERSION" not in env


class TestWindowsAutostartRepair:
    def test_existing_task_skips_elevated_powershell_repair(self):
        from hermes_cli import tools_config

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return SimpleNamespace(returncode=0)

        with patch.object(tools_config.sys, "platform", "win32"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch.object(tools_config.shutil, "which") as which:
            ok = tools_config._repair_cua_driver_autostart_windows(
                "cua-driver", verbose=False
            )

        assert ok is True
        assert [cmd for cmd, _kwargs in calls] == [
            ["schtasks.exe", "/Query", "/TN", "cua-driver-serve"]
        ]
        which.assert_not_called()

    def test_windows_installer_runs_autostart_repair_after_success(self):
        from unittest.mock import MagicMock
        from hermes_cli import tools_config

        captured = {}
        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return fake_proc

        def fake_which(name: str):
            if name == "cua-driver":
                return r"C:\Users\Ha Trung\AppData\Local\Programs\Cua\cua-driver\bin\cua-driver.exe"
            return None

        with patch("platform.system", return_value="Windows"), \
             patch.object(tools_config.shutil, "which", side_effect=fake_which), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_repair_cua_driver_autostart_windows", return_value=True) as repair, \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"), \
             patch.object(tools_config, "_print_success"):
            ok = tools_config._run_cua_driver_installer(label="Refreshing", verbose=False)

        assert ok is True
        assert captured["kwargs"].get("shell") is False
        assert isinstance(captured["cmd"], list)
        assert captured["cmd"][:4] == [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        ]
        repair.assert_called_once_with("cua-driver", verbose=False)

    def test_autostart_repair_quotes_username_space_path_via_file_path(self):
        from hermes_cli import tools_config

        calls = []
        driver = (
            r"C:\Users\Ha Trung\AppData\Local\Programs\Cua"
            r"\cua-driver\bin\cua-driver.exe"
        )

        def fake_which(name: str):
            if name == "cua-driver":
                return driver
            if name == "powershell":
                return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            return None

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[0] == "schtasks.exe":
                return SimpleNamespace(returncode=1)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(tools_config.sys, "platform", "win32"), \
             patch.object(tools_config.shutil, "which", side_effect=fake_which), \
             patch("subprocess.run", side_effect=fake_run), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            ok = tools_config._repair_cua_driver_autostart_windows(
                "cua-driver", verbose=False
            )

        assert ok is True
        ps_calls = [cmd for cmd, _kwargs in calls if cmd[0].endswith("powershell.exe")]
        assert len(ps_calls) == 1
        ps_command = ps_calls[0][-1]
        assert "-FilePath $exe" in ps_command
        assert "-ArgumentList @('autostart','enable')" in ps_command
        assert f"$exe = '{driver}'" in ps_command
        assert f"& {driver}" not in ps_command
