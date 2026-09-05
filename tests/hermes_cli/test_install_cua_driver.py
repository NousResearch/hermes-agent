"""Tests for ``install_cua_driver`` upgrade semantics.

The cua-driver upstream installer always pulls the latest release tag, so
re-running it is the canonical upgrade path. ``install_cua_driver(upgrade=True)``
must:

* Be supported-platform-only — no-op silently elsewhere so ``hermes update``
  can call it unconditionally without warning unsupported-platform users.
* Re-run the installer even when the binary is already on PATH (this is the
  fix for the "we only pulled cua-driver once on enable" complaint).
* For ``upgrade=False``, keep compatible installations, repair old or
  incomplete installations, and install when missing.

The pre-install arch probe that used to live alongside this function was
deleted (see top-of-file comment in tools_config.py) — the upstream
installer has CUA_DRIVER_RS_BAKED_VERSION baked in by CD and errors
cleanly on missing-arch assets, and the upgrade path uses
``cua_driver_update_check()`` (which shells `cua-driver check-update
--json` against the already-installed binary).
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _runtime_manifest(version="0.20.0", *, omit=None):
    omit = set(omit or ())
    required = {
        "mcp": {"--socket", "--grant"},
        "serve": {
            "--socket",
            "--permission-mode",
            "--capability-manifest",
            "--approve-capability-manifest",
            "--embedded",
        },
        "stop": {"--socket"},
    }
    return {
        "binary_version": version,
        "mcp_invocation": {"command": "/opt/cua-driver", "args": ["mcp"]},
        "subcommands": [
            {
                "name": command,
                "args": [
                    {"name": arg}
                    for arg in sorted(args - omit)
                ],
            }
            for command, args in required.items()
        ],
    }


class TestCuaDriverRuntimeContract:
    def test_current_manifest_is_ready(self):
        from hermes_cli import tools_config

        result = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_runtime_manifest()),
            stderr="",
        )
        with patch("subprocess.run", return_value=result):
            state = tools_config._cua_driver_contract_status("/opt/cua-driver")

        assert state == {
            "ready": True,
            "binary": "/opt/cua-driver",
            "version": "0.20.0",
            "reason": "",
        }

    @pytest.mark.parametrize("version", ["0.19.4", "bad-version"])
    def test_old_or_unversioned_driver_needs_repair(self, version):
        from hermes_cli import tools_config

        result = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_runtime_manifest(version)),
            stderr="",
        )
        with patch("subprocess.run", return_value=result):
            state = tools_config._cua_driver_contract_status("/opt/cua-driver")

        assert state["ready"] is False
        assert state["reason"]

    def test_incomplete_manifest_needs_repair(self):
        from hermes_cli import tools_config

        result = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                _runtime_manifest(omit={"--approve-capability-manifest"})
            ),
            stderr="",
        )
        with patch("subprocess.run", return_value=result):
            state = tools_config._cua_driver_contract_status("/opt/cua-driver")

        assert state["ready"] is False
        assert "serve --approve-capability-manifest" in state["reason"]


class TestInstallCuaDriverUpgrade:
    # ``install_cua_driver`` supports macOS, Windows AND Linux. For everything
    # below except the two unsupported-platform cases, the Linux host takes a
    # byte-identical path to macOS — same ``fetch_tool`` ("curl"), same
    # ``_cua_install_target_writable()`` verdict, same branch — so the old
    # ``patch("platform.system", return_value="Darwin")`` bought nothing but a
    # fake host. Dropped, and the names no longer claim macOS.

    def test_upgrade_on_unsupported_platform_is_silent_noop(self):
        """The one branch no CI runner can reach for real.

        ``platform.system`` is still faked here, deliberately and narrowly: we
        run Linux/macOS/Windows lanes, and every one of them is a *supported*
        platform, so the refusal path is unreachable on all three. The fake is
        sound because the function returns before touching any OS facility —
        no subprocess, no path handling, no import — so there is nothing
        underneath the branch for a real host to falsify.
        """
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="FreeBSD"):
            assert tools_config.install_cua_driver(upgrade=True) is False
            warn.assert_not_called()

    def test_non_upgrade_on_unsupported_platform_warns(self):
        """Same narrow exception as above — see that test's docstring."""
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="FreeBSD"):
            assert tools_config.install_cua_driver(upgrade=False) is False
            warn.assert_called()

    def test_upgrade_with_binary_present_runs_installer(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 return_value={"ready": True, "version": "0.20.0", "reason": ""},
             ), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()
            kwargs = runner.call_args.kwargs
            assert kwargs.get("verbose") is False

    def test_upgrade_without_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()

    @pytest.mark.linux_only
    def test_quiet_refresh_prints_single_contextual_progress_line(self):
        """``linux_only``: reaches Popen through the POSIX download-then-exec
        branch, which this lane takes for real."""
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        with patch(
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

    @pytest.mark.linux_only
    def test_quiet_refresh_can_suppress_progress_line(self):
        """``linux_only``: same POSIX Popen path as the test above."""
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        fake_proc = MagicMock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = ("", None)

        with patch(
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

        with patch.object(
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
                 "_cua_driver_contract_status",
                 return_value={"ready": True, "version": "0.20.0", "reason": ""},
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

    def test_upgrade_non_writable_install_target_skips_refresh(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
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

    def test_fresh_install_non_writable_install_target_skips_install(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
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

    @pytest.mark.macos_only
    def test_install_target_writability_is_probed_for_real_on_macos(self):
        """The ``_cua_install_target_writable`` seam the two tests above patch.

        ``macos_only``: ``/Applications`` is the only install target Hermes
        checks, and the probe short-circuits to True on every other platform —
        so this is the one host where the real filesystem answer means
        anything.
        """
        import os

        from hermes_cli import tools_config

        writable = tools_config._cua_install_target_writable()
        if os.path.isdir("/Applications"):
            assert writable is os.access("/Applications", os.W_OK)
        else:
            assert writable is True

    def test_non_upgrade_with_binary_skips_install(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner, \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 return_value={"ready": True, "version": "0.20.0", "reason": ""},
             ), \
             patch.object(
                 tools_config,
                 "_repair_cua_driver_autostart_windows",
                 return_value=True,
             ), \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_not_called()

    def test_non_upgrade_repairs_incompatible_existing_driver(self):
        from hermes_cli import tools_config

        incompatible = {
            "ready": False,
            "version": "0.19.4",
            "reason": "Hermes computer use requires cua-driver 0.20.0 or newer",
        }
        repaired = {"ready": True, "version": "0.20.0", "reason": ""}
        with patch.object(
                 tools_config.shutil,
                 "which",
                 side_effect=lambda name: f"/usr/bin/{name}",
             ), \
             patch.object(
                 tools_config,
                 "_resolved_cua_driver_cmd",
                 return_value="/usr/bin/cua-driver",
             ), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 side_effect=[incompatible, repaired],
             ), \
             patch.object(
                 tools_config,
                 "_run_cua_driver_installer",
                 return_value=True,
             ) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True

        assert runner.call_args.kwargs["label"] == "Repairing"

    def test_incompatible_explicit_override_is_not_replaced(self, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", "/opt/custom/cua-driver")
        incompatible = {
            "ready": False,
            "version": "0.19.4",
            "reason": "Hermes computer use requires cua-driver 0.20.0 or newer",
        }
        with patch.object(
                 tools_config,
                 "_resolved_cua_driver_cmd",
                 return_value="/opt/custom/cua-driver",
             ), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 return_value=incompatible,
             ), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner:
            assert tools_config.install_cua_driver(upgrade=False) is False

        runner.assert_not_called()

    @pytest.mark.parametrize("upgrade", [False, True])
    def test_missing_explicit_override_does_not_install_standard_driver(
        self, monkeypatch, upgrade
    ):
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", "/missing/custom/cua-driver")
        with patch.object(
                 tools_config,
                 "_resolved_cua_driver_cmd",
                 return_value=None,
             ), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner:
            assert tools_config.install_cua_driver(upgrade=upgrade) is False

        runner.assert_not_called()

    def test_non_upgrade_without_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
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

    def _install(self, check_state, require_confirmed):
        """Drive ``install_cua_driver`` on the host, whatever it is.

        The old signature took a ``system`` string and faked
        ``platform.system`` with it, so callers picked "Windows"/"Darwin"
        arbitrarily. Nothing in the confirmed-update gate is
        platform-dependent — it's ``check-update`` state plus a flag — so the
        fake only decided which lie the test told itself. ``which`` answers
        for every fetch tool so the host's own branch resolves cleanly.
        """
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/x/" + n
                          if n in {"cua-driver", "curl", "powershell"} else None), \
             patch.object(tools_config, "_resolved_cua_driver_cmd",
                          return_value="/x/cua-driver"), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=True), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 return_value={"ready": True, "version": "0.20.0", "reason": ""},
             ), \
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
        ok, runner, info = self._install(None, require_confirmed=True)
        assert ok is True
        runner.assert_not_called()
        assert any(
            "keeping the installed version" in call.args[0]
            for call in info.call_args_list
        )

    def test_indeterminate_check_points_at_force_path(self):
        ok, runner, info = self._install(None, require_confirmed=True)
        assert ok is True
        runner.assert_not_called()
        assert any(
            "computer-use install --upgrade" in call.args[0]
            for call in info.call_args_list
        )

    def test_confirmed_update_still_runs_installer(self):
        state = {"current_version": "0.5.0", "latest_version": "0.6.0",
                 "update_available": True}
        ok, runner, _ = self._install(state, require_confirmed=True)
        assert ok is True
        runner.assert_called_once()

    def test_up_to_date_short_circuits(self):
        state = {"current_version": "0.6.0", "latest_version": "0.6.0",
                 "update_available": False}
        ok, runner, _ = self._install(state, require_confirmed=True)
        assert ok is True
        runner.assert_not_called()

    def test_explicit_upgrade_still_falls_through_on_indeterminate(self):
        # `hermes computer-use install --upgrade` (default flag): the old
        # behaviour — indeterminate check re-runs the installer.
        ok, runner, _ = self._install(None, require_confirmed=False)
        assert ok is True
        runner.assert_called_once()

    def test_incompatible_driver_repairs_despite_indeterminate_check(self):
        """Hermes' own version floor is the confirmation. When the installed
        driver fails the runtime contract, the `hermes update` refresh must
        repair it even though ``check-update`` can't confirm a newer release
        (its ~20h cache routinely lags a same-day floor bump — the 0.19.3
        wedge)."""
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        incompatible = {
            "ready": False,
            "version": "0.19.3",
            "reason": "Hermes computer use requires cua-driver 0.20.0 or newer",
        }
        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/x/" + n
                          if n in {"cua-driver", "curl", "powershell"} else None), \
             patch.object(tools_config, "_resolved_cua_driver_cmd",
                          return_value="/x/cua-driver"), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=True), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 side_effect=[incompatible,
                              {"ready": True, "version": "0.20.0", "reason": ""}],
             ), \
             patch("tools.computer_use.cua_backend.cua_driver_update_check",
                   return_value=None) as check, \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run",
                   return_value=MagicMock(stdout="cua-driver 0.19.3",
                                          returncode=0)), \
             patch.object(tools_config, "_print_success"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            ok = tools_config.install_cua_driver(
                upgrade=True, require_confirmed_update=True
            )

        assert ok is True
        runner.assert_called_once()
        assert runner.call_args.kwargs["label"] == "Repairing"
        # The confirmed-update gate must not even consult check-update:
        # the contract failure already confirmed the need.
        check.assert_not_called()


class TestUpdateCheckTimeoutDefaults:
    """cua_driver_update_check: platform-sensitive default timeout.

    8s is fine on POSIX but too tight for Windows first-spawn (Defender /
    SmartScreen scanning), and a false timeout is what used to trigger the
    full reinstall fall-through during `hermes update`.
    """

    def _captured_timeout(self):
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
             patch("tools.computer_use.cua_backend.subprocess.run",
                   side_effect=fake_run):
            cua_backend.cua_driver_update_check()
        return captured.get("timeout")

    @pytest.mark.windows_only
    def test_windows_default_is_generous(self):
        """``windows_only``: the 25s default exists because a real Windows
        first-spawn is delayed by Defender/SmartScreen scanning — a faked
        platform asserted the constant, never the host it is chosen for.
        """
        assert self._captured_timeout() == 25.0

    def test_posix_default_unchanged(self):
        # Unmarked: the POSIX default is what this (Linux) host already picks,
        # so no platform faking is involved.
        assert self._captured_timeout() == 8.0

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

        # No platform fake: "does Python hit the GitHub API?" is host-agnostic,
        # and ``which`` is stubbed so the host's own fetch tool resolves.
        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/" + n
                                                 if n in ("curl", "powershell") else None), \
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

        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in ("cua-driver", "curl", "powershell") else None), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 return_value={"ready": True, "version": "0.20.0", "reason": ""},
             ), \
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
    @pytest.mark.windows_only
    def test_windows_branch_uses_file_lock_probe(self):
        """``windows_only``: which lock protocol applies IS the host fact under
        test — on Linux the faked platform asserted the dispatch and skipped
        the ``.install.lock.d`` directory that really exists here.
        """
        from hermes_cli import tools_config

        with patch.object(
                 tools_config, "_clear_stale_windows_cua_install_lock"
             ) as clear_windows:
            tools_config._clear_stale_cua_install_lock()

        clear_windows.assert_called_once_with()


# ``windows_only`` rather than ``skipif(sys.platform != "win32")``: the
# dedicated Windows CI job selects ``-m windows_only``, so a bare skipif left
# these real-CreateFileW tests running on no host at all.
@pytest.mark.windows_only
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
    `curl | bash` grandchildren can't survive holding the install lock.

    The POSIX cases drop the old ``platform.system`` → "Linux" fake: this lane
    IS Linux, so the branch is selected for real. The Windows cases are
    ``windows_only`` — the psutil tree-kill only runs when ``is_windows``, and
    on Linux the fake picked that branch on a host with no such process model.
    """

    @pytest.mark.linux_only
    def test_timeout_kills_process_group_and_returns_false(self):
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

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")), \
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

    @pytest.mark.linux_only
    def test_installer_runs_in_new_session_on_posix(self):
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

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch.object(tools_config, "_clear_stale_cua_install_lock"), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"):
            tools_config._run_cua_driver_installer(label="Refreshing", verbose=False)

        assert captured.get("start_new_session") is True

    @pytest.mark.windows_only
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

        with patch("subprocess.Popen", return_value=fake_proc), \
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

    @pytest.mark.windows_only
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

        with patch("subprocess.Popen", return_value=fake_proc), \
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


@pytest.mark.linux_only
class TestInstallerNoShell:
    """The POSIX installer path must not use shell=True or command
    substitution: the script is downloaded to a mkstemp file and exec'd
    as a plain argv list (salvage of #34974's intent, without the fixed
    /tmp path TOCTOU that PR introduced).

    ``linux_only``: the download-then-exec argv IS the POSIX branch, and this
    lane already takes it — the old ``platform.system`` → "Linux" fake was
    asserting a branch the host had already selected.
    """

    def _run(self, download_rc=0):
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

        with patch("subprocess.run", side_effect=fake_run), \
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

    def test_temp_script_removed_after_run(self):
        import os
        captured = {}
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

        with patch("subprocess.run", side_effect=fake_run), \
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
        """Host-agnostic: version pinning is string handling, not an OS branch.

        The old ``platform.system`` → "Windows" fake was incidental — the
        pin flows into ``CUA_DRIVER_RS_VERSION`` identically on every host
        (both upstream installers honour it), and this test never reaches the
        installer anyway because ``_run_cua_driver_installer`` is mocked.
        """
        from unittest.mock import MagicMock

        from hermes_cli import tools_config

        with patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/x/" + n
                          if n in {"cua-driver", "curl", "powershell"} else None), \
             patch.object(tools_config, "_resolved_cua_driver_cmd",
                          return_value="/x/cua-driver"), \
             patch.object(tools_config, "_cua_install_target_writable",
                          return_value=True), \
             patch.object(
                 tools_config,
                 "_cua_driver_contract_status",
                 return_value={"ready": True, "version": "0.20.0", "reason": ""},
             ), \
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


@pytest.mark.linux_only
class TestRunInstallerPinEnv:
    """_run_cua_driver_installer(pin_version=...) exports CUA_DRIVER_RS_VERSION
    into the installer child env; unpinned runs leave it untouched.

    ``linux_only``: the helper reaches Popen through the POSIX
    download-then-exec branch, which this lane takes for real — no
    ``platform.system`` fake needed. The pin itself is host-agnostic
    (``TestConfirmedVersionPinning`` covers the caller side unmarked).
    """

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

        with patch("subprocess.run", side_effect=fake_run), \
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
    @pytest.mark.windows_only
    def test_existing_task_skips_elevated_powershell_repair(self):
        """``windows_only``: ``_repair_cua_driver_autostart_windows`` returns
        True unconditionally off Windows, so only the fake made the schtasks
        probe run at all.
        """
        from hermes_cli import tools_config

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return SimpleNamespace(returncode=0)

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(tools_config.shutil, "which") as which:
            ok = tools_config._repair_cua_driver_autostart_windows(
                "cua-driver", verbose=False
            )

        assert ok is True
        assert [cmd for cmd, _kwargs in calls] == [
            ["schtasks.exe", "/Query", "/TN", "cua-driver-serve"]
        ]
        which.assert_not_called()

    @pytest.mark.windows_only
    def test_windows_installer_runs_autostart_repair_after_success(self):
        """``windows_only``: the PowerShell install argv and the autostart
        repair hook are both inside the ``is_windows`` branch, so on Linux the
        fake selected a branch whose `powershell` doesn't exist on PATH."""
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

        with patch.object(tools_config.shutil, "which", side_effect=fake_which), \
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
        repair.assert_called_once_with(
            r"C:\Users\Ha Trung\AppData\Local\Programs\Cua\cua-driver\bin\cua-driver.exe",
            verbose=False,
        )

    @pytest.mark.windows_only
    def test_autostart_repair_quotes_username_space_path_via_file_path(self):
        """``windows_only``: same early return off Windows — the elevated
        PowerShell command string is only built on a real Windows host.
        """
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

        with patch.object(tools_config.shutil, "which", side_effect=fake_which), \
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


class TestCuaVersionSummary:
    """`hermes computer-use status` prints one line, whatever the binary says.

    A binary chosen by HERMES_CUA_DRIVER_CMD is under no obligation to answer
    `--version` the way cua-driver does, and its output used to be spliced
    verbatim into the status line.
    """

    @staticmethod
    def _summary(raw, **kw):
        from hermes_cli import tools_config

        return tools_config._cua_version_summary(raw, **kw)

    def test_plain_version_passes_through(self):
        assert self._summary("cua-driver 0.20.0") == "cua-driver 0.20.0"

    def test_multiline_banner_collapses_to_first_line(self):
        banner = (
            "Microsoft Windows [Version 10.0.26200.9168]\n"
            "(c) Microsoft Corporation. All rights reserved.\n"
            "\n"
            "C:\\Users\\demo>"
        )
        summary = self._summary(banner)
        assert summary == "Microsoft Windows [Version 10.0.26200.9168]"
        assert "\n" not in summary

    def test_leading_blank_lines_skipped(self):
        assert self._summary("\n\n  cua-driver 0.20.0  ") == "cua-driver 0.20.0"

    def test_long_line_is_bounded(self):
        assert len(self._summary("x" * 500)) == 120

    def test_empty_output_stays_empty(self):
        assert self._summary("") == ""
        assert self._summary("   \n  \n") == ""
