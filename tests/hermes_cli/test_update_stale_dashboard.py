"""Tests for the stale-dashboard handling run at the end of ``hermes update``.

``hermes update`` detects ``hermes dashboard`` processes left over from the
previous version and kills them (SIGTERM + SIGKILL grace, or ``taskkill /F``
on Windows).  Without this, the running backend silently serves stale Python
against a freshly-updated JS bundle, producing 401s / empty data.

History:
- #16872 introduced the warn-only helper (``_warn_stale_dashboard_processes``).
- #17049 fixed a Windows wmic UnicodeDecodeError crash on non-UTF-8 locales.
- This file now also covers the kill semantics that replaced the warning.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.main_dashboard import _find_stale_dashboard_pids
from hermes_cli.dashboard_procs import _kill_stale_dashboard_processes
from hermes_cli import dashboard_procs
from hermes_cli import main_dashboard
from hermes_cli import update_cmd_maint


def test_dashboard_import_tolerates_stale_cli_output_without_line_input(monkeypatch):
    """Post-update imports must not bind a helper absent from a cached old module."""
    import hermes_cli

    original_dashboard = sys.modules["hermes_cli.main_dashboard"]
    stale_cli_output = types.ModuleType("hermes_cli.cli_output")
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "hermes_cli.cli_output", stale_cli_output)
        isolated.delitem(sys.modules, "hermes_cli.main_dashboard")
        isolated.delattr(hermes_cli, "main_dashboard", raising=False)
        imported = importlib.import_module("hermes_cli.main_dashboard")
        assert callable(imported._line_input)
    sys.modules["hermes_cli.main_dashboard"] = original_dashboard
    hermes_cli.main_dashboard = original_dashboard


@pytest.fixture(autouse=True)
def _refresh_bindings_against_live_module():
    """Rebind module-level names to the *current* defining modules.

    Other tests in the suite reload modules from ``sys.modules``; when that
    happens on the same xdist worker before we run, our top-of-file bindings
    end up pointing at the *old* module object and ``patch("<module>.X")``
    patches the *new* one, so every patch becomes a no-op and the kill path
    silently returns early. Refreshing the bindings keeps them consistent.
    """
    global _find_stale_dashboard_pids
    global _kill_stale_dashboard_processes
    global _restart_managed_dashboard_service

    _find_stale_dashboard_pids = main_dashboard._find_stale_dashboard_pids
    _kill_stale_dashboard_processes = dashboard_procs._kill_stale_dashboard_processes
    _restart_managed_dashboard_service = main_dashboard._restart_managed_dashboard_service
    yield


@pytest.fixture(autouse=True)
def _no_real_launchd_jobs():
    """The kill path now scans the host's launchd jobs on macOS; a developer's own LaunchAgents
    must never leak into these tests. Tests that need jobs patch the scan themselves."""
    with patch.object(main_dashboard, "_loaded_launchd_backend_jobs", return_value=[]):
        yield


def _ps_line(pid: int, cmd: str) -> str:
    """Format a line as it would appear in ``ps -A -o pid=,command=`` output."""
    return f"{pid:>7} {cmd}"


def _ps_runner(stdout: str):
    """Build a subprocess.run side_effect that only stubs ps -A calls.

    Any other subprocess.run invocation (e.g. taskkill on Windows) is
    handed back as a successful no-op.  This lets tests exercise the real
    scan path without having to re-stub every unrelated subprocess call
    made later in ``_kill_stale_dashboard_processes``.
    """
    def _side_effect(args, *a, **kw):
        if isinstance(args, (list, tuple)) and args and args[0] == "ps":
            return MagicMock(returncode=0, stdout=stdout, stderr="")
        # Any other subprocess.run (e.g. taskkill) — benign success stub.
        return MagicMock(returncode=0, stdout="", stderr="")
    return _side_effect


def _write_valid_ssh_backend_lock(tmp_path, monkeypatch) -> int:
    pid = 4242
    ownership_id = "a" * 32
    spawn_nonce = "b" * 16
    lock_dir = tmp_path / "desktop-ssh" / ownership_id
    lock_dir.mkdir(parents=True)
    (lock_dir / "backend.lock.json").write_text(json.dumps({
        "schemaVersion": 2,
        "protocolVersion": 1,
        "ownershipId": ownership_id,
        "spawnNonce": spawn_nonce,
        "tokenFingerprint": "c" * 32,
        "pid": pid,
        "port": 46369,
        "profile": "default",
        "hermesPath": "/opt/hermes/bin/hermes",
        "hermesHome": str(tmp_path),
        "logPath": f"{tmp_path}/desktop-ssh/{ownership_id}/{spawn_nonce}.log",
        "startedAt": "2026-08-21T15:27:39Z",
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    return pid


def test_update_cleanup_spares_backend_owned_by_valid_ssh_lock(tmp_path, monkeypatch):
    pid = _write_valid_ssh_backend_lock(tmp_path, monkeypatch)

    def assert_owned_pid_is_excluded(*, exclude_pids=None, **_):
        assert exclude_pids is not None
        assert pid in exclude_pids
        return []

    with patch(
        "hermes_cli.main_dashboard._find_stale_dashboard_pids",
        side_effect=assert_owned_pid_is_excluded,
    ):
        result = _kill_stale_dashboard_processes(restart_managed=True)

    assert result == {"matched": [], "killed": [], "failed": []}


def test_explicit_stop_does_not_spare_backend_owned_by_valid_ssh_lock(
    tmp_path,
    monkeypatch,
):
    pid = _write_valid_ssh_backend_lock(tmp_path, monkeypatch)

    def assert_owned_pid_is_not_excluded(*, exclude_pids=None, **_):
        assert exclude_pids is None or pid not in exclude_pids
        return []

    with patch(
        "hermes_cli.main_dashboard._find_stale_dashboard_pids",
        side_effect=assert_owned_pid_is_not_excluded,
    ):
        result = _kill_stale_dashboard_processes(restart_managed=False)

    assert result == {"matched": [], "killed": [], "failed": []}


class TestFindStaleDashboardPids:
    """Unit tests for the ps/wmic-based detection step."""



    @pytest.mark.skipif(sys.platform == "win32", reason="ps-based scan path")
    def test_self_pid_excluded(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(os.getpid(), "python3 -m hermes_cli.main dashboard"),
                    _ps_line(12345, "hermes dashboard --port 9119"),
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert os.getpid() not in pids
        assert 12345 in pids


    def _assert_ps_timeout_returns_empty(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ps", 10)):
            assert _find_stale_dashboard_pids() == []

    @pytest.mark.platforms("linux")
    def test_ps_timeout_returns_empty_linux(self):
        self._assert_ps_timeout_returns_empty()

    @pytest.mark.platforms("macos")
    def test_ps_timeout_returns_empty_macos(self):
        self._assert_ps_timeout_returns_empty()




@pytest.mark.skipif(sys.platform == "win32", reason="POSIX kill semantics")
class TestKillStaleDashboardPosix:
    """Kill path on Linux / macOS: SIGTERM then SIGKILL any survivors."""


    def test_sigterm_graceful_exit(self, capsys):
        """Processes that exit on SIGTERM (the probe gets ProcessLookupError)
        are reported as stopped and SIGKILL is never sent."""
        import signal as _signal

        killed_signals: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            killed_signals.append((pid, sig))
            if sig == 0:
                # Probe after SIGTERM → "process gone".
                raise ProcessLookupError
            # SIGTERM itself: succeed silently.

        with patch("hermes_cli.main_dashboard._find_stale_dashboard_pids",
                   return_value=[12345, 12346]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            result = _kill_stale_dashboard_processes()

        # Both got SIGTERM.
        sigterms = [pid for pid, sig in killed_signals if sig == _signal.SIGTERM]
        assert sorted(sigterms) == [12345, 12346]
        # No SIGKILL was needed.
        assert not any(sig == _signal.SIGKILL for _, sig in killed_signals)
        assert result["matched"] == [12345, 12346]
        assert result["killed"] == [12345, 12346]
        assert result["failed"] == []





    def test_user_scope_restart_never_falls_back_to_system_or_sudo(self, capsys):
        """A user unit is discovered and restarted through ``systemctl --user``.

        Since #92145 the managed restart no longer ends the pass — the scan
        for OTHER stale serve/dashboard backends continues (with the restarted
        unit recorded in ``already_restarted_units``), so the scan being
        reached is now part of the contract rather than a violation of it.
        The invariant this test pins is unchanged: nothing here may touch the
        system scope or sudo.
        """
        calls: list[list[str]] = []

        def fake_run(args, *a, **kw):
            calls.append(list(args))
            if args == ["systemctl", "--user", "list-unit-files", "hermes-dashboard.service", "--no-legend", "--no-pager"]:
                return MagicMock(returncode=0, stdout="hermes-dashboard.service enabled enabled\n", stderr="")
            if args == ["systemctl", "--user", "is-active", "hermes-dashboard.service"]:
                return MagicMock(returncode=0, stdout="active\n", stderr="")
            if args == ["systemctl", "--user", "is-enabled", "hermes-dashboard.service"]:
                return MagicMock(returncode=0, stdout="enabled\n", stderr="")
            if args == ["systemctl", "--user", "restart", "hermes-dashboard.service"]:
                return MagicMock(returncode=0, stdout="", stderr="")
            raise AssertionError(f"unexpected subprocess.run call: {args}")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("hermes_cli.main_dashboard._find_stale_dashboard_pids", return_value=[]), \
             patch("os.kill") as kill:
            _kill_stale_dashboard_processes(restart_managed=True)

        assert ["systemctl", "--user", "restart", "hermes-dashboard.service"] in calls
        assert all(call[:1] != ["sudo"] and call[:2] == ["systemctl", "--user"] for call in calls)
        kill.assert_not_called()




class TestKillStaleDashboardWindows:
    """Kill path on Windows: taskkill /F."""

    @pytest.mark.platforms("windows")
    def test_taskkill_invoked_for_each_pid(self, capsys):
        """``platforms("windows")``: ``taskkill.exe`` only exists on Windows, and the
        faked platform also silently skipped the POSIX-only cgroup/argv
        snapshot the real Windows path must not take.
        """

        def fake_run(args, *a, **kw):
            # taskkill returns 0 on success
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("hermes_cli.main_dashboard._find_stale_dashboard_pids",
                   return_value=[12345, 12346]), \
             patch("gateway.status.get_process_start_time", return_value=123), \
             patch("hermes_cli._subprocess_compat.pid_is_hermes", return_value=True), \
             patch("subprocess.run", side_effect=fake_run) as mock_run:
            _kill_stale_dashboard_processes()

        # Each PID triggered a taskkill /PID <n> /F invocation.
        taskkill_calls = [
            c for c in mock_run.call_args_list
            if c.args and isinstance(c.args[0], list) and c.args[0][:1] == ["taskkill"]
        ]
        assert len(taskkill_calls) == 2
        assert ["taskkill", "/PID", "12345", "/F"] in [c.args[0] for c in taskkill_calls]
        assert ["taskkill", "/PID", "12346", "/F"] in [c.args[0] for c in taskkill_calls]




class TestDashboardUpdateCleanup:
    """The git and Windows ZIP update paths share this final cleanup."""

    def test_all_failed_stops_do_not_claim_the_dashboard_was_stopped(self, capsys, monkeypatch, tmp_path):
        own_home = tmp_path / "profiles" / "work"
        monkeypatch.setenv("HERMES_HOME", str(own_home))
        with patch(
            "hermes_cli.main._kill_stale_dashboard_processes",
            return_value={"matched": [12345], "killed": [], "failed": [(12345, "denied")],
                          "unrecovered": []},
        ) as kill:
            update_cmd_maint._refresh_dashboard_after_update()

        # The sweep only touches this home's backends (#113978).
        assert kill.call_args.kwargs["scope_home"] == str(own_home)
        assert "stopped during update" not in capsys.readouterr().out




@pytest.mark.skipif(sys.platform == "win32", reason="POSIX kill + systemd restart")
class TestSupervisedBackendRestart:
    """After the kill, systemd-supervised PIDs get their owning unit
    restarted (#68934) — SIGTERM reads as a clean stop to systemd, so
    Restart=on-failure never fires on its own."""

    def _live(self):
        return main_dashboard

    def test_supervised_pid_restarts_owning_unit(self, capsys):
        """A killed PID whose cgroup names a custom unit → systemctl restart."""
        live = self._live()

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError

        with patch.object(main_dashboard, "_restart_managed_dashboard_service", return_value=False), \
             patch.object(live, "_find_stale_dashboard_pids", return_value=[4321]), \
             patch.object(main_dashboard, "_get_pid_cgroup_path",
                          return_value="/system.slice/hermes-serve.service"), \
             patch.object(main_dashboard, "_get_systemd_service_for_pid",
                          return_value="hermes-serve.service"), \
             patch.object(main_dashboard, "_try_restart_systemd_service", return_value=True) as restart, \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes(restart_managed=True)

        restart.assert_called_once_with(
            "hermes-serve.service", "/system.slice/hermes-serve.service"
        )
        # Supervised restart succeeded — no manual hint.
        assert "when you're ready" not in capsys.readouterr().out

    def test_already_restarted_unit_is_left_untouched(self):
        """Review on #83595: hermes update's systemd fleet-restart loop may
        already have restarted this PID's owning unit directly (e.g. a
        Serve-only install). Passing it via already_restarted_units must
        skip killing/restarting it again here."""
        live = self._live()

        with patch.object(main_dashboard, "_restart_managed_dashboard_service", return_value=False), \
             patch.object(live, "_find_stale_dashboard_pids", return_value=[4321]), \
             patch.object(main_dashboard, "_get_pid_cgroup_path",
                          return_value="/system.slice/hermes-serve.service"), \
             patch.object(main_dashboard, "_get_systemd_service_for_pid",
                          return_value="hermes-serve.service"), \
             patch.object(main_dashboard, "_try_restart_systemd_service") as restart, \
             patch("os.kill") as kill, \
             patch("time.sleep"):
            result = _kill_stale_dashboard_processes(
                restart_managed=True, already_restarted_units={"hermes-serve"}
            )

        kill.assert_not_called()
        restart.assert_not_called()
        assert result == {"matched": [], "killed": [], "failed": []}


class TestManualBackendRespawn:
    """Manually-started dashboards/serves have their argv captured before the
    kill and are respawned detached after the update (#40449)."""

    def _live(self):
        return main_dashboard


    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX cmdline capture + respawn")
    def test_argv_capture_failure_falls_back_to_hint(self, capsys):
        live = self._live()

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError

        with patch.object(main_dashboard, "_restart_managed_dashboard_service", return_value=False), \
             patch.object(live, "_find_stale_dashboard_pids", return_value=[5555]), \
             patch.object(main_dashboard, "_get_pid_cgroup_path", return_value=None), \
             patch.object(main_dashboard, "_get_systemd_service_for_pid", return_value=None), \
             patch.object(main_dashboard, "_dashboard_cmdline_for_pid", return_value=None), \
             patch.object(live, "_respawn_dashboard_processes") as respawn, \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            result = _kill_stale_dashboard_processes(restart_managed=True)

        respawn.assert_not_called()
        assert result["unrecovered"] == [5555]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX cmdline capture + respawn")
    def test_non_orphan_fixed_port_still_respawns(self, capsys):
        """A supervised-by-shell dashboard with a fixed port is still restarted."""
        live = self._live()
        argv = ["hermes", "dashboard", "--port", "8300"]

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError

        with patch.object(main_dashboard, "_restart_managed_dashboard_service", return_value=False), \
             patch.object(live, "_find_stale_dashboard_pids", return_value=[6001]), \
             patch.object(main_dashboard, "_get_pid_cgroup_path", return_value=None), \
             patch.object(main_dashboard, "_get_systemd_service_for_pid", return_value=None), \
             patch.object(main_dashboard, "_dashboard_cmdline_for_pid", return_value=argv), \
             patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=None), \
             patch.object(live, "_respawn_dashboard_processes", return_value=[]) as respawn, \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes(restart_managed=True)

        respawn.assert_called_once_with([argv])
        assert "when you're ready" not in capsys.readouterr().out

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX cmdline capture + respawn")
    def test_port_zero_serves_killed_without_respawn(self, capsys):
        """``serve --port 0`` backends are stopped but not resurrected (#78821)."""
        live = self._live()
        argv = [
            "python", "-m", "hermes_cli.main",
            "serve", "--host", "127.0.0.1", "--port", "0",
        ]

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError

        with patch.object(main_dashboard, "_restart_managed_dashboard_service", return_value=False), \
             patch.object(live, "_find_stale_dashboard_pids",
                          return_value=[7001, 7002, 7003]), \
             patch.object(main_dashboard, "_get_pid_cgroup_path", return_value=None), \
             patch.object(main_dashboard, "_get_systemd_service_for_pid", return_value=None), \
             patch.object(main_dashboard, "_dashboard_cmdline_for_pid", return_value=argv), \
             patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=None), \
             patch.object(live, "_respawn_dashboard_processes") as respawn, \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            result = _kill_stale_dashboard_processes(restart_managed=True)

        respawn.assert_not_called()
        assert sorted(result["killed"]) == [7001, 7002, 7003]
        # Intentional skips are not "unrecovered" — no noisy manual hint.
        assert result["unrecovered"] == []
        assert "when you're ready" not in capsys.readouterr().out


    def test_respawn_adds_no_open_to_dashboard_commands(self, tmp_path, monkeypatch):
        """Respawned `dashboard` argv gains --no-open; `serve` argv untouched."""
        live = self._live()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        spawned: list[list[str]] = []

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                spawned.append(list(cmd))

        with patch.object(live.subprocess, "Popen", _FakePopen):
            failed = live._respawn_dashboard_processes([
                ["hermes", "dashboard", "--port", "8300"],
                ["hermes", "serve", "--host", "0.0.0.0"],
            ])

        assert failed == []
        assert spawned[0] == ["hermes", "dashboard", "--port", "8300", "--no-open"]
        assert spawned[1] == ["hermes", "serve", "--host", "0.0.0.0"]

    def test_respawn_failure_returned(self, tmp_path, monkeypatch, capsys):
        live = self._live()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        with patch.object(live.subprocess, "Popen", side_effect=OSError("no such file")):
            failed = live._respawn_dashboard_processes([["hermes", "serve"]])

        assert failed == [["hermes", "serve"]]


class TestFilterDashboardRespawnCandidates:
    """Unit tests for respawn filtering / dedupe / orphan skip (#78821)."""

    def test_skips_serve_port_zero(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        argv = [
            "python", "-m", "hermes_cli.main",
            "--profile", "mini-cat",
            "serve", "--host", "127.0.0.1", "--port", "0",
        ]
        assert _filter_dashboard_respawn_candidates([
            (42, argv, "/home/u/.hermes/profiles/mini-cat"),
        ]) == []

    def test_skips_legacy_dashboard_port_zero(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        argv = [
            "hermes", "--profile", "coder",
            "dashboard", "--no-open", "--host", "127.0.0.1", "--port", "0",
        ]
        assert _filter_dashboard_respawn_candidates([(7, argv, None)]) == []

    def test_skips_serve_port_equals_zero(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        argv = ["hermes", "serve", "--port=0"]
        assert _filter_dashboard_respawn_candidates([(1, argv, None)]) == []


    def test_dedupes_identical_normalized_cmdlines(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        a = ["/usr/bin/python3", "-m", "hermes_cli.main", "dashboard", "--port", "8300"]
        b = ["/other/python", "-m", "hermes_cli.main", "dashboard", "--port", "8300"]
        out = _filter_dashboard_respawn_candidates([
            (1, a, None),
            (2, b, None),
        ])
        assert out == [a]

    def test_caps_one_per_profile(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        a = ["hermes", "--profile", "coder", "dashboard", "--port", "8300"]
        b = ["hermes", "--profile", "coder", "dashboard", "--port", "8301"]
        c = ["hermes", "--profile", "writer", "dashboard", "--port", "8302"]
        out = _filter_dashboard_respawn_candidates([
            (1, a, None),
            (2, b, None),
            (3, c, None),
        ])
        assert out == [a, c]

    def test_caps_one_per_hermes_home(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        home = "/tmp/hermes-home-a"
        a = ["hermes", "dashboard", "--port", "8300"]
        b = ["hermes", "dashboard", "--port", "8301"]
        out = _filter_dashboard_respawn_candidates(
            [
                (1, a, home),
                (2, b, home),
            ],
            own_home=home,
        )
        assert out == [a]

    def test_profile_flag_and_profiles_home_share_cap(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        a = ["hermes", "--profile", "coder", "dashboard", "--port", "8300"]
        b = ["hermes", "dashboard", "--port", "8301"]
        out = _filter_dashboard_respawn_candidates([
            (1, a, None),
            (2, b, "/home/u/.hermes/profiles/coder"),
        ])
        assert out == [a]

    def test_default_profile_same_root_home_caps(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        a = ["hermes", "--profile", "default", "dashboard", "--port", "8300"]
        b = ["hermes", "dashboard", "--port", "8301"]
        home = "/home/u/.hermes"
        out = _filter_dashboard_respawn_candidates(
            [
                (1, a, home),
                (2, b, home),
            ],
            own_home=home,
        )
        assert out == [a]

    def test_foreign_home_backend_is_not_replayed(self):
        """A backend from another HERMES_HOME is never respawned (#94030).

        Its supervisor/user owns its lifecycle; an argv-only replay would
        run on the updating install's home and steal the foreign install's
        fixed port.  Supersedes the old "distinct homes don't share a cap"
        pin — a foreign home is no longer replayed at all.
        """
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        a = ["hermes", "dashboard", "--port", "8300"]
        b = ["hermes", "dashboard", "--port", "8301"]
        out = _filter_dashboard_respawn_candidates(
            [
                (1, a, "/home/u/.hermes"),
                (2, b, "/work/project/.hermes"),
            ],
            own_home="/home/u/.hermes",
        )
        assert out == [a]

    def test_skips_sidecar_fixed_port_serve_on_foreign_home(self):
        """The reported case: launchd-supervised sidecar serve, fixed port."""
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        argv = [
            "/Users/u/.hermes-sidecar/hermes-agent/venv/bin/python",
            "-m", "hermes_cli.main",
            "serve", "--host", "127.0.0.1", "--port", "9118", "--skip-build",
        ]
        out = _filter_dashboard_respawn_candidates(
            [(15364, argv, "/Users/u/.hermes-lifeos")],
            own_home="/Users/u/.hermes",
        )
        assert out == []

    def test_matching_hermes_home_is_kept(self):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        argv = ["hermes", "serve", "--host", "127.0.0.1", "--port", "9118"]
        out = _filter_dashboard_respawn_candidates(
            [(15364, argv, "/Users/u/.hermes")],
            own_home="/Users/u/.hermes",
        )
        assert out == [argv]

    def test_symlinked_hermes_home_compares_equal(self, tmp_path):
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        real = tmp_path / "real-home"
        real.mkdir()
        link = tmp_path / "linked-home"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")

        argv = ["hermes", "serve", "--port", "9118"]
        out = _filter_dashboard_respawn_candidates(
            [(15364, argv, str(real))],
            own_home=str(link),
        )
        assert out == [argv]

    def test_unknown_home_stays_eligible(self):
        """Unreadable HERMES_HOME (env probe failed) keeps pre-#94030 behaviour."""
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        argv = ["hermes", "dashboard", "--port", "8300"]
        out = _filter_dashboard_respawn_candidates(
            [(1, argv, None)],
            own_home="/home/u/.hermes",
        )
        assert out == [argv]

    def test_own_home_defaults_to_get_hermes_home(self, monkeypatch):
        from pathlib import Path

        import hermes_constants
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates

        monkeypatch.setattr(
            hermes_constants, "get_hermes_home", lambda: Path("/home/u/.hermes")
        )
        argv = ["hermes", "serve", "--port", "9118"]
        foreign = _filter_dashboard_respawn_candidates([
            (1, argv, "/Users/u/.hermes-lifeos"),
        ])
        assert foreign == []
        own = _filter_dashboard_respawn_candidates([
            (1, argv, "/home/u/.hermes"),
        ])
        assert own == [argv]




class TestCmdlineCapture:
    """_dashboard_cmdline_for_pid reads /proc on Linux, ps on macOS."""

    def _live(self):
        return main_dashboard

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX /proc cmdline path")
    def test_reads_proc_cmdline_when_available(self, tmp_path, monkeypatch):
        live = self._live()
        proc_file = tmp_path / "cmdline"
        proc_file.write_bytes(b"/usr/bin/python3\x00-m\x00hermes_cli.main\x00serve\x00")

        real_exists = os.path.exists

        def fake_exists(path):
            if path == "/proc/777/cmdline":
                return True
            return real_exists(path)

        real_open = open

        def fake_open(path, *a, **kw):
            if path == "/proc/777/cmdline":
                return real_open(proc_file, *a, **kw)
            return real_open(path, *a, **kw)

        with patch.object(live.os.path, "exists", fake_exists), \
             patch("builtins.open", fake_open):
            argv = main_dashboard._dashboard_cmdline_for_pid(777)

        assert argv == ["/usr/bin/python3", "-m", "hermes_cli.main", "serve"]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX ps cmdline fallback")
    def test_falls_back_to_ps_without_proc(self, monkeypatch):
        live = self._live()

        def fake_run(args, *a, **kw):
            assert args == ["ps", "-p", "888", "-o", "command="]
            return MagicMock(returncode=0, stdout="hermes serve --port 8300\n", stderr="")

        with patch.object(live.os.path, "exists", return_value=False), \
             patch("subprocess.run", side_effect=fake_run):
            argv = main_dashboard._dashboard_cmdline_for_pid(888)

        assert argv == ["hermes", "serve", "--port", "8300"]

    @pytest.mark.platforms("windows")
    def test_returns_none_on_windows(self):
        """``platforms("windows")``: the contract is "no graceful-argv capture on a
        real Windows host" — asserting it against a faked platform only
        restated the branch condition.
        """
        assert main_dashboard._dashboard_cmdline_for_pid(123) is None


class TestPostUpdateDashboardCleanupIsolation:
    """Post-update dashboard cleanup is one isolated step among the others."""

    def test_cleanup_failure_is_isolated_and_recorded(self, capsys):
        """A failure inside the dashboard scan (#112604) must not abort the fleet-verification
        tail (matrix, reconciliation, inner receipt finalize): contained, visible, recorded as
        a failed step on the open receipt."""
        import hermes_cli.update_receipt as ur

        ur._current.set(None)
        try:
            ur.begin_update_receipt()
            with patch(
                "hermes_cli.main._kill_stale_dashboard_processes",
                side_effect=AttributeError(
                    "module 'hermes_cli.main_dashboard' has no attribute '_loaded_launchd_backend_jobs'"
                ),
            ):
                update_cmd_maint._refresh_dashboard_after_update()  # must not raise

            steps = {s["name"]: s for s in ur._current.get().data["steps"]}
        finally:
            ur._current.set(None)

        assert steps["dashboard_cleanup"]["ok"] is False
        assert "_loaded_launchd_backend_jobs" in steps["dashboard_cleanup"]["detail"]
        assert "_loaded_launchd_backend_jobs" in capsys.readouterr().out


class TestLaunchdSupervisedBackends:
    """macOS (#111689): a backend supervised by a launchd job must come back through launchd. Respawning
    its argv detached leaves a copy holding the job's port, so every KeepAlive restart of the job
    fails with "port already in use" and the backend that IS running is no longer supervised."""

    ARGV = [
        "/opt/hermes/venv/bin/python", "-m", "hermes_cli.main",
        "dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open", "--skip-build",
    ]

    @staticmethod
    def _fake_kill(pid, sig):
        if sig == 0:
            raise ProcessLookupError

    def _run(self, pid, jobs, *, restart_ok=True, restart_managed=True):
        live = main_dashboard
        with patch.object(main_dashboard, "_restart_managed_dashboard_service", return_value=False), \
             patch.object(live, "_find_stale_dashboard_pids", return_value=[pid]), \
             patch.object(main_dashboard, "_get_pid_cgroup_path", return_value=None), \
             patch.object(main_dashboard, "_get_systemd_service_for_pid", return_value=None), \
             patch.object(main_dashboard, "_dashboard_cmdline_for_pid", return_value=list(self.ARGV)), \
             patch.object(main_dashboard, "_loaded_launchd_backend_jobs", return_value=jobs), \
             patch.object(main_dashboard, "_restart_launchd_job", return_value=restart_ok) as restart, \
             patch("hermes_cli.dashboard_procs._process_ancestors", return_value=[]), \
             patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=None), \
             patch.object(live, "_respawn_dashboard_processes", return_value=[]) as respawn, \
             patch("os.kill", side_effect=self._fake_kill), \
             patch("time.sleep"):
            result = _kill_stale_dashboard_processes(restart_managed=restart_managed)
        return result, restart, respawn

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX cmdline capture + respawn")
    def test_launchd_owned_backend_restarts_through_launchd_never_as_a_detached_respawn(self, capsys):
        """The reporter's state: the job is loaded but has no live process (it keeps failing on the
        port) and a detached copy runs its exact ProgramArguments. The copy is stopped and the JOB
        is kickstarted; an argv respawn would recreate the port conflict. A failed kickstart counts
        the PID as unrecovered and prints the manual command; a loaded job with a different argv
        and PID leaves the manual-backend respawn untouched (control)."""
        job = ("system", "ai.hermes.dashboard", list(self.ARGV), None)
        result, restart, respawn = self._run(9102, [job])
        respawn.assert_not_called()
        restart.assert_called_once_with("system", "ai.hermes.dashboard", None)
        assert result["killed"] == [9102] and result["unrecovered"] == []
        assert "when you're ready" not in capsys.readouterr().out

        result, restart, respawn = self._run(9103, [("gui/501", "ai.hermes.dashboard", list(self.ARGV), 9103)],
                                             restart_ok=False)
        respawn.assert_not_called()
        assert result["unrecovered"] == [9103]
        assert "run: launchctl kickstart -k gui/501/ai.hermes.dashboard" in capsys.readouterr().out

        # A LaunchDaemon lives in the system domain: kickstart needs root, so a non-root hint says sudo.
        with patch("hermes_cli.dashboard_procs.os.geteuid", return_value=501, create=True):  # windows-footgun: ok — patch target string, create=True
            self._run(9105, [("system", "ai.hermes.serve", list(self.ARGV), None)], restart_ok=False)
        assert "run: sudo launchctl kickstart -k system/ai.hermes.serve" in capsys.readouterr().out

        other = ("gui/501", "ai.hermes.other", ["hermes", "dashboard", "--port", "8300"], 777)
        result, restart, respawn = self._run(9104, [other])
        restart.assert_not_called()
        respawn.assert_called_once_with([list(self.ARGV)])
        assert result["unrecovered"] == []

    def test_launchd_job_attribution_is_by_live_pid_ancestor_or_exact_argv(self):
        """A PID is attributed to a loaded launchd backend job by launchd's live PID, by a live-PID
        ancestor (exec-less ``/bin/sh -c`` wrapper plist) or by exact argv (the detached copy),
        never otherwise."""
        uid = 501
        backend_argv = ["/opt/hermes/venv/bin/python", "-m", "hermes_cli.main", "dashboard", "--port", "9119"]
        serve_argv = ["/opt/hermes/venv/bin/python", "-m", "hermes_cli.main", "serve", "--port", "8642"]
        jobs = [
            (f"user/{uid}", "ai.hermes.dashboard", backend_argv, 4242),
            ("system", "ai.hermes.serve", serve_argv, None),
        ]

        owning = main_dashboard._launchd_job_owning_backend
        assert owning(4242, ["something", "else"], jobs) == (f"user/{uid}", "ai.hermes.dashboard", 4242)
        assert owning(9999, ["python"], jobs, ancestors=[4242, 1]) == (f"user/{uid}", "ai.hermes.dashboard", 4242)
        assert owning(9999, serve_argv, jobs) == ("system", "ai.hermes.serve", None)
        # An earlier detached respawn runs the plist argv plus the ``--no-open`` the respawn path
        # appends; it must still be attributed to the job, or every update respawns it again.
        assert owning(9999, backend_argv + ["--no-open"], jobs) == (f"user/{uid}", "ai.hermes.dashboard", 4242)
        assert owning(9999, serve_argv[:-1] + ["8643"], jobs) is None
        assert owning(9999, None, jobs) is None
        assert owning(4242, None, []) is None
