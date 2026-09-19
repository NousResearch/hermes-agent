# -*- coding: utf-8 -*-
"""Regression tests for the fail-closed PID-ownership guard.

Refs #90471 / #89614.  The three patched Windows ``taskkill`` boundaries:

- ``hermes_cli/_subprocess_compat.pid_is_hermes`` / ``kill_process_tree``
- ``hermes_cli/dashboard_procs._kill_stale_dashboard_processes`` (win32)
- ``hermes_cli/update_cmd._stop_process_trees``

Acceptance from #90471:
1. missing / unreadable / non-matching identity fails closed -> no taskkill
2. a recycled or foreign PID control process remains untouched
3. probe failure or timeout is never converted into permission to kill
"""
import subprocess
import sys
from unittest import mock

import pytest

from hermes_cli import _subprocess_compat
from hermes_cli import dashboard_procs
from hermes_cli import update_cmd


def _probe_stdout(value: str) -> mock.Mock:
    return mock.Mock(stdout=value)


class TestPidIsHermes:
    """The shared identity probe must fail closed on every ambiguity."""

    def test_non_windows_is_unconditional_pass(self):
        # Non-Windows callers have no taskkill path at all.
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", False):
            assert _subprocess_compat.pid_is_hermes(1234) is True

    def test_non_windows_still_rejects_recycled_identity(self):
        # An explicit fingerprint mismatch is a recycled PID on any platform.
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", False), mock.patch.object(
            _subprocess_compat, "_process_start_time", return_value=456
        ):
            assert _subprocess_compat.pid_is_hermes(
                1234, expected_start_time=123
            ) is False

    def test_hermes_match_requires_token_boundary(self):
        # "hermes" buried inside an unrelated path segment must not match.
        assert _subprocess_compat._text_names_hermes(
            r"c:\users\shermesa\app.exe"
        ) is False
        assert _subprocess_compat._text_names_hermes(
            r"C:\Users\x\.hermes-runtime\python.exe -m hermes_cli.main"
        ) is True
        assert _subprocess_compat._text_names_hermes(
            "/opt/hermes-agent/venv/bin/python"
        ) is True

    def test_invalid_pid_inputs_do_not_crash(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True):
            assert _subprocess_compat.pid_is_hermes(-1) is False
            assert _subprocess_compat.pid_is_hermes(0) is False
            assert _subprocess_compat.pid_is_hermes("not-a-pid") is False
            assert _subprocess_compat.pid_is_hermes(True) is False

    def test_probe_matches_hermes_like_process(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "_process_start_time", return_value=123
        ), mock.patch.object(
            _subprocess_compat, "_process_command_is_hermes", return_value=True
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is True

    def test_probe_rejects_recycled_process_identity(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "_process_start_time", return_value=456
        ), mock.patch.object(
            _subprocess_compat, "_process_command_is_hermes", return_value=True
        ):
            assert _subprocess_compat.pid_is_hermes(
                1234, expected_start_time=123
            ) is False

    def test_probe_rejects_foreign_process(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "_process_start_time", return_value=123
        ), mock.patch.object(
            _subprocess_compat, "_process_command_is_hermes", return_value=False
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    def test_probe_blank_stdout_fails_closed(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "_process_start_time", return_value=None
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    def test_probe_timeout_fails_closed(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "_process_start_time", return_value=None
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    def test_probe_oserror_fails_closed(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "_process_start_time", side_effect=OSError("broken pipe")
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    @pytest.mark.skipif(sys.platform != "win32", reason="real probe is windows-only")
    def test_missing_pid_real_probe_fails_closed(self):
        # A PID that cannot exist must never be judged Hermes-owned.
        assert _subprocess_compat.pid_is_hermes(2**24) is False


class TestKillProcessTree:
    """kill_process_tree operates on our own retained Popen handle.

    A retained handle pins the PID (the child cannot be reaped while the
    handle is open), so PID recycling is impossible there and the identity
    guard deliberately does NOT apply — it could only false-refuse a
    legitimate cleanup. These tests pin that contract for the legacy
    Windows fallback path.
    """

    def _proc(self, pid=4321):
        return mock.Mock(pid=pid)

    def test_retained_handle_is_taskkilled_without_probe(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "pid_is_hermes"
        ) as guard, mock.patch.object(_subprocess_compat.subprocess, "run") as run:
            _subprocess_compat._legacy_kill_process_tree(self._proc())
            guard.assert_not_called()
            run.assert_called_once()
            argv = run.call_args.args[0]
            assert argv[0] == "taskkill"
            assert "/PID" in argv
            assert str(4321) in argv


class TestStopProcessTrees:
    """update_cmd._stop_process_trees guard behaviour."""

    def test_foreign_pids_only_probed(self):
        with mock.patch(
            "gateway.status.get_process_start_time", return_value=123
        ), mock.patch(
            "hermes_cli._subprocess_compat.pid_is_hermes", return_value=False
        ), mock.patch.object(update_cmd.subprocess, "run") as run:
            update_cmd._stop_process_trees([1111, 2222])
        run.assert_not_called()

    def test_hermes_pid_probed_then_taskkilled(self):
        with mock.patch(
            "gateway.status.get_process_start_time", return_value=123
        ), mock.patch(
            "hermes_cli._subprocess_compat.pid_is_hermes", return_value=True
        ), mock.patch.object(
            update_cmd.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            update_cmd._stop_process_trees([1111])
        assert len(run.call_args_list) == 1
        assert run.call_args.args[0][0] == "taskkill"

    def test_probe_timeout_skips_taskkill(self):
        with mock.patch(
            "gateway.status.get_process_start_time", return_value=123
        ), mock.patch(
            "hermes_cli._subprocess_compat.pid_is_hermes", return_value=False
        ), mock.patch.object(update_cmd.subprocess, "run") as run:
            update_cmd._stop_process_trees([1111, 2222])  # must not raise
        run.assert_not_called()


class TestKillStaleDashboardProcesses:
    """dashboard_procs win32 kill branch guard behaviour."""

    def _patch_find(self, pids=(12345,)):
        from hermes_cli import main_dashboard

        return mock.patch.object(main_dashboard, "_find_stale_dashboard_pids", return_value=list(pids))

    def test_foreign_pid_reported_not_killed(self):
        with self._patch_find(), mock.patch.object(
            dashboard_procs.sys, "platform", "win32"
        ), mock.patch(
            "gateway.status.get_process_start_time", return_value=123
        ), mock.patch(
            "hermes_cli._subprocess_compat.pid_is_hermes", return_value=False
        ), mock.patch.object(dashboard_procs.subprocess, "run") as run:
            result = dashboard_procs._kill_stale_dashboard_processes()
        assert result["killed"] == []
        assert result["failed"] == [
            (12345, "not hermes-owned or process identity changed")
        ]
        run.assert_not_called()

    def test_hermes_pid_killed(self):
        with self._patch_find(), mock.patch.object(
            dashboard_procs.sys, "platform", "win32"
        ), mock.patch(
            "gateway.status.get_process_start_time", return_value=123
        ), mock.patch(
            "hermes_cli._subprocess_compat.pid_is_hermes", return_value=True
        ), mock.patch.object(
            dashboard_procs.subprocess, "run", return_value=mock.Mock(
                returncode=0, stderr="", stdout=""
            )
        ) as run:
            result = dashboard_procs._kill_stale_dashboard_processes()
        taskkill_calls = [c for c in run.call_args_list if c.args[0][0] == "taskkill"]
        assert len(taskkill_calls) == 1
        assert result["killed"] == [12345]
        assert result["failed"] == []

    @pytest.mark.parametrize("include_pids", [None, {12345, 12346, 12347}, {12346, 12347}, set()])
    def test_stop_only_targets_the_invoking_hermes_home(self, monkeypatch, include_pids):
        """An argv match from another profile is never a ``--stop`` target."""
        own_home = "/tmp/hermes-own"
        foreign_home = "/tmp/hermes-foreign"
        monkeypatch.setenv("HERMES_HOME", own_home)

        with mock.patch.object(
            dashboard_procs, "_scan_dashboard_processes",
            return_value=[(12345, "hermes serve"), (12346, "hermes serve"), (12347, "hermes serve")],
        ), mock.patch.object(dashboard_procs, "_caller_ancestor_pids", return_value=set()), mock.patch.object(
            dashboard_procs, "_hermes_home_for_pid",
            side_effect=lambda pid: {
                12345: own_home,
                12346: foreign_home,
                12347: None,
            }[pid],
        ), mock.patch.object(
            dashboard_procs, "_kill_pids_posix"
        ) as kill:
            result = dashboard_procs._kill_stale_dashboard_processes(
                scope_home=own_home, include_pids=include_pids,
            )

        if include_pids is None or 12345 in include_pids:
            kill.assert_called_once()
            assert kill.call_args.args[0] == [12345]
            assert result["matched"] == [12345]
        else:
            kill.assert_not_called()
            assert result["matched"] == []


class TestHermesHomeForPid:
    """Tri-state owner resolution: a readable environment always names a home."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX default home is $HOME/.hermes")
    def test_readable_env_without_var_resolves_to_that_process_default_home(self, monkeypatch, tmp_path):
        """The common install shape exports no HERMES_HOME: the backend lives in its user's
        platform default home, and a default-home ``--stop`` must still find it (#113978)."""
        home = str(tmp_path / "alice")
        monkeypatch.setattr(dashboard_procs, "_pid_environ", lambda pid: {"HOME": home})
        from hermes_cli import main_dashboard
        monkeypatch.setattr(main_dashboard, "_dashboard_cmdline_for_pid",
                            lambda pid: ["hermes", "--profile", "work", "serve"] if pid == 2 else ["hermes", "serve"])

        assert dashboard_procs._hermes_home_for_pid(1) == f"{home}/.hermes"
        # ``-p``/``--profile`` is applied to os.environ after exec, invisible in /proc environ.
        assert dashboard_procs._hermes_home_for_pid(2) == f"{home}/.hermes/profiles/work"
        assert dashboard_procs._pids_owned_by_hermes_home([1, 2], f"{home}/.hermes") == [1]

    def test_root_shaped_hermes_home_follows_the_flag_and_the_sticky_active_profile(self, monkeypatch, tmp_path):
        """Mirror ``_apply_profile_override``: an exported root ``HERMES_HOME`` is the root, not the
        home — ``-p work`` and ``hermes profile use work`` both land in ``<root>/profiles/work``."""
        root = tmp_path / ".hermes"
        root.mkdir()
        monkeypatch.setattr(dashboard_procs, "_pid_environ",
                            lambda pid: {"HOME": str(tmp_path), "HERMES_HOME": str(root)})
        from hermes_cli import main_dashboard
        monkeypatch.setattr(main_dashboard, "_dashboard_cmdline_for_pid",
                            lambda pid: ["hermes", "-p", "work", "serve"] if pid == 2 else ["hermes", "serve"])

        assert dashboard_procs._hermes_home_for_pid(2) == str(root / "profiles" / "work")
        assert dashboard_procs._hermes_home_for_pid(1) == str(root)  # no flag, no active_profile
        (root / "active_profile").write_text("work", encoding="utf-8")
        assert dashboard_procs._hermes_home_for_pid(1) == str(root / "profiles" / "work")
        # A profile-shaped HERMES_HOME without a flag is the home itself (root = its grandparent).
        monkeypatch.setattr(dashboard_procs, "_pid_environ",
                            lambda pid: {"HERMES_HOME": str(root / "profiles" / "ops")})
        assert dashboard_procs._hermes_home_for_pid(1) == str(root / "profiles" / "ops")

    def test_unreadable_env_is_none_and_spared(self, monkeypatch):
        monkeypatch.setattr(dashboard_procs, "_pid_environ", lambda pid: None)
        assert dashboard_procs._hermes_home_for_pid(7) is None
        assert dashboard_procs._pids_owned_by_hermes_home([7], "/home/alice/.hermes") == []
