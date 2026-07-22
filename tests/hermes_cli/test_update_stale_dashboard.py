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
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.main import (
    _find_stale_dashboard_pids,
    _kill_stale_dashboard_processes,
    _restart_dashboard_via_service_manager,
    _warn_stale_dashboard_processes,  # back-compat alias
)


@pytest.fixture(autouse=True)
def _refresh_bindings_against_live_module():
    """Rebind module-level names to the *current* ``hermes_cli.main``.

    Other tests in the suite (notably ``test_env_loader.py`` and
    ``test_skills_subparser.py``) reload or delete ``hermes_cli.main`` from
    ``sys.modules``.  When that happens on the same xdist worker before we
    run, our top-of-file ``from hermes_cli.main import ...`` bindings end
    up pointing at the *old* module object.  ``patch(\"hermes_cli.main.X\")``
    then patches the *new* module, but the function we call still resolves
    ``_find_stale_dashboard_pids`` via its stale ``__globals__``, so every
    patch becomes a no-op and the kill path silently returns early.

    Refreshing the bindings (and the patch target) to the live module
    object — and keeping them consistent — makes the tests immune to
    ordering within the worker.  The fix lives in the test module because
    the two pollutants above are load-bearing for their own tests.
    """
    global _find_stale_dashboard_pids
    global _kill_stale_dashboard_processes
    global _warn_stale_dashboard_processes

    live = sys.modules.get("hermes_cli.main")
    if live is None:
        live = importlib.import_module("hermes_cli.main")

    _find_stale_dashboard_pids = live._find_stale_dashboard_pids
    _kill_stale_dashboard_processes = live._kill_stale_dashboard_processes
    _warn_stale_dashboard_processes = live._warn_stale_dashboard_processes
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


class TestFindStaleDashboardPids:
    """Unit tests for the ps/wmic-based detection step."""

    def test_no_matches_returns_empty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(111, "/usr/bin/python3 -m some.other.module")
                + "\n"
                + _ps_line(222, "/usr/bin/bash")
                + "\n",
                stderr="",
            )
            assert _find_stale_dashboard_pids() == []

    def test_matches_running_dashboard(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119") + "\n",
                stderr="",
            )
            assert _find_stale_dashboard_pids() == [12345]

    def test_multiple_matches(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119"),
                    _ps_line(12346, "hermes dashboard --port 9120 --no-open"),
                    _ps_line(12347, "python /home/x/hermes_cli/main.py dashboard"),
                ]) + "\n",
                stderr="",
            )
            assert sorted(_find_stale_dashboard_pids()) == [12345, 12346, 12347]

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

    def test_ps_not_found_returns_empty(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _find_stale_dashboard_pids() == []

    def test_ps_timeout_returns_empty(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ps", 10)):
            assert _find_stale_dashboard_pids() == []

    def test_unrelated_process_containing_word_dashboard_not_matched(self):
        """Guards against greedy pgrep-style matching catching chat sessions
        or unrelated processes whose cmdline happens to contain 'dashboard'.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119"),
                    _ps_line(22222, "python3 -m hermes_cli.main chat -q 'rewrite my dashboard'"),
                    _ps_line(33333, "node /opt/grafana/dashboard-server.js"),
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert pids == [12345]

    def test_grep_lines_ignored(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(99999, "grep hermes dashboard"),
                    _ps_line(12345, "hermes dashboard --port 9119"),
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert 99999 not in pids
        assert 12345 in pids

    def test_invalid_pid_lines_skipped(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    "notapid hermes dashboard --bad",
                    _ps_line(12345, "hermes dashboard --port 9119"),
                    "   ",
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert pids == [12345]

    def test_exclude_pids_filters_specified_pids(self):
        """exclude_pids removes specific PIDs from the result — used by
        the Desktop Electron app to protect its own backend child.  (#37532)
        """
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(11111, "hermes dashboard --port 9119"),
                    _ps_line(22222, "hermes dashboard --port 9120"),
                    _ps_line(33333, "hermes dashboard --port 9121"),
                ]) + "\n",
                stderr="",
            )
            # Exclude the desktop-managed backend PID
            pids = _find_stale_dashboard_pids(exclude_pids={22222})
        assert 11111 in pids
        assert 22222 not in pids
        assert 33333 in pids

    def test_exclude_pids_none_is_noop(self):
        """Passing exclude_pids=None (the default) changes nothing."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(12345, "hermes dashboard --port 9119") + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids(exclude_pids=None)
        assert pids == [12345]

    def test_exclude_all_pids_returns_empty(self):
        """If all matched PIDs are excluded, the result is empty."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(12345, "hermes dashboard --port 9119") + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids(exclude_pids={12345})
        assert pids == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX kill semantics")
class TestKillStaleDashboardPosix:
    """Kill path on Linux / macOS: SIGTERM then SIGKILL any survivors."""

    def test_no_stale_processes_is_a_noop(self, capsys):
        with patch("hermes_cli.main._find_stale_dashboard_pids", return_value=[]):
            _kill_stale_dashboard_processes()
        assert capsys.readouterr().out == ""

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

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345, 12346]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes()

        # Both got SIGTERM.
        sigterms = [pid for pid, sig in killed_signals if sig == _signal.SIGTERM]
        assert sorted(sigterms) == [12345, 12346]
        # No SIGKILL was needed.
        assert not any(sig == _signal.SIGKILL for _, sig in killed_signals)

        out = capsys.readouterr().out
        assert "Stopping 2 dashboard" in out
        assert "✓ stopped PID 12345" in out
        assert "✓ stopped PID 12346" in out
        assert "Restart the dashboard" in out

    def test_sigkill_fallback_for_survivors(self, capsys):
        """If a process survives SIGTERM + the grace window, SIGKILL is sent."""
        import signal as _signal

        sent: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            sent.append((pid, sig))
            # Simulate stubborn process: probe (sig 0) always succeeds,
            # SIGTERM does nothing, SIGKILL is where it "dies".
            if sig in {_signal.SIGTERM, 0, _signal.SIGKILL}:
                return
            # Any other signal — also fine.

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[99999]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"), \
             patch("time.monotonic", side_effect=[0.0] + [10.0] * 20):
            # monotonic jumps past the 3s deadline on the second read so the
            # grace loop exits immediately after one iteration.
            _kill_stale_dashboard_processes()

        signals_sent = [sig for _, sig in sent]
        assert _signal.SIGTERM in signals_sent
        assert _signal.SIGKILL in signals_sent

        out = capsys.readouterr().out
        assert "✓ stopped PID 99999" in out

    def test_permission_error_is_reported_not_raised(self, capsys):
        """os.kill raising PermissionError (e.g. another user's process)
        must not abort hermes update — it's reported as a failure and we
        move on."""
        def fake_kill(pid, sig):
            raise PermissionError("Operation not permitted")

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes()  # must not raise

        out = capsys.readouterr().out
        assert "✗ failed to stop PID 12345" in out
        assert "Operation not permitted" in out

    def test_process_already_gone_counts_as_stopped(self, capsys):
        """ProcessLookupError on the initial SIGTERM means the process
        already exited between detection and the kill — treat as success."""
        def fake_kill(pid, sig):
            raise ProcessLookupError

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes()

        out = capsys.readouterr().out
        assert "✓ stopped PID 12345" in out
        assert "failed to stop" not in out


class TestKillStaleDashboardWindows:
    """Kill path on Windows: taskkill /F."""

    def test_taskkill_invoked_for_each_pid(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "win32")

        def fake_run(args, *a, **kw):
            # taskkill returns 0 on success
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345, 12346]), \
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

        out = capsys.readouterr().out
        assert "✓ stopped PID 12345" in out
        assert "✓ stopped PID 12346" in out

    def test_taskkill_failure_is_reported(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "win32")

        def fake_run(args, *a, **kw):
            return MagicMock(returncode=128, stdout="",
                             stderr="ERROR: Access is denied.")

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345]), \
             patch("subprocess.run", side_effect=fake_run):
            _kill_stale_dashboard_processes()  # must not raise

        out = capsys.readouterr().out
        assert "✗ failed to stop PID 12345" in out
        assert "Access is denied" in out


class TestBackCompatAlias:
    """``_warn_stale_dashboard_processes`` is kept as an alias for the
    new kill function so old imports don't break."""

    def test_alias_is_the_kill_function(self):
        assert _warn_stale_dashboard_processes is _kill_stale_dashboard_processes


class TestWindowsWmicEncoding:
    """Regression tests for #17049 — the Windows wmic branch must not crash
    `hermes update` on non-UTF-8 system locales (e.g. cp936 on zh-CN).
    """

    def test_wmic_invoked_with_utf8_ignore_errors(self, monkeypatch):
        """The wmic subprocess.run call must pass encoding='utf-8' and
        errors='ignore' so the subprocess reader thread cannot raise
        UnicodeDecodeError on non-UTF-8 wmic output."""
        monkeypatch.setattr(sys, "platform", "win32")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "CommandLine=python -m hermes_cli.main dashboard\n"
                    "ProcessId=12345\n"
                ),
                stderr="",
            )
            _find_stale_dashboard_pids()

        # The wmic call is the first subprocess.run invocation.
        assert mock_run.called, "subprocess.run was not invoked"
        wmic_call = mock_run.call_args_list[0]
        kwargs = wmic_call.kwargs
        assert kwargs.get("encoding") == "utf-8", (
            "encoding kwarg must be 'utf-8' so wmic output is decoded "
            "deterministically rather than via the implicit reader-thread "
            "default that crashes on non-UTF-8 locales (#17049)."
        )
        assert kwargs.get("errors") == "ignore", (
            "errors kwarg must be 'ignore' so undecodable bytes don't take "
            "down the reader thread (#17049)."
        )

    def test_wmic_returns_none_stdout_does_not_crash(self, monkeypatch):
        """If subprocess.run returns successfully but stdout is None — which
        is what Python 3.11 leaves behind when the reader thread silently
        crashed on UnicodeDecodeError before this fix landed — detection
        must short-circuit instead of raising AttributeError on
        ``None.split('\\n')`` and aborting `hermes update` (#17049)."""
        monkeypatch.setattr(sys, "platform", "win32")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=None, stderr=""
            )
            # Must not raise.
            assert _find_stale_dashboard_pids() == []


class TestRestartDashboardViaServiceManager:
    """Regression tests for the post-``hermes update`` dashboard restart.

    Repro of the original bug: ``hermes update`` only auto-restarted
    ``hermes-gateway*`` systemd units and SIGTERM-killed the dashboard by PID,
    leaving a systemd-managed ``hermes-dashboard.service`` dead after every
    update.  ``_restart_dashboard_via_service_manager()`` must detect an
    active ``hermes-dashboard*`` unit and restart it, and return ``False`` (so
    the caller falls back to the PID-kill) only when no unit owns the
    dashboard.
    """

    @staticmethod
    def _fake_systemctl(unit_line: str, *, is_active: bool = True, sudo_ok: bool = True, scope: str = "user"):
        """Build a subprocess.run side_effect for one dashboard unit.

        Routes the systemctl / sudo calls the function makes:
          - list-units hermes-dashboard*  -> emit ``unit_line`` only for the
            scope under test (user vs system), empty otherwise
          - is-active <svc>               -> "active" / "inactive"
          - reset-failed <svc>            -> rc 0
          - restart <svc>                 -> rc 0
          - sudo -n true                  -> rc 0 (sudo available) / rc 1
        """
        def _side_effect(cmd, *a, **kw):
            if isinstance(cmd, (list, tuple)) and cmd:
                if cmd[0] == "sudo":
                    # Both sudo probes (``sudo -n true`` and the targeted
                    # ``sudo -n systemctl ... reset-failed``) must reflect the
                    # same privilege outcome, otherwise the fake lies about
                    # sudo availability and the restart path proceeds.
                    return SimpleNamespace(returncode=0 if sudo_ok else 1,
                                           stdout="", stderr="")
                if "list-units" in cmd:
                    # Only advertise the unit in the scope under test so the
                    # function restarts exactly one unit (mirrors reality: a
                    # unit is active in at most one scope).
                    in_user_scope = "--user" in cmd
                    emit = unit_line if (
                        (scope == "user" and in_user_scope)
                        or (scope == "system" and not in_user_scope)
                        or scope == "both"
                    ) else ""
                    return SimpleNamespace(returncode=0, stdout=emit, stderr="")
                if "is-active" in cmd:
                    return SimpleNamespace(
                        returncode=0,
                        stdout="active\n" if is_active else "inactive\n",
                        stderr="",
                    )
                if "reset-failed" in cmd:
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                if "restart" in cmd:
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return _side_effect

    def _patch_systemd(self, monkeypatch, *, supported: bool = True):
        monkeypatch.setattr(
            "hermes_cli.gateway.supports_systemd_services",
            lambda: supported,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway._ensure_user_systemd_env",
            lambda: None,
        )

    def test_no_systemd_support_returns_false(self, monkeypatch):
        """When the platform has no systemd (e.g. termux), the function must
        short-circuit and report 'no service manager' (False)."""
        self._patch_systemd(monkeypatch, supported=False)
        assert _restart_dashboard_via_service_manager() is False

    def test_no_dashboard_unit_returns_false(self, monkeypatch):
        """No hermes-dashboard* unit present -> caller should fall back to the
        legacy PID-kill (returns False)."""
        self._patch_systemd(monkeypatch, supported=True)
        with patch("hermes_cli.main.subprocess.run",
                   side_effect=self._fake_systemctl("", is_active=False)):
            assert _restart_dashboard_via_service_manager() is False

    def test_user_unit_active_is_restarted(self, monkeypatch, capsys):
        """An active user-scope hermes-dashboard.service must be restarted via
        reset-failed + restart, and the function returns True."""
        self._patch_systemd(monkeypatch, supported=True)
        with (
            patch("hermes_cli.main.subprocess.run",
                  side_effect=self._fake_systemctl(
                      "hermes-dashboard.service  loaded active running")) as mock_run,
            patch("hermes_cli.main._time.monotonic", side_effect=[0.0, 0.1]),
            patch("hermes_cli.main._time.sleep"),
        ):
            result = _restart_dashboard_via_service_manager()

        assert result is True
        invoked = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "--no-ask-password", "reset-failed", "hermes-dashboard"] in invoked
        assert ["systemctl", "--user", "--no-ask-password", "restart", "hermes-dashboard"] in invoked
        out = capsys.readouterr().out
        assert "Restarting dashboard service hermes-dashboard" in out
        assert "Restarted hermes-dashboard" in out

    def test_system_unit_active_without_sudo_skips_and_hints(self, monkeypatch, capsys):
        """A system-scope unit with no non-interactive sudo path must not hang
        on a polkit prompt: it prints a manual-restart hint and returns True
        (the unit is 'owned' by systemd, so the PID-kill fallback is skipped)."""
        self._patch_systemd(monkeypatch, supported=True)
        monkeypatch.setattr("hermes_cli.main.os.geteuid", lambda: 1000)
        with (
            patch("hermes_cli.main.subprocess.run",
                  side_effect=self._fake_systemctl(
                      "hermes-dashboard.service  loaded active running",
                      sudo_ok=False, scope="system")) as mock_run,
            patch("hermes_cli.main._time.monotonic", side_effect=[0.0, 0.1]),
            patch("hermes_cli.main._time.sleep"),
        ):
            result = _restart_dashboard_via_service_manager()

        assert result is True
        # No restart/reset-failed issued: user scope has no unit and the
        # system-scope unit has no non-interactive privilege path.
        invoked = [c.args[0] for c in mock_run.call_args_list]
        assert not any("restart" in c for c in invoked)
        out = capsys.readouterr().out
        assert "needs root to restart" in out
        assert "systemctl restart hermes-dashboard" in out

    def test_restart_fails_then_succeeds_on_retry(self, monkeypatch, capsys):
        """If the first restart exits non-zero (transient), the function must
        retry once and succeed."""
        self._patch_systemd(monkeypatch, supported=True)
        state = {"attempt": 0}

        def _flaky_restart(cmd, *a, **kw):
            if isinstance(cmd, (list, tuple)) and "list-units" in cmd:
                # Only the user-scope unit exists (system scope finds nothing).
                emit = "hermes-dashboard.service  loaded active running" if "--user" in cmd else ""
                return SimpleNamespace(returncode=0, stdout=emit, stderr="")
            if isinstance(cmd, (list, tuple)) and "is-active" in cmd:
                return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
            if isinstance(cmd, (list, tuple)) and "reset-failed" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if isinstance(cmd, (list, tuple)) and "restart" in cmd:
                state["attempt"] += 1
                if state["attempt"] == 1:
                    return SimpleNamespace(returncode=1, stdout="", stderr="Job failed")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch("hermes_cli.main.subprocess.run", side_effect=_flaky_restart) as mock_run,
            patch("hermes_cli.main._time.monotonic", side_effect=[0.0, 0.1, 0.2, 0.3]),
            patch("hermes_cli.main._time.sleep"),
        ):
            result = _restart_dashboard_via_service_manager()

        assert result is True
        restart_calls = [
            c.args[0] for c in mock_run.call_args_list
            if c.args and isinstance(c.args[0], list) and "restart" in c.args[0]
        ]
        assert len(restart_calls) == 2
        out = capsys.readouterr().out
        assert "Restarted hermes-dashboard (retry)" in out

    def test_inactive_unit_is_ignored(self, monkeypatch):
        """A hermes-dashboard* unit that exists but is not active must not be
        restarted and must not count as 'found'."""
        self._patch_systemd(monkeypatch, supported=True)
        with patch("hermes_cli.main.subprocess.run",
                   side_effect=self._fake_systemctl(
                       "hermes-dashboard.service  loaded inactive dead",
                       is_active=False)) as mock_run:
            result = _restart_dashboard_via_service_manager()
        assert result is False
        invoked = [c.args[0] for c in mock_run.call_args_list if c.args]
        assert not any("restart" in c for c in invoked)
