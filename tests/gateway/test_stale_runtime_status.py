"""Stale-gateway-status fix: dead-PID records must not read as live.

Covers:
1. ``read_runtime_status`` marks a dead-PID live-claiming record ``stale`` /
   ``alive: False`` (pid-liveness + start-time PID-reuse guard) and annotates a
   live record ``alive: True`` without touching its state.
2. ``retained_gateway_state`` judges a ``stale`` record by ``recorded_state`` (a
   watchdog ``degraded`` stays ``degraded``).
3. ``read_lifecycle_status`` / ``get_loop_heartbeat_age_s`` give the lifecycle
   ledger phase + heartbeat age the same liveness treatment.
4. ``cron status`` consults ticker freshness before claiming jobs will NOT fire
   (serve/desktop-hosted ticker fires without a gateway).
5. The loop-liveness watchdog warns + leaves forensics when no restart
   supervisor exists (Windows login-item install).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

_DEAD_PID = 2**30  # unassignable; _pid_exists() reports it dead without touching a real process


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _live_self_payload(**overrides):
    from gateway.status import _get_process_start_time

    payload = {
        "gateway_state": "running",
        "pid": os.getpid(),
        "start_time": _get_process_start_time(os.getpid()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(overrides)
    return payload


class TestReadRuntimeStatusLiveness:
    def test_dead_pid_running_marks_stale(self, tmp_path):
        from gateway.status import read_runtime_status

        path = _write(tmp_path / "gateway_state.json", {
            "gateway_state": "running", "pid": _DEAD_PID, "start_time": 111,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "platforms": {"telegram": {"state": "connected"}},
        })
        payload = read_runtime_status(path)
        assert payload["gateway_state"] == "stale"
        assert payload["alive"] is False
        assert payload["recorded_state"] == "running"
        # Untouched claims survive for forensics.
        assert payload["platforms"]["telegram"]["state"] == "connected"
        # Read-path only: the file on disk still holds the raw record.
        assert json.loads(path.read_text(encoding="utf-8"))["gateway_state"] == "running"

    def test_pid_reuse_marks_stale(self, tmp_path):
        """Live PID but a conflicting start_time is a recycled PID, not the gateway."""
        from gateway.status import read_runtime_status

        path = _write(tmp_path / "gateway_state.json", _live_self_payload(start_time=1))
        payload = read_runtime_status(path)
        assert payload["gateway_state"] == "stale"
        assert payload["alive"] is False
        assert payload["recorded_state"] == "running"

    def test_live_pid_annotated_alive_state_kept(self, tmp_path):
        from gateway.status import read_runtime_status

        path = _write(tmp_path / "gateway_state.json", _live_self_payload())
        payload = read_runtime_status(path)
        assert payload["gateway_state"] == "running"
        assert payload["alive"] is True
        assert "recorded_state" not in payload

    def test_non_live_claiming_state_gets_alive_false_only(self, tmp_path):
        from gateway.status import read_runtime_status

        path = _write(tmp_path / "gateway_state.json", {
            "gateway_state": "stopped", "pid": _DEAD_PID, "exit_reason": "operator_stop"})
        payload = read_runtime_status(path)
        assert payload["gateway_state"] == "stopped"
        assert payload["alive"] is False

    def test_missing_file_and_pidless_record_untouched(self, tmp_path):
        from gateway.status import read_runtime_status

        assert read_runtime_status(tmp_path / "absent.json") is None
        path = _write(tmp_path / "gateway_state.json", {"gateway_state": "running"})
        assert read_runtime_status(path) == {"gateway_state": "running"}


class TestRetainedStaleState:
    def test_stale_watchdog_degraded_stays_degraded(self):
        from gateway.status import WATCHDOG_EXIT_REASONS, retained_gateway_state

        reason = next(iter(WATCHDOG_EXIT_REASONS))
        assert retained_gateway_state({
            "gateway_state": "stale", "recorded_state": "degraded",
            "exit_reason": reason, "alive": False}) == "degraded"

    def test_stale_running_is_stopped(self):
        from gateway.status import retained_gateway_state

        assert retained_gateway_state({
            "gateway_state": "stale", "recorded_state": "running", "alive": False}) == "stopped"

    def test_stale_runtime_pid_probe_is_none(self, tmp_path):
        from gateway.status import get_runtime_status_running_pid, read_runtime_status

        path = _write(tmp_path / "gateway_state.json", {
            "gateway_state": "running", "pid": _DEAD_PID, "start_time": 111})
        assert get_runtime_status_running_pid(read_runtime_status(path)) is None


class TestLifecycleStatusReader:
    def _sentinel(self, home: Path, **fields) -> Path:
        return _write(home / "state" / "gateway.lifecycle.json", fields)

    def test_dead_pid_running_sentinel_not_alive(self, tmp_path):
        from gateway.lifecycle_ledger import read_lifecycle_status

        self._sentinel(tmp_path, phase="running", pid=_DEAD_PID,
                       start_time=time.time() - 7200, started_at="then")
        status = read_lifecycle_status(tmp_path)
        assert status["phase"] == "running"
        assert status["alive"] is False

    def test_exited_sentinel_not_alive(self, tmp_path):
        from gateway.lifecycle_ledger import read_lifecycle_status

        self._sentinel(tmp_path, phase="exited", pid=os.getpid(), exit_reason="graceful_shutdown")
        assert read_lifecycle_status(tmp_path)["alive"] is False

    def test_heartbeat_age_reported(self, tmp_path):
        from gateway.lifecycle_ledger import read_lifecycle_status

        self._sentinel(tmp_path, phase="running", pid=_DEAD_PID, start_time=1)
        hb = tmp_path / "state" / "gateway.heartbeat"
        _write(hb, {"updated_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()})
        status = read_lifecycle_status(tmp_path)
        assert status["heartbeat_age_s"] is not None
        assert status["heartbeat_age_s"] > 2 * 3600

    def test_unreadable_home_is_none(self, tmp_path):
        from gateway.lifecycle_ledger import read_lifecycle_status

        assert read_lifecycle_status(tmp_path / "nope") is None


class TestLoopHeartbeatAge:
    def test_missing_is_none(self, tmp_path):
        from gateway.shutdown_watchdog import get_loop_heartbeat_age_s

        assert get_loop_heartbeat_age_s(tmp_path) is None

    def test_fresh_and_stale(self, tmp_path):
        from gateway.shutdown_watchdog import get_loop_heartbeat_age_s

        hb = _write(tmp_path / "state" / "gateway.heartbeat",
                    {"updated_at": datetime.now(timezone.utc).isoformat()})
        assert get_loop_heartbeat_age_s(tmp_path) < 60
        hb.write_text(json.dumps(
            {"updated_at": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()}),
            encoding="utf-8")
        assert get_loop_heartbeat_age_s(tmp_path) > 4 * 3600

    def test_garbage_is_none(self, tmp_path):
        from gateway.shutdown_watchdog import get_loop_heartbeat_age_s

        hb = tmp_path / "state" / "gateway.heartbeat"
        hb.parent.mkdir(parents=True, exist_ok=True)
        hb.write_text("not json", encoding="utf-8")
        assert get_loop_heartbeat_age_s(tmp_path) is None


class TestCronStatusTickerFreshness:
    def _run_status(self, monkeypatch, capsys, heartbeat_age):
        from hermes_cli import cron as cron_mod

        monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")
        with (
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[]),
            patch("hermes_cli.gateway.named_profile_served_by_running_multiplexer", return_value=False),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=False),
            patch("cron.jobs.get_ticker_heartbeat_age", return_value=heartbeat_age),
            patch("hermes_cli.profiles.get_active_profile_name", return_value="default"),
            patch("cron.jobs.list_jobs", return_value=[]),
        ):
            cron_mod.cron_status()
        return capsys.readouterr().out

    def test_fresh_ticker_without_gateway_says_will_fire(self, monkeypatch, capsys):
        out = self._run_status(monkeypatch, capsys, heartbeat_age=5)
        assert "will fire automatically" in out
        assert "will NOT fire" not in out

    def test_no_ticker_without_gateway_says_will_not_fire(self, monkeypatch, capsys):
        out = self._run_status(monkeypatch, capsys, heartbeat_age=None)
        assert "will NOT fire" in out


class TestTaskLauncherSupervisionContract:
    """The installed task VBS must wait on the child and propagate its exit code
    (#91099) before registration counts as supervision (#91097)."""

    def _launcher_result(self, monkeypatch, tmp_path, vbs_text=None):
        import hermes_cli.gateway_windows as gw

        cmd_path = tmp_path / "Hermes_Gateway.cmd"
        cmd_path.write_text("rem placeholder", encoding="utf-8")
        if vbs_text is not None:
            cmd_path.with_suffix(".vbs").write_text(vbs_text, encoding="utf-8")
        monkeypatch.setattr(gw, "get_task_script_path", lambda: cmd_path)
        return gw.task_launcher_supervises()

    def _current_template_vbs(self):
        import hermes_cli.gateway_windows as gw
        from pathlib import Path

        orig = gw._resolve_detached_python
        gw._resolve_detached_python = lambda exe: (r"C:\venv\Scripts\python.exe", Path(r"C:\venv"), [])
        try:
            return gw._build_gateway_vbs_script(
                r"C:\venv\Scripts\python.exe", r"C:\work", r"C:\h", "--profile work")
        finally:
            gw._resolve_detached_python = orig

    def test_current_template_detaches(self, monkeypatch, tmp_path):
        assert self._launcher_result(monkeypatch, tmp_path, self._current_template_vbs()) is False

    def test_synchronous_launcher_supervises(self, monkeypatch, tmp_path):
        vbs = ('Option Explicit\r\n'
               'Dim sh, exitCode\r\n'
               'Set sh = CreateObject("WScript.Shell")\r\n'
               'exitCode = sh.Run("C:\\venv\\Scripts\\python.exe -m hermes_cli.main gateway run", 0, True)\r\n'
               'WScript.Quit exitCode\r\n')
        assert self._launcher_result(monkeypatch, tmp_path, vbs) is True

    def test_missing_launcher_fails_open(self, monkeypatch, tmp_path):
        assert self._launcher_result(monkeypatch, tmp_path, None) is True


class TestWatchdogNoSupervisor:
    def _fake_windows(self, monkeypatch, registered, supervises):
        import types

        fake_windows = types.ModuleType("hermes_cli.gateway_windows")
        fake_windows.is_task_registered = lambda: registered
        fake_windows.task_launcher_supervises = lambda: supervises
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", fake_windows)

    def _quiet_exit(self, monkeypatch):
        import gateway.lifecycle_ledger as ledger
        import gateway.status as status_mod

        calls: dict = {}
        monkeypatch.setattr(ledger, "mark_exited",
                            lambda *a, **k: calls.setdefault("mark_exited", (a, k)))
        monkeypatch.setattr(ledger, "_append_exit_diag",
                            lambda *a, **k: calls.setdefault("exit_diag", (a, k)))
        monkeypatch.setattr(status_mod, "write_runtime_status",
                            lambda *a, **k: calls.setdefault("runtime", (a, k)))
        return calls

    def test_exit_without_supervisor_leaves_forensics(self, monkeypatch):
        from gateway import shutdown_watchdog as wd

        monkeypatch.setattr(sys, "platform", "win32")
        self._fake_windows(monkeypatch, registered=False, supervises=False)
        calls = self._quiet_exit(monkeypatch)

        assert wd._restart_supervisor_present() is False
        wd._mark_exited_quietly(75, "loop_liveness_watchdog")

        record = calls["exit_diag"][0][0]
        assert record["tag"] == "gateway.watchdog_exit_no_supervisor"
        assert calls["runtime"][1]["gateway_state"] == "degraded"

    def test_registered_but_detaching_launcher_leaves_forensics(self, monkeypatch):
        """#91097: a registered task whose VBS detaches is not a supervisor."""
        from gateway import shutdown_watchdog as wd

        monkeypatch.setattr(sys, "platform", "win32")
        self._fake_windows(monkeypatch, registered=True, supervises=False)
        calls = self._quiet_exit(monkeypatch)

        assert wd._restart_supervisor_present() is False
        wd._mark_exited_quietly(75, "loop_liveness_watchdog")

        assert calls["exit_diag"][0][0]["tag"] == "gateway.watchdog_exit_no_supervisor"

    def test_registered_supervising_launcher_is_quiet(self, monkeypatch):
        """#91099 contract: task + synchronous launcher really supervises."""
        from gateway import shutdown_watchdog as wd

        monkeypatch.setattr(sys, "platform", "win32")
        self._fake_windows(monkeypatch, registered=True, supervises=True)
        calls = self._quiet_exit(monkeypatch)

        assert wd._restart_supervisor_present() is True
        wd._mark_exited_quietly(75, "loop_liveness_watchdog")

        assert "exit_diag" not in calls
        assert calls["runtime"][1]["gateway_state"] == "degraded"

    def test_supervised_platform_assumes_supervisor(self, monkeypatch):
        from gateway import shutdown_watchdog as wd

        monkeypatch.setattr(sys, "platform", "linux")
        assert wd._restart_supervisor_present() is True
