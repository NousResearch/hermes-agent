"""Tests for the Windows gateway post-update job-object escape (issue #84185).

The bug: ``_cold_start_windows_gateway_after_update`` and
``_spawn_gateway_restart_watcher`` both spawn the gateway via
``subprocess.Popen`` + ``CREATE_BREAKAWAY_FROM_JOB``, which CreateProcess
accepts silently even when the parent job denies breakaway. The spawned
child lands inside the parent's job and is hard-killed when the updater
exits. The printed ✓ is therefore a lie.

The fix routes both spawn points through the Scheduled Task when one is
registered — ``schtasks /Run`` goes through the Task Scheduler service and
is never a child of any job containing the updater. Falls back to the
direct spawn only when no Scheduled Task exists, and even then reports
survival honestly.

Follow-up (issue #84185 review): the task is re-registered before /Run so it
never replays a stale Python path from task-creation time, and the post-
trigger poll checks only for NEW gateway PIDs (not one that was already
running) so a pre-update gateway draining in the background does not satisfy
the check on its own.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

MODULE_UPDATE_CMD = "hermes_cli.update_cmd"

@pytest.fixture
def cold_start_mocks(monkeypatch):
    """Patch the three gateway_windows helpers the cold-start path uses.

    Returns a namespace with the three mocks. The low-level spawn helpers
    on ``hermes_cli.gateway_windows`` are patched directly (same pattern
    as ``tests/hermes_cli/test_gateway_windows.py``);
    ``hermes_cli.gateway.find_gateway_pids`` is stubbed to report nothing
    running; and ``update_cmd._m`` is patched to report Windows.
    """
    from hermes_cli import gateway, gateway_windows, update_cmd

    m_main = MagicMock(name="hermes_cli.main")
    m_main._is_windows.return_value = True

    spawn_via_schtasks = MagicMock(name="_spawn_via_scheduled_task", return_value=False)
    spawn_detached = MagicMock(name="_spawn_detached", return_value=0)
    wait_for_ready = MagicMock(name="_wait_for_gateway_ready", return_value=[])

    monkeypatch.setattr(gateway_windows, "_spawn_via_scheduled_task", spawn_via_schtasks)
    monkeypatch.setattr(gateway_windows, "_spawn_detached", spawn_detached)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", wait_for_ready)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
    monkeypatch.setattr(update_cmd, "_m", lambda: m_main)

    class _NS:
        pass

    ns = _NS()
    ns.spawn_via_schtasks = spawn_via_schtasks
    ns.spawn_detached = spawn_detached
    ns.wait_for_ready = wait_for_ready
    return ns

class TestColdStartEscape:
    """_cold_start_windows_gateway_after_update must prefer the Scheduled Task."""

    def test_cold_start_via_scheduled_task_when_task_exists(
        self, capsys, cold_start_mocks
    ):
        """schtasks path spawns and survives → prints task-based ✓ and returns."""
        from hermes_cli import update_cmd

        cold_start_mocks.spawn_via_schtasks.return_value = True
        update_cmd._cold_start_windows_gateway_after_update()

        out = capsys.readouterr().out
        cold_start_mocks.spawn_via_schtasks.assert_called_once()
        cold_start_mocks.spawn_detached.assert_not_called()
        assert "Scheduled Task" in out
        assert "✓" in out
        assert "did not survive" not in out

    def test_cold_start_falls_back_to_spawn_detached_when_no_task(
        self, capsys, cold_start_mocks
    ):
        """No Scheduled Task registered → fall back to _spawn_detached + survival check."""
        from hermes_cli import update_cmd

        cold_start_mocks.spawn_via_schtasks.return_value = False
        cold_start_mocks.spawn_detached.return_value = 54321
        cold_start_mocks.wait_for_ready.return_value = [54321]
        update_cmd._cold_start_windows_gateway_after_update()

        out = capsys.readouterr().out
        cold_start_mocks.spawn_via_schtasks.assert_called_once()
        cold_start_mocks.spawn_detached.assert_called_once()
        cold_start_mocks.wait_for_ready.assert_called_once()
        assert "54321" in out
        assert "did not survive" not in out

    def test_cold_start_reports_failure_when_spawn_does_not_survive(
        self, capsys, cold_start_mocks
    ):
        """Direct spawn returns a PID but the gateway never comes up → ✗, no ✓."""
        from hermes_cli import update_cmd

        cold_start_mocks.spawn_via_schtasks.return_value = False
        cold_start_mocks.spawn_detached.return_value = 54321
        cold_start_mocks.wait_for_ready.return_value = []
        update_cmd._cold_start_windows_gateway_after_update()

        out = capsys.readouterr().out
        cold_start_mocks.spawn_detached.assert_called_once()
        cold_start_mocks.wait_for_ready.assert_called_once()
        assert "did not survive" in out
        assert "hermes gateway start" in out
        assert "✓ Starting Windows gateway after update" not in out

class TestSpawnViaScheduledTaskHelper:
    """_spawn_via_scheduled_task returns False unless a NEW gateway actually shows up."""

    def test_returns_false_when_no_task_registered(self, monkeypatch):
        from hermes_cli import gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
        exec_mock = MagicMock()
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", exec_mock)
        assert gateway_windows._spawn_via_scheduled_task() is False
        exec_mock.assert_not_called()

    def test_returns_false_when_schtasks_run_fails(self, monkeypatch):
        from hermes_cli import gateway_windows

        wait_mock = MagicMock(return_value=[])
        write_mock = MagicMock(return_value=Path("/fake/script.cmd"))
        install_mock = MagicMock(return_value=(True, "created"))
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        # Launcher is NOT ours -> refresh path runs (write + reinstall).
        monkeypatch.setattr(gateway_windows, "_launcher_is_ours", lambda: False)
        monkeypatch.setattr(gateway_windows, "_write_task_script", write_mock)
        monkeypatch.setattr(gateway_windows, "_install_scheduled_task", install_mock)
        monkeypatch.setattr(
            gateway_windows, "_exec_schtasks", lambda *a, **kw: (1, "", "error")
        )
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", wait_mock)
        assert gateway_windows._spawn_via_scheduled_task() is False
        write_mock.assert_called_once()
        install_mock.assert_called_once()
        wait_mock.assert_not_called()  # didn't even wait — run failed

    def test_returns_false_when_task_triggered_but_no_new_pid_appears(self, monkeypatch):
        from hermes_cli import gateway, gateway_windows

        write_mock = MagicMock(return_value=Path("/fake/script.cmd"))
        install_mock = MagicMock(return_value=(True, "created"))
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(gateway_windows, "_write_task_script", write_mock)
        monkeypatch.setattr(gateway_windows, "_install_scheduled_task", install_mock)
        monkeypatch.setattr(
            gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", "")
        )
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [])
        # A PRE-EXISTING gateway is still running (draining): /Run succeeded and
        # no NEW pid appeared, so we must NOT report success — the pre-existing
        # gateway satisfies nothing and a fallback direct spawn would race with
        # IgnoreNew (the task's start would be suppressed).
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [999])
        assert gateway_windows._spawn_via_scheduled_task() is False

    def test_run_accepted_no_prior_gateway_counts_as_success(self, monkeypatch):
        """Cold starts can exceed the poll window; /Run accepted + empty
        pre-set must count as success instead of racing a direct spawn."""
        from hermes_cli import gateway, gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(
            gateway_windows,
            "_task_action_matches_expected",
            lambda: True,
        )
        monkeypatch.setattr(gateway_windows, "_launcher_is_ours", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", "")
        )
        # Poll window expires without a visible PID (still importing).
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [])
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
        assert gateway_windows._spawn_via_scheduled_task() is True

    def test_skips_reinstall_when_task_action_and_launcher_are_current(self, monkeypatch):
        """No delete+create when /Query /XML matches our .vbs and the .cmd is
        ours — avoids UAC-protected re-register on hosts where it fails."""
        from hermes_cli import gateway, gateway_windows

        install_mock = MagicMock()
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(gateway_windows, "_task_action_matches_expected", lambda: True)
        monkeypatch.setattr(gateway_windows, "_launcher_is_ours", lambda: True)
        monkeypatch.setattr(gateway_windows, "_install_scheduled_task", install_mock)
        monkeypatch.setattr(
            gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", "")
        )
        wait_mock = MagicMock(return_value=[12345])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", wait_mock)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
        assert gateway_windows._spawn_via_scheduled_task() is True
        install_mock.assert_not_called()

    def test_snapshots_pids_before_triggering(self, monkeypatch):
        """pre_pids must be captured BEFORE schtasks /Run fires."""
        from hermes_cli import gateway, gateway_windows

        order = []

        def fake_find(**kw):
            order.append("find")
            return []

        def fake_exec(*a, **kw):
            order.append("run")
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(gateway_windows, "_task_action_matches_expected", lambda: True)
        monkeypatch.setattr(gateway_windows, "_launcher_is_ours", lambda: True)
        monkeypatch.setattr(gateway_windows, "_write_task_script", MagicMock())
        monkeypatch.setattr(gateway_windows, "_install_scheduled_task", MagicMock())
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_exec)
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [7])
        monkeypatch.setattr(gateway, "find_gateway_pids", fake_find)
        gateway_windows._spawn_via_scheduled_task()
        assert order == ["find", "run"] or order[:2] == ["find", "run"]

    def test_returns_true_when_task_triggered_and_new_pid_appears(self, monkeypatch):
        from hermes_cli import gateway, gateway_windows

        write_mock = MagicMock(return_value=Path("/fake/script.cmd"))
        install_mock = MagicMock(return_value=(True, "created"))
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        # Launcher is NOT ours -> refresh path runs (write + reinstall).
        monkeypatch.setattr(gateway_windows, "_launcher_is_ours", lambda: False)
        monkeypatch.setattr(gateway_windows, "_write_task_script", write_mock)
        monkeypatch.setattr(gateway_windows, "_install_scheduled_task", install_mock)
        monkeypatch.setattr(
            gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", "")
        )
        # Pre-trigger: no gateway; post-trigger: PID 12345 appears (new).
        wait_mock = MagicMock(return_value=[12345])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", wait_mock)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
        assert gateway_windows._spawn_via_scheduled_task() is True
        write_mock.assert_called_once()
        install_mock.assert_called_once()

    def test_returns_false_when_only_preexisting_gateway_detected(self, monkeypatch):
        """The pre-update gateway still draining must NOT satisfy the check."""
        from hermes_cli import gateway, gateway_windows

        write_mock = MagicMock(return_value=Path("/fake/script.cmd"))
        install_mock = MagicMock(return_value=(True, "created"))
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(gateway_windows, "_write_task_script", write_mock)
        monkeypatch.setattr(gateway_windows, "_install_scheduled_task", install_mock)
        monkeypatch.setattr(
            gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", "")
        )
        # find_gateway_pids returns the SAME PID before and after → no new gateway.
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [9999])
        wait_mock = MagicMock(return_value=[9999])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", wait_mock)
        assert gateway_windows._spawn_via_scheduled_task() is False

    def test_launcher_is_ours_true_for_current_template_vbs(self, tmp_path, monkeypatch):
        """_launcher_is_ours() must return True when the on-disk .vbs is
        byte-identical to what _write_task_script would generate.

        Regression for a live-host bug (jrleal10, PR #84409): the helper
        passed the get_hermes_home FUNCTION to _profile_arg instead of its
        result, raising TypeError inside Path(); the broad except swallowed
        it and returned False unconditionally — sending every profile with
        an official .vbs down the delete+create/UAC path on every update.
        This test exercises the REAL body (no mocking of _launcher_is_ours).
        """
        from hermes_cli import gateway, gateway_windows
        from hermes_cli.config import get_hermes_home

        hermes_home = str(Path(get_hermes_home()))
        expected_vbs = gateway_windows._build_gateway_vbs_script(
            gateway_windows._preserve_hermes_home_path(gateway.get_python_path()),
            gateway_windows._stable_gateway_working_dir(gateway.PROJECT_ROOT),
            hermes_home,
            gateway._profile_arg(hermes_home),
        )
        fake_script = tmp_path / "Hermes_Gateway.cmd"
        fake_script.write_text("@echo off\n", encoding="utf-8")
        fake_vbs = fake_script.with_suffix(".vbs")
        fake_vbs.write_bytes(expected_vbs.encode("utf-8"))

        monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: fake_script)
        # The real body must run: no monkeypatch of _launcher_is_ours here.
        assert gateway_windows._launcher_is_ours() is True

    def test_launcher_is_ours_false_when_vbs_customized(self, tmp_path, monkeypatch):
        """A user-customized .vbs must NOT be considered ours."""
        from hermes_cli import gateway, gateway_windows

        fake_script = tmp_path / "Hermes_Gateway.cmd"
        fake_script.write_text("@echo off\n", encoding="utf-8")
        fake_vbs = fake_script.with_suffix(".vbs")
        fake_vbs.write_text("' custom supervisor launcher\nWScript.Quit 0\n", encoding="utf-8")

        monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: fake_script)
        assert gateway_windows._launcher_is_ours() is False

    def test_returns_false_when_script_write_fails(self, monkeypatch):
        """If _write_task_script fails, _spawn_via_scheduled_task must bail."""
        from hermes_cli import gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "_write_task_script", MagicMock(side_effect=OSError("disk full"))
        )
        assert gateway_windows._spawn_via_scheduled_task() is False

    def test_returns_false_when_task_registration_fails(self, monkeypatch):
        """If _install_scheduled_task fails, _spawn_via_scheduled_task must bail."""
        from hermes_cli import gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "_write_task_script", MagicMock(return_value=Path("/fake/script.cmd"))
        )
        monkeypatch.setattr(
            gateway_windows, "_install_scheduled_task", MagicMock(return_value=(False, "error"))
        )
        assert gateway_windows._spawn_via_scheduled_task() is False


class TestWatcherSchtasksBlock:
    """The watcher inline script must trigger the task and check for NEW pids."""

    def test_watcher_script_contains_task_refresh_and_pid_snapshot(self):
        """The embedded watcher script snapshots pre-existing gateway PIDs
        BEFORE the /Run trigger and only counts NEW processes. Script refresh
        was intentionally removed from the hot path (delete+create can hit
        UAC 'Access is denied' on live hosts where /Run alone works, and
        re-registering clobbers user-customized launchers)."""
        from hermes_cli import gateway

        import inspect
        source = inspect.getsource(gateway._spawn_gateway_restart_watcher)

        # 1. No delete+create re-register in the hot path.
        assert "_write_task_script" not in source
        assert "_install_scheduled_task" not in source

        # 2. Pre-existing PIDs must be snapshotted BEFORE the trigger.
        assert source.index("_pre_pids = set(_fgp())") < source.index('"/Run"')
        assert "_new = set(_fgp()) - _pre_pids" in source
        # 3. /Run accepted + empty pre-set counts as success (cold-start race).
        assert "_started_via_task = _ok or not _pre_pids" in source
