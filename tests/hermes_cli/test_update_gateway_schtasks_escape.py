"""Windows gateway job-object escape via Scheduled Task (issue #84185).

The bug: post-update spawns used ``subprocess.Popen`` + ``CREATE_BREAKAWAY_FROM_JOB``.
``CreateProcess`` accepts that flag silently even when the parent job denies
breakaway, so the child lands inside the job and is killed when the updater
exits.

The fix: prefer ``schtasks /Run`` when a task is registered (Task Scheduler
runs outside any job holding the updater). Falls back to direct spawn only
when no task exists, gated on the shared liveness poll. The poll counts
only NEW PIDs so a draining pre-update gateway cannot satisfy the check.

This suite keeps the two critical invariants only — no launcher-reinstall,
no /End retry, no source-inspection tests (AGENTS.md).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.windows_only


@pytest.fixture
def cold_start_mocks(monkeypatch):
    from hermes_cli import gateway, gateway_windows, update_cmd

    m_main = MagicMock(name="hermes_cli.main")
    m_main._is_windows.return_value = True
    spawn_via = MagicMock(name="_spawn_via_scheduled_task", return_value=False)
    spawn_detached = MagicMock(name="_spawn_detached", return_value=0)
    report = MagicMock(name="_report_gateway_start")
    monkeypatch.setattr(gateway_windows, "_spawn_via_scheduled_task", spawn_via)
    monkeypatch.setattr(gateway_windows, "_spawn_detached", spawn_detached)
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", report)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
    monkeypatch.setattr(update_cmd, "_m", lambda: m_main)

    class _NS:
        pass

    ns = _NS()
    ns.spawn_via_schtasks = spawn_via
    ns.spawn_detached = spawn_detached
    ns.report = report
    return ns


class TestColdStartEscape:
    def test_prefers_scheduled_task_when_registered(self, capsys, cold_start_mocks):
        from hermes_cli import update_cmd

        cold_start_mocks.spawn_via_schtasks.return_value = True
        update_cmd._cold_start_windows_gateway_after_update()
        cold_start_mocks.spawn_via_schtasks.assert_called_once()
        cold_start_mocks.spawn_detached.assert_not_called()
        cold_start_mocks.report.assert_called_once()
        assert "Scheduled Task" in cold_start_mocks.report.call_args.args[0]
        assert cold_start_mocks.report.call_args.kwargs.get("all_profiles") is True

    def test_task_path_fails_loudly_when_repoll_misses(self, cold_start_mocks):
        """Task accepted but the shared re-poll sees nothing: raise, never
        silently claim success and never fall back to a direct spawn (which
        cannot escape the parent job and would race the task-spawned process)."""
        from hermes_cli import update_cmd

        cold_start_mocks.spawn_via_schtasks.return_value = True
        cold_start_mocks.report.return_value = []
        with pytest.raises(RuntimeError, match="via Scheduled Task did not become ready"):
            update_cmd._cold_start_windows_gateway_after_update()
        cold_start_mocks.spawn_detached.assert_not_called()

    def test_falls_back_to_direct_spawn_with_liveness_gate_when_no_task(self, cold_start_mocks):
        from hermes_cli import update_cmd

        cold_start_mocks.spawn_via_schtasks.return_value = False
        cold_start_mocks.spawn_detached.return_value = 54321
        update_cmd._cold_start_windows_gateway_after_update()
        cold_start_mocks.spawn_via_schtasks.assert_called_once()
        cold_start_mocks.spawn_detached.assert_called_once()
        cold_start_mocks.report.assert_called_once()
        assert "54321" in cold_start_mocks.report.call_args.args[0]


class TestSpawnViaScheduledTaskHelper:
    def test_new_pid_only_preexisting_does_not_count(self, monkeypatch):
        """A pre-update gateway still draining must not satisfy the task-route check."""
        from hermes_cli import gateway, gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: True)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", ""))
        # same pid before and after -> no new gateway
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [9999])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [9999])
        assert gateway_windows._spawn_via_scheduled_task() is False

    def test_snapshots_pids_before_trigger(self, monkeypatch):
        from hermes_cli import gateway, gateway_windows

        order: list[str] = []

        def fake_find(**kw):
            order.append("find")
            return []

        def fake_exec(*a, **kw):
            order.append("run")
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: True)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_exec)
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [7])
        monkeypatch.setattr(gateway, "find_gateway_pids", fake_find)
        assert gateway_windows._spawn_via_scheduled_task() is True
        assert order[:2] == ["find", "run"]
