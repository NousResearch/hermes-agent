"""Windows gateway job-object escape via Scheduled Task (issue #84185).

The bug: post-update spawns used ``subprocess.Popen`` + ``CREATE_BREAKAWAY_FROM_JOB``.
``CreateProcess`` accepts that flag silently even when the parent job denies
breakaway, so the child lands inside the job and is killed when the updater
exits.

The fix: prefer ``schtasks /Run`` when a task is registered (Task Scheduler
runs outside any job holding the updater). Falls back to direct spawn only
when nothing was triggered (no task, or ``/Run`` rejected with nothing
spawned); an accepted trigger with no NEW pid within the poll window raises
instead, so no caller ever falls back into a racing second spawn. The poll
counts only NEW PIDs so a draining pre-update gateway cannot satisfy the
check.

This suite keeps the critical invariants only — no launcher-reinstall,
no /End retry, no source-inspection tests (AGENTS.md).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.platforms("windows")


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
        assert cold_start_mocks.report.call_args.args[0] == "cold-start after update"  # main's success-line contract


class TestSpawnViaScheduledTaskHelper:
    def test_accepted_trigger_without_new_pid_raises(self, monkeypatch):
        """/Run accepted but the poll sees only a pre-existing (draining) PID:
        the task-spawned gateway may still be starting — raise so no caller
        falls back into a racing second spawn (#84185)."""
        from hermes_cli import gateway, gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: True)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", ""))
        # same pid before and after -> no NEW gateway
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [9999])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [9999])
        with pytest.raises(RuntimeError, match="no new gateway PID"):
            gateway_windows._spawn_via_scheduled_task()

    def test_no_task_registered_returns_false_without_triggering(self, monkeypatch):
        """No task at all: the contract ``False`` callers use to decide the direct-spawn fallback."""
        from hermes_cli import gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: False)
        exec_schtasks = MagicMock(name="_exec_schtasks")
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", exec_schtasks)
        assert gateway_windows._spawn_via_scheduled_task() is False
        exec_schtasks.assert_not_called()

    def test_task_name_derives_from_spawned_home_not_caller_env(self, monkeypatch):
        """Profile-scoped callers pass only ``hermes_home``: the task name must come
        from THAT home — resolving it against the caller's env would trigger task A
        while starting profile B (the profile-blind class)."""
        from hermes_cli import gateway_windows

        seen: dict = {}
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)

        def fake_get_task_name(home=None):
            seen["task_name_home"] = home
            return "TASK"

        def fake_is_registered(task_name=None):
            seen["task_name"] = task_name
            return False

        monkeypatch.setattr(gateway_windows, "get_task_name", fake_get_task_name)
        monkeypatch.setattr(gateway_windows, "is_task_registered", fake_is_registered)
        assert gateway_windows._spawn_via_scheduled_task(hermes_home="/x/profile-home") is False
        assert seen == {"task_name_home": "/x/profile-home", "task_name": "TASK"}

    def test_rejected_trigger_with_nothing_spawned_returns_false(self, monkeypatch):
        """/Run rejected and no NEW pid after the wait: nothing to race, the fallback is safe."""
        from hermes_cli import gateway, gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: True)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda *a, **kw: (1, "", "denied"))
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [9999])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [9999])
        assert gateway_windows._spawn_via_scheduled_task() is False

    def test_rejected_trigger_but_new_pid_returns_true(self, monkeypatch):
        """/Run rejected because the task is ALREADY RUNNING from a previous trigger,
        yet a NEW gateway appeared: that is the wanted outcome — never race it."""
        from hermes_cli import gateway, gateway_windows

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: True)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda *a, **kw: (1, "", "already running"))
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [7])
        assert gateway_windows._spawn_via_scheduled_task() is True

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


class TestAttestedMultiProfileEscape:
    """#110959: the attested multi-profile cold-start must take the same job-object
    escape (#84185) as the active-profile one, and a helper raise must keep the
    profile pending for the aggregate error — never fall back to a second spawn."""

    @pytest.fixture
    def attested_env(self, monkeypatch, tmp_path):
        from hermes_cli import gateway_windows
        import hermes_cli.profiles as profiles

        monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda **kw: [])
        spawn_via = MagicMock(name="_spawn_via_scheduled_task", return_value=True)
        spawn_detached = MagicMock(name="_spawn_detached", return_value=777)
        monkeypatch.setattr(gateway_windows, "_spawn_via_scheduled_task", spawn_via)
        monkeypatch.setattr(gateway_windows, "_spawn_detached", spawn_detached)
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kw: [777])
        monkeypatch.setattr(gateway_windows, "_consume_start_attestation", lambda *a, **kw: None)
        monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **kw: None)
        monkeypatch.setattr(profiles, "get_profile_dir", lambda name: tmp_path / name)

        return SimpleNamespace(spawn_via=spawn_via, spawn_detached=spawn_detached)

    def test_uses_task_route_with_per_profile_home(self, attested_env):
        from hermes_cli import update_cmd_windows

        update_cmd_windows._cold_start_attested_profiles({"cold_start_profiles": {"alpha": "gen1"}})
        attested_env.spawn_via.assert_called_once()
        kwargs = attested_env.spawn_via.call_args.kwargs
        assert set(kwargs) == {"hermes_home"}
        assert kwargs["hermes_home"].endswith("alpha")
        attested_env.spawn_detached.assert_not_called()

    def test_falls_back_only_when_task_route_declines(self, attested_env):
        from hermes_cli import update_cmd_windows

        attested_env.spawn_via.return_value = False
        update_cmd_windows._cold_start_attested_profiles({"cold_start_profiles": {"alpha": "gen1"}})
        attested_env.spawn_detached.assert_called_once()
        assert str(attested_env.spawn_detached.call_args.kwargs["home"]).endswith("alpha")

    def test_helper_raise_keeps_profile_pending_and_never_double_spawns(self, attested_env):
        from hermes_cli import update_cmd_windows

        attested_env.spawn_via.side_effect = RuntimeError("no new gateway PID within 30.0s")
        token = {"cold_start_profiles": {"alpha": "gen1"}}
        with pytest.raises(RuntimeError, match="not verified for profile"):
            update_cmd_windows._cold_start_attested_profiles(token)
        attested_env.spawn_detached.assert_not_called()
        assert "alpha" in str(token.get("cold_start_profiles"))


class TestWatcherTaskRoute:
    """The restart-watcher template must never fall through from a failed task route
    into a direct Popen that races the task-spawned gateway (#84185). Renders the
    real script without spawning it, then execs it with the spawn points recorded."""

    @pytest.fixture
    def watcher_env(self, monkeypatch, tmp_path):
        import sys as _sys

        from hermes_cli import gateway, gateway_windows
        from gateway import status as gw_status

        records: list[list] = []

        def record_popen(*args, **kwargs):
            records.append(list(args))
            return MagicMock(name="popen-child")

        monkeypatch.setattr(gateway.subprocess, "Popen", record_popen)
        assert gateway._spawn_gateway_restart_watcher(
            987654, ["python.exe", "-m", "hermes_cli.main", "gateway", "run"]
        )
        assert records, "the watcher itself must spawn"
        watcher_argv = records.pop()[0]
        script = watcher_argv[2]

        monkeypatch.setattr(gw_status, "_pid_exists", lambda _p: False)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
        monkeypatch.setattr(gateway_windows, "get_task_name", lambda home=None: f"TASK:{home}")

        def run(*, task_registered: bool, helper):
            monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: task_registered)
            monkeypatch.setattr(gateway_windows, "_spawn_via_scheduled_task", helper)
            g = {"__name__": "__main__"}
            monkeypatch.setattr(_sys, "argv", ["-c", *watcher_argv[3:]])
            try:
                exec(compile(script, "<watcher>", "exec"), g)
            finally:
                g.clear()

        return SimpleNamespace(run=run, records=records, expected_cmd=watcher_argv[4:])

    def test_task_timeout_exits_without_direct_popen(self, watcher_env):
        helper = MagicMock(
            name="_spawn_via_scheduled_task",
            side_effect=RuntimeError("no new gateway PID within 30.0s"),
        )
        with pytest.raises(SystemExit) as excinfo:
            watcher_env.run(task_registered=True, helper=helper)
        assert excinfo.value.code == 1
        helper.assert_called_once()
        assert helper.call_args.kwargs["task_name"].startswith("TASK:")
        assert watcher_env.records == [], (
            "a direct Popen after an unconfirmed /Run would race the task-spawned gateway"
        )

    def test_direct_respawn_when_no_task(self, watcher_env):
        helper = MagicMock(name="_spawn_via_scheduled_task")
        watcher_env.run(task_registered=False, helper=helper)
        helper.assert_not_called()
        assert watcher_env.records, "no task must fall back to the direct respawn"
        assert watcher_env.records[0][0] == watcher_env.expected_cmd
