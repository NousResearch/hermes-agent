"""Release of a restart-safe cron worker's transient scope unit once its ledger row is terminal.

A ``--collect`` scope stays ``loaded active running`` while anything lives in its cgroup, so a
background job that outlived the execution held the unit alive for days after the row
completed. The gateway's waiter releases it, by the exact unit name only.
"""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest

from cron import scheduler_detached_worker as detached


class _FakeProcess:
    def __init__(self, pid: int, *, hangs: bool = False):
        self.pid = pid
        self._hangs = hangs
        self.killed = False

    def wait(self, timeout=None):
        if self._hangs and not self.killed:
            raise subprocess.TimeoutExpired("worker", timeout)
        return 0

    def kill(self):
        self.killed = True


def _run_release(monkeypatch, process, unit, *, active: bool, grace: float = 0.05):
    stops: list = []
    monkeypatch.setattr(detached, "_scope_unit_is_active", lambda name: active)
    monkeypatch.setattr("tools.process_registry._stop_systemd_unit", lambda *a: stops.append(a) or True)
    detached._release_scope_unit(process, unit, grace_seconds=grace, poll_seconds=0.01, deadline_seconds=0.01)
    return stops


def test_release_stops_exact_unit_when_active_after_worker_exit(monkeypatch):
    unit = "hermes-worker-cron-j1-exec-e1.scope"
    stops = _run_release(monkeypatch, _FakeProcess(4101), unit, active=True)
    assert stops == [(unit,)]


def test_release_skips_when_unit_already_inactive(monkeypatch):
    stops = _run_release(monkeypatch, _FakeProcess(4102), "hermes-worker-cron-j1-exec-e2.scope", active=False)
    assert stops == []


def test_release_kills_worker_at_deadline(monkeypatch):
    process = _FakeProcess(4103, hangs=True)
    unit = "hermes-worker-cron-j1-exec-e3.scope"
    stops = _run_release(monkeypatch, process, unit, active=True, grace=0.02)
    assert process.killed
    assert stops == [(unit,)]


def test_body_schedules_release_on_terminal(monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(scheduler, "get_execution", lambda _eid: {"id": "e1", "status": "completed"},
                        raising=False)
    release, reap = Mock(), Mock()
    monkeypatch.setattr(detached, "schedule_scope_unit_release", release)
    monkeypatch.setattr(detached, "reap_terminal_worker_in_background", reap)
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired("worker", 1.0)
    process.poll.return_value = None

    unit = "hermes-worker-cron-j1-exec-e1.scope"
    assert scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=unit)
    release.assert_called_once_with(process, unit)
    reap.assert_not_called()

    release.reset_mock()
    assert scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=None)
    release.assert_not_called()
    reap.assert_called_once_with(process)


def test_body_schedules_release_after_worker_exit_with_terminal_row(monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(scheduler, "get_execution", lambda _eid: {"id": "e1", "status": "failed"},
                        raising=False)
    release = Mock()
    monkeypatch.setattr(detached, "schedule_scope_unit_release", release)
    process = Mock()
    process.wait.return_value = 1

    unit = "hermes-worker-cron-j1-exec-e1.scope"
    assert scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=unit)
    release.assert_called_once_with(process, unit)


def test_scope_unit_name_matches_dispatch_argv(monkeypatch):
    import tools.process_registry as pr

    monkeypatch.setattr(pr, "_IS_LINUX", True)
    monkeypatch.setenv("INVOCATION_ID", "test")
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(pr, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    dispatch = pr.restart_safe_gateway_child_argv(
        ["true"], unit_suffix="cron-j1-exec-e1", require_restart_safe_scope=True)
    # systemd appends ``.scope`` to a ``--scope`` unit name given without a suffix.
    assert dispatch.mode == "scoped"
    assert dispatch.unit == f"{dispatch.argv[dispatch.argv.index('--unit') + 1]}.scope"
    assert dispatch.unit == "hermes-worker-cron-j1-exec-e1.scope"


def test_body_releases_scope_on_dead_owner_terminalization(monkeypatch):
    """The crash path terminalizes the row and RAISES for bookkeeping (#128509) — the
    release must still be scheduled there: the dead worker's cgroup stragglers are the
    leak class, and this is where they most often come from."""
    from cron import scheduler

    monkeypatch.setattr(scheduler, "get_execution",
                        lambda _eid: {"id": "e1", "status": "running"}, raising=False)
    monkeypatch.setattr(scheduler, "terminalize_dead_owner",
                        lambda _eid, reason=None: True, raising=False)
    release = Mock()
    monkeypatch.setattr(detached, "schedule_scope_unit_release", release)
    process = Mock()
    process.wait.return_value = -15  # worker died first with the row still open

    unit = "hermes-worker-cron-j1-exec-e1.scope"
    with pytest.raises(RuntimeError):
        scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=unit)
    release.assert_called_once_with(process, unit)

    # scope_unit=None keeps the pre-fix behavior on the raise path: no release, same raise.
    release.reset_mock()
    with pytest.raises(RuntimeError):
        scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=None)
    release.assert_not_called()


def _dead_worker_body(monkeypatch, status: str):
    from cron import scheduler

    monkeypatch.setattr(scheduler, "get_execution",
                        lambda _eid: {"id": "e1", "status": status}, raising=False)
    monkeypatch.setattr(scheduler, "terminalize_dead_owner",
                        lambda _eid, reason=None: False, raising=False)
    monkeypatch.setattr(scheduler, "recover_interrupted_executions", lambda: None, raising=False)
    release = Mock()
    monkeypatch.setattr(detached, "schedule_scope_unit_release", release)
    process = Mock()
    process.wait.return_value = -9
    return scheduler, process, release


def test_body_releases_scope_before_raise_on_unknown_terminal_row(monkeypatch):
    """A worker that recorded ``unknown`` itself exits cleanly; the narrow re-read gates
    still raise (death evidence), but the raise must not leak the scope unit."""
    scheduler, process, release = _dead_worker_body(monkeypatch, "unknown")

    unit = "hermes-worker-cron-j1-exec-e1.scope"
    with pytest.raises(RuntimeError):
        scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=unit)
    release.assert_called_once_with(process, unit)

    release.reset_mock()
    with pytest.raises(RuntimeError):
        scheduler._wait_for_external_cron_worker_body(process, execution_id="e1", scope_unit=None)
    release.assert_not_called()


def test_body_never_releases_while_row_open(monkeypatch):
    scheduler, process, release = _dead_worker_body(monkeypatch, "running")

    with pytest.raises(RuntimeError):
        scheduler._wait_for_external_cron_worker_body(
            process, execution_id="e1", scope_unit="hermes-worker-cron-j1-exec-e1.scope")
    release.assert_not_called()


@pytest.mark.parametrize(("state", "expected"), [
    ("active", True), ("activating", True), ("inactive", False),
])
def test_scope_unit_is_active_reads_systemctl_state(monkeypatch, state, expected):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(detached.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=f"{state}\n".encode()))
    assert detached._scope_unit_is_active("hermes-worker-cron-j1-exec-e1.scope") is expected


def test_scope_unit_is_active_fails_open_loudly_when_systemctl_unreachable(monkeypatch, caplog):
    def _boom(*_a, **_k):
        raise OSError("no user bus")

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(detached.subprocess, "run", _boom)
    with caplog.at_level("WARNING", logger=detached.logger.name):
        assert detached._scope_unit_is_active("hermes-worker-cron-j1-exec-e1.scope") is False
    assert any(r.levelname == "WARNING" for r in caplog.records)
