"""Windows gateway job-object escape via Scheduled Task (issue #84185).

``CreateProcess`` accepts ``CREATE_BREAKAWAY_FROM_JOB`` silently even when the
parent job denies breakaway, so a directly spawned post-update gateway can land
inside the updater's job and die with it. ``schtasks /Run`` starts outside that
job, so the cold-start prefers the registered task and direct-spawns only when
there is no task. The task route counts only NEW PIDs so a draining pre-update
gateway or a live sibling profile cannot satisfy the check (#110959).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.windows_only


def _cold_start(monkeypatch, *, task_result):
    from hermes_cli import gateway, gateway_windows, update_cmd

    m_main = MagicMock(name="hermes_cli.main")
    m_main._is_windows.return_value = True
    spawn_via = MagicMock(name="_spawn_via_scheduled_task", return_value=task_result)
    spawn_detached = MagicMock(name="_spawn_detached", return_value=54321)
    monkeypatch.setattr(gateway_windows, "_spawn_via_scheduled_task", spawn_via)
    monkeypatch.setattr(gateway_windows, "_spawn_detached", spawn_detached)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: [777])
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **k: None)
    monkeypatch.setattr(gateway_windows, "attested_death_generation", lambda **_k: None)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [])
    monkeypatch.setattr(update_cmd, "_m", lambda: m_main)
    assert update_cmd._cold_start_windows_gateway_after_update({}) is True
    return spawn_via, spawn_detached


def test_task_route_first_direct_spawn_only_when_no_task(monkeypatch):
    spawn_via, spawn_detached = _cold_start(monkeypatch, task_result=[777])
    spawn_via.assert_called_once()
    spawn_detached.assert_not_called()  # a direct spawn beside the task-spawned gateway would race the port

    spawn_via, spawn_detached = _cold_start(monkeypatch, task_result=None)  # no task registered
    spawn_via.assert_called_once()
    spawn_detached.assert_called_once()


def test_preexisting_pid_never_counts_as_newly_spawned(monkeypatch):
    """A live sibling (or draining pre-update gateway) must not satisfy the task-route check on
    its own, and a new pid appearing beside it must."""
    from hermes_cli import gateway_windows

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda home=None: "HermesGateway")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda task_name=None: True)
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda *a, **kw: (0, "", ""))
    monkeypatch.setattr(gateway_windows, "_confirm_gateway_stable", lambda pids, *a, **k: pids)
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda s: None)

    monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda **kw: [9999])
    assert gateway_windows._spawn_via_scheduled_task(home="C:\\h", timeout_s=0.05) == []

    probes = iter([[9999], [9999], [9999, 10001]])
    monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda **kw: next(probes))
    assert gateway_windows._spawn_via_scheduled_task(home="C:\\h", timeout_s=5) == [10001]
