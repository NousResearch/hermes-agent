"""An orphaned planned-stop marker (dead stopper) must self-cancel the restart drain.

Regression for the 2026-09-14 incident: ``hermes gateway restart`` run through a terminal
with a short timeout wrote the planned-stop marker, the CLI (the only process that performs
a plain restart) was killed before the after-turn drain finished, and the gateway spent the
whole cap shedding new runs with 503s — then would have exited cleanly with nothing left to
revive it.

Contracts:
  * plain restart + dead stopper -> drain cancels: flags reset, marker cleared, stop() never
    called, service resumes
  * via_service + live marker naming a dead stopper -> same cancel (2026-09-15 incident: the
    stopper that took responsibility died mid-drain, so nobody performs the service restart)
  * live stopper / detached / no marker (incl. marker-less via_service requesters) -> the
    drain proceeds exactly as before
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import status
from gateway.status import planned_stop_stopper_alive
from gateway.run_shutdown import GatewayShutdownMixin, _RestartRequesterGone
from tests.gateway.restart_test_helpers import make_restart_runner

MARKER_NAME = status._PLANNED_STOP_MARKER_FILENAME


def _write_marker(path: Path, stopper_pid: int) -> None:
    path.write_text(
        json.dumps({
            "target_pid": os.getpid(),
            "target_start_time": None,
            "stopper_pid": stopper_pid,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
    )


def _dead_pid() -> int:
    for _ in range(10):
        proc = subprocess.Popen(["true"])
        proc.wait()
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            return proc.pid
    pytest.fail("could not obtain a dead pid")


def _runner_with_active_work():
    """Runner whose drain sees one active work unit on the first poll, zero afterwards."""
    runner, _adapter = make_restart_runner()
    polls = {"n": 0}

    def _count() -> int:
        polls["n"] += 1
        return 1 if polls["n"] < 2 else 0

    runner._active_work_count = MagicMock(side_effect=_count)
    runner._awaitable_work_count = MagicMock(side_effect=_count)
    runner._wedged_agent_count = MagicMock(return_value=0)
    runner._scale_to_zero_status = MagicMock()
    runner.stop = AsyncMock()
    return runner


def _marker_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    marker = tmp_path / MARKER_NAME
    monkeypatch.setattr(status, "_get_planned_stop_marker_path", lambda: marker)
    return marker


# ── status.planned_stop_stopper_alive ────────────────────────────────────────


def test_stopper_alive_with_no_marker(tmp_path, monkeypatch):
    _marker_path(tmp_path, monkeypatch)
    assert planned_stop_stopper_alive() is True


def test_stopper_alive_false_for_dead_stopper(tmp_path, monkeypatch):
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, _dead_pid())
    assert planned_stop_stopper_alive() is False


def test_stopper_alive_true_for_live_stopper(tmp_path, monkeypatch):
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, os.getpid())  # this test process is a live stopper
    assert planned_stop_stopper_alive() is True


def test_stopper_alive_true_for_malformed_marker(tmp_path, monkeypatch):
    marker = _marker_path(tmp_path, monkeypatch)
    marker.write_text(json.dumps({"target_pid": os.getpid(), "written_at": "now"}))
    assert planned_stop_stopper_alive() is True


# ── drain self-cancel ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dead_stopper_cancels_drain_and_resumes(tmp_path, monkeypatch):
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, _dead_pid())
    runner = _runner_with_active_work()

    assert runner.request_restart() is True
    await asyncio.wait_for(runner._restart_task, 5)

    assert marker.exists() is False  # orphaned marker cleared
    assert runner._draining is False
    assert runner._restart_requested is False
    assert runner._restart_task_started is False
    runner.stop.assert_not_awaited()  # gateway keeps serving


@pytest.mark.asyncio
async def test_live_stopper_still_drains_and_stops(tmp_path, monkeypatch):
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, os.getpid())
    runner = _runner_with_active_work()

    assert runner.request_restart() is True
    await asyncio.wait_for(runner._restart_task, 5)

    runner.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_via_service_cancels_when_live_markers_stopper_is_dead(tmp_path, monkeypatch):
    """2026-09-15 incident: a via_service restart whose live marker names a dead stopper is
    orphaned in the plain-restart sense — nobody performs the service restart — so the drain
    cancels exactly like a plain restart (flags reset, marker cleared, service resumes)."""
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, _dead_pid())
    runner = _runner_with_active_work()

    assert runner.request_restart(via_service=True) is True
    await asyncio.wait_for(runner._restart_task, 5)

    assert marker.exists() is False  # orphaned marker cleared
    assert runner._draining is False
    assert runner._restart_requested is False
    assert runner._restart_task_started is False
    runner.stop.assert_not_awaited()  # gateway keeps serving


@pytest.mark.asyncio
async def test_via_service_no_marker_drains_untouched(tmp_path, monkeypatch):
    """Marker-less via_service requesters (updater SIGUSR1, control-socket pause-for-update)
    write no marker: nothing opts them into the orphan probe, drain/stop exactly as before."""
    _marker_path(tmp_path, monkeypatch)
    runner = _runner_with_active_work()

    assert runner.request_restart(via_service=True) is True
    await asyncio.wait_for(runner._restart_task, 5)

    runner.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_detached_restart_ignores_dead_stopper(tmp_path, monkeypatch):
    """A detached restart is completed by the helper, not the stopper: never cancel."""
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, _dead_pid())
    runner = _runner_with_active_work()
    runner._launch_detached_restart_command = AsyncMock()

    assert runner.request_restart(detached=True) is True
    await asyncio.wait_for(runner._restart_task, 5)

    runner._launch_detached_restart_command.assert_awaited_once()
    runner.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_marker_restart_unaffected(tmp_path, monkeypatch):
    """Marker-less restarts (API, slash command) have no stopper to lose: drain as before."""
    _marker_path(tmp_path, monkeypatch)
    runner = _runner_with_active_work()

    assert runner.request_restart() is True
    await asyncio.wait_for(runner._restart_task, 5)

    runner.stop.assert_awaited_once()


def test_await_raises_for_dead_stopper_direct_call(tmp_path, monkeypatch):
    """The wait itself raises ``_RestartRequesterGone`` so any future caller inherits the guard."""
    marker = _marker_path(tmp_path, monkeypatch)
    _write_marker(marker, _dead_pid())
    runner = _runner_with_active_work()

    coro = GatewayShutdownMixin._await_active_work_before_restart(runner)
    with pytest.raises(_RestartRequesterGone):
        asyncio.run(coro)
