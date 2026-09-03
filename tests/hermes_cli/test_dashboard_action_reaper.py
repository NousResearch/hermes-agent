"""Tests for the background dashboard-action reaper (#89060).

Action children are spawned detached and, before this loop existed, were only
``wait()``ed when the UI polled ``GET /api/actions/{name}/status``. A headless
dashboard has no browser to poll it, so finished children stayed ``<defunct>``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, cast
from unittest.mock import patch

import pytest


class _FinishedProc:
    def __init__(self, pid: int, code: int = 0):
        self.pid = pid
        self._code = code
        self.wait_calls = 0

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        assert timeout == 0
        self.wait_calls += 1
        return self._code


class _RunningThenFinishedProc(_FinishedProc):
    def __init__(self, pid: int, code: int = 0):
        super().__init__(pid, code)
        self._running = True

    def poll(self):
        if self._running:
            return None
        return self._code

    def finish(self):
        self._running = False


class _NeverExitsProc(_FinishedProc):
    def poll(self):
        return None


@pytest.mark.asyncio
async def test_action_reaper_reaps_completed_registered_action():
    from hermes_cli import web_server

    proc = _FinishedProc(pid=1234, code=7)
    web_server._ACTION_PROCS["unit-action"] = cast(Any, proc)
    web_server._ACTION_COMMANDS["unit-action"] = ("doctor",)
    web_server._ACTION_IDS["unit-action"] = "deadbeef"
    web_server._ACTION_RESULTS.pop("unit-action", None)

    task = asyncio.create_task(web_server.run_action_reaper(interval=0.001))
    try:
        for _ in range(20):
            await asyncio.sleep(0.01)
            if "unit-action" not in web_server._ACTION_PROCS:
                break
        assert "unit-action" not in web_server._ACTION_PROCS
        assert "unit-action" not in web_server._ACTION_COMMANDS
        # Every other retirement path (_record_completed_action, the status
        # endpoint) drops the action id with the handle; so does this one.
        assert "unit-action" not in web_server._ACTION_IDS
        assert web_server._ACTION_RESULTS["unit-action"] == {"exit_code": 7, "pid": 1234}
        assert proc.wait_calls == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        web_server._ACTION_PROCS.pop("unit-action", None)
        web_server._ACTION_COMMANDS.pop("unit-action", None)
        web_server._ACTION_IDS.pop("unit-action", None)
        web_server._ACTION_RESULTS.pop("unit-action", None)


@pytest.mark.asyncio
async def test_action_reaper_leaves_a_live_registered_action_alone():
    """A running child keeps its registry slot — reaping it would break status."""
    from hermes_cli import web_server

    proc = _NeverExitsProc(pid=4321)
    web_server._ACTION_PROCS["unit-live"] = cast(Any, proc)
    web_server._ACTION_COMMANDS["unit-live"] = ("gateway", "restart")
    web_server._ACTION_RESULTS.pop("unit-live", None)

    task = asyncio.create_task(web_server.run_action_reaper(interval=0.001))
    try:
        await asyncio.sleep(0.05)
        assert web_server._ACTION_PROCS.get("unit-live") is proc
        assert "unit-live" in web_server._ACTION_COMMANDS
        assert "unit-live" not in web_server._ACTION_RESULTS
        assert proc.wait_calls == 0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        web_server._ACTION_PROCS.pop("unit-live", None)
        web_server._ACTION_COMMANDS.pop("unit-live", None)
        web_server._ACTION_RESULTS.pop("unit-live", None)


@pytest.mark.asyncio
async def test_action_reaper_drains_retired_same_name_orphan():
    from hermes_cli import web_server

    proc = _RunningThenFinishedProc(pid=5678)
    with web_server._ACTION_ORPHANS_LOCK:
        web_server._ACTION_ORPHANS.clear()

    web_server._retire_action_proc(cast(Any, proc))
    with web_server._ACTION_ORPHANS_LOCK:
        assert web_server._ACTION_ORPHANS == [proc]

    proc.finish()
    task = asyncio.create_task(web_server.run_action_reaper(interval=0.001))
    try:
        for _ in range(20):
            await asyncio.sleep(0.01)
            with web_server._ACTION_ORPHANS_LOCK:
                if not web_server._ACTION_ORPHANS:
                    break
        with web_server._ACTION_ORPHANS_LOCK:
            assert web_server._ACTION_ORPHANS == []
        assert proc.wait_calls == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with web_server._ACTION_ORPHANS_LOCK:
            web_server._ACTION_ORPHANS.clear()


def test_lifespan_starts_the_action_reaper():
    """Without this, dropping the ``create_task`` line leaves every unit test green.

    Scoped to startup on purpose: the loop's own teardown cancels a pending
    task regardless, so a shutdown-cancellation assertion here would pass even
    with ``action_reaper_task.cancel()`` deleted.
    """
    from fastapi.testclient import TestClient

    from hermes_cli import web_server

    state: dict[str, Any] = {"task": None}

    async def _stub_reaper():
        state["task"] = asyncio.current_task()
        await asyncio.sleep(3600)

    with patch.object(web_server, "run_action_reaper", _stub_reaper):
        with TestClient(web_server.app, raise_server_exceptions=False):
            for _ in range(200):
                if state["task"] is not None:
                    break
                time.sleep(0.01)
            assert state["task"] is not None, "_lifespan never started run_action_reaper"
