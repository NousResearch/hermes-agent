"""Fail-closed restart accounting for embedded Kanban workers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.run_shutdown import GatewayShutdownMixin
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


class _Watcher(GatewayKanbanWatchersMixin):
    pass


class _Conn:
    def close(self):
        pass


def test_worker_scan_fails_closed_on_board_enumeration_error(monkeypatch):
    def fail_enumeration(*, include_archived=False):
        raise OSError("boards root unavailable")

    monkeypatch.setattr(kb, "list_boards", fail_enumeration)

    with pytest.raises(kbd.KanbanWorkerScanError, match="board enumeration failed"):
        _Watcher()._kanban_running_workers()


def test_worker_scan_fails_closed_on_later_board_error(monkeypatch):
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda **_: [{"slug": "jarvis-os"}, {"slug": "sycode-trading"}],
    )
    monkeypatch.setattr(kbc, "connect", lambda *, board: _Conn())

    calls = []

    def scan_board(_conn, *, include_wedged=False):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("second board unavailable")
        return []

    monkeypatch.setattr(kbd, "list_running_workers", scan_board)

    with pytest.raises(kbd.KanbanWorkerScanError, match="board sycode-trading"):
        _Watcher()._kanban_running_workers()


def test_force_cleanup_propagates_scan_failure_without_returning_zero(monkeypatch):
    monkeypatch.setattr(kb, "list_boards", lambda **_: [{"slug": "jarvis-os"}])
    monkeypatch.setattr(kbc, "connect", lambda *, board: _Conn())
    monkeypatch.setattr(
        kbd,
        "list_running_workers",
        lambda _conn, **_: (_ for _ in ()).throw(OSError("scan unavailable")),
    )

    with pytest.raises(kbd.KanbanWorkerScanError):
        _Watcher()._interrupt_kanban_workers_for_restart()


def test_active_work_count_propagates_kanban_scan_error():
    class Runner(GatewayShutdownMixin):
        def _running_agent_count(self):
            return 0

        def _active_cron_job_count(self):
            return 0

        def _active_api_run_count(self):
            return 0

        def _active_deferred_agent_worker_count(self):
            return 0

        def _active_kanban_worker_count(self):
            raise kbd.KanbanWorkerScanError("board scan failed")

    with pytest.raises(kbd.KanbanWorkerScanError):
        Runner()._active_work_count()


@pytest.mark.asyncio
async def test_restart_stays_draining_when_kanban_scan_fails():
    class Runner(GatewayShutdownMixin):
        _restart_after_turn_timeout = 1800

        def _running_agent_count(self):
            return 0

        def _active_cron_job_count(self):
            return 0

        def _active_api_run_count(self):
            return 0

        def _active_deferred_agent_worker_count(self):
            return 0

        def _active_kanban_worker_count(self):
            raise kbd.KanbanWorkerScanError("board scan failed")

        def _scale_to_zero_status(self, *args, **kwargs):
            pass

    runner = Runner()
    runner._restart_task_started = False
    runner.stop = AsyncMock()

    assert runner.request_restart(detached=False, via_service=True) is True
    await runner._restart_task

    runner.stop.assert_not_awaited()
    assert runner._draining is True
    assert runner._restart_task_started is False
