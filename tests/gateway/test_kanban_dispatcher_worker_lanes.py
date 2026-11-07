"""The gateway's embedded Kanban dispatcher hands matching tasks to plugin worker lanes and counts
them as spawnable work, through the same thread hop the gateway uses."""
from __future__ import annotations

import asyncio
from pathlib import Path

from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _to_thread_process_service
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_worker_lane_registry as registry
from hermes_cli.kanban_worker_lanes import KanbanWorkerLane
from hermes_constants import hermes_home_key


class _Lane(KanbanWorkerLane):
    assignees = ("codex-*",)
    name = "external"

    def __init__(self):
        self.claims = []

    def spawn(self, claim):
        self.claims.append(claim)


def test_embedded_dispatcher_runs_a_lane_task_and_reports_it_spawnable(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    registry._reset_for_tests()
    lane = _Lane()
    registry.register_provider(lane, scope=hermes_home_key())
    with kbc.connect_closing(board="default") as conn:
        task_id = kb.create_task(conn, title="external work", assignee="codex-1")
    dispatcher = kwd._KanbanDispatcher(kb, kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None))
    try:
        assert dispatcher.ready_nonempty()
        result = asyncio.run(_to_thread_process_service(dispatcher.tick_once_for_board, "default"))
    finally:
        registry._reset_for_tests()

    assert [c.task_id for c in lane.claims] == [task_id]
    assert [s[0] for s in result.spawned] == [task_id]
    assert not dispatcher.ready_nonempty()
