"""Plugin worker lanes inside one dispatcher tick: who may run a task, and the spawn call.

Contract and plugin API: :mod:`hermes_cli.kanban_worker_lanes`. Each lane's code runs under the
Hermes home of the profile that registered it, as plugin loading does, because the dispatcher's
thread is bound to the launch profile.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from hermes_cli.kanban_worker_lanes import NOT_HANDLED, KanbanWorkerLane, WorkerLaneClaim

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

logger = logging.getLogger(__name__)


def _board_slug(board: Optional[str]) -> str:
    """The slug a lane's ``boards`` filter sees: ``board``, else the board this process resolves."""
    if board:
        return board
    from hermes_cli import kanban_db
    return kanban_db.get_current_board()


@contextlib.contextmanager
def _lane_scope(scope: Optional[str]):
    if scope is None:
        yield
        return
    from hermes_cli.plugins_loader import _plugin_home_scope
    with _plugin_home_scope(Path(scope)):
        yield


def _lanes(assignee: Any, board: Optional[str]) -> list[tuple[Optional[str], KanbanWorkerLane]]:
    from hermes_cli.kanban_worker_lane_registry import matching_lanes
    return matching_lanes(assignee, _board_slug(board))


def _accepting(task: Task, board: Optional[str], lane_kind: str) -> list[tuple[Optional[str], KanbanWorkerLane]]:
    slug = _board_slug(board)
    accepted = []
    for scope, lane in _lanes(task.assignee, slug):
        try:
            with _lane_scope(scope):
                ok = lane.accepts(task, board=slug, lane=lane_kind)
        except Exception:  # health: allow BLE001 -- a broken accepts() leaves the task to other lanes or the native spawn
            logger.warning("Kanban worker lane %r: accepts() raised on %s; skipping it", lane.name, task.id,
                           exc_info=True)
            continue
        if ok:
            accepted.append((scope, lane))
    return accepted


def has_lane(assignee: Any, board: Optional[str]) -> bool:
    """A lane's ``assignees`` and ``boards`` match. Readiness telemetry uses this; it does not
    load tasks, so it skips :meth:`KanbanWorkerLane.accepts`."""
    return bool(_lanes(assignee, board))


def runnable(conn: sqlite3.Connection, task_id: str, assignee: str, *, board: Optional[str], lane: str,
             native: Optional[Callable[[str], bool]]) -> bool:
    """Whether this tick may claim the row: the native profile check (``None`` = unavailable, which
    keeps the historic fail-open), else a lane that matches and accepts the task."""
    if native is None or native(assignee):
        return True
    if not has_lane(assignee, board):
        return False
    from hermes_cli import kanban_db
    task = kanban_db.get_task(conn, task_id)
    return task is not None and bool(_accepting(task, board, lane))


def spawn(conn: sqlite3.Connection, claimed: Task, workspace: str, *, board: Optional[str], lane: str,
          native: Callable[..., Optional[int]]) -> Optional[int]:
    """Offer the claimed task to each accepting lane in order; the first that does not return
    :data:`NOT_HANDLED` runs it. Otherwise the native spawn runs it."""
    from hermes_cli.kanban_db_connect import _main_db_file
    for scope, worker_lane in _accepting(claimed, board, lane):
        claim = WorkerLaneClaim(task=claimed, workspace=workspace, board=board, lane=lane,
                                db_path=_main_db_file(conn))
        with _lane_scope(scope):
            pid = worker_lane.spawn(claim)
        if pid is NOT_HANDLED:
            continue
        logger.info("kanban dispatcher: %s %s ran by worker lane %r", lane, claimed.id, worker_lane.name)
        return pid
    return _call_spawn_fn(native, claimed, workspace, board)


def _call_spawn_fn(spawn_fn, task: Task, workspace: str, board: Optional[str]) -> Optional[int]:
    """Back-compat: older spawn_fn signatures (and test stubs) accept only
    ``(task, workspace)``; pass ``board`` only when the callable supports it."""
    try:
        sig = inspect.signature(spawn_fn)
        if "board" in sig.parameters:
            return spawn_fn(task, workspace, board=board)
        return spawn_fn(task, workspace)
    except (TypeError, ValueError):
        return spawn_fn(task, workspace)
