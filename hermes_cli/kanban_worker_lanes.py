"""Plugin Kanban worker lanes: the public contract.

A plugin registers a :class:`KanbanWorkerLane` with ``ctx.register_kanban_worker_lane(lane)``. The
dispatcher (the gateway's embedded one, ``hermes kanban dispatch`` and the dashboard) hands a claimed
task to the lane when the lane's ``assignees`` and ``boards`` match the task and
:meth:`KanbanWorkerLane.accepts` returns True. Every other task keeps the native ``hermes -p
<assignee>`` spawn. Hermes keeps the claim, the run record, the caps, the respawn guard, retries and
the failure breaker; the lane only starts the work.

A lane starts work in one of two ways:

- **A child process.** :meth:`KanbanWorkerLane.spawn` returns the child's PID. Crash detection and
  the live-worker claim extension follow that PID, as for a native worker.
- **In-process work** (for example a turn that a platform plugin runs inside the gateway).
  ``spawn`` returns ``None`` and the work reports through the :class:`WorkerLaneClaim` it received:
  :meth:`~WorkerLaneClaim.heartbeat`, :meth:`~WorkerLaneClaim.complete`, :meth:`~WorkerLaneClaim.request_review`,
  :meth:`~WorkerLaneClaim.block` and :meth:`~WorkerLaneClaim.fail`. With no PID the claim TTL is the
  liveness signal: each heartbeat extends the claim, and a run that stops heartbeating is reclaimed
  after the TTL (``kanban_db.DEFAULT_CLAIM_TTL_SECONDS``) and counted as a failure.

``spawn`` may return :data:`NOT_HANDLED` to decline a task after the claim; the next matching lane,
then the native spawn, runs it. Long form: ``website/docs/user-guide/features/kanban-worker-lanes.md``.
"""

from __future__ import annotations

import abc
import contextlib
import fnmatch
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

#: The dispatch lanes a task can be claimed from: ``ready`` work, or a ``review`` run.
LANE_KINDS = ("ready", "review")


class _NotHandled:
    """Type of :data:`NOT_HANDLED`."""

    def __repr__(self) -> str:
        return "NOT_HANDLED"


#: Returned by :meth:`KanbanWorkerLane.spawn` to give a claimed task back to the next lane.
NOT_HANDLED = _NotHandled()


@dataclass(frozen=True)
class WorkerLaneClaim:
    """One claimed run handed to a lane, and the handle its in-process work reports through.

    Each method opens its own connection to the board the task was claimed from, so it is safe to
    call from any thread after :meth:`KanbanWorkerLane.spawn` returns. Every write is fenced on
    :attr:`run_id`: a call for a run that Hermes already reclaimed, or that another run replaced,
    changes nothing and returns False.
    """

    task: Task
    workspace: str
    board: Optional[str]
    lane: str
    db_path: Optional[str] = None

    @property
    def task_id(self) -> str:
        return self.task.id

    @property
    def run_id(self) -> Optional[int]:
        return self.task.current_run_id

    @property
    def claim_lock(self) -> Optional[str]:
        return self.task.claim_lock

    @contextlib.contextmanager
    def _board(self):
        from hermes_cli import kanban_db_connect
        path = Path(self.db_path) if self.db_path else None
        with kanban_db_connect.connect_closing(db_path=path, board=self.board) as conn:
            yield conn

    def heartbeat(self, note: Optional[str] = None) -> bool:
        """Record progress and extend the claim. False when this run no longer owns the task."""
        from hermes_cli import kanban_db, kanban_db_dispatch
        with self._board() as conn:
            if not kanban_db_dispatch.heartbeat_worker(conn, self.task_id, note=note, expected_run_id=self.run_id):
                return False
            return kanban_db.heartbeat_claim(conn, self.task_id, claimer=self.claim_lock)

    def complete(self, summary: str, *, result: Optional[str] = None, metadata: Optional[dict] = None) -> bool:
        """End the run as ``done`` (a ``review`` run approves the card)."""
        from hermes_cli import kanban_db
        with self._board() as conn:
            return kanban_db.complete_task(conn, self.task_id, summary=summary, result=result,
                                           metadata=metadata, expected_run_id=self.run_id)

    def request_review(self, summary: str, *, metadata: Optional[dict] = None, reviewer: Optional[str] = None) -> bool:
        """End the run and move the card to ``review`` (same-card review)."""
        from hermes_cli import kanban_db
        with self._board() as conn:
            return bool(kanban_db.request_review(conn, self.task_id, summary=summary, metadata=metadata,
                                                 reviewer=reviewer, expected_run_id=self.run_id))

    def block(self, reason: str, *, kind: Optional[str] = None) -> bool:
        """End the run as ``blocked`` (``kind``: one of ``kanban_db.VALID_BLOCK_KINDS``)."""
        from hermes_cli import kanban_db
        with self._board() as conn:
            return kanban_db.block_task(conn, self.task_id, reason=reason, kind=kind, expected_run_id=self.run_id)

    def fail(self, error: str) -> bool:
        """End the run as a failed attempt: release the claim, count it toward the failure breaker
        (``max_retries``, else ``kanban.failure_limit``), and let the dispatcher retry it."""
        from hermes_cli import kanban_db_dispatch
        with self._board() as conn:
            if not _owns_run(conn, self.task_id, self.run_id):
                return False
            kanban_db_dispatch._record_task_failure(
                conn, self.task_id, f"lane: {error}", outcome="crashed", failure_limit=_failure_limit(),
                release_claim=True, end_run=True, expected_run_id=self.run_id)
            return True


def _owns_run(conn: sqlite3.Connection, task_id: str, run_id: Optional[int]) -> bool:
    row = conn.execute("SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return bool(row) and row["status"] == "running" and run_id is not None and row["current_run_id"] == run_id


def _failure_limit() -> Optional[int]:
    """``kanban.failure_limit`` when it is a positive integer; ``None`` keeps the default."""
    from hermes_cli.config import load_config_readonly
    section = (load_config_readonly() or {}).get("kanban")
    value = section.get("failure_limit") if isinstance(section, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


class KanbanWorkerLane(abc.ABC):
    """A plugin-owned way to run Kanban tasks.

    ``assignees`` and ``boards`` are exact names or shell-style globs (``"codex-*"``). An empty
    tuple matches every value, but a lane must name at least one assignee or one board, so a lane
    never takes a whole fleet by accident. A lane may match assignees that are also Hermes profiles:
    on the boards it names, it runs their tasks instead of the native spawn.
    """

    assignees: tuple[str, ...] = ()
    boards: tuple[str, ...] = ()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier; the registry keys lanes by it."""

    def accepts(self, task: Task, *, board: Optional[str], lane: str) -> bool:
        """Optional finer filter, checked before the claim on a task that ``assignees`` and
        ``boards`` already match (``lane``: ``"ready"`` or ``"review"``). Default: accept."""
        return True

    @abc.abstractmethod
    def spawn(self, claim: WorkerLaneClaim) -> int | None | _NotHandled:
        """Start the work for ``claim``. Return a child PID, ``None`` for in-process work that
        reports through ``claim``, or :data:`NOT_HANDLED` to give the task to the next lane or
        the native spawn. An exception is a failed spawn: the claim is released and the attempt
        counts toward the failure breaker. Runs on the dispatcher's thread while it holds the
        board's tick lock, so start the work and return; do not wait for it."""

    def matches(self, assignee: Any, board: Optional[str]) -> bool:
        """True when ``assignee`` and ``board`` fall inside ``assignees`` and ``boards``."""
        if not isinstance(assignee, str) or not assignee.strip():
            return False
        return _any_match(assignee.strip(), self.assignees) and _any_match(board or "", self.boards)


def _any_match(value: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)
