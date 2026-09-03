"""Fail-closed runtime reconcile for running Kanban workers.

Inspects running tasks against live host processes, classifying runtime state
into safe/repairable vs ambiguous/fail-closed categories.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Sequence

from hermes_cli import kanban_db as kb

_log = logging.getLogger(__name__)

FIX_ALLOWED_CLASSIFICATIONS: frozenset[str] = frozenset({
    "dead_registered_worker",
    "orphaned_claim_no_process",
    "missing_pid_no_process",
})


@dataclass(frozen=True)
class ReconcileFinding:
    board: str
    db_path: str
    task_id: str
    classification: str
    task_status: str
    current_run_id: int | None
    registered_pid: int | None
    matching_pids: tuple[int, ...]
    fix_allowed: bool
    fixed: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "db_path": self.db_path,
            "task_id": self.task_id,
            "classification": self.classification,
            "task_status": self.task_status,
            "current_run_id": self.current_run_id,
            "registered_pid": self.registered_pid,
            "matching_pids": list(self.matching_pids),
            "fix_allowed": self.fix_allowed,
            "fixed": self.fixed,
            "detail": self.detail,
        }


def argv_matches_task(argv: Sequence[str], task_id: str) -> bool:
    """Return True if argv contains the consecutive tokens ('work', 'kanban', 'task', task_id).

    Never matches a substring of the task id; matching requires exact token equality.
    """
    if not argv or not task_id:
        return False
    target = ("work", "kanban", "task", task_id)
    n = len(target)
    for i in range(len(argv) - n + 1):
        if tuple(argv[i : i + n]) == target:
            return True
    return False


class ProcessSnapshot:
    """Snapshot of host processes for consistent point-in-time runtime inspection."""

    def __init__(self, processes: dict[int, tuple[str, ...]] | None = None) -> None:
        if processes is not None:
            self._procs = dict(processes)
        else:
            self._procs = {}
            self._collect()

    def _collect(self) -> None:
        try:
            import psutil

            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline")
                    if cmdline:
                        self._procs[proc.pid] = tuple(str(x) for x in cmdline)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as exc:
            _log.debug("ProcessSnapshot collect error: %s", exc)

    def is_pid_alive(self, pid: int | None) -> bool:
        if pid is None or pid <= 0:
            return False
        if pid in self._procs:
            return True
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def get_cmdline(self, pid: int | None) -> tuple[str, ...] | None:
        if pid is None or pid <= 0:
            return None
        return self._procs.get(pid)

    def find_matching_pids(self, task_id: str) -> tuple[int, ...]:
        matches: list[int] = []
        for pid, argv in self._procs.items():
            if argv_matches_task(argv, task_id):
                matches.append(pid)
        return tuple(sorted(matches))


def inspect_task_runtime(
    conn: sqlite3.Connection,
    task: kb.Task,
    *,
    board: str,
    process_snapshot: ProcessSnapshot | None = None,
) -> ReconcileFinding:
    """Inspect one Kanban task runtime identity and return its reconciliation finding."""
    db_path = str(kb.kanban_db_path(board=board).resolve())
    snapshot = process_snapshot if process_snapshot is not None else ProcessSnapshot()
    matching_pids = snapshot.find_matching_pids(task.id)
    matching_count = len(matching_pids)
    registered_pid = task.worker_pid

    if task.status != "running":
        if matching_count > 1:
            return ReconcileFinding(
                board=board,
                db_path=db_path,
                task_id=task.id,
                classification="duplicate_live_workers",
                task_status=task.status,
                current_run_id=task.current_run_id,
                registered_pid=registered_pid,
                matching_pids=matching_pids,
                fix_allowed=False,
                detail=f"Found {matching_count} live processes matching non-running task {task.id}",
            )
        if matching_count == 1:
            return ReconcileFinding(
                board=board,
                db_path=db_path,
                task_id=task.id,
                classification="live_process_unregistered",
                task_status=task.status,
                current_run_id=task.current_run_id,
                registered_pid=registered_pid,
                matching_pids=matching_pids,
                fix_allowed=False,
                detail=f"Live worker process {matching_pids[0]} detected for non-running task {task.id}",
            )
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="healthy",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=(),
            fix_allowed=False,
            detail=f"Task status is {task.status}",
        )

    # Status is 'running'
    if matching_count > 1:
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="duplicate_live_workers",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=matching_pids,
            fix_allowed=False,
            detail=f"Found {matching_count} live processes matching task {task.id}",
        )

    host_prefix = f"{kb._claimer_id().split(':', 1)[0]}:"
    if task.claim_lock is not None and not task.claim_lock.startswith(host_prefix):
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="remote_or_unreadable",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=matching_pids,
            fix_allowed=False,
            detail=f"Task claimed by remote host: {task.claim_lock}",
        )

    if task.claim_lock is None or task.claim_expires is None:
        if matching_count == 1:
            return ReconcileFinding(
                board=board,
                db_path=db_path,
                task_id=task.id,
                classification="live_process_unregistered",
                task_status=task.status,
                current_run_id=task.current_run_id,
                registered_pid=registered_pid,
                matching_pids=matching_pids,
                fix_allowed=False,
                detail=f"Broken claim lock but live worker process {matching_pids[0]} detected",
            )
        if registered_pid and snapshot.is_pid_alive(registered_pid):
            return ReconcileFinding(
                board=board,
                db_path=db_path,
                task_id=task.id,
                classification="registered_pid_mismatch",
                task_status=task.status,
                current_run_id=task.current_run_id,
                registered_pid=registered_pid,
                matching_pids=(),
                fix_allowed=False,
                detail=f"Registered worker {registered_pid} is alive but argv does not match task {task.id}",
            )
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="orphaned_claim_no_process",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=(),
            fix_allowed=True,
            detail="Broken claim lock with no live worker process",
        )

    # Valid local claim
    if registered_pid is not None:
        if snapshot.is_pid_alive(registered_pid):
            if registered_pid in matching_pids:
                return ReconcileFinding(
                    board=board,
                    db_path=db_path,
                    task_id=task.id,
                    classification="healthy",
                    task_status=task.status,
                    current_run_id=task.current_run_id,
                    registered_pid=registered_pid,
                    matching_pids=matching_pids,
                    fix_allowed=False,
                    detail=f"Registered worker {registered_pid} is alive and healthy",
                )
            return ReconcileFinding(
                board=board,
                db_path=db_path,
                task_id=task.id,
                classification="registered_pid_mismatch",
                task_status=task.status,
                current_run_id=task.current_run_id,
                registered_pid=registered_pid,
                matching_pids=matching_pids,
                fix_allowed=False,
                detail=f"Registered worker {registered_pid} is alive but argv does not match task {task.id}",
            )
        # registered_pid is dead
        if matching_count == 1:
            return ReconcileFinding(
                board=board,
                db_path=db_path,
                task_id=task.id,
                classification="live_process_unregistered",
                task_status=task.status,
                current_run_id=task.current_run_id,
                registered_pid=registered_pid,
                matching_pids=matching_pids,
                fix_allowed=False,
                detail=f"Registered worker {registered_pid} is dead but unregistered worker {matching_pids[0]} is alive",
            )
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="dead_registered_worker",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=(),
            fix_allowed=True,
            detail=f"Registered worker {registered_pid} is dead with no live worker process",
        )

    # registered_pid is None
    if matching_count == 1:
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="live_process_unregistered",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=matching_pids,
            fix_allowed=False,
            detail=f"Unregistered live worker process {matching_pids[0]} detected",
        )

    started_at = task.started_at
    grace = kb._resolve_crash_grace_seconds()
    if started_at is not None and (time.time() - started_at < grace):
        return ReconcileFinding(
            board=board,
            db_path=db_path,
            task_id=task.id,
            classification="healthy",
            task_status=task.status,
            current_run_id=task.current_run_id,
            registered_pid=registered_pid,
            matching_pids=(),
            fix_allowed=False,
            detail=f"Task within launch grace window ({grace}s)",
        )

    return ReconcileFinding(
        board=board,
        db_path=db_path,
        task_id=task.id,
        classification="missing_pid_no_process",
        task_status=task.status,
        current_run_id=task.current_run_id,
        registered_pid=registered_pid,
        matching_pids=(),
        fix_allowed=True,
        detail="Missing worker PID with no live worker process",
    )


def reconcile_board(
    *,
    board: str,
    task_id: str | None,
    fix: bool,
) -> list[ReconcileFinding]:
    """Reconcile tasks on a given board against live process snapshot."""
    findings: list[ReconcileFinding] = []
    snapshot = ProcessSnapshot()

    with kb.connect_closing(board=board) as conn:
        if task_id:
            task = kb.get_task(conn, task_id)
            if task is None:
                raise ValueError(f"Task {task_id!r} not found on board {board!r}")
            tasks = [task]
        else:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status = 'running' ORDER BY priority DESC, id ASC"
            ).fetchall()
            tasks = []
            for r in rows:
                t = kb.get_task(conn, r["id"])
                if t is not None:
                    tasks.append(t)

        for task in tasks:
            finding = inspect_task_runtime(
                conn, task, board=board, process_snapshot=snapshot
            )
            # Enforce hard invariant: only approved classifications may be fixed
            is_fixable = (
                finding.fix_allowed
                and finding.classification in FIX_ALLOWED_CLASSIFICATIONS
            )
            if not is_fixable:
                finding = ReconcileFinding(
                    board=finding.board,
                    db_path=finding.db_path,
                    task_id=finding.task_id,
                    classification=finding.classification,
                    task_status=finding.task_status,
                    current_run_id=finding.current_run_id,
                    registered_pid=finding.registered_pid,
                    matching_pids=finding.matching_pids,
                    fix_allowed=False,
                    fixed=False,
                    detail=finding.detail,
                )

            if fix and is_fixable:
                repaired = kb.reconcile_running_task_if_unchanged(
                    conn,
                    task.id,
                    expected_run_id=task.current_run_id,
                    expected_claim_lock=task.claim_lock,
                    expected_worker_pid=task.worker_pid,
                    reason=finding.classification,
                )
                if repaired:
                    finding = ReconcileFinding(
                        board=finding.board,
                        db_path=finding.db_path,
                        task_id=finding.task_id,
                        classification=finding.classification,
                        task_status="ready",
                        current_run_id=None,
                        registered_pid=None,
                        matching_pids=finding.matching_pids,
                        fix_allowed=True,
                        fixed=True,
                        detail=f"Reconciled and requeued to ready ({finding.classification})",
                    )
                else:
                    finding = ReconcileFinding(
                        board=finding.board,
                        db_path=finding.db_path,
                        task_id=finding.task_id,
                        classification=finding.classification,
                        task_status=finding.task_status,
                        current_run_id=finding.current_run_id,
                        registered_pid=finding.registered_pid,
                        matching_pids=finding.matching_pids,
                        fix_allowed=True,
                        fixed=False,
                        detail="CAS failed: task state changed during reconciliation",
                    )
            findings.append(finding)

    return findings
