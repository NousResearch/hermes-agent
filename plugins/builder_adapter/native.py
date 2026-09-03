"""Narrow Hermes-owned facade over native Kanban APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


BUILDER_WORKER_POLICY = {
    "policy_id": "hermes.builder_dispatch.v1",
    "tool_allowlist": [
        "builder_patch",
        "builder_read_execution_packet",
        "builder_read_file",
        "builder_run_validation_profile",
        "builder_search_files",
        "builder_write_file",
        "kanban_block",
        "kanban_complete",
        "kanban_heartbeat",
    ],
    "completion_requires_exit_proof": True,
}


@dataclass
class TaskSnapshot:
    task_id: str
    status: str
    run_ids: list[str]
    attempt_count: int
    worker_pid: int | None = None
    claim_lock: str | None = None


class KanbanBackend(Protocol):
    def create_task(self, request_sha256: str, request) -> str: ...
    def snapshot(self, task_id: str) -> TaskSnapshot: ...
    def cancel(self, task_id: str, reason: str) -> "CancellationProof": ...
    def completion_exclusive(self, task_id: str) -> bool: ...
    def release_completion_lease(self, task_id: str) -> bool: ...


@dataclass(frozen=True)
class CancellationProof:
    confirmed: bool
    process_tree_terminated: bool
    task_archived: bool
    detail: str


class NativeKanbanBackend:
    """Uses Hermes internals only inside the Hermes-owned provider boundary."""

    def __init__(self, *, board: str):
        self.board = board

    def create_task(self, request_sha256: str, request) -> str:
        from hermes_cli import kanban_db

        with kanban_db.connect_closing(board=self.board) as conn:
            return kanban_db.create_task(
                conn,
                title=f"Governed implementation {request.cycle_id}",
                body=(
                    "Load the immutable execution packet with "
                    "builder_read_execution_packet. "
                    f"dispatch_id={request.dispatch_id}; request_sha256={request_sha256}"
                ),
                assignee="deepseek-builder",
                created_by="hermes.builder_dispatch.v1",
                workspace_kind="dir",
                workspace_path=request.worktree_path,
                branch_name=None,
                idempotency_key=request.idempotency_key,
                max_runtime_seconds=request.timeout_policy.max_runtime_seconds,
                max_retries=request.retry_policy.max_attempts,
                initial_status="running",
                worker_policy=BUILDER_WORKER_POLICY,
                board=self.board,
            )

    def snapshot(self, task_id: str) -> TaskSnapshot:
        from hermes_cli import kanban_db

        with kanban_db.connect_closing(board=self.board) as conn:
            task = kanban_db.get_task(conn, task_id)
            if task is None:
                raise RuntimeError("native Kanban task missing")
            runs = kanban_db.list_runs(conn, task_id)
            return TaskSnapshot(
                task_id=task_id,
                status=task.status,
                run_ids=[str(run.id) for run in runs],
                attempt_count=len(runs),
                worker_pid=task.worker_pid,
                claim_lock=task.claim_lock,
            )

    def completion_exclusive(self, task_id: str) -> bool:
        from hermes_cli import kanban_db

        with kanban_db.connect_closing(board=self.board) as conn:
            task = kanban_db.get_task(conn, task_id)
            return bool(
                task is not None
                and task.status == "done"
                and not task.worker_pid
                and not task.claim_lock
                and conn.execute(
                    "SELECT 1 FROM governed_worker_lifecycle "
                    "WHERE task_id=? AND state='terminated' "
                    "AND completion_lease IS NOT NULL",
                    (task_id,),
                ).fetchone()
            )

    def release_completion_lease(self, task_id: str) -> bool:
        from hermes_cli import kanban_db

        with kanban_db.connect_closing(board=self.board) as conn:
            with kanban_db.write_txn(conn):
                current = conn.execute(
                    "SELECT state,completion_lease FROM governed_worker_lifecycle "
                    "WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if (
                    current is not None
                    and current["state"] == "attested"
                    and current["completion_lease"] == ""
                ):
                    return True
                changed = conn.execute(
                    "UPDATE governed_worker_lifecycle SET state='attested', "
                    "completion_lease='' WHERE task_id=? AND state='terminated' "
                    "AND completion_lease != ''",
                    (task_id,),
                )
                return changed.rowcount == 1

    def cancel(self, task_id: str, reason: str) -> CancellationProof:
        # Native reclaim_task is intentionally not called: its public boolean
        # result does not prove process-tree termination and may clear PID/claim
        # state after a best-effort single-PID attempt. Until Hermes exposes a
        # structured proof interface, cancellation must remain non-terminal.
        return CancellationProof(
            confirmed=False,
            process_tree_terminated=False,
            task_archived=False,
            detail=(
                "native Hermes does not expose a process-tree termination proof; "
                f"cancellation reason {reason!r} was not applied to {task_id}"
            ),
        )
