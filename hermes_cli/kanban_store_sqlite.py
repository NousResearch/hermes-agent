"""SQLite Kanban storage adapter.

This wrapper is intentionally thin: it delegates to the existing
``hermes_cli.kanban_db`` facade so current behavior and call sites remain
unchanged while contract tests can exercise a store object boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_store import StoreCapabilities


@dataclass(frozen=True)
class SQLiteKanbanStore:
    """Adapter object for the legacy SQLite-backed Kanban implementation."""

    capabilities: StoreCapabilities = StoreCapabilities(
        backend="sqlite",
        supports_row_level_locking=False,
        supports_skip_locked=False,
        supports_concurrent_writers=False,
        production_ready=True,
    )

    def __getattr__(self, name: str) -> Any:
        """Delegate legacy constants and helper functions during migration."""
        return getattr(kb, name)

    def connect(self, *args: Any, **kwargs: Any):
        return kb.connect(*args, **kwargs)

    def init_db(self, *args: Any, **kwargs: Any):
        return kb.init_db(*args, **kwargs)

    def write_txn(self, conn):
        return kb.write_txn(conn)

    def run_daemon(self, *args: Any, **kwargs: Any):
        return kb.run_daemon(*args, **kwargs)

    def create_task(
        self,
        conn,
        *,
        title: str,
        body: Optional[str] = None,
        assignee: Optional[str] = None,
        created_by: Optional[str] = None,
        workspace_kind: str = "scratch",
        workspace_path: Optional[str] = None,
        branch_name: Optional[str] = None,
        tenant: Optional[str] = None,
        priority: int = 0,
        parents: Iterable[str] = (),
        triage: bool = False,
        idempotency_key: Optional[str] = None,
        max_runtime_seconds: Optional[int] = None,
        skills: Optional[Iterable[str]] = None,
        max_retries: Optional[int] = None,
        initial_status: str = "running",
        session_id: Optional[str] = None,
        board: Optional[str] = None,
    ) -> str:
        return kb.create_task(
            conn,
            title=title,
            body=body,
            assignee=assignee,
            created_by=created_by,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            branch_name=branch_name,
            tenant=tenant,
            priority=priority,
            parents=parents,
            triage=triage,
            idempotency_key=idempotency_key,
            max_runtime_seconds=max_runtime_seconds,
            skills=skills,
            max_retries=max_retries,
            initial_status=initial_status,
            session_id=session_id,
            board=board,
        )

    def get_task(self, conn, task_id: str):
        return kb.get_task(conn, task_id)

    def list_tasks(self, conn, *args: Any, **kwargs: Any):
        return kb.list_tasks(conn, *args, **kwargs)

    def recompute_ready(self, conn) -> int:
        return kb.recompute_ready(conn)

    def claim_task(
        self,
        conn,
        task_id: str,
        *,
        ttl_seconds: Optional[int] = None,
        claimer: Optional[str] = None,
    ):
        return kb.claim_task(
            conn,
            task_id,
            ttl_seconds=ttl_seconds,
            claimer=claimer,
        )

    def heartbeat_claim(
        self,
        conn,
        task_id: str,
        *,
        ttl_seconds: Optional[int] = None,
        claimer: Optional[str] = None,
    ) -> bool:
        return kb.heartbeat_claim(
            conn,
            task_id,
            ttl_seconds=ttl_seconds,
            claimer=claimer,
        )

    def release_stale_claims(self, conn, *args: Any, **kwargs: Any) -> int:
        return kb.release_stale_claims(conn, *args, **kwargs)

    def complete_task(
        self,
        conn,
        task_id: str,
        *,
        result: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
        created_cards: Optional[Iterable[str]] = None,
        expected_run_id: Optional[int] = None,
    ) -> bool:
        return kb.complete_task(
            conn,
            task_id,
            result=result,
            summary=summary,
            metadata=metadata,
            created_cards=created_cards,
            expected_run_id=expected_run_id,
        )

    def block_task(self, conn, task_id: str, *args: Any, **kwargs: Any) -> bool:
        return kb.block_task(conn, task_id, *args, **kwargs)

    def unblock_task(self, conn, task_id: str) -> bool:
        return kb.unblock_task(conn, task_id)

    def add_comment(self, conn, task_id: str, author: str, body: str) -> int:
        return kb.add_comment(conn, task_id, author, body)

    def list_comments(self, conn, task_id: str):
        return kb.list_comments(conn, task_id)

    def list_events(self, conn, task_id: str):
        return kb.list_events(conn, task_id)

    def dispatch_once(self, conn, *args: Any, **kwargs: Any):
        return kb.dispatch_once(conn, *args, **kwargs)


__all__ = ["SQLiteKanbanStore"]
