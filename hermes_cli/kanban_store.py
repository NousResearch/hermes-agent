"""Kanban storage backend contract.

This module defines the adapter boundary for Kanban persistence without
changing existing call sites.  The current SQLite implementation in
``hermes_cli.kanban_db`` remains the runtime path; future backends should
implement these protocols and pass the store contract tests.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from hermes_cli.kanban_db import Comment, DispatchResult, Event, Task


@runtime_checkable
class KanbanStoreConnection(Protocol):
    """Minimal connection shape required by current Kanban operations."""

    def execute(self, sql: str, parameters: Iterable[Any] = (), /) -> Any:
        """Execute one statement and return a cursor-like result."""
        ...

    def executescript(self, sql_script: str, /) -> Any:
        """Execute multiple statements during initialization/migrations."""
        ...

    def close(self) -> None:
        """Release backend resources for this connection."""
        ...


@dataclass(frozen=True)
class StoreCapabilities:
    """Static capability metadata for a Kanban storage backend."""

    backend: str = "unknown"
    supports_row_level_locking: bool = False
    supports_skip_locked: bool = False
    supports_concurrent_writers: bool = False
    production_ready: bool = False


@runtime_checkable
class KanbanStore(Protocol):
    """Protocol for Kanban storage backends.

    Operations are grouped by the domain behavior currently provided by
    ``hermes_cli.kanban_db``. Implementations must preserve task/run/event
    ledger semantics and make claim/complete/reclaim transitions serializable.
    """

    @property
    def capabilities(self) -> StoreCapabilities:
        """Return backend capability metadata."""
        ...

    def connect(self, *args: Any, **kwargs: Any) -> AbstractContextManager[KanbanStoreConnection]:
        """Open a storage connection scoped to a board or explicit DB path."""
        ...

    def init_db(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize or migrate backend schema."""
        ...

    def create_task(
        self,
        conn: KanbanStoreConnection,
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
        """Create a task and return its id."""
        ...

    def get_task(self, conn: KanbanStoreConnection, task_id: str) -> Optional[Task]:
        """Return one task by id, or None."""
        ...

    def list_tasks(self, conn: KanbanStoreConnection, *args: Any, **kwargs: Any) -> list[Task]:
        """List tasks using backend-supported filters/order."""
        ...

    def recompute_ready(self, conn: KanbanStoreConnection) -> int:
        """Promote todo tasks whose parents are complete."""
        ...

    def claim_task(
        self,
        conn: KanbanStoreConnection,
        task_id: str,
        *,
        ttl_seconds: Optional[int] = None,
        claimer: Optional[str] = None,
    ) -> Optional[Task]:
        """Atomically claim a ready task for execution."""
        ...

    def heartbeat_claim(
        self,
        conn: KanbanStoreConnection,
        task_id: str,
        *,
        ttl_seconds: Optional[int] = None,
        claimer: Optional[str] = None,
    ) -> bool:
        """Extend the current claim when still owned by the caller."""
        ...

    def release_stale_claims(self, conn: KanbanStoreConnection, *args: Any, **kwargs: Any) -> int:
        """Release expired claims and return the number reclaimed."""
        ...

    def complete_task(
        self,
        conn: KanbanStoreConnection,
        task_id: str,
        *,
        result: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
        created_cards: Optional[Iterable[str]] = None,
        expected_run_id: Optional[int] = None,
    ) -> bool:
        """Mark a task complete and close the active run."""
        ...

    def block_task(self, conn: KanbanStoreConnection, task_id: str, *args: Any, **kwargs: Any) -> bool:
        """Move a task to blocked state."""
        ...

    def unblock_task(self, conn: KanbanStoreConnection, task_id: str) -> bool:
        """Move a blocked task back to a schedulable state."""
        ...

    def add_comment(self, conn: KanbanStoreConnection, task_id: str, author: str, body: str) -> int:
        """Append a task comment and emit its event."""
        ...

    def list_comments(self, conn: KanbanStoreConnection, task_id: str) -> list[Comment]:
        """Return comments for a task."""
        ...

    def list_events(self, conn: KanbanStoreConnection, task_id: str) -> list[Event]:
        """Return ledger events for a task."""
        ...

    def dispatch_once(self, conn: KanbanStoreConnection, *args: Any, **kwargs: Any) -> DispatchResult:
        """Run one dispatcher tick against the backend."""
        ...


__all__ = [
    "Comment",
    "DispatchResult",
    "Event",
    "KanbanStore",
    "KanbanStoreConnection",
    "StoreCapabilities",
    "Task",
]
