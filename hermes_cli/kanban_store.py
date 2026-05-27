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

    @property
    def DEFAULT_BOARD(self) -> str:
        """Default board slug for board-aware helpers."""
        ...

    @property
    def DEFAULT_CLAIM_TTL_SECONDS(self) -> int:
        """Default claim lease duration."""
        ...

    @property
    def DEFAULT_FAILURE_LIMIT(self) -> int:
        """Default task retry limit."""
        ...

    @property
    def DEFAULT_SPAWN_FAILURE_LIMIT(self) -> int:
        """Default dispatcher spawn failure limit."""
        ...

    @property
    def VALID_INITIAL_STATUSES(self) -> Any:
        """Statuses accepted for task creation."""
        ...

    @property
    def VALID_SORT_ORDERS(self) -> Any:
        """Supported task list sort orders."""
        ...

    @property
    def VALID_STATUSES(self) -> Any:
        """Supported task statuses."""
        ...

    def add_notify_sub(self, *args: Any, **kwargs: Any) -> Any:
        """Register a notification subscription."""
        ...

    def archive_task(self, *args: Any, **kwargs: Any) -> Any:
        """Archive a task."""
        ...

    def assign_task(self, *args: Any, **kwargs: Any) -> Any:
        """Assign a task to a profile."""
        ...

    def board_exists(self, *args: Any, **kwargs: Any) -> Any:
        """Return whether a board exists."""
        ...

    def board_stats(self, *args: Any, **kwargs: Any) -> Any:
        """Return aggregate board statistics."""
        ...

    def build_worker_context(self, *args: Any, **kwargs: Any) -> Any:
        """Build worker spawn context."""
        ...

    def child_ids(self, *args: Any, **kwargs: Any) -> Any:
        """Return child task ids."""
        ...

    def create_board(self, *args: Any, **kwargs: Any) -> Any:
        """Create or update board metadata."""
        ...

    def delete_archived_task(self, *args: Any, **kwargs: Any) -> Any:
        """Delete an archived task."""
        ...

    def edit_completed_task_result(self, *args: Any, **kwargs: Any) -> Any:
        """Edit result metadata for a completed task."""
        ...

    def gc_events(self, *args: Any, **kwargs: Any) -> Any:
        """Garbage-collect task events."""
        ...

    def gc_worker_logs(self, *args: Any, **kwargs: Any) -> Any:
        """Garbage-collect worker logs."""
        ...

    def get_current_board(self, *args: Any, **kwargs: Any) -> Any:
        """Return the current board slug."""
        ...

    def has_spawnable_ready(self, *args: Any, **kwargs: Any) -> Any:
        """Return whether spawnable ready work exists."""
        ...

    def heartbeat_worker(self, *args: Any, **kwargs: Any) -> Any:
        """Record a dispatcher/worker heartbeat."""
        ...

    def kanban_db_path(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the backend database path for a board."""
        ...

    def known_assignees(self, *args: Any, **kwargs: Any) -> Any:
        """Return known assignee profiles."""
        ...

    def latest_summary(self, *args: Any, **kwargs: Any) -> Any:
        """Return the latest task summary."""
        ...

    def link_tasks(self, *args: Any, **kwargs: Any) -> Any:
        """Link parent/child tasks."""
        ...

    def list_boards(self, *args: Any, **kwargs: Any) -> Any:
        """List board metadata."""
        ...

    def list_notify_subs(self, *args: Any, **kwargs: Any) -> Any:
        """List notification subscriptions."""
        ...

    def list_profiles_on_disk(self, *args: Any, **kwargs: Any) -> Any:
        """List profiles visible to dispatcher spawn."""
        ...

    def list_runs(self, *args: Any, **kwargs: Any) -> Any:
        """List task runs."""
        ...

    def parent_ids(self, *args: Any, **kwargs: Any) -> Any:
        """Return parent task ids."""
        ...

    def promote_task(self, *args: Any, **kwargs: Any) -> Any:
        """Promote a task status."""
        ...

    def read_board_metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Read metadata for one board."""
        ...

    def read_worker_log(self, *args: Any, **kwargs: Any) -> Any:
        """Read one worker log."""
        ...

    def reassign_task(self, *args: Any, **kwargs: Any) -> Any:
        """Reassign a task."""
        ...

    def reclaim_task(self, *args: Any, **kwargs: Any) -> Any:
        """Reclaim a running task."""
        ...

    def remove_board(self, *args: Any, **kwargs: Any) -> Any:
        """Remove or archive a board."""
        ...

    def remove_notify_sub(self, *args: Any, **kwargs: Any) -> Any:
        """Remove a notification subscription."""
        ...

    def resolve_workspace(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve workspace path for a task."""
        ...

    def run_daemon(self, *args: Any, **kwargs: Any) -> Any:
        """Run the dispatcher daemon loop."""
        ...

    def schedule_task(self, *args: Any, **kwargs: Any) -> Any:
        """Schedule a task for future execution."""
        ...

    def set_current_board(self, *args: Any, **kwargs: Any) -> Any:
        """Set the current board slug."""
        ...

    def set_workspace_path(self, *args: Any, **kwargs: Any) -> Any:
        """Set a task workspace path."""
        ...

    def unlink_tasks(self, *args: Any, **kwargs: Any) -> Any:
        """Remove a parent/child link."""
        ...

    def workspaces_root(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the workspaces root."""
        ...

    def write_board_metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Write board metadata."""
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
