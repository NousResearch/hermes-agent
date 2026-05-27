"""PostgreSQL Kanban storage adapter placeholder.

This module reserves the backend plug point without enabling a migration.  The
class advertises the concurrency capabilities expected from PostgreSQL, but all
runtime operations raise until the implementation is built and contract-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes_cli.kanban_store import StoreCapabilities


def _not_implemented() -> NotImplementedError:
    return NotImplementedError(
        "PostgresKanbanStore is a placeholder; PostgreSQL backend is not implemented or migration-enabled yet"
    )


@dataclass(frozen=True)
class PostgresKanbanStore:
    """Non-runtime placeholder for the future PostgreSQL-backed store."""

    capabilities: StoreCapabilities = StoreCapabilities(
        backend="postgres",
        supports_row_level_locking=True,
        supports_skip_locked=True,
        supports_concurrent_writers=True,
        production_ready=False,
    )

    def connect(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def init_db(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def create_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def get_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_tasks(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def recompute_ready(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def claim_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def heartbeat_claim(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def release_stale_claims(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def complete_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def block_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def unblock_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def add_comment(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_comments(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_events(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def dispatch_once(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def add_notify_sub(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def archive_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def assign_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def board_exists(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def board_stats(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def build_worker_context(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def child_ids(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def create_board(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def delete_archived_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def edit_completed_task_result(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def gc_events(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def gc_worker_logs(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def get_current_board(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def has_spawnable_ready(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def heartbeat_worker(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def kanban_db_path(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def known_assignees(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def latest_summary(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def link_tasks(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_boards(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_notify_subs(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_profiles_on_disk(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def list_runs(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def parent_ids(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def promote_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def read_board_metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def read_worker_log(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def reassign_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def reclaim_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def remove_board(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def remove_notify_sub(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def resolve_workspace(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def run_daemon(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def schedule_task(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def set_current_board(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def set_workspace_path(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def unlink_tasks(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def workspaces_root(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()

    def write_board_metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for the future PostgreSQL implementation."""
        raise _not_implemented()


__all__ = ["PostgresKanbanStore"]
