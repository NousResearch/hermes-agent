"""Off-dispatch maintenance runner for merged-PR worktree reconciliation."""

from __future__ import annotations

import contextlib
import threading


def start(
    previous: threading.Thread | None,
) -> threading.Thread:
    """Keep at most one profile-scoped maintenance pass alive."""
    if previous is not None and previous.is_alive():
        return previous

    def run() -> None:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_workspace as kbw

        try:
            with contextlib.closing(kbc.connect()) as conn:
                kbw.reap_merged_pr_worktrees(conn, max_checks=1)
        except Exception:  # health: allow BLE001 -- daemon boundary logs and retries next tick
            kb._log.exception("standalone Kanban PR maintenance failed")

    from agent.memory_provider import spawn_context_thread

    thread = spawn_context_thread(
        run,
        name="kanban-pr-maintenance",
        daemon=True,
    )
    thread.start()
    return thread
