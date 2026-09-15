"""Bounded turn-end nudge for workers that have not handed off their run.

The run, not the card or an attempted tool call, owns the obligation to finish.
A reviewer may already own the same card by the time the old worker stops.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from typing import Any, Iterable, Optional

from agent.delegation_context import is_dispatcher_owned_worker_context


_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete", "kanban_block", "kanban_request_review", "kanban_request_changes",
})
_DEFAULT_MAX_ATTEMPTS = 2
logger = logging.getLogger(__name__)


def kanban_stop_nudge_enabled() -> bool:
    """Only the owning worker, not an in-process delegate or cron job, needs a handoff."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip()) and is_dispatcher_owned_worker_context()


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """Legacy history fallback: require an acknowledged success, never just an attempt.

    Dispatched runs use durable state instead: history can be compacted, inherited
    from an older run, or report a successor's id in the post-transition receipt.
    """
    calls = {}
    tid = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if call_id:
                    calls[call_id] = _tool_call_name(tc)
        if msg.get("role") != "tool":
            continue
        name = msg.get("name") or calls.get(msg.get("tool_call_id"))
        if name not in _TERMINAL_KANBAN_TOOLS:
            continue
        try:
            result = json.loads(msg.get("content") or "")
        except (TypeError, ValueError):
            continue
        if (isinstance(result, dict) and result.get("ok") is True
                and not result.get("error") and (not tid or result.get("task_id") == tid)):
            return True
    return False


def _run_needs_handoff(task_id: str, raw_run_id: str) -> bool:
    """Read the pinned board without creating, migrating, or repairing it.

    No cache: an open run can hand off between two stop attempts. On a failed
    lookup we cannot safely order more writes; the dispatcher still detects an
    unfinished clean exit. The read is advisory, not a replacement for write fencing.
    """
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli.sqlite_safe_read import connect_tracked

        run_id = int(raw_run_id)
        path = kb.kanban_db_path().resolve()
        with closing(connect_tracked(path.as_uri() + "?mode=ro", uri=True, timeout=1.0)) as conn:
            conn.row_factory = sqlite3.Row
            # goal_run_status supports legacy cards with no run row. A pinned
            # worker must have the matching durable record, not just a pointer.
            run = conn.execute(
                "SELECT 1 FROM task_runs WHERE id = ? AND task_id = ?",
                (run_id, task_id),
            ).fetchone()
            return run is not None and kb.goal_run_status(conn, task_id, run_id) == "running"
    except Exception:
        logger.debug("kanban stop guard could not verify run ownership", exc_info=True)
        return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Continue an unfinished owning run, never a handed-off or superseded worker.

    Older/standalone workers without a run id retain a success-only history
    fallback. With an id, durable board state takes precedence over all history.
    """
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None
    tid = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if task_id is not None and task_id.strip() != tid:
        return None
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if raw_run_id:
        if not _run_needs_handoff(tid, raw_run_id):
            return None
    elif session_called_kanban_terminal(messages):
        return None

    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "lifecycle handoff for the board.\n\n"
        f"Before exiting task `{tid}`, verify your own run still owns the work "
        "with `kanban_show`. Never act on a successor's run. An unfinished "
        "clean exit causes a protocol violation.\n\n"
        "If you still own the work, finish any remaining deliverable, then use "
        "the appropriate lifecycle tool: `kanban_complete(summary=..., artifacts=[...])` "
        "for finished work; `kanban_request_review(summary=...)` for review; "
        "`kanban_request_changes(reason=...)` for reviewer rework; or "
        "`kanban_block(reason=...)` for a genuine blocker. A rejected call does "
        "not end your run.\n\n"
        "Never end a turn with only a promise of future action.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
