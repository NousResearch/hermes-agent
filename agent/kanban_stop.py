"""Turn-end guard for kanban workers, which must record a lifecycle handoff.
Some models narrate the next step and stop with no tool calls;
Hermes treats that as a clean exit → ``rc=0`` → dispatcher ``protocol_violation``.
Policy-only: return a bounded synthetic nudge so the loop continues instead of exiting.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

_DEFAULT_MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool" and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Nudge an unfinished worker, not a handed-off or superseded run.

    Dispatcher-pinned runs use durable lifecycle state, which survives context
    compression and successor claims. Unpinned sessions keep the legacy tool scan.
    """
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if raw_run_id:
        from hermes_cli import kanban_db as kb
        from hermes_cli.sqlite_safe_read import connect_tracked, UntrackableConnectionError

        try:
            run_id = int(raw_run_id)
            # A stop check must not initialize/migrate a missing board. Reuse
            # the goal loop's run-scoped lifecycle policy, not task.status alone.
            uri = kb.kanban_db_path().resolve().as_uri() + "?mode=ro"
            with closing(connect_tracked(uri, uri=True, timeout=1.0)) as conn:
                conn.row_factory = sqlite3.Row
                status = kb.goal_run_status(conn, tid, run_id)
            if status in {"done", "blocked", "review", "changes_requested", "superseded"}:
                return None
        except (OSError, ValueError, OverflowError, sqlite3.Error, UntrackableConnectionError):
            logger.debug("Cannot verify kanban worker run handoff", exc_info=True)
        # Failed/foreign tool calls are not proof that this pinned run ended.
    elif session_called_kanban_terminal(messages):
        return None

    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"No finished run or lifecycle handoff was verified for task `{tid}`. "
        "Exiting an active run without a handoff causes a protocol violation.\n\n"
        "Do this in your next response — do not narrate intent:\n"
        "1. Use `kanban_show` to verify your run still owns the work, especially "
        "if a handoff was already attempted. Do not mutate a successor's run.\n"
        "2. If you still own the work, finish any remaining deliverable and call "
        "`kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, `kanban_request_review(summary=...)` when handing work to review, "
        "`kanban_request_changes(reason=...)` when returning a review to its implementer, "
        "OR `kanban_block(reason=...)` if you are genuinely blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
