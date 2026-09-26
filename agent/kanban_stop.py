"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import os
import sqlite3
from enum import Enum
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


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


class RunStanding(Enum):
    ACTIVE = "active"
    HANDED_OFF = "handed_off"
    UNKNOWN = "unknown"


def _worker_run_id() -> Optional[int]:
    raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    return int(raw) if raw.isdigit() else None


def read_run_standing(task_id: str) -> RunStanding:
    run_id = _worker_run_id()
    if not task_id or run_id is None:
        return RunStanding.UNKNOWN
    try:
        from hermes_cli.kanban_db import kanban_db_path

        uri = kanban_db_path().resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
    except Exception:
        return RunStanding.UNKNOWN
    try:
        row = conn.execute(
            "SELECT r.ended_at, t.current_run_id FROM task_runs r JOIN tasks t ON t.id = r.task_id "
            "WHERE r.id = ? AND r.task_id = ?",
            (run_id, task_id),
        ).fetchone()
    except sqlite3.Error:
        return RunStanding.UNKNOWN
    finally:
        conn.close()
    if row is None:
        return RunStanding.UNKNOWN
    ended_at, current_run_id = row
    return RunStanding.ACTIVE if ended_at is None and current_run_id == run_id else RunStanding.HANDED_OFF


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None
    owned = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    standing = read_run_standing(owned)
    if standing is RunStanding.HANDED_OFF:
        return None
    if standing is RunStanding.UNKNOWN and session_called_kanban_terminal(messages):
        return None

    tid = owned or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: your run is still open, with no accepted "
        "terminal board call (`kanban_complete` / `kanban_request_review` / `kanban_block`). Ending now "
        "causes a protocol violation (clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "RunStanding",
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "read_run_standing",
    "session_called_kanban_terminal",
]
