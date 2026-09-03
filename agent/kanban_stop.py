"""Turn-end guard for kanban workers.

Kanban workers must end with ``kanban_complete`` or ``kanban_block``. Models
(especially GLM / Qwen families) sometimes narrate the next step
("Let me write the report now") and stop with ``finish_reason=stop`` and no
tool calls. Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

# Board statuses that are terminal for the purposes of this guard. A task in
# one of these can no longer be completed or blocked by THIS session, so the
# nudge ("Task ... is still running — call kanban_complete/kanban_block")
# would be factually wrong and, on models that comply, can even cause an
# improper board mutation attempt. ``review`` is deliberately NOT terminal:
# a reviewer session must still end with ``kanban_complete`` /
# ``kanban_request_changes``.
_TERMINAL_TASK_STATUSES = frozenset({"done", "archived"})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """Return whether the kanban stop-guard is active for this process.

    On when ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), unless
    ``HERMES_KANBAN_STOP_NUDGE`` explicitly disables it.
    """
    env = os.environ.get("HERMES_KANBAN_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return bool(task)


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    return True
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                return True
    return False


def _live_task_is_terminal(task_id: str) -> bool:
    """True only on a POSITIVE live read of a terminal board status.

    Consults the board DB (resolved the same way every other kanban surface
    resolves it — ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` env, then
    the default board). Fail-open by design: any failure to consult the
    board — import error, unreadable DB, missing task row — returns ``False``
    so the caller still nudges. Over-suppressing the nudge on a board we
    cannot read would hide genuine protocol violations from workers whose
    task really is still running.
    """
    try:
        from hermes_cli import kanban_db

        conn = kanban_db.connect()
        try:
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return False
    if row is None:
        return False
    return str(row["status"]) in _TERMINAL_TASK_STATUSES


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a synthetic follow-up when a kanban worker exits without a terminal tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed/blocked, nudge budget exhausted, or the task's LIVE
    board status is already terminal — e.g. a session that inherited
    ``HERMES_KANBAN_TASK`` from a card another run completed).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # Suppress only on a positive live read: the session env can outlive the
    # card it names (child processes inherit it; sessions resume after
    # completion). Never suppress merely because the board is unreadable.
    if tid != "this task" and _live_task_is_terminal(tid):
        return None
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
