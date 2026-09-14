"""Turn-end guards for dispatcher-owned kanban workers.

A worker that narrates and stops without a lifecycle tool gets a bounded nudge.
Conversely, once ``complete``, ``block``, ``request_review``, or
``request_changes`` durably succeeds, the worker must exit without another
provider round. This module owns both sides of that terminal boundary.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2


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


def _tool_call_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def successful_kanban_terminal_transition(
    tool_calls: Iterable[Any] | None,
    messages: Iterable[dict] | None,
    *,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return the successful terminal tool from *this* tool round.

    A terminal lifecycle result is safe to use as the worker's natural exit
    boundary only when the handler returned its structured ``ok`` receipt for
    the dispatcher-owned task. Matching current tool-call ids prevents an old
    successful transition in resumed history from ending an unrelated round.
    """
    expected_task = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not expected_task:
        return None

    current: dict[str, str] = {}
    names_without_ids: set[str] = set()
    for tc in tool_calls or ():
        name = _tool_call_name(tc)
        if name not in _TERMINAL_KANBAN_TOOLS:
            continue
        call_id = _tool_call_id(tc)
        if call_id:
            current[call_id] = name
        else:
            names_without_ids.add(name)
    if not current and not names_without_ids:
        return None

    for msg in reversed(list(messages or ())):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = str(msg.get("name") or "")
        call_id = str(msg.get("tool_call_id") or "")
        if call_id:
            if current.get(call_id) != name:
                continue
        elif name not in names_without_ids:
            continue
        try:
            payload = json.loads(str(msg.get("content") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("ok") is True
            and str(payload.get("task_id") or "") == expected_task
        ):
            return name
    return None


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
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
    "successful_kanban_terminal_transition",
]
