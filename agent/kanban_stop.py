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

import json
import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

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


def _tool_call_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def is_successful_terminal_result(tool_name: str, content: Any) -> bool:
    """Return whether a terminal kanban tool actually reached ``ok=true``."""
    if tool_name not in _TERMINAL_KANBAN_TOOLS or not isinstance(content, str):
        return False
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("ok") is True


def terminal_success_for_tool_calls(
    messages: Iterable[dict] | None,
    tool_calls: Iterable[Any] | None,
) -> bool:
    """Match successful terminal results to assistant calls by id and name."""
    expected = {
        (_tool_call_id(tool_call), _tool_call_name(tool_call))
        for tool_call in (tool_calls or [])
        if _tool_call_id(tool_call)
        and _tool_call_name(tool_call) in _TERMINAL_KANBAN_TOOLS
    }
    if not expected:
        return False
    return any(
        (
            str(message.get("tool_call_id") or ""),
            str(message.get("name") or ""),
        )
        in expected
        and is_successful_terminal_result(
            str(message.get("name") or ""),
            message.get("content"),
        )
        for message in (messages or [])
        if isinstance(message, dict) and message.get("role") == "tool"
    )


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True only after a matching terminal kanban tool succeeded."""
    message_list = list(messages or [])
    tool_calls = [
        tool_call
        for message in message_list
        if isinstance(message, dict) and message.get("role") == "assistant"
        for tool_call in (message.get("tool_calls") or [])
    ]
    return terminal_success_for_tool_calls(message_list, tool_calls)


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a synthetic follow-up when a kanban worker exits without a terminal tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed/blocked, or nudge budget exhausted).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
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
    "is_successful_terminal_result",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "terminal_success_for_tool_calls",
]
