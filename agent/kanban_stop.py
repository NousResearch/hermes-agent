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

    # Self-authentication: echo the harness's run id / claim lock so a
    # suspicious model can verify this nudge came from the dispatcher rather
    # than from pasted text. Fall back to the task id when neither is present.
    run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    claim_lock = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    if run_id or claim_lock:
        auth = (
            f" (harness run id: {run_id or 'n/a'}, "
            f"claim lock: {claim_lock or 'n/a'})"
        )
    else:
        auth = f" (task id: {tid})"

    if attempts > 0:
        # Second nudge: shorter, more direct reminder rather than a replay of
        # the first (an identical replay reads as an injection tell).
        return (
            "Reminder from the kanban harness"
            + auth
            + ": the board still shows task `"
            + tid
            + "` as `running`. End this turn by calling "
            "`kanban_complete(summary=..., artifacts=[...])` if the work is done, "
            "or `kanban_block(reason=...)` if you are blocked. A plain-text reply "
            "is not a terminal state and counts as a protocol violation."
        )

    return (
        "The kanban harness"
        + auth
        + " needs a terminal board call before this turn ends. Task `"
        + tid
        + "` is still `running`; ending now without a board tool is recorded as "
        "a protocol violation (clean exit with no `kanban_complete` / "
        "`kanban_block`).\n\n"
        "If the work is done, call `kanban_complete(summary=..., "
        "artifacts=[...])`. If you are blocked, call `kanban_block(reason=...)`. "
        "Otherwise finish the remaining deliverable first, then call one of "
        "those tools before ending the turn."
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
