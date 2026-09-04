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

import logging
import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

_DEFAULT_MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)


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


def record_context_recovery_exhausted(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
) -> bool:
    """Run-fenced diagnostic block after a stop continuation exhausts context.

    This fallback never infers completion from assistant prose. It only prevents
    a dispatcher-owned worker run from disappearing as an opaque dead PID after
    the runtime already issued a terminal-tool continuation. A valid run id is
    mandatory so a reclaimed worker cannot mutate its successor.
    """
    from agent.delegation_context import is_dispatcher_owned_worker_context

    if (
        attempts <= 0
        or not is_dispatcher_owned_worker_context()
        or not kanban_stop_nudge_enabled()
    ):
        return False
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if not task_id or not raw_run_id:
        return False
    try:
        run_id = int(raw_run_id)
    except ValueError:
        logger.warning("invalid HERMES_KANBAN_RUN_ID=%r", raw_run_id)
        return False

    session_id = (os.environ.get("HERMES_SESSION_ID") or "").strip() or "unavailable"
    workspace = (os.environ.get("HERMES_KANBAN_WORKSPACE") or "").strip() or "unavailable"
    reason = (
        "Runtime diagnostic: the Kanban terminal-tool continuation exhausted "
        "context overflow recovery before an explicit kanban_complete or "
        "kanban_block call could be committed. No completion was inferred from "
        "assistant prose. "
        f"Evidence handles: task={task_id}; run={run_id}; session={session_id}; "
        f"workspace={workspace}. "
        "Clearing gate: inspect the preserved task workspace/session evidence, "
        "resume or re-run with enough context for the terminal turn, and commit "
        "an explicit kanban_complete or kanban_block transition."
    )

    try:
        from agent.redact import redact_sensitive_text
        from hermes_cli import kanban_db as kb

        conn = kb.connect()
        try:
            return kb.block_task(
                conn,
                task_id,
                reason=redact_sensitive_text(reason, force=True),
                # Goal-mode workers already permit needs_input as a genuine
                # external stop. Use the same lifecycle ownership instead of
                # creating a runtime-only escape around the completion judge.
                kind="needs_input",
                expected_run_id=run_id,
            )
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "Failed to record context-recovery diagnostic block for task %s run %s",
            task_id,
            run_id,
            exc_info=True,
        )
        return False


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "record_context_recovery_exhausted",
    "session_called_kanban_terminal",
]
