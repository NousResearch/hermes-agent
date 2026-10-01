"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
KANBAN_TERMINAL_HANDOFF_TOOLS = (
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
)
_TERMINAL_KANBAN_TOOLS = frozenset(KANBAN_TERMINAL_HANDOFF_TOOLS)

KANBAN_STOP_MAX_ATTEMPTS = 2
_DEFAULT_MAX_ATTEMPTS = KANBAN_STOP_MAX_ATTEMPTS


@dataclass(frozen=True)
class KanbanStopTarget:
    """A board task/run pair whose lifecycle tools can still accept a handoff."""

    task_id: str
    run_id: int
    status: str
    terminal_handoff_accepted: bool = False


def kanban_stop_target() -> Optional[KanbanStopTarget]:
    """Resolve the current dispatcher-owned task and run from the existing board, read-only.

    ``None`` means ownership, run identity, or board state could not be proved. The probe must
    never initialize a board for an env-only task id. For a running task, the board's current
    run must match this worker. A non-running status is accepted only when this exact run has a
    closed, successful terminal handoff outcome and the card's resulting status agrees.
    """
    from agent.delegation_context import owned_kanban_task

    task_id = owned_kanban_task()
    # Lifecycle tools compare this env value verbatim; a stripped-but-different id cannot hand off.
    if not task_id or os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        run_id = int(raw_run_id)
    except (TypeError, ValueError):
        return None

    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_readonly_closing() as conn:
            task = kb.get_task(conn, task_id)
            run = kb.get_run(conn, run_id)
    except Exception:
        return None
    if task is None:
        return None
    status = str(task.status or "")
    if status not in kb.VALID_STATUSES:
        return None
    if status == "running":
        if task.current_run_id != run_id:
            return None
        return KanbanStopTarget(task_id=task_id, run_id=run_id, status=status)

    accepted_statuses = {
        "completed": {"done"},
        "review_requested": {"review"},
        "changes_requested": {"ready"},
        "blocked": {"blocked", "todo", "triage"},
    }
    expected_statuses = accepted_statuses.get(str(getattr(run, "outcome", "") or ""), set())
    if (
        run is None or run.task_id != task_id or run.ended_at is None
        or status not in expected_statuses or task.current_run_id is not None
    ):
        return None
    return KanbanStopTarget(task_id=task_id, run_id=run_id, status=status, terminal_handoff_accepted=True)


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


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
    target: Optional[KanbanStopTarget] = None,
    allow_after_terminal_attempt: bool = False,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or (not allow_after_terminal_attempt and session_called_kanban_terminal(messages))
    ):
        return None

    if target is not None:
        # A caller that needs a stronger status requirement supplies a previously validated target.
        task_id = target.task_id

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # Native Hermes uses transcript handoff detection. The Codex app-server caller supplies a
    # read-only validated target and may retry after a rejected terminal call only while that
    # board read proves the run remains live.
    if target is None:
        task_state = (
            "has not been handed off: this session made no terminal board call "
            "(`kanban_complete` / `kanban_request_review` / `kanban_block`)."
        )
    else:
        task_state = "is still running and has not been handed off."
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` {task_state} Ending now "
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
    "KANBAN_STOP_MAX_ATTEMPTS", "KANBAN_TERMINAL_HANDOFF_TOOLS", "KanbanStopTarget", "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled", "kanban_stop_target", "session_called_kanban_terminal",
]
