"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close — a reviewer
# legitimately never calls ``kanban_complete`` when rejecting work (t_e90216b9, 2026-09-15).
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


def _own_run_is_stale(task_id: str, own_run_id: int) -> bool:
    """True when this worker's run is no longer the live one: the run row is
    already closed (terminal ``outcome``), or the task's ``current_run_id``
    moved to a different run (another worker was spawned for the same task).
    Never nudging a worker about a task whose live run is not theirs stops the
    reminder from pressuring a finished reviewer into a false
    ``kanban_complete`` that would target someone else's live run."""
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
    except Exception:
        return False
    try:
        conn = kbc.connect(board=None)
    except Exception:
        # An unreadable board must never wedge a worker's exit; the
        # message-history checks below remain the only gate in that case.
        return False
    try:
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE id = ? AND task_id = ?",
            (int(own_run_id), task_id),
        ).fetchone()
        if run is None or run["outcome"] is not None:
            # Unknown run or already terminal (changes_requested,
            # review_requested, completed, blocked, ...): our run is done.
            return True
        task = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        current_run_id = task["current_run_id"] if task is not None else None
        return current_run_id is None or int(current_run_id) != int(own_run_id)
    except Exception:
        # DB hiccup: fail OPEN (nudge may fire) — the guard must not mask a
        # genuine protocol violation over a transient read error.
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted,
    or this worker's own run is no longer the task's live run)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"

    # Run-ownership gate: the reminder keys on task-level status, so it fired at
    # a reviewer whose own run had already been closed by
    # kanban_request_changes — and could fire at an agent whose task had been
    # handed to a different run entirely. If OUR run is known and no longer the
    # live one (or already ended with a terminal outcome), this worker has
    # nothing left to do: stay silent instead of pressuring a stale worker into
    # a false kanban_complete against someone else's run (t_e90216b9).
    own_run_raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if own_run_raw:
        try:
            own_run_id = int(own_run_raw)
        except ValueError:
            own_run_id = None
        if own_run_id is not None and _own_run_is_stale(tid, own_run_id):
            return None

    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no terminal board "
        "call (`kanban_complete` / `kanban_request_review` / `kanban_block`). Ending now "
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


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
