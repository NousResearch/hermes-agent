"""Turn-end guard for kanban workers, which must end with a *board-terminal* tool:
``kanban_complete``, ``kanban_block``, ``kanban_request_review``, or
``kanban_request_changes``. Some models narrate the next step and stop with no tool
calls; Hermes treats that as a clean exit -> rc=0 -> dispatcher protocol_violation.
Policy-only: return a bounded synthetic nudge so the loop continues instead of exiting.

Two suppression rules keep the guard from re-arming a session that already reached a
terminal state (#98107, #98750): the session history already invoked any board-terminal
tool (including the review-lane hand-offs, which end the run like complete/block), or the
worker's own run is already terminal on the board (task_runs.outcome set for
HERMES_KANBAN_RUN_ID - a terminal run must never be told its work is still owed).
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


# Board-terminal tools: invoking any of these ends the worker's run.
# Review-lane hand-offs (request_review / request_changes) route the card
# and close the run exactly like complete / block, so they count here too.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2

# Memoized run-outcome reads, keyed by (board DB path, HERMES_KANBAN_RUN_ID).
# A closed run never reopens, so a non-null outcome is cached for the process
# lifetime; misses are not cached (the run may legitimately close later).
_RUN_OUTCOME_CACHE: "dict[tuple[str, str], Optional[str]]" = {}


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def reset_run_outcome_cache() -> None:
    """Clear memoized board run-outcome reads (tests, long-lived hosts)."""
    _RUN_OUTCOME_CACHE.clear()


def _run_outcome_from_board() -> Optional[str]:
    """Return this worker's ``task_runs.outcome``, or None while still open.

    Reads the run row pinned by ``HERMES_KANBAN_RUN_ID`` from the board DB.
    None means "no evidence the run is terminal": no run id in the env
    (older dispatcher), unknown run id, board error, or an open run
    (``outcome IS NULL`` — the normal live-worker case). Every failure mode
    degrades to None so the guard fails open, never against the worker.
    """
    run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if not run_id:
        return None
    try:
        from hermes_cli import kanban_db

        board_path = str(kanban_db.kanban_db_path())
    except Exception:
        # Unreadable board → cannot prove the run terminal → nudge stays.
        return None
    cache_key = (board_path, run_id)
    if cache_key in _RUN_OUTCOME_CACHE:
        return _RUN_OUTCOME_CACHE[cache_key]
    outcome: Optional[str] = None
    try:
        conn = kanban_db.connect()
        try:
            run = kanban_db.get_run(conn, int(run_id))
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if run is not None:
            outcome = run.outcome
    except Exception:
        # Unreadable board → cannot prove the run terminal → nudge stays.
        return None
    _RUN_OUTCOME_CACHE[cache_key] = outcome
    return outcome


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a board-terminal kanban tool
    (complete/block plus the review-lane hand-offs request_review/request_changes)."""
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
    """Synthetic follow-up when a kanban worker exits without a board-terminal tool;
``None`` when the guard should not fire (not a kanban worker, no terminal tool in the
session, the worker's own run already terminal on the board - outcome set, #98750 - or
budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
        or _run_outcome_from_board() is not None
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # The template asserts only what this module actually read: that no
    # board-terminal tool appears in the session history and this run has
    # no outcome yet. It makes no claim about card status — the card may
    # have moved on (review, reassignment) while this session lagged.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not received a board-terminal tool call in this "
        "session. Ending now without one causes a protocol violation "
        "(clean exit with no `kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked — or, for "
        "review-lane hand-offs, `kanban_request_review(...)` / "
        "`kanban_request_changes(...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual "
        "intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "reset_run_outcome_cache", "session_called_kanban_terminal"]
