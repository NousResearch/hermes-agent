"""Turn-end guard for kanban workers, which must end a run with a terminal board call.
Some models narrate the next step and stop with no tool calls; Hermes treats that as a
clean exit → ``rc=0`` → dispatcher ``protocol_violation``. Policy-only: return a bounded
synthetic nudge so the loop continues instead of exiting.

Terminal means "this run is over as far as the board is concerned": ``kanban_complete``,
``kanban_block``, and the review transitions ``kanban_request_review`` /
``kanban_request_changes`` (both end the run and move the card out of ``running``). The
guard also re-reads the task row before firing, so a run that already landed a terminal
state can never be nagged toward ``kanban_complete`` — that nag used to push workers into
marking unmerged, review-rejected work DONE.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from agent.delegation_context import is_dispatcher_owned_worker_context

logger = logging.getLogger(__name__)


# Every board call that ends the current run. ``kanban_request_review`` parks the card in
# ``review`` and ``kanban_request_changes`` returns it to the implementer; both clear
# ``current_run_id``, so demanding kanban_complete/kanban_block afterwards is a false nag.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

# Task statuses that mean "this worker's run is still open". Anything else (review, done,
# blocked, todo, ready-after-requeue) is already terminal for the run that just ended.
_LIVE_TASK_STATUSES = frozenset({"running"})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip()) and is_dispatcher_owned_worker_context()


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


def _expected_run_id() -> Optional[int]:
    raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def task_run_is_live(task_id: str) -> bool:
    """True when the board still shows an open run for ``task_id`` (durable check).

    The nag must never be emitted on stale dispatch-time state: a worker that ended its run
    with ``kanban_request_review`` leaves the card in ``review`` with ``current_run_id``
    cleared, and telling that worker it is "still running, call kanban_complete" is how
    false-DONE happens. Fails OPEN (returns True) when the board cannot be read or the row
    is not found, so a DB/board-resolution problem degrades to the previous behaviour rather
    than silencing the guard fleet-wide: only a positively-observed terminal state suppresses
    the nag.
    """
    if not task_id:
        return True
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
        if task is None:
            return True
        if str(task.status or "") not in _LIVE_TASK_STATUSES or task.current_run_id is None:
            return False
        expected = _expected_run_id()
        # A different run owns the task now (reclaim / requeue): ours is over.
        return expected is None or int(task.current_run_id) == expected
    except Exception:
        logger.debug("kanban stop guard: liveness read failed, assuming live", exc_info=True)
        return True


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already terminal on the board, budget
    exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_run_is_live(tid):
        return None
    tid = tid or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked. If the work "
        "needs review first, `kanban_request_review(summary=...)` is also a "
        "terminal board action.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "task_run_is_live",
]
