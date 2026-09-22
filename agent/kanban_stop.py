"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.

The nudge orders a terminal call, so it may only fire when such a call can succeed:
:func:`kanban_stop_target` proves the card is on this worker's board under this worker's run
first. ``HERMES_KANBAN_TASK`` alone is not identity — the raw env value must match what the
tools scope on, name a live board row under this worker's run id, and carry a
handoff-accepting status; anything less leaves every terminal call refused by the ownership
guard and must not become a demand. The loop is still bounded when a terminal call is
*attempted and refused*: :func:`session_called_kanban_terminal` matches the call, not its
success, and ``_DEFAULT_MAX_ATTEMPTS`` caps the rest. That escape is load-bearing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task

logger = logging.getLogger(__name__)


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

# Statuses a terminal handoff can still land on: the widest list any of the four tools accepts
# (``kanban_db.complete_task`` writes ``status IN ('running','ready','blocked','review')``;
# ``block_task`` / ``request_review`` require ``('running','ready')``; ``request_changes``
# requires ``'running'``). A row outside this set has no legal terminal call left.
_TERMINAL_HANDOFF_STATUSES = frozenset({"running", "ready", "blocked", "review"})


@dataclass(frozen=True)
class KanbanStopTarget:
    """A card this worker can actually hand off: board-validated ``task_id`` + ``run_id``."""

    task_id: str
    run_id: int


def _nudge_suppressed_by_env() -> bool:
    """``HERMES_KANBAN_STOP_NUDGE=0/false/no/off`` switches the gate off outright."""
    return (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}


def _worker_run_id(task_id: str) -> Optional[int]:
    """This worker's run id, read exactly as ``tools.kanban_tools._worker_run_id`` reads it:
    only when the *raw* ``HERMES_KANBAN_TASK`` env value is this task, and only when
    ``HERMES_KANBAN_RUN_ID`` parses as an int."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _board_run_and_status(task_id: str) -> Optional[tuple[Optional[int], str]]:
    """``(current_run_id, status)`` for ``task_id`` on this worker's board; ``None`` when the
    board file or the row is missing. Never creates a board: a probe must not initialize a DB
    (``kanban_db_connect.connect`` does that) for a card it cannot even name."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    if not kb.kanban_db_path().exists():
        return None
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return None
    return task.current_run_id, str(task.status or "")


def kanban_stop_target() -> Optional[KanbanStopTarget]:
    """The card this worker can legally terminate right now, or ``None`` when that cannot be
    proven. Every terminal kanban call is refused unless all of:

    * this process is the dispatcher-owned worker (``owned_kanban_task()``),
    * the raw ``HERMES_KANBAN_TASK`` env value is that same task — the tools' scope compare
      (``_own_task_env`` / ``_enforce_worker_task_ownership``) is verbatim, so an id that
      differs at all (e.g. padding) is refused by every lifecycle tool,
    * ``HERMES_KANBAN_RUN_ID`` resolves to an int (``tools.kanban_tools._worker_guard``
      rejects an unbound worker on all four lifecycle tools),
    * the task has a row on this worker's board,
    * that row's ``current_run_id`` is this worker's run — the ``AND current_run_id = ?`` CAS
      the lifecycle tools apply through ``expected_run_id``, and
    * the status is one a handoff can still write (``_TERMINAL_HANDOFF_STATUSES``).

    A phantom ``HERMES_KANBAN_TASK`` (env presence, no run id, no row) fails one of those and
    yields ``None``, so the stop gate stays silent instead of ordering a call that cannot
    succeed. Any failure is reported as ``None`` on purpose: an unverifiable board must not
    become a demand.
    """
    task_id = owned_kanban_task()
    if not task_id:
        return None
    # ``owned_kanban_task()`` strips; the lifecycle tools compare the raw env value, so a
    # padded (or otherwise differing) id resolves here and is refused there: every terminal
    # call impossible. Fail closed on that mismatch instead of ordering it.
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    run_id = _worker_run_id(task_id)
    if run_id is None:
        return None
    try:
        row = _board_run_and_status(task_id)
    except Exception:
        logger.debug("kanban stop-gate board probe failed for %s", task_id, exc_info=True)
        return None
    if row is None:
        return None
    current_run_id, status = row
    if current_run_id != run_id or status not in _TERMINAL_HANDOFF_STATUSES:
        return None
    return KanbanStopTarget(task_id=task_id, run_id=run_id)


def kanban_stop_nudge_enabled() -> bool:
    """On when the guard would actually nudge: the dispatcher-owned worker identity (not a
    delegate_task child or an in-process cron run) **and** a card/run this worker can prove it
    owns, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it. ``HERMES_KANBAN_TASK`` alone is not
    enough — a worker that cannot legally terminate must not be ordered to."""
    if _nudge_suppressed_by_env():
        return False
    return kanban_stop_target() is not None


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
    target: Optional[KanbanStopTarget] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (no card this worker can terminate, already handed off, budget
    exhausted). ``target`` is a caller's already-resolved :func:`kanban_stop_target` (the stop
    gate passes it so its log line names exactly the card the probe validated); resolved here
    when omitted."""
    if (
        attempts >= max_attempts
        or _nudge_suppressed_by_env()
        or session_called_kanban_terminal(messages)
    ):
        return None
    if target is None:
        target = kanban_stop_target()
    if target is None:
        return None

    tid = target.task_id
    # The transcript is the status source: this text is only reached when the session made no
    # handoff call, so it never tells a worker to close a card it already sent to review.
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


__all__ = [
    "KanbanStopTarget",
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "kanban_stop_target",
    "session_called_kanban_terminal",
]
