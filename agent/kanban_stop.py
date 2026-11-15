"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.

Reconciled ownership (the point of this guard): a nudge is
only issued to a worker that STILL owns the live run it was launched on. The guard
reads the trusted execution identity (``agent.kanban_execution_identity``) and re-checks
it against the board's current task/run/claim/process state. A worker whose claim was
completed, reclaimed or superseded gets no instruction to complete/review/block — it is
detached, and nudging it would tell a stale process to mutate a successor's card.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
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
    "kanban_schedule",
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


def worker_still_owns_claim() -> bool:
    """True only while this execution is the live OWNER of its task/run/claim.

    Uses the trusted identity when the process bound one (the normal worker path); with no
    bound identity it falls back to the exact read-only claim proof
    (``worker_claim_is_live``), which compares the dispatcher-pinned task/run/lock against
    the DB rather than trusting the environment alone. Any unreadable state is NOT
    ownership.
    """
    from agent.kanban_execution_identity import (
        ROLE_OWNER,
        current_execution_context,
        owner_is_current,
    )

    try:
        ctx = current_execution_context()
        if ctx is not None:
            return ctx.role == ROLE_OWNER and owner_is_current()
        from agent.kanban_turn_recovery import worker_claim_is_live

        return bool(worker_claim_is_live())
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        # An unreadable board is not proof of ownership: stay detached, and say why.
        logger.debug("kanban stop guard: claim proof unavailable: %s", exc)
        return False


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def _tool_call_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def _result_is_failure(tool_name: str, content: Any) -> bool:
    """Classify a tool result: a refused/failed handoff must not satisfy the guard."""
    if content is None:
        return True
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return True
        try:
            json.loads(text)
        except (TypeError, ValueError):
            return _text_result_is_failure(text)
        raw = text
    else:
        raw = json.dumps(content, ensure_ascii=False)
    from agent.tool_guardrails import classify_tool_failure

    failed, _ = classify_tool_failure(tool_name, raw)
    return failed


def _text_result_is_failure(text: str) -> bool:
    """A plain-text lifecycle result is a handoff unless it names its own refusal."""
    lowered = text.lower()
    if lowered.startswith(("error", "kanban_error", "[error]")):
        return True
    markers = ("refused:", "not authorized", "authorization_denied",
               "no trusted kanban execution identity")
    return any(marker in lowered[:160] for marker in markers)


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already completed a SUCCESSFUL terminal kanban handoff.

    A tool call whose result never arrived (the turn died mid-call) or whose result is a
    refusal/failure is not a handoff: the card is still ``running`` and the worker still
    owes one.
    """
    called_ids: set[str] = set()
    called_names: set[str] = set()
    failed_ids: set[str] = set()
    failed_names: set[str] = set()
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                name = _tool_call_name(tc)
                if name not in _TERMINAL_KANBAN_TOOLS:
                    continue
                cid = _tool_call_id(tc)
                (called_ids.add(cid) if cid else called_names.add(name))
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS and _result_is_failure(name, msg.get("content")):
                cid = str(msg.get("tool_call_id") or "")
                (failed_ids.add(cid) if cid else failed_names.add(name))
    return bool(called_ids - failed_ids) or bool(called_names - failed_names)


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
    ownership_check=None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget
    exhausted, successful handoff recorded, or ownership detached/lost)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None
    still_owner = (ownership_check or worker_still_owns_claim)()
    if not still_owner:
        # Detached: no instruction to complete/review/block. Saying nothing here is the
        # whole point — the run no longer owns a card to hand off.
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # The transcript is the status source: this text is only reached when the session made no
    # successful handoff call, so it never tells a worker to close a card it already sent to review.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no successful terminal board "
        "call (`kanban_complete` / `kanban_request_review` / `kanban_block`). Ending now "
        "causes a protocol violation (clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Terminal subprocesses intentionally do not carry Kanban ownership credentials: the "
        "absent `HERMES_KANBAN_TASK` inside a terminal does NOT mean you lost ownership. "
        "Kanban tool calls in THIS session run under the parent worker's verified identity; "
        "`kanban_show()` reports `execution_identity` with your role and claim state.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "worker_still_owns_claim",
]
