"""Turn-end guard for Kanban workers.

A Kanban worker may leave normally only after exactly one successful terminal
board transition. A model naming a terminal tool is not evidence: the matching
tool result must carry Hermes' durable JSON receipt for this task/run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import json
import os
from typing import Any, Optional


_TERMINAL_KANBAN_TOOLS = frozenset(
    {
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    }
)

_DEFAULT_MAX_ATTEMPTS = 2


class HandoffStatus(str, Enum):
    """State of the durable terminal receipts in a worker transcript."""

    NOT_REQUIRED = "not_required"
    MISSING = "missing"
    VALID = "valid"
    CONFLICT = "conflict"


class StopAction(str, Enum):
    """Action the conversation loop must take at a candidate stop."""

    ALLOW = "allow"
    NUDGE = "nudge"
    VIOLATION = "violation"


@dataclass(frozen=True)
class HandoffAssessment:
    status: HandoffStatus
    successful_count: int
    tool_name: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class StopDecision:
    action: StopAction
    assessment: HandoffAssessment
    nudge: Optional[str] = None
    reason: str = ""


def kanban_stop_nudge_enabled() -> bool:
    """Return whether the Kanban stop guard is active for this process."""
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


def _receipt_payload(content: Any) -> Optional[dict]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _receipt_matches_worker(
    payload: dict,
    *,
    task_id: str,
    run_id: str,
) -> bool:
    if payload.get("ok") is not True:
        return False
    receipt_task = str(payload.get("task_id") or "").strip()
    if not receipt_task or (task_id and receipt_task != task_id):
        return False
    if run_id:
        receipt_run = str(payload.get("run_id") or "").strip()
        if receipt_run != run_id:
            return False
    return True


def assess_kanban_handoff(
    messages: Iterable[dict] | None,
    *,
    task_id: Optional[str] = None,
    run_id: Optional[str | int] = None,
) -> HandoffAssessment:
    """Validate exactly one successful terminal receipt for this worker run."""
    history = list(messages or [])
    expected_task = (
        task_id or os.environ.get("HERMES_KANBAN_TASK") or ""
    ).strip()
    expected_run = str(
        run_id if run_id is not None else os.environ.get("HERMES_KANBAN_RUN_ID") or ""
    ).strip()

    terminal_calls: dict[str, str] = {}
    duplicate_terminal_call_ids: set[str] = set()
    for msg in history:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            name = _tool_call_name(tool_call)
            call_id = _tool_call_id(tool_call)
            if name in _TERMINAL_KANBAN_TOOLS and call_id:
                if call_id in terminal_calls:
                    duplicate_terminal_call_ids.add(call_id)
                terminal_calls[call_id] = name

    successful: list[tuple[str, str]] = []
    for msg in history:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or "")
        called_name = terminal_calls.get(call_id)
        result_name = str(msg.get("name") or msg.get("tool_name") or "")
        if not called_name or result_name != called_name:
            continue
        payload = _receipt_payload(msg.get("content"))
        if payload is None:
            continue
        if _receipt_matches_worker(
            payload,
            task_id=expected_task,
            run_id=expected_run,
        ):
            successful.append((call_id, called_name))

    if duplicate_terminal_call_ids:
        return HandoffAssessment(
            status=HandoffStatus.CONFLICT,
            successful_count=len(successful),
            reason=(
                "duplicate terminal tool_call_id values make the handoff "
                "transcript ambiguous"
            ),
        )
    if len(successful) == 1:
        return HandoffAssessment(
            status=HandoffStatus.VALID,
            successful_count=1,
            tool_name=successful[0][1],
            reason="exactly one successful durable terminal receipt",
        )
    if len(successful) > 1:
        return HandoffAssessment(
            status=HandoffStatus.CONFLICT,
            successful_count=len(successful),
            reason=(
                "expected exactly one successful durable terminal receipt, "
                f"found {len(successful)}"
            ),
        )
    if terminal_calls:
        reason = "terminal tool call had no matching successful durable receipt"
    else:
        reason = "no terminal Kanban tool call was made"
    return HandoffAssessment(
        status=HandoffStatus.MISSING,
        successful_count=0,
        reason=reason,
    )


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """Compatibility helper: true only for one valid durable handoff receipt."""
    return assess_kanban_handoff(messages).status is HandoffStatus.VALID


def _nudge_text(*, task_id: str, reason: str) -> str:
    return (
        "[System: You are a Hermes Kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{task_id}` is still `running`. Handoff validation failed: "
        f"{reason}.\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call exactly one lifecycle tool appropriate to the task: "
        "`kanban_complete`, `kanban_block`, `kanban_request_review`, or "
        "`kanban_request_changes`.\n"
        "3. If a lifecycle tool returns an error, correct it and retry; a rejected "
        "tool call is not a handoff.\n\n"
        "Never end a turn with only a promise of future action. Exhausting this "
        "retry budget is an explicit Kanban protocol failure, not a clean exit.]"
    )


def evaluate_kanban_stop(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
    run_id: Optional[str | int] = None,
) -> StopDecision:
    """Return the fail-closed action for a candidate Kanban worker stop."""
    if not kanban_stop_nudge_enabled():
        assessment = HandoffAssessment(
            HandoffStatus.NOT_REQUIRED,
            successful_count=0,
            reason="Kanban stop guard disabled for this process",
        )
        return StopDecision(StopAction.ALLOW, assessment, reason=assessment.reason)

    assessment = assess_kanban_handoff(
        messages,
        task_id=task_id,
        run_id=run_id,
    )
    if assessment.status is HandoffStatus.VALID:
        return StopDecision(StopAction.ALLOW, assessment, reason=assessment.reason)
    if assessment.status is HandoffStatus.CONFLICT:
        return StopDecision(
            StopAction.VIOLATION,
            assessment,
            reason=assessment.reason,
        )
    if attempts >= max_attempts:
        reason = f"Kanban terminal retry budget exhausted: {assessment.reason}"
        return StopDecision(StopAction.VIOLATION, assessment, reason=reason)

    tid = (
        task_id or os.environ.get("HERMES_KANBAN_TASK") or "this task"
    ).strip()
    nudge = _nudge_text(task_id=tid, reason=assessment.reason)
    return StopDecision(
        StopAction.NUDGE,
        assessment,
        nudge=nudge,
        reason=assessment.reason,
    )


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Backward-compatible nudge-only view of :func:`evaluate_kanban_stop`."""
    return evaluate_kanban_stop(
        messages=messages,
        attempts=attempts,
        max_attempts=max_attempts,
        task_id=task_id,
    ).nudge


__all__ = [
    "HandoffAssessment",
    "HandoffStatus",
    "StopAction",
    "StopDecision",
    "assess_kanban_handoff",
    "build_kanban_stop_nudge",
    "evaluate_kanban_stop",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
