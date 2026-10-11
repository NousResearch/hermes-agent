"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


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


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def _strict_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _tool_result_failed(content: Any) -> bool:
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        return True
    try:
        data = json.loads(content, parse_constant=_strict_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError):
        return True
    return not (isinstance(data, dict) and data.get("ok") is True)


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True only when a terminal Kanban tool call has exactly one matching successful result.

    Fail closed: tool-call/result ids must each be globally unique across the whole
    transcript; the matching result must exist exactly once, have the same tool name,
    and contain a strict JSON object with ``ok is True``.
    """
    msgs = [m for m in (messages or ()) if isinstance(m, dict)]

    call_id_counts: dict[str, int] = {}
    result_id_counts: dict[str, int] = {}
    terminal_calls: dict[str, list[str]] = {}
    terminal_results: dict[str, list[dict]] = {}

    for msg in msgs:
        role = msg.get("role")

        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    call_id = str(tc.get("id") or "").strip()
                else:
                    call_id = str(getattr(tc, "id", "") or "").strip()
                if not call_id:
                    continue

                call_id_counts[call_id] = call_id_counts.get(call_id, 0) + 1
                name = _tool_call_name(tc)
                if name in _TERMINAL_KANBAN_TOOLS:
                    terminal_calls.setdefault(call_id, []).append(name)

        elif role == "tool":
            result_id = str(msg.get("tool_call_id") or "").strip()
            if not result_id:
                continue

            result_id_counts[result_id] = result_id_counts.get(result_id, 0) + 1
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                terminal_results.setdefault(result_id, []).append(msg)

    for call_id, names in terminal_calls.items():
        if call_id_counts.get(call_id) != 1 or len(names) != 1:
            continue

        if result_id_counts.get(call_id) != 1:
            continue

        results = terminal_results.get(call_id, [])
        if len(results) != 1:
            continue

        result = results[0]
        if str(result.get("name") or "") != names[0]:
            continue

        if not _tool_result_failed(result.get("content")):
            return True

    return False

def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # The transcript is the status source: this text is only reached when the session made no
    # handoff call, so it never tells a worker to close a card it already sent to review.
    if attempts + 1 >= max_attempts:
        return (
            "[System: FINAL Kanban terminal guard. Your next response MUST contain exactly "
            "one structured terminal Kanban tool call. Do NOT reply with plain text and do NOT "
            "call `kanban_show` again.\n\n"
            f"Task `{tid}` is still running and has no successful terminal handoff. Choose the "
            "terminal action from the ACTUAL task state now: "
            "`kanban_complete(...)` when the work is complete and needs no review; "
            "`kanban_request_review(...)` for a code change that requires same-card review; "
            "`kanban_request_changes(...)` when you are the reviewer returning the card for "
            "changes; or `kanban_block(...)` when genuinely blocked.\n\n"
            "Issue the structured tool call NOW. Do not narrate intent, do not make another "
            "status-only call, and do not end the turn without a terminal tool call.]"
        )
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
