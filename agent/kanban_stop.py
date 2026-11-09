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


def _is_tool_error_content(content: Any) -> bool:
    """True if tool message content indicates an error/rejection."""
    if not content:
        return False
    if isinstance(content, dict):
        return bool(content.get("error")) or content.get("ok") is False
    if isinstance(content, str):
        trimmed = content.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                data = json.loads(trimmed)
                if isinstance(data, dict):
                    if "error" in data or data.get("ok") is False:
                        return True
            except Exception:
                pass
        lower = trimmed.lower()
        if lower.startswith("error:") or lower.startswith("error "):
            return True
    return False


def _tool_call_args(tc: Any) -> dict[str, Any]:
    """Parse tool call arguments dictionary."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        raw = fn.get("arguments", {}) if isinstance(fn, dict) else tc.get("arguments", {})
    else:
        fn = getattr(tc, "function", None)
        raw = getattr(fn, "arguments", {}) if fn is not None else getattr(tc, "arguments", {})
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def _terminal_tool_result_succeeded(content: Any, expected_task: str) -> bool:
    """Only a recorded non-error result for the owned card can verify handoff.

    Some Kanban adapters return short human-readable success text rather than JSON;
    preserve that contract, but reject empty results, explicit errors and foreign IDs.
    The dispatcher's independent board-state gate remains authoritative.
    """
    if content is None or content == "":
        return False
    if _is_tool_error_content(content):
        return False
    data = content
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return False
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
            except (ValueError, TypeError):
                return False
        elif stripped.lower().startswith(
            ("schema_validation_error", "tool error", "failed:", "rejected:", "exception:")
        ):
            return False
    if isinstance(data, dict):
        if data.get("ok") is False or data.get("success") is False:
            return False
        actual_task = str(data.get("task_id") or data.get("id") or "").strip()
        if expected_task and actual_task and actual_task != expected_task:
            return False
    return True


def session_called_kanban_terminal(
    messages: Iterable[dict] | None,
    task_id: Optional[str] = None,
) -> bool:
    """True only for a matched, successful terminal-tool response for the owned task.

    A merely proposed tool call, an orphan tool result, a mismatched response ID,
    or a failed response cannot satisfy the lifecycle handoff.
    """
    expected_task = (task_id or owned_kanban_task() or "").strip()
    calls: dict[str, tuple[str, str]] = {}
    successful_results: dict[str, tuple[str, Any]] = {}

    for msg in messages or ():
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or ():
                name = _tool_call_name(tc)
                if name not in _TERMINAL_KANBAN_TOOLS:
                    continue
                call_id = str(
                    (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")) or ""
                ).strip()
                if not call_id:
                    continue
                args = _tool_call_args(tc)
                target = str(args.get("task_id") or "").strip()
                if expected_task and target and target != expected_task:
                    continue
                calls[call_id] = (name, target)
        elif msg.get("role") == "tool":
            call_id = str(msg.get("tool_call_id") or "").strip()
            name = str(msg.get("name") or "")
            if call_id and name in _TERMINAL_KANBAN_TOOLS:
                successful_results[call_id] = (name, msg.get("content"))

    for call_id, (name, _) in calls.items():
        result = successful_results.get(call_id)
        if result is None or result[0] != name:
            continue
        if _terminal_tool_result_succeeded(result[1], expected_task):
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
    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages, task_id=tid)
    ):
        return None

    # The transcript is the status source: this text is only reached when the session made no
    # handoff call, so it never tells a worker to close a card it already sent to review.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no terminal board "
        "call (`kanban_complete` / `kanban_request_review` / `kanban_request_changes` / "
        "`kanban_schedule` / `kanban_block`). Ending now causes a protocol violation "
        "(clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`; park recurring tasks with `kanban_schedule(until=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


def kanban_handoff_exhausted(
    messages: Iterable[dict] | None,
    *,
    attempts: int,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> bool:
    """True when the stop gate has already nudged ``max_attempts`` times and this worker still
    made no terminal board call.

    This is the boundary where a prose-only exit must NOT be accepted as a clean completion:
    the worker's narrative answer is a protocol violation the dispatcher would otherwise have to
    book, so the caller fails the turn honestly (non-zero exit) instead of letting the card stay
    ``running`` behind a clean ``rc=0``. ``False`` when this is not a dispatcher-owned worker,
    when the worker already made a terminal call, or when nudges remain.
    """
    return (
        kanban_stop_nudge_enabled()
        and attempts >= max_attempts
        and not session_called_kanban_terminal(messages, task_id=task_id)
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_handoff_exhausted",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
