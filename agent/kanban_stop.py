"""Turn-end guards for kanban workers, whose turn must END on the terminal board tool that
hands the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``), and must NOT end without one.

``build_kanban_stop_nudge`` covers the missing-handoff direction: some models narrate the
next step and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` →
dispatcher ``protocol_violation``, so we return a bounded synthetic nudge instead of
exiting. ``terminal_handoff_status`` covers the landed-handoff direction: the agent loop
breaks its turn once a terminal call has succeeded, because the card already has a new
owner and a worker that keeps iterating acts on a task it no longer holds.

Policy-only: no board access, no state.
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


def _landed_status(content: Any) -> Optional[str]:
    """The card status a SUCCESS payload reports, else ``None``.

    Success payloads are ``{"ok": true, ...}`` (``tools/kanban_tools._ok`` / ``_ok_landed``
    carry the status the card actually landed in); a refused call is a ``tool_error`` row
    (``{"error": ...}``) and must NOT read as a handoff — the worker has to retry. Anything
    that is not a success payload (spilled/stubbed result, multimodal content list) counts
    the same way: no evidence, no handoff.
    """
    if isinstance(content, dict):
        payload: Any = content
    elif isinstance(content, str) and '"ok"' in content:
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    status = payload.get("status")
    return status.strip() if isinstance(status, str) else ""


def terminal_handoff_status(
    messages: Iterable[dict] | None,
    tool_calls: Iterable[Any] | None,
) -> Optional[str]:
    """The status landed by a terminal board call in the round that just ran, else ``None``.

    The result rows are read from the tail of ``messages`` — the block after the newest
    assistant row carrying tool calls, i.e. THIS round's results, never an earlier round's —
    and are matched to a terminal call by call id (``tool_call_id_variants``, the single
    pairing policy), not by tool name: the mixed-invalid-batch path appends an error row
    under the terminal tool's own name, and name alone would read that as a handoff.
    """
    from agent.message_sanitization import tool_call_id_variants, tool_result_id_variants

    terminal_calls = [
        tc for tc in (tool_calls or ()) if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS
    ]
    if not terminal_calls:
        return None

    history = [m for m in (messages or ()) if isinstance(m, dict)]
    start = 0
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "assistant" and history[idx].get("tool_calls"):
            start = idx + 1
            break
    rows = [m for m in history[start:] if m.get("role") == "tool"]

    for tc in terminal_calls:
        wanted = tool_call_id_variants(tc)
        for row in rows:
            if not (wanted & tool_result_id_variants(row.get("tool_call_id"))):
                continue
            status = _landed_status(row.get("content"))
            if status is not None:
                return status
    return None


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
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "terminal_handoff_status",
]
