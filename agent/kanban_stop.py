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
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2

# Tool calls the worker must make after a nudge for it to count as "resumed work" (the
# budget then resets). Low on purpose: one call is a worker reading the nudge and poking
# the board; a few are real progress.
_PROGRESS_RESET_CALLS = 3


def kanban_stop_max_attempts(default: int = _DEFAULT_MAX_ATTEMPTS) -> int:
    """``HERMES_KANBAN_STOP_NUDGE_MAX`` (>=1) or the default."""
    raw = (os.environ.get("HERMES_KANBAN_STOP_NUDGE_MAX") or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value >= 1 else default


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


def consecutive_stop_attempts(messages: Iterable[dict] | None, attempts: int) -> int:
    """``attempts`` minus the nudges the worker answered with real work.

    The budget exists to stop a worker that narrates instead of calling a terminal tool.
    A worker that resumes for dozens of tool calls after a nudge is not that worker — it is
    mid-task — so only a stop *without* work since the last nudge counts against it. Measured
    on a fleet: 6/6 rc=0 crashes spent both nudges mid-task (one pair 18s apart), then worked
    10–25 more minutes with no guard left and exited narrating.
    """
    if attempts <= 0:
        return 0
    streak = 0
    calls_since = 0
    seen_nudge = False
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        if msg.get("_kanban_stop_synthetic") and msg.get("role") == "user":
            if seen_nudge and calls_since >= _PROGRESS_RESET_CALLS:
                streak = 0
            streak += 1
            calls_since = 0
            seen_nudge = True
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls_since += len(msg.get("tool_calls") or [])
    if seen_nudge and calls_since >= _PROGRESS_RESET_CALLS:
        streak = 0
    # The caller's counter is authoritative when the transcript carries no markers
    # (e.g. tests passing bare messages).
    return streak if seen_nudge else attempts


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: Optional[int] = None,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted).

    ``attempts`` counts nudges issued this session; only the *consecutive* ones (no work in
    between) are charged against ``max_attempts``. The last charged nudge is restrictive:
    hand the card off now, do not resume work."""
    if max_attempts is None:
        max_attempts = kanban_stop_max_attempts()
    if not kanban_stop_nudge_enabled() or session_called_kanban_terminal(messages):
        return None
    effective = consecutive_stop_attempts(messages, attempts)
    if effective >= max_attempts:
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    if effective == max_attempts - 1:
        return (
            "[System: You are a Hermes kanban worker. FINAL notice — the next plain-text "
            "reply ends this process with the card still `running` (protocol violation, "
            "work discarded from the board's point of view).\n\n"
            f"Do NOT resume implementation. In this response, make exactly one board call "
            f"for `{tid}` describing the current state: `kanban_request_review(summary=...)` "
            "if a code change exists on the branch, `kanban_complete(summary=..., "
            "artifacts=[...])` if the work is done, otherwise `kanban_block(reason=...)` "
            "stating what remains. A partial handoff is recoverable; a narrated exit is not.]"
        )
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
    "consecutive_stop_attempts",
    "kanban_stop_max_attempts",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
