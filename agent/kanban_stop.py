"""Turn-end guard for kanban workers, which must end with ``kanban_complete`` or
``kanban_block``. Some models narrate the next step and stop with no tool calls;
Hermes treats that as a clean exit → ``rc=0`` → dispatcher ``protocol_violation``.
Policy-only: return a bounded synthetic nudge so the loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


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
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


# ---- Terminal provider-wall guard -----------------------------------------------

# ``FailureReason`` values that mean the provider/credential itself is unusable, so
# retrying the run now would end the same way. Auth is included: a dead key blocks
# every model call, and the dispatcher requeues instead of looping a requeue.
_PROVIDER_WALL_REASONS = frozenset({"billing", "rate_limit", "auth", "auth_permanent"})

# Text markers for provider walls whose result carries no ``failure_reason``
# (e.g. a non-retryable abort built by ``_failed_turn_result`` directly).
# Keep these specific — ``failed=True`` is already required, but a task-level
# error whose body happens to mention "quota" must not trip this.
_PROVIDER_WALL_TEXT = (
    "out of credits", "insufficient balance", "spending limit",
    "usage limit", "exceeded your current quota", "invalid api key",
    "incorrect api key", "personal-team-blocked",
    "http 402", "http 403", "http 429", "error code: 402", "error code: 403",
    "error code: 429",
)


def _looks_like_provider_wall(reason: Any, error_text: str) -> bool:
    if isinstance(reason, str) and reason in _PROVIDER_WALL_REASONS:
        return True
    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _PROVIDER_WALL_TEXT)


def maybe_block_kanban_on_provider_failure(result: Any, *, received_provider_response: bool | None = None) -> bool:
    """Terminal provider-wall guard for kanban workers.

    When every model call failed with a payment / credit / auth / rate-limit
    error, ``run_conversation`` returns early from the retry loop and never
    reaches ``finalize_turn`` — so the budget-exhausted kanban bookkeeping
    there is skipped, and a plain-text exit reads as ``rc=0`` → dispatcher
    ``protocol_violation`` → requeue loop. This closes the run as a
    real ``kanban_block`` (``kind="transient"``) carrying the last provider
    error instead.

    Fires only when: this process is a kanban worker, the turn result says
    the turn failed, the failure looks like a provider wall, no model
    response survived validation this turn, and the conversation never
    invoked a terminal kanban tool itself. Every step is best-effort: the
    exit path must never crash here. Returns True when a block was recorded.
    """
    try:
        task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        if not task_id or not isinstance(result, dict):
            return False
        if not result.get("failed"):
            return False
        error_text = str(result.get("error") or result.get("final_response") or "")
        if not _looks_like_provider_wall(result.get("failure_reason"), error_text):
            return False
        if received_provider_response is None:
            received_provider_response = result.get("received_provider_response") is True
        if received_provider_response:
            return False
        if result.get("messages") and session_called_kanban_terminal(result.get("messages")):
            return False
        reason = str(result.get("failure_reason") or "unknown")
        block_reason = (
            f"All model calls failed with a provider error ({reason}) — blocking with "
            f"the last provider error instead of exiting cleanly: {error_text[:500]}"
        )
        from hermes_cli import kanban_db as _kb
        from hermes_cli import kanban_db_connect as _kbc
        with _kbc.connect_closing() as conn:
            return bool(_kb.block_task(conn, task_id, reason=block_reason, kind="transient"))
    except Exception:
        return False


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "maybe_block_kanban_on_provider_failure",
    "session_called_kanban_terminal",
]
