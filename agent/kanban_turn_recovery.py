"""In-place turn recovery for kanban workers whose API call failed retryably.

A worker session that dies on an exhausted API call (provider stream storm,
upstream idle-kill, transport reset) used to end the run: cli.py's one-shot
single-query path returned, the process exited rc=0 without any terminal kanban
call, and the dispatcher booked a "protocol violation" and cold-restarted the
task from scratch — discarding the session's entire context (files read,
analysis done). Under a provider storm the cold restarts multiplied a morning's
runtime by 2-4x and every stage missed its deadline.

This module decides when a failed worker turn should be retried IN PLACE (same
session, same conversation history, bounded budget) and builds the continuation
nudge. Policy only: the caller owns sleeping, re-entering the turn, and the
final exit code.

Budget: ``HERMES_KANBAN_TURN_RECOVERY`` (default 3 attempts; 0/false disables).
Only retryable provider failures are recovered; quota walls and billing blocks
are left to the dispatcher's cooldown/breaker path.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_RECOVERY_ATTEMPTS = 3

#: Backoff before recovery attempt N (1-based). The last entry repeats.
RECOVERY_DELAYS_SECONDS: tuple[float, ...] = (15.0, 45.0, 90.0)

#: Failure reasons that must NOT be retried in place: a quota wall or a billing
#: block needs the dispatcher's cooldown/breaker accounting, not another call.
_NON_RECOVERABLE_REASONS = frozenset({"rate_limit", "billing"})

_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def kanban_task_id() -> str | None:
    """The dispatcher-set kanban task id, or ``None`` when this is not a worker run.

    Single source of truth for every kanban-worker predicate. The env value is
    stripped exactly once, here, so a whitespace-only value can never read as
    "worker" to one caller and "not a worker" to another (exit-code guards must
    agree with the recovery gate).
    """
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return task_id or None


def kanban_turn_recovery_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set and the attempt budget is non-zero."""
    if kanban_task_id() is None:
        return False
    return max_recovery_attempts() > 0


def max_recovery_attempts() -> int:
    """``HERMES_KANBAN_TURN_RECOVERY`` parsed as an attempt count (0 disables).

    Unset/blank -> :data:`DEFAULT_MAX_RECOVERY_ATTEMPTS`; explicit 0/false/no/off
    -> 0; anything unparseable -> the default. Clamped to [0, 10] so a bad value
    can never create an unbounded loop.
    """
    raw = (os.environ.get("HERMES_KANBAN_TURN_RECOVERY") or "").strip()
    if not raw:
        return DEFAULT_MAX_RECOVERY_ATTEMPTS
    if raw.lower() in _OFF_VALUES:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RECOVERY_ATTEMPTS
    return max(0, min(value, 10))


def recovery_delay_seconds(attempt: int) -> float:
    """Backoff before recovery attempt ``attempt`` (1-based); last entry repeats."""
    if attempt < 1:
        attempt = 1
    index = min(attempt - 1, len(RECOVERY_DELAYS_SECONDS) - 1)
    return RECOVERY_DELAYS_SECONDS[index]


def turn_is_unfinished(result: Any) -> bool:
    """True when a settled one-shot result says the turn did NOT finish its job.

    Three shapes count as unfinished: nothing settled (``None``/non-dict), a
    failed turn, and an INCOMPLETE turn (``partial`` / ``completed=False`` —
    truncation, deferred or exhausted compression, tool-validation give-up).
    Used by both the exit-code guards and the recovery policy so they cannot
    disagree (see ``kanban_task_id`` for the same rule on the env side).
    """
    if not isinstance(result, dict):
        return True
    if result.get("failed") or result.get("partial"):
        return True
    # ``== 0`` catches the canonical ``False`` flag AND an int ``0`` marker (a
    # producer writing 0 must not slip through as "finished"); absent/None means
    # "completed" — normal results omit the field entirely.
    return result.get("completed") == 0


def should_recover_turn(result: Any, *, attempt: int) -> bool:
    """True when a settled worker turn result should be retried in place.

    ``attempt`` is the number of recovery attempts ALREADY made. All must hold:
    the recovery is enabled, the budget is not exhausted, and the turn is
    unfinished — either a retryable failed turn (quota/billing walls excluded),
    or an incomplete turn. An incomplete turn is the same class of "work not
    done": retrying in place keeps the session context a cold restart discards.
    """
    if not kanban_turn_recovery_enabled():
        return False
    if attempt >= max_recovery_attempts():
        return False
    if not isinstance(result, dict):
        return False
    if result.get("failed"):
        if result.get("failure_retryable") is not True:
            return False
        if str(result.get("failure_reason") or "") in _NON_RECOVERABLE_REASONS:
            return False
        return True
    return turn_is_unfinished(result)


def _truncate(text: str, limit: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[: limit - 1] + "…" if len(text) > limit else text


def build_recovery_nudge(result: Any, *, attempt: int, max_attempts: int) -> str:
    """The synthetic user turn sent after a failed worker turn.

    Mirrors the stop-nudge contract (agent/kanban_stop.py): plain text is not a
    terminal state; the worker must finish with ``kanban_complete`` /
    ``kanban_block``. Emphasises "same session, do not start over" so the model
    reuses everything it already read/wrote instead of re-doing the work.
    """
    error = ""
    if isinstance(result, dict):
        error = _truncate(str(result.get("error") or result.get("final_response") or ""))
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: the previous turn ended UNFINISHED — either the API call failed after "
        f"all retries, or the turn stopped incomplete mid-work (recovery attempt {attempt}/{max_attempts}). "
        f"Detail: {error or 'the turn did not complete'}.\n\n"
        f"Task `{task_id}` is still `running`. This is the SAME session, with your full "
        "context — nothing you already read, wrote, or computed is lost. Do NOT start over.\n\n"
        "Do this immediately:\n"
        "1. If the interrupted turn was mid tool-call, re-check the on-disk state of "
        "whatever it was writing before continuing (the action may not have executed).\n"
        "2. Continue the task from where it stopped and finish any remaining deliverable.\n"
        "3. End with a terminal `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, or `kanban_block(reason=...)` if you are blocked.]"
    )


def recover_failed_kanban_turns(
    turn_fn: Callable[[str], Any],
    get_result: Callable[[], Any],
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    emit: Optional[Callable[[str], None]] = None,
) -> int:
    """Retry a failed kanban worker turn in place; returns attempts made.

    ``turn_fn(nudge)`` runs one more turn with the given synthetic user message.
    ``get_result()`` returns the LATEST settled turn result (``None`` when the
    turn never settled — that stops the loop). Loop is bounded by the attempt
    budget even when ``get_result`` never changes.
    """
    attempts = 0
    while True:
        result = get_result()
        if not should_recover_turn(result, attempt=attempts):
            return attempts
        attempts += 1
        delay = recovery_delay_seconds(attempts)
        task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        message = (
            f"[kanban] unfinished turn on {task_id or 'task'} — retrying in place "
            f"(attempt {attempts}/{max_recovery_attempts()}) after {int(delay)}s; session context preserved"
        )
        logger.warning("%s", message)
        if emit is not None:
            try:
                emit(message)
            except Exception:
                logger.debug("kanban turn-recovery emit failed", exc_info=True)
        sleep_fn(delay)
        turn_fn(build_recovery_nudge(result, attempt=attempts, max_attempts=max_recovery_attempts()))


__all__ = [
    "DEFAULT_MAX_RECOVERY_ATTEMPTS",
    "RECOVERY_DELAYS_SECONDS",
    "build_recovery_nudge",
    "kanban_task_id",
    "kanban_turn_recovery_enabled",
    "max_recovery_attempts",
    "recover_failed_kanban_turns",
    "recovery_delay_seconds",
    "should_recover_turn",
    "turn_is_unfinished",
]
