"""In-place recovery for kanban worker turns that die on a retryable failure.

A worker spawned by the kanban dispatcher used to end its run silently when a
provider call failed: the one-shot CLI returned, the process exited ``rc=0``
with no terminal ``kanban_complete`` / ``kanban_block`` call, the dispatcher
booked a *protocol violation* and respawned the task from scratch — discarding
the whole session context the worker had accumulated. Under a provider stream
storm that turns one slow morning into several cold restarts per card.

This module retries such a turn IN PLACE: same session, same conversation
history, bounded budget, explicit backoff, and a continuation nudge that tells
the model to finish the task with a real terminal call. When the budget is
exhausted (or the retry is not authorised) the caller exits NON-ZERO so the run
is booked honestly instead of masquerading as a clean exit.

Retry authority is POSITIVE and TYPED — never inferred from an outcome shape:

* only a ``failed`` turn whose classifier marked the failure ``retryable`` is
  eligible, and quota/billing walls are left to the dispatcher's
  cooldown/breaker accounting (``KANBAN_RATE_LIMIT_EXIT_CODE``, see
  ``hermes_cli/cli_single_query.py::_single_query_exit_code``);
* ``interrupted=True`` is never retried (the interrupt was persisted and
  cleared by ``agent/turn_recovery.py::abort_turn_on_interrupt`` — re-entering
  the model would resurrect explicitly cancelled work);
* a turn that ended through the bounded terminal settlement is never retried
  (``agent/turn_finalizer.py::_record_kanban_budget_exhausted`` already recorded
  ``outcome="timed_out"`` and released the claim — #87096);
* before EVERY attempt the worker proves it still owns the exact live run and
  unexpired claim lease (``worker_claim_is_live``). The proof is the carrier the
  dispatcher pinned at spawn — ``HERMES_KANBAN_DB`` + ``HERMES_KANBAN_RUN_ID`` +
  ``HERMES_KANBAN_CLAIM_LOCK`` are REQUIRED and compared exactly, and the
  task/run ``claim_expires`` lease must be unexpired; any missing coordinate or
  expired lease means no proof, no retry (fail closed).

Incomplete-but-not-failed results (``partial`` / ``completed=False``) are
deliberately NOT retry authority: truncation and compression repair are owned
inside the conversation loop (#89289) and by the dispatcher. They still fail the
process exit code (``hermes_cli/cli_single_query.py::_single_query_exit_code``) so such a run is booked
honestly instead of ending as a silent ``rc=0``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

#: Recovery attempts when ``HERMES_KANBAN_TURN_RECOVERY`` is unset.
DEFAULT_MAX_RECOVERY_ATTEMPTS = 3

#: Backoff before recovery attempt N (1-based). The last entry repeats.
RECOVERY_DELAYS_SECONDS: tuple[float, ...] = (15.0, 45.0, 90.0)

#: Failure reasons that must NOT be retried in place: quota walls and billing
#: blocks need the dispatcher's cooldown/breaker accounting, not another call.
#: Aligned with the exit side of main's ``cli._TRANSIENT_PROVIDER_REASONS``
#: (which also releases ``timeout`` / ``overloaded`` / ``server_error`` runs as
#: neutral 75s AFTER in-place recovery has been exhausted). The two sets answer
#: different questions: this one says who may re-enter the model here; main's
#: says how the process exit is booked.
_NON_RECOVERABLE_REASONS = frozenset({
    FailoverReason.rate_limit.value,
    FailoverReason.billing.value,
    FailoverReason.upstream_rate_limit.value,
})

#: ``turn_exit_reason`` prefixes that mean the turn already went through a
#: durable terminal settlement (the run may be closed and its claim released):
#: ``agent/turn_finalizer.py`` writes ``max_iterations_reached(N/M)`` right after
#: recording ``outcome="timed_out"`` for an out-of-budget kanban worker (#87096).
_TERMINAL_TURN_EXIT_PREFIXES = ("max_iterations_reached",)

_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def kanban_task_id() -> Optional[str]:
    """The dispatcher-set kanban task id, or ``None`` when this is not a worker run.

    Single source of truth for every kanban-worker predicate. The env value is
    stripped exactly once, here, so a whitespace-only value can never read as
    "worker" to one caller and "not a worker" to another (the exit-code guard and
    the recovery gate must agree).
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


def _turn_exit_is_terminal(result: dict) -> bool:
    reason = str(result.get("turn_exit_reason") or "")
    return reason.startswith(_TERMINAL_TURN_EXIT_PREFIXES)


def should_recover_turn(result: Any, *, attempt: int) -> bool:
    """True when a settled worker turn grants in-place retry authority.

    ``attempt`` is the number of recovery attempts ALREADY made. Authority is
    positive and typed: an enabled recovery within budget, a FAILED turn whose
    classifier marked the failure ``retryable``, no quota/billing wall, no
    explicit cancellation, and no durable terminal settlement. Everything else —
    including ``partial`` / ``completed=False`` — is not authority to re-enter
    the model. See the module docstring for the full rule set.
    """
    if not kanban_turn_recovery_enabled():
        return False
    if attempt >= max_recovery_attempts():
        return False
    if not isinstance(result, dict):
        return False
    if result.get("interrupted") is True:
        return False
    if _turn_exit_is_terminal(result):
        return False
    if result.get("failed") is not True:
        return False
    if result.get("failure_retryable") is not True:
        return False
    if str(result.get("failure_reason") or "") in _NON_RECOVERABLE_REASONS:
        return False
    return True


def _kanban_db_path() -> Optional[str]:
    """The dispatcher-pinned board DB — no ambient resolution, ever.

    The dispatcher pins ``HERMES_KANBAN_DB`` (plus run id + claim lock) at spawn
    (``hermes_cli/kanban_db_dispatch.py``); that pin IS the exact authority
    carrier the retry proof must re-check. If the pin is absent there is no
    exact carrier to re-prove, so this returns ``None`` and the caller fails
    closed: resolving whatever board is ambient *now* would silently move the
    proof onto a different board — precisely the weaker proof this fail-closed
    contract exists to prevent.
    """
    pinned = (os.environ.get("HERMES_KANBAN_DB") or "").strip()
    return pinned or None


def _now() -> int:
    """Wall-clock seconds — one seam so tests can advance the clock across a backoff."""
    return int(time.time())


def worker_claim_is_live() -> bool:
    """True when THIS process still owns a live, unexpired run/claim lease.

    Verified read-only against the dispatcher-pinned board before EVERY retry.
    All three pinned coordinates are REQUIRED — ``HERMES_KANBAN_DB``,
    ``HERMES_KANBAN_RUN_ID``, ``HERMES_KANBAN_CLAIM_LOCK`` — and compared with
    exact equality; a missing pin means there is no exact authority carrier to
    re-prove, never a fallback to ambient board state.

    The lease itself is part of the liveness contract: the canonical stale-claim
    selector (``hermes_cli/kanban_db.py::release_stale_claims``) treats
    ``status='running' AND claim_expires < now`` as stale and reclaims the task,
    and the worker-liveness path treats an expired claim as non-live. So both
    the task row AND the open run row must carry an UNEXPIRED ``claim_expires``
    (``heartbeat_claim`` extends the task expiry and mirrors it onto the active
    run row). An expired lease means mutation authority has lapsed even before
    the reconciler processes the row; a NULL expiry is not a provable live
    lease either and fails closed.

    FAIL-CLOSED: any missing coordinate, missing/unreadable DB, mismatched
    run/lock/pid, closed run, or expired lease returns False and the caller
    does not re-enter the model — an honest non-zero exit is always available,
    whereas mutating state under a released or expired claim is not recoverable.
    """
    task_id = kanban_task_id()
    if task_id is None:
        return False
    db_path = _kanban_db_path()
    if not db_path or not Path(db_path).exists():
        logger.warning("kanban claim check: pinned board db %r not found — not retrying in place", db_path)
        return False
    run_id_env = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    lock_env = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    if not run_id_env or not lock_env:
        logger.warning(
            "kanban claim check: missing dispatcher-pinned run-id/claim-lock carrier "
            "— no exact authority to re-prove, not retrying in place"
        )
        return False
    now = _now()
    try:
        conn = sqlite3.connect(f"{Path(db_path).absolute().as_uri()}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status, worker_pid, current_run_id, claim_lock, claim_expires "
                "FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            if (row["status"] or "") != "running":
                return False
            worker_pid = row["worker_pid"]
            if worker_pid is not None and int(worker_pid) != os.getpid():
                return False
            # A running task with no run pointer is not proof of a live run: the
            # dispatcher writes it on claim and clears it on release/end.
            current_run_id = row["current_run_id"]
            if current_run_id is None:
                return False
            if str(current_run_id) != run_id_env:
                return False
            if (row["claim_lock"] or "") != lock_env:
                return False
            if row["claim_expires"] is None or int(row["claim_expires"]) < now:
                return False
            run = conn.execute(
                "SELECT ended_at, worker_pid, claim_expires FROM task_runs WHERE id = ?",
                (int(current_run_id),),
            ).fetchone()
            if run is None:
                return False
            if run["ended_at"] is not None:
                return False
            run_pid = run["worker_pid"]
            if run_pid is not None and int(run_pid) != os.getpid():
                return False
            if run["claim_expires"] is None or int(run["claim_expires"]) < now:
                return False
            return True
        finally:
            conn.close()
    except Exception:
        logger.warning("kanban claim check failed — not retrying in place", exc_info=True)
        return False


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
    task_id = kanban_task_id() or "this task"
    return (
        "[System: the previous turn ended UNFINISHED — the API call failed after "
        f"all retries and the turn was interrupted (recovery attempt {attempt}/{max_attempts}). "
        f"Error: {error or 'provider stream failure'}.\n\n"
        f"Task `{task_id}` is still `running`. This is the SAME session, with your full "
        "context — nothing you already read, wrote, or computed is lost. Do NOT start over.\n\n"
        "Do this immediately:\n"
        "1. If the interrupted turn was mid tool-call, re-check the on-disk state of "
        "whatever it was writing before continuing (the action may not have executed).\n"
        "2. Continue the task from where it stopped and finish any remaining deliverable.\n"
        "3. End with a terminal `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, or `kanban_block(reason=...)` if you are blocked.]"
    )


def _emit_recovery_skipped(emit: Optional[Callable[[str], None]]) -> None:
    """The one "no proof, no retry" line — fail-closed is worth saying out loud."""
    message = (
        f"[kanban] in-place recovery skipped for {kanban_task_id() or 'task'}: this worker no "
        "longer holds a live run/claim (already settled, reclaimed, or unverifiable)"
    )
    logger.warning("%s", message)
    if emit is not None:
        try:
            emit(message)
        except Exception:
            logger.debug("kanban turn-recovery emit failed", exc_info=True)


def recover_failed_kanban_turns(
    turn_fn: Callable[[str], Any],
    get_result: Callable[[], Any],
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    emit: Optional[Callable[[str], None]] = None,
    claim_check: Optional[Callable[[], bool]] = None,
) -> int:
    """Retry an authorised kanban worker turn in place; returns attempts made.

    ``turn_fn(nudge)`` runs one more turn with the given synthetic user message.
    ``get_result()`` returns the LATEST settled turn result (``None`` stops the
    loop). Before EVERY attempt the worker must still own its live run/claim
    (``claim_check``, default :func:`worker_claim_is_live`); the loop is bounded
    by the attempt budget even when ``get_result`` never changes. The loop
    deliberately does NOT extend its own claim lease during the backoff —
    self-holding a claim is claim-lifetime policy (#95318), not this module's;
    if the lease expires while sleeping, the re-proof fails and the attempt is
    not made.
    """
    attempts = 0
    check = claim_check or worker_claim_is_live
    while True:
        result = get_result()
        if not should_recover_turn(result, attempt=attempts):
            return attempts
        if not check():
            _emit_recovery_skipped(emit)
            return attempts
        attempts += 1
        delay = recovery_delay_seconds(attempts)
        message = (
            f"[kanban] provider failure on {kanban_task_id() or 'task'} — retrying the turn in place "
            f"(attempt {attempts}/{max_recovery_attempts()}) after {int(delay)}s; session context preserved"
        )
        logger.warning("%s", message)
        if emit is not None:
            try:
                emit(message)
            except Exception:
                logger.debug("kanban turn-recovery emit failed", exc_info=True)
        sleep_fn(delay)
        # Re-prove ownership AFTER the backoff and immediately before model re-entry: the
        # dispatcher can reclaim or settle this run while we sleep, and the pre-sleep proof
        # alone would let that one attempt run without live ownership (round-5 finding F1 —
        # TOCTOU window). No proof, no retry; the caller's honest exit still applies.
        if not check():
            _emit_recovery_skipped(emit)
            return attempts
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
    "worker_claim_is_live",
]
