"""Write a queued cron delivery's terminal outcome back onto the run that queued it.

An agent run hands its send to ``cron.delivery_queue`` and books ``last_status =
"delivery_queued"`` / ledger ``delivery_outcome = "queued"``. Once a gateway drain sends,
refuses, suppresses or fences the send ``unknown``, the job record and the execution
ledger must carry the real result (main's inline ``ok``/``delivered`` and
``delivery_failed``/``failed`` + ``last_delivery_error``); otherwise ``hermes cron list``
says "delivery still in progress" forever and refused sends never surface.

Both sides call :func:`settle_queued_delivery`: the drain right after it terminalizes a
row, and the run right after its own bookkeeping (a drain can finish before the run's
``mark_job_run`` lands). Every write is fenced on the execution id, so the call is
idempotent and never rewrites a later run's status.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Queue terminal status -> ledger delivery_outcome. ``unknown`` (owner died mid-send, never
# retried) is surfaced as a failure: the user cannot be told the result arrived.
_LEDGER_OUTCOME = {
    "delivered": "delivered", "suppressed": "suppressed", "failed": "failed", "unknown": "failed",
}


def _settle_job_record(job_id: str, execution_id: str, delivery_error: Optional[str]) -> bool:
    """Stamp the outcome onto the job only while *execution_id* is still its current run:
    the ledger's latest attempt for the job AND the job's latest recorded completion."""
    from cron.executions import latest_execution
    from cron.jobs import _with_job, save_jobs

    latest = latest_execution(job_id)
    if latest is None or latest.get("id") != execution_id:
        return False

    def apply(jobs, _i, job):
        completions = job.get("canonical_completions") or []
        if not job.get("no_agent") and (not completions or completions[-1] != execution_id):
            return False
        status = job.get("last_status")
        if status == "delivery_queued":
            status = "delivery_failed" if delivery_error else "ok"
        if (status, delivery_error, None) == (
            job.get("last_status"), job.get("last_delivery_error"), job.get("last_delivery_queued")
        ):
            return True
        job["last_status"] = status
        job["last_delivery_error"] = delivery_error
        job["last_delivery_queued"] = None
        save_jobs(jobs)
        return True

    return bool(_with_job(job_id, apply, missing=False))


def _bot_chat_terminal(job_id: Optional[str]) -> Optional[tuple[str, Optional[str]]]:
    """Check ``last_delivery_queued`` Bot Chat receipts in ``bot_live_delivery``; return
    ``(ledger_outcome, delivery_error)`` once all queued Bot Chat targets are terminal."""
    if not job_id:
        return None
    from cron.jobs import get_job
    from hermes_constants import get_hermes_home
    from tools.bot_live_delivery import read_delivery_result

    job = get_job(str(job_id)) or {}
    ldq = job.get("last_delivery_queued")
    if not isinstance(ldq, dict) or not ldq or "canonical" in ldq:
        return None
    home = get_hermes_home()
    errors: list[str] = []
    saw_target = False
    for target, rec in ldq.items():
        if not isinstance(rec, dict):
            return None
        did = rec.get("delivery_id")
        if not did:
            return None
        live = read_delivery_result(home, did)
        if not live:
            return None
        st = live.get("status")
        if st in ("queued", "claimed", "canonical"):
            return None
        saw_target = True
        if st != "settled":
            err = live.get("error") or live.get("reason") or f"bot-chat delivery {st}"
            errors.append(f"{target} {st} (receipt {did}): {err}")
    if not saw_target:
        return None
    if errors:
        return "failed", "; ".join(errors)
    return "delivered", None


def _job_id_for_execution(execution_id: str) -> Optional[str]:
    try:
        from cron.executions import _connect
        with _connect() as conn:
            row = conn.execute(
                "SELECT job_id FROM executions WHERE id = ?", (str(execution_id),)
            ).fetchone()
            if not row:
                return None
            return str(row["job_id"] if hasattr(row, "keys") else row[0])
    except Exception:
        return None


def _terminal(execution_id: str, job_id: Optional[str] = None) -> Optional[tuple[str, Optional[str]]]:
    """``(ledger_outcome, delivery_error)`` once the queue row is terminal, else None."""
    from cron.delivery_queue import get_status

    row = get_status(str(execution_id))
    if row and row["status"] in _LEDGER_OUTCOME:
        delivery_error = None
        if row["status"] in ("failed", "unknown"):
            delivery_error = str(row.get("error") or f"delivery {row['status']}")
        return _LEDGER_OUTCOME[row["status"]], delivery_error
    resolved_job_id = job_id or _job_id_for_execution(str(execution_id))
    return _bot_chat_terminal(resolved_job_id)


def settled_outcome(execution_id: str, job_id: Optional[str] = None) -> Optional[str]:
    """The ledger outcome a drain already reached for *execution_id* (None while queued)."""
    terminal = _terminal(execution_id, job_id=job_id)
    return terminal[0] if terminal else None


def settle_queued_delivery(job_id: Optional[str], execution_id: Optional[str]) -> bool:
    """Apply the queue's terminal outcome for *execution_id*; False while still in flight."""
    if not job_id or not execution_id:
        return False
    terminal = _terminal(str(execution_id), job_id=str(job_id))
    if terminal is None:
        return False
    from cron.executions import settle_delivery_outcome

    settle_delivery_outcome(str(execution_id), terminal[0])
    _settle_job_record(str(job_id), str(execution_id), terminal[1])
    return True


def settle_quietly(job_id: Optional[str], execution_id: Optional[str]) -> None:
    """Bookkeeping failure must not undo a terminal at-most-once send or fail its run."""
    try:
        settle_queued_delivery(job_id, execution_id)
    except Exception:
        logger.warning("Cron delivery %s: outcome write-back failed", execution_id, exc_info=True)


def settle_bot_chat_delivery_by_id(delivery_id: Optional[str]) -> int:
    """Reconcile any cron job whose ``last_delivery_queued`` references *delivery_id* (or all
    jobs with settled Bot Chat receipts when *delivery_id* is None). Returns count of settled jobs."""
    from cron.executions import latest_execution, settle_delivery_outcome
    from cron.jobs import _with_job, load_jobs, save_jobs

    settled_jobs = 0
    candidate_ids: list[str] = []
    for job in load_jobs():
        ldq = job.get("last_delivery_queued")
        if not isinstance(ldq, dict) or not ldq or "canonical" in ldq:
            continue
        if delivery_id is not None and not any(
            isinstance(r, dict) and r.get("delivery_id") == delivery_id for r in ldq.values()
        ):
            continue
        if job.get("id"):
            candidate_ids.append(str(job["id"]))

    for jid in candidate_ids:
        terminal = _bot_chat_terminal(jid)
        if terminal is None:
            continue
        outcome, delivery_error = terminal
        latest = latest_execution(jid)
        if latest and latest.get("delivery_outcome") == "queued":
            settle_delivery_outcome(str(latest["id"]), outcome)

        def _apply(jobs, _i, j, _err=delivery_error):
            if j.get("last_status") == "delivery_queued":
                j["last_status"] = "delivery_failed" if _err else "ok"
            if _err:
                j["last_delivery_error"] = _err
            j["last_delivery_queued"] = None
            save_jobs(jobs)
            return True

        if _with_job(jid, _apply, missing=False):
            settled_jobs += 1
    return settled_jobs
