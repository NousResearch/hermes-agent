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
    from cron.jobs_error_text import sanitize_persisted_error

    # The queue row is redacted but not length-bounded; jobs.json gets the same cap as every
    # other last_delivery_error writer, and the no-op check compares the stored form.
    delivery_error = sanitize_persisted_error(delivery_error)
    latest = latest_execution(job_id)
    if latest is None or latest.get("id") != execution_id:
        return False

    def apply(jobs, _i, job):
        completions = job.get("canonical_completions") or []
        if not completions or completions[-1] != execution_id:
            return False
        status = job.get("last_status")
        if status == "delivery_queued":
            status = "delivery_failed" if delivery_error else "ok"
        if (status, delivery_error) == (job.get("last_status"), job.get("last_delivery_error")):
            return True
        job["last_status"] = status
        job["last_delivery_error"] = delivery_error
        save_jobs(jobs)
        return True

    return bool(_with_job(job_id, apply, missing=False))


def _terminal(execution_id: str) -> Optional[tuple[str, Optional[str]]]:
    """``(ledger_outcome, delivery_error)`` once the queue row is terminal, else None."""
    from cron import delivery_queue as queue

    row = queue.get_execution_status(str(execution_id))
    if not row:
        return None
    if row.get("bot_chat_receipt"):
        import json
        from tools.bot_live_delivery import read_delivery_result
        from cron.bot_chat_legacy import _require_key
        try:
            reference = json.loads(row["bot_chat_receipt"])
            targets = reference.get("targets", [reference])
            if not isinstance(targets, list) or not targets:
                return None
            for target in targets:
                _require_key(target.get("delivery_id"))
                if not isinstance(target.get("home"), str) or not target["home"]:
                    return None
        except (ValueError, TypeError, AttributeError):
            return None  # Corrupt cron evidence cannot use the broader live protocol.
        # A compatibility queue key is a delivery digest, never a ledger execution.
        if reference.get("execution_id") != execution_id:
            return None
        handoff = row["status"] == "pending" and reference.get("handoff") is True
        if row["status"] == "delivered" or handoff:
            # Locally suppressed targets never entered the live authority. Their
            # retained disposition is terminal without inventing a live receipt.
            receipts = [read_delivery_result(target["home"], target["delivery_id"])
                        for target in targets if target.get("local_disposition") != "suppressed"]
            if any(not receipt or receipt.get("status") in ("queued", "claimed") for receipt in receipts):
                return None  # SQL completion is transport handoff, not inference completion.
            failed = next((r for r in receipts if r.get("status") not in ("settled", "suppressed")), None)
            if failed is not None:
                outcome, error = "failed", str(failed.get("error") or "Bot Chat completion unverified; do not resend")
            elif all(r.get("status") == "suppressed" for r in receipts) and not reference.get("other_targets"):
                outcome, error = "suppressed", None
            else:
                outcome, error = "delivered", None
            if handoff:
                from agent.redact import redact_sensitive_text
                from hermes_time import now
                error = redact_sensitive_text(error, force=True, redact_url_credentials=True) if error else None
                with queue._transaction() as conn:
                    cur = conn.execute(
                        "UPDATE deliveries SET status=?, finished_at=?, error=? WHERE execution_id=? "
                        "AND status='pending' AND next_attempt_at IS NULL AND bot_chat_receipt=?",
                        (outcome, now().isoformat(), error, row["execution_id"], row["bot_chat_receipt"]))
                    if cur.rowcount != 1:
                        return None  # Another disposition won; re-observe on the next poll.
                    queue._prune_terminal_unlocked(conn)
            return outcome, error
    if row["status"] not in _LEDGER_OUTCOME:
        return None
    delivery_error = None
    if row["status"] in ("failed", "unknown"):
        delivery_error = str(row.get("error") or f"delivery {row['status']}")
    return _LEDGER_OUTCOME[row["status"]], delivery_error


def settled_outcome(execution_id: str) -> Optional[str]:
    """The ledger outcome a drain already reached for *execution_id* (None while queued)."""
    terminal = _terminal(execution_id)
    return terminal[0] if terminal else None


def settle_queued_delivery(job_id: Optional[str], execution_id: Optional[str]) -> bool:
    """Apply the queue's terminal outcome for *execution_id*; False while still in flight."""
    if not job_id or not execution_id:
        return False
    terminal = _terminal(str(execution_id))
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
