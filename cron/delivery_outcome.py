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


def _settle_job_record(job_id: str, execution_id: str, delivery_error: Optional[str],
                       *, expected_pending: Optional[dict] = None) -> bool:
    """Stamp the outcome onto the job only while *execution_id* is still its current run:
    the ledger's latest attempt for the job AND the job's latest recorded completion."""
    from cron.executions import latest_execution
    from cron.jobs import _with_job, save_jobs

    def apply(jobs, _i, job):
        if expected_pending is not None and job.get("last_delivery_queued") != expected_pending:
            return False
        latest = latest_execution(job_id)
        if latest is None or latest.get("id") != execution_id or job.get("fire_claim"):
            return False
        completions = job.get("canonical_completions") or []
        if not completions or completions[-1] != execution_id:
            # Older no_agent runs have no canonical completion. The validated,
            # execution-keyed Bot Chat receipts and terminal ledger fence them instead.
            if (not job.get("no_agent") or job.get("fire_claim")
                    or latest.get("delivery_outcome") not in ("delivered", "failed")
                    or job.get("last_status") != "delivery_queued"
                    or _bot_terminal(job_id, execution_id, job=job) is None):
                return False
        status = job.get("last_status")
        if status == "delivery_queued":
            status = "delivery_failed" if delivery_error else "ok"
        if (status, delivery_error) == (job.get("last_status"), job.get("last_delivery_error")):
            return True
        job["last_status"] = status
        job["last_delivery_error"] = delivery_error
        if any(key.startswith("bot-chat:") for key in (job.get("last_delivery_queued") or {})):
            job["last_delivery_queued"] = None
        save_jobs(jobs)
        return True

    from cron.jobs import _under_fire_fence
    return bool(_under_fire_fence(job_id, lambda: _with_job(job_id, apply, missing=False)))


def _legacy_bot_envelope(job: dict, run: dict, target_home):
    """Older successful script runs stored only the key. Require ONE archived output
    inside this attempt's time window; missing/pruned/ambiguous evidence stays queued.
    Never infer a failure-notice payload from successful stdout."""
    import hashlib
    from datetime import datetime
    from cron.jobs import _job_output_dir
    from cron.scheduler_delivery import bot_chat_message

    if not job.get("no_agent") or run.get("status") != "completed":
        return None
    start = datetime.fromisoformat(run["claimed_at"])
    end = datetime.fromisoformat(run["finished_at"])
    candidates = []
    for path in _job_output_dir(job["id"]).glob("*.md"):
        try:
            stamp = datetime.strptime(path.stem, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=end.tzinfo)
        except ValueError:
            continue
        if start.replace(microsecond=0) <= stamp <= end:
            candidates.append(path)
    if len(candidates) != 1:
        return None
    text = candidates[0].read_text(encoding="utf-8-sig")
    header, separator, body = text.partition("\n---\n\n")
    if (not separator or f"**Job ID:** {job['id']}\n" not in header
            or "**Mode:** no_agent (script)\n" not in header):
        return None
    message = bot_chat_message(job, body.strip())
    return dict(execution_id=run["id"], profile_home=str(target_home),
                message_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
                notification_category="result")


def _bot_terminal(job_id: str, execution_id: str, *, job: Optional[dict] = None):
    """Inspect exact Bot Chat receipts only; never discover an owner or replay admission."""
    import hashlib
    import json
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_profile_dir
    from cron.executions import get_execution
    from cron.jobs import get_job
    from tools.bot_live_delivery import read_delivery_result

    job = get_job(job_id) if job is None else job
    run = get_execution(execution_id)
    if (not job or job.get("last_status") != "delivery_queued"
            or not run or run["job_id"] != job_id
            or run.get("status") not in ("completed", "failed", "unknown")):
        return None
    pending = job.get("last_delivery_queued") or {}
    if not pending or any(not target.startswith("bot-chat:") for target in pending):
        return None
    home = get_hermes_home().resolve()
    errors = []
    for target, expected in pending.items():
        profile = target[len("bot-chat:"):]
        target_home = (home if profile == "(own)" else get_profile_dir(profile)).resolve()
        key = hashlib.sha256(json.dumps(
            [str(home), job_id, execution_id, str(target_home)],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if not isinstance(expected, dict) or expected.get("delivery_id") != key:
            return None
        if "message_sha256" not in expected:
            if set(expected) != {"status", "delivery_id"}:
                return None  # Incomplete modern metadata must never downgrade to legacy.
            legacy = _legacy_bot_envelope(job, run, target_home)
            if legacy is None:
                return None
            expected = {**expected, **legacy}
        if (expected.get("delivery_id") != key
                or expected.get("execution_id") != execution_id
                or expected.get("profile_home") != str(target_home)):
            return None
        receipt = read_delivery_result(target_home, key)
        if (not isinstance(receipt, dict) or receipt.get("delivery_id") != key
                or receipt.get("profile_home") != str(target_home)
                or not isinstance(receipt.get("message"), str)
                or hashlib.sha256(receipt["message"].encode("utf-8")).hexdigest() != expected.get("message_sha256")
                or receipt.get("notification_category", "result") != expected.get("notification_category")):
            return None
        status = receipt.get("status")
        if status not in ("settled", "failed", "cancelled", "ambiguous"):
            return None
        if status != "settled":
            errors.append(f"{target} {status}: {receipt.get('error') or receipt.get('reason') or 'completion unverified; do not resend'}")
    return ("failed", "; ".join(errors)) if errors else ("delivered", None)


def _terminal(execution_id: str) -> Optional[tuple[str, Optional[str]]]:
    """``(ledger_outcome, delivery_error)`` once the queue row is terminal, else None."""
    from cron.delivery_queue import get_status

    row = get_status(str(execution_id))
    if not row or row["status"] not in _LEDGER_OUTCOME:
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
    from cron.jobs import get_job
    job = get_job(str(job_id)) or {}
    bot_pending = any(key.startswith("bot-chat:") for key in (job.get("last_delivery_queued") or {}))
    terminal = (_bot_terminal(str(job_id), str(execution_id), job=job)
                if bot_pending else _terminal(str(execution_id)))
    if terminal is None:
        return False
    from cron.executions import settle_delivery_outcome

    settle_delivery_outcome(str(execution_id), terminal[0])
    _settle_job_record(str(job_id), str(execution_id), terminal[1],
                       expected_pending=job.get("last_delivery_queued") if bot_pending else None)
    return True


def reconcile_bot_chat_deliveries() -> None:
    """Tick/restart repair for the current queued receipt of each profile-local job.
    This is status write-back, NOT a drain: no send, owner discovery or admission."""
    from cron.jobs import load_jobs
    from cron.executions import latest_execution

    for job in load_jobs():
        pending = job.get("last_delivery_queued")
        if not isinstance(pending, dict) or not any(target.startswith("bot-chat:") for target in pending):
            continue
        try:
            run = latest_execution(job["id"])
            if run is not None:
                settle_quietly(job["id"], run["id"])
        except Exception:
            logger.warning("Cron Bot Chat %s: reconciliation failed", job.get("id"), exc_info=True)


def settle_quietly(job_id: Optional[str], execution_id: Optional[str]) -> None:
    """Bookkeeping failure must not undo a terminal at-most-once send or fail its run."""
    try:
        settle_queued_delivery(job_id, execution_id)
    except Exception:
        logger.warning("Cron delivery %s: outcome write-back failed", execution_id, exc_info=True)
