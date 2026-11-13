"""Receipt-driven settle of a cron job's ``last_delivery_queued`` marker (#134092).

A Bot Chat hand-off admits with a queued/claimed receipt and the run ends
``last_status="delivery_queued"``; the owner's settle lands only in the receipt file. This
module re-reads those receipts so the job record graduates as soon as the delivery is
terminal, instead of waiting for the job's next run.
"""

import logging

logger = logging.getLogger("cron.scheduler")

_PENDING_RECEIPT_STATUSES = ("queued", "claimed")
_RECEIPT_SETTLED_STATUSES = frozenset({"settled", "suppressed"})


def reconcile_delivery_receipts(job: dict) -> bool:
    """Settle ``last_delivery_queued`` against the receipts Bot Chat already terminal-wrote.

    A Bot Chat hand-off admits with a queued/claimed receipt and the run ends
    ``last_status="delivery_queued"``; the owner's settle lands only in the receipt file, so a
    weekly job keeps telling `cron list` its delivery is in progress until its next run fires
    (#134092). The receipt stays the source of truth — no new state: re-read every queued
    entry, keep entries still in flight (or unreadable, e.g. the deferred lane's records), and
    once no entry is pending anymore graduate ``last_status`` — settled → ok, a terminal
    failure/cancel/ambiguity → delivery_failed with the receipt's reason. Returns True when the
    job record changed. Never raises: bookkeeping must not fail a tick or `cron list`.
    """
    queued = job.get("last_delivery_queued")
    if not isinstance(queued, dict) or not queued:
        return False
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_profile_dir
    from tools.bot_live_delivery import read_delivery_result

    pending: dict = {}
    failures: list = []
    for target, entry in queued.items():
        delivery_id = entry.get("delivery_id") if isinstance(entry, dict) else None
        receipt = None
        if delivery_id:
            try:
                # ``last_delivery_queued`` keys mirror the producer's targets: ``bot-chat:(own)``
                # lives in this home, ``bot-chat:<name>`` in the target profile's home.
                label = str(target).partition(":")[2]
                if label and label != "(own)":
                    home = get_profile_dir(label)
                else:
                    home = get_hermes_home()
                receipt = read_delivery_result(home, delivery_id)
            except Exception as exc:  # health: allow BLE001 -- an unreadable receipt is routine (in flight, deferred lane) and only means "keep the marker"; logged at debug to stay quiet on every tick
                logger.debug("Job '%s': could not read delivery receipt for %s: %s",
                             job.get("id"), target, exc)
                receipt = None
        status = (receipt or {}).get("status")
        if status is None or status in _PENDING_RECEIPT_STATUSES:
            pending[target] = entry
            continue
        if status not in _RECEIPT_SETTLED_STATUSES:
            detail = receipt.get("error") or receipt.get("reason") or "not completed"
            failures.append(f"{target} {status} (receipt {delivery_id}): {detail}")
    updates: dict = {}
    if pending:
        if len(pending) != len(queued):
            updates["last_delivery_queued"] = pending
    else:
        updates["last_delivery_queued"] = None
        # Only the delivery_queued marker graduates; a later run's ok/error status is newer
        # truth and must survive.
        if job.get("last_status") == "delivery_queued":
            if failures:
                updates["last_status"] = "delivery_failed"
                updates["last_delivery_error"] = "; ".join(failures)
            else:
                updates["last_status"] = "ok"
    if not updates:
        return False
    job.update(updates)
    try:
        from cron.jobs import update_job

        update_job(job["id"], updates)
    except Exception as exc:  # health: allow BLE001 -- bookkeeping must not fail a tick or `cron list`; the reconcile simply retries on the next read
        logger.debug(
            "Job '%s': could not record reconciled delivery state: %s", job.get("id"), exc)
    return True
