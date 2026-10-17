"""Terminal bookkeeping phases shared by scheduler execution paths."""

from typing import Optional

EMPTY_RESPONSE_ERROR = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"


def fail_empty_response(d, final_response):
    """A "successful" run with a blank answer is a soft failure (never ``ok``): the ordinary tail and
    receipt recovery both book it, so a recovered whitespace answer cannot turn the job green."""
    if d.success and not str(final_response or "").strip():
        d.success = False
        d.error = EMPTY_RESPONSE_ERROR


def finish_interrupted_run(job, execution_id, delivery_error):
    """Shutdown already advanced this run; update a failed notice without spending another fire."""
    from cron import scheduler
    logger = scheduler.logger
    if delivery_error:
        try:
            from cron.jobs import update_job
            update_job(job['id'], {'last_delivery_error': delivery_error})
        except Exception as exc:
            logger.debug('Failed recording delivery_error for interrupted job %s: %s', job['id'], exc, exc_info=True)
    scheduler.finish_execution(execution_id, success=False,
                               error='Interrupted by gateway shutdown before terminal completion.')


def _classify_delivery_outcome(
    *, delivery_error, should_deliver: bool, unresolved_origin: bool,
    normalized_deliver: str, incident_acked: bool, success: bool,
    delivery_queued=None, notification_suppressed: bool = False,
) -> str:
    if delivery_error:
        return "failed"
    if should_deliver and delivery_queued:
        return "queued"
    if notification_suppressed:
        return "suppressed"
    if should_deliver and unresolved_origin:
        return "not_configured"
    if should_deliver and normalized_deliver != "local":
        return "delivered"
    if incident_acked and not success:
        # Failure ping withheld for a known signature: operator acked it, or it was already
        # alerted inside the reminder cooldown (vs. plain "suppressed").
        return "suppressed_acked"
    return "suppressed"


def finish_completed_run(d, fire_owner: Optional[str], execution_id: str, *, recovered=False) -> bool:
    """mark_job_run (owner-fenced) + execution ledger row for a run that reached delivery.

    ``d`` is the scheduler's ``_RunDelivery``. Store and ledger writers are read off the
    ``cron.scheduler`` facade, which is where tests and the receipt-recovery path patch them."""
    from cron import scheduler
    job = d.job
    if not d.should_deliver and job.get("last_delivery_queued"):
        from cron.jobs import update_job
        update_job(job["id"], {"last_delivery_queued": None})
        job["last_delivery_queued"] = None
    mark_kwargs: dict = {"delivery_error": d.delivery_error}
    from cron.scheduler_authority import journal_path
    journal = journal_path(job['id'], execution_id)
    if not job.get('no_agent'):
        mark_kwargs['execution_id'] = execution_id
    if not d.success and job.pop("_model_unreachable", False):
        # Never-reached-the-model failure: schedule the Cowork-style bounded re-run
        # (cron/unreachable_retry.py) inside the same fenced store write.
        mark_kwargs["model_unreachable"] = True
    from cron.unreachable_retry import is_retry_run
    if is_retry_run(job):
        # A re-run of an occurrence that already counted: must not spend another repeat slot.
        mark_kwargs["ladder_rung"] = True
    _hold_s = job.pop("_quota_hold_seconds", None)
    if not d.success and _hold_s:
        # Provider window closed for a known duration: park past it (cron/quota_hold.py, #89376).
        mark_kwargs["quota_hold_seconds"] = _hold_s
        mark_kwargs["recover_consumed_fire"] = bool(job.get("_scheduled_instant"))
    if d.success and not d.delivery_error and d.should_deliver and job.get("last_delivery_queued"):
        mark_kwargs["status"] = "delivery_queued"
    if fire_owner is not None:
        mark_kwargs["expected_fire_owner"] = fire_owner
    if d.blocked_config:
        mark_kwargs["status"] = "blocked_config"
    # A run that removed its own record has nothing left to mark; the delivery above is its result.
    marked = scheduler.self_removal_delivery_allowed(job["id"]) or scheduler.mark_job_run(
        job["id"], d.success, d.error, **mark_kwargs)
    if fire_owner is not None and not marked:
        scheduler.finish_execution(
            execution_id, success=False,
            error="Fire claim ownership lost before terminal completion.")
        return True
    delivery_outcome = _classify_delivery_outcome(
        delivery_error=d.delivery_error,
        delivery_queued=job.get("last_delivery_queued"),
        notification_suppressed=bool(job.get("_notification_all_targets_suppressed")),
        should_deliver=d.should_deliver,
        unresolved_origin=d.unresolved_origin,
        # Read the lane the notice was actually routed through (failure_deliver on failure).
        normalized_deliver=scheduler._normalize_deliver_value(
            scheduler._delivery_lane_value(job, for_failure=not d.success)),
        incident_acked=d.incident_acked,
        success=d.success,
    )
    from cron.delivery_outcome import settle_quietly, settled_outcome
    if delivery_outcome == "queued":
        # A drain that already finished this send (cron/delivery_outcome.py) is the real outcome.
        delivery_outcome = settled_outcome(execution_id) or "queued"
    if delivery_outcome in ("delivered", "not_configured") and not d.success:
        # Failure ping left the process (or had a configured target): mark the incident alerted.
        scheduler._mark_incident_alerted(d.failure_incident_id)
    from functools import partial
    from cron.executions import get_execution, recover_receipted_execution
    finish = partial(recover_receipted_execution, job_id=job['id']) if recovered else scheduler.finish_execution
    finished = finish(
        execution_id, success=d.success, error=d.error, delivery_outcome=delivery_outcome)
    if recovered and finished is None:
        current = get_execution(execution_id)
        # A live foreign firer still owns this row. Keep its only recovery link until settlement.
        # Legacy direct calls may have no execution row; an already-finished row is also safe.
        if current is not None and current['status'] not in {'completed', 'failed'}:
            return False
    if job.get("last_delivery_queued"):
        # A drain that settled before this run's own bookkeeping landed found nothing to fence on.
        settle_quietly(job["id"], execution_id)
    journal.unlink(missing_ok=True)
    return True
