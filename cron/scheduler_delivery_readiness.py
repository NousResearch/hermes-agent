"""Keep never-attempted script transitions durable until platform registration finishes."""
from __future__ import annotations
import logging

logger = logging.getLogger("cron.scheduler")


def unregistered_delivery_platforms(job: dict, *, for_failure: bool = False) -> list[str]:
    from cron.scheduler_delivery import BOT_CHAT_PLATFORM, ROOM_PLATFORM, _resolve_delivery_targets
    from gateway.platform_registry import platform_registry

    targets = _resolve_delivery_targets(job, for_failure=for_failure)
    return sorted({
        name for target in targets
        if (name := target["platform"].lower()) not in {BOT_CHAT_PLATFORM, ROOM_PLATFORM, "local"}
        and platform_registry.get(name) is None
    })


def queued_delivery_ready(job: dict) -> bool:
    from gateway.platform_registry import platform_registry

    # Only startup-deferred sends carry this marker. No transport has run yet, so waiting is safe;
    # once claimed, the ordinary at-most-once/unknown-send fence still owns the outcome.
    return all(platform_registry.get(name) is not None
               for name in job.get("_delivery_wait_for_platforms", []))


def queue_run_delivery(d, fence, content: str) -> bool:
    """Hand canonical output or a startup-deferred script transition to the existing queue."""
    from cron import scheduler
    from cron.delivery_queue import enqueue

    job = d.job
    execution_id = job.get("execution_id")
    if not execution_id:
        return False
    waiting_for = (unregistered_delivery_platforms(job, for_failure=not d.success)
                   if job.get("no_agent") else [])
    if job.get("no_agent") and not waiting_for:
        return False
    # An origin-less/local job has no send to queue.
    if d.unresolved_origin or scheduler._normalize_deliver_value(
            scheduler._delivery_lane_value(job, for_failure=not d.success)) == "local":
        return True
    queued_job = dict(job)
    if waiting_for:
        queued_job["_delivery_wait_for_platforms"] = waiting_for
    if d.failure_incident_id:
        queued_job["_failure_incident_id"] = d.failure_incident_id
    # Publishing IS the send: a stolen claim must not leave a stale notice queued.
    with fence.side_effect_fence() as owns_delivery:
        if not owns_delivery:
            raise scheduler._FireClaimLostDuringSideEffect
        queued = enqueue(execution_id, queued_job, content, for_failure=not d.success)
        d.delivery_attempted = True
    job["last_delivery_queued"] = {"canonical": {
        "status": queued["status"], "execution_id": execution_id,
    }}
    if waiting_for:
        logger.warning("Job '%s': delivery deferred until platform registration: %s",
                       job["id"], ", ".join(waiting_for))
    return True
