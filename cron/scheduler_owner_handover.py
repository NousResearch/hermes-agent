"""A restart-safe cron worker waits out a gateway handover instead of booking a failed run.

An update installs the new checkout before the gateway is replaced, so a ``scoped`` worker the old
gateway spawned in that window runs newer code than its parent. ``ensure_gateway_runtime`` then reports
that parent as ``incompatible``/``parent_outdated`` (it is never the child's to stop), or the install as
``draining``/``update_paused`` while the update fence is up. Either confirms a handover in progress:
the old gateway is about to be replaced, and only a ``scoped`` worker outlives that. So the worker that
owns the fire, before anything is submitted, waits (bounded) for the replacement, tolerating the
``starting``/``draining`` states it passes through, then submits as usual.

Nothing else waits: an interactive caller, a ``degraded`` worker (it shares the gateway's cgroup, so
waiting would race its own teardown), a permanent refusal (``owner_newer``, ``foreign_install``,
``session_authority_unavailable``, ...), and a bare ``starting``/``draining`` with no handover behind it
all fail fast as before. If the budget runs out or the fire is cancelled first, the run fails exactly as
it would have without the wait: the occurrence was dispatched, and nothing was submitted.
"""
import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

OWNER_HANDOVER_WAIT_S = 900.0
_POLL_MAX_S = 5.0
_HANDOVER = {("incompatible", "parent_outdated"), ("draining", "update_paused")}
# States a confirmed handover passes through: the update fence, the old owner draining, and the
# replacement booting (``ensure`` gives up on a slow boot with ``starting``/``deadline``).
_IN_HANDOVER = {"incompatible", "draining", "starting"}


def _clock():
    return time.monotonic()


async def _pause(seconds):
    await asyncio.sleep(seconds)


def _may_wait(request_id) -> bool:
    from cron.scheduler_worker_env import DISPATCH_MARKER
    return (os.environ.get("_HERMES_CRON_EXTERNAL_WORKER") == request_id
            and os.environ.get(DISPATCH_MARKER) == "scoped")


async def dial_owner(run, *, job_id, request_id, cancel_event, submitted):
    """``run(client)`` against the owning gateway, waiting out a confirmed handover first."""
    from cron.executions import touch_execution_progress
    from hermes_cli.gateway_client import GatewayUnavailableError, connect_gateway

    refused, deadline, delay = None, 0.0, 0.5

    def stop_waiting():
        return (cancel_event is not None and cancel_event.is_set()) or _clock() >= deadline

    while refused is None or not stop_waiting():
        try:
            async with connect_gateway() as client:
                # Cancellation or the budget can land while connecting: re-check before submitting.
                if refused is None or not stop_waiting():
                    return await run(client)
            break
        except GatewayUnavailableError as exc:
            verdict = (getattr(exc, "state", None), getattr(exc, "reason_code", None))
            handover = verdict in _HANDOVER or (refused is not None and verdict[0] in _IN_HANDOVER)
            if submitted() or not handover or not _may_wait(request_id):
                raise
            if refused is None:
                deadline = _clock() + OWNER_HANDOVER_WAIT_S
                logger.info("Cron job %s: gateway %s (%s); waiting up to %ds for its replacement",
                            job_id, verdict[0], verdict[1], int(OWNER_HANDOVER_WAIT_S))
            refused = exc
            if stop_waiting():
                break
            # The worker is waiting, not wedged: keep its ledger row out of the stale-claim sweep.
            await asyncio.to_thread(touch_execution_progress, request_id)
            await _pause(min(delay, max(deadline - _clock(), 0.0)))
            delay = min(delay * 2, _POLL_MAX_S)
    raise refused