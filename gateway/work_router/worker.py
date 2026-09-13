"""Gateway-owned processing loop for the durable Work Router inbox."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from .service import ProcessResult, WorkRouter


logger = logging.getLogger(__name__)


class WorkRouterWorker:
    """Poll the durable inbox and invoke the generation-fenced processor.

    The loop is intentionally single-worker: SQLite remains the source of truth,
    while one task avoids needless lease contention inside a single Gateway. A
    second process is still fenced by the store's thread leases and CAS checks.
    """

    def __init__(
        self,
        router: WorkRouter,
        *,
        sender: Callable[..., Any],
        ready: Callable[[], bool] | None = None,
        poll_interval_seconds: float = 0.25,
        startup_freshness_seconds: float = 5 * 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(sender):
            raise RuntimeError("Work Router native sender is not bound")
        self.router = router
        self.sender = sender
        self.ready = ready
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self.startup_freshness_seconds = max(0.0, float(startup_freshness_seconds))
        self.clock = clock
        self.worker_id = f"gateway-router-{uuid.uuid4().hex}"

    async def run(self) -> None:
        started_at = self.clock()
        cutoff = started_at - self.startup_freshness_seconds
        skipped = self.router.store.quarantine_stale_events_before(
            cutoff,
            now=started_at,
        )
        logger.info(
            "Work Router worker started: worker=%s poll_interval=%.2fs "
            "startup_freshness=%.0fs stale_quarantined=%d",
            self.worker_id,
            self.poll_interval_seconds,
            self.startup_freshness_seconds,
            len(skipped),
        )
        if skipped:
            logger.warning(
                "Work Router startup backlog quarantined: count=%d "
                "reason=stale_before_start",
                len(skipped),
            )
            for event_id in skipped:
                logger.warning(
                    "Work Router event processed: event_id=%s status=quarantined "
                    "action=none target=none operation_id=none "
                    "error=startup_stale_event",
                    event_id,
                )

        while True:
            # Receivers are wired before connect/registration. Retain durable
            # work until the receiving transport is actually usable, including
            # during reconnect; do not spend an outbox attempt on startup order.
            if self.ready is not None and not self.ready():
                await asyncio.sleep(self.poll_interval_seconds)
                continue
            event_ids = self.router.store.processable_event_ids(now=self.clock())
            if not event_ids:
                await asyncio.sleep(self.poll_interval_seconds)
                continue

            for event_id in event_ids:
                try:
                    result = await self.router.process_once(
                        event_id,
                        worker_id=self.worker_id,
                        sender=self.sender,
                    )
                    self._log_result(result)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One malformed event must not kill the durable worker loop.
                    logger.exception(
                        "Work Router event processing crashed: event_id=%s worker=%s",
                        event_id,
                        self.worker_id,
                    )
            # A live lease or transient store contention can leave the same
            # candidate visible.  Back off between batches instead of turning
            # that condition into a CPU/log busy-spin.
            await asyncio.sleep(self.poll_interval_seconds)

    @staticmethod
    def _log_result(result: ProcessResult) -> None:
        decision = result.decision
        action = decision.action if decision is not None else None
        target = action.target_profile if action is not None else None
        kind = action.kind if action is not None else None
        log = logger.info
        if result.status in {
            "quarantined",
            "recovery_quarantined",
            "stale_worker_rejected",
            "summary_quarantined",
        }:
            log = logger.warning
        elif result.status == "blocked" or result.error:
            log = logger.error
        log(
            "Work Router event processed: event_id=%s status=%s action=%s "
            "target=%s operation_id=%s error=%s",
            result.event_id,
            result.status,
            kind or "none",
            target or "none",
            result.operation_id or "none",
            result.error or "none",
        )
