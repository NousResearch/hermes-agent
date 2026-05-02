from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable

from .control_plane import ControlPlaneApiError, ControlPlaneClient
from .executor import HermesCoryExecutor

logger = logging.getLogger(__name__)


class WorkerOutcome(str, Enum):
    NO_JOB = "no_job"
    COMPLETED = "completed"
    FAILED = "failed"


class CoryControlPlaneWorker:
    def __init__(
        self,
        *,
        client: ControlPlaneClient,
        executor: HermesCoryExecutor,
        poll_interval_seconds: float = 5.0,
        max_backoff_seconds: float = 60.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._executor = executor
        self._poll_interval_seconds = poll_interval_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep_fn

    def run_once(self) -> WorkerOutcome:
        claim = self._client.claim_next_job()
        if claim is None:
            logger.debug("No queued interpretation job available")
            return WorkerOutcome.NO_JOB

        assert claim.job is not None
        logger.info("Claimed Cory interpretation job %s", claim.job.id)

        try:
            submission = self._executor.run(claim)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            logger.exception("Hermes failed while interpreting job %s", claim.job.id)
            try:
                self._client.fail_job(claim, message)
            except Exception:
                logger.exception("Failed to report interpretation failure for job %s", claim.job.id)
                raise
            return WorkerOutcome.FAILED

        self._client.complete_job(claim, submission)
        logger.info("Completed Cory interpretation job %s", claim.job.id)
        return WorkerOutcome.COMPLETED

    def run_forever(self) -> None:
        backoff = self._poll_interval_seconds
        while True:
            try:
                outcome = self.run_once()
            except ControlPlaneApiError:
                logger.exception("Control-plane API error while running Cory worker")
                self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_seconds)
                continue
            except Exception:
                logger.exception("Unexpected Cory worker failure")
                self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_seconds)
                continue

            backoff = self._poll_interval_seconds
            self._sleep(self._poll_interval_seconds if outcome == WorkerOutcome.NO_JOB else 0.5)
