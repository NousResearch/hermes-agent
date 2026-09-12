"""Ownership of post-turn reviews, including queued work and receipt delivery.

The request handshake still gives foreground turns priority. Worker completion is
separate: process exit and automatic session reclamation must also await receipts.
"""
from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from typing import Any

logger = logging.getLogger(__name__)
DEFAULT_SHUTDOWN_TIMEOUT_S = 120.0


def shutdown_timeout(task_cfg: dict | None = None) -> float:
    if task_cfg is None:
        from agent.background_review import _background_review_task_config
        task_cfg = _background_review_task_config()
    try:
        value = float(task_cfg.get("shutdown_timeout_s", DEFAULT_SHUTDOWN_TIMEOUT_S))
    except (TypeError, ValueError):
        return DEFAULT_SHUTDOWN_TIMEOUT_S
    return min(600.0, max(0.0, value)) if math.isfinite(value) else DEFAULT_SHUTDOWN_TIMEOUT_S


def _lock(agent: Any):
    return getattr(agent, "_background_review_lock", None) or nullcontext()


def has_pending_review(agent: Any) -> bool:
    if agent is None:
        return False
    from agent.review_idle_queue import QUEUE
    with _lock(agent):
        workers = getattr(agent, "_background_review_workers", ())
        if any(not run.worker_done.is_set() for run in workers):
            return True
    return QUEUE.has_pending_for(agent)


def publish_review_status(agent: Any) -> None:
    """A small lifecycle snapshot for the isolated compute-host's parent mirror."""
    callback = vars(agent).get("background_review_status_callback")
    if not callable(callback):
        return
    try:
        # Serialize snapshots, not delivery; a slow transport cannot block startup.
        with _lock(agent):
            pending = has_pending_review_unlocked(agent)
            event = {"pending": pending, "revision": time.monotonic_ns(),
                     "shutdown_timeout_s": getattr(agent, "_review_shutdown_timeout_s", DEFAULT_SHUTDOWN_TIMEOUT_S)}
        callback(event)
    except Exception:
        logger.debug("Could not publish background review lifecycle", exc_info=True)


def has_pending_review_unlocked(agent: Any) -> bool:
    from agent.review_idle_queue import QUEUE
    return (any(not run.worker_done.is_set() for run in getattr(agent, "_background_review_workers", ()))
            or QUEUE.has_pending_for(agent))


def finish_review_worker(agent: Any, run: Any) -> None:
    from agent.background_review import finish_background_review_run
    finish_background_review_run(agent, run)  # also covers pre-request skips / setup failures
    with _lock(agent):
        getattr(agent, "_background_review_workers", set()).discard(run)
        run.worker_done.set()
    publish_review_status(agent)


def cancel_owned_reviews(agent: Any) -> None:
    """Fence queued/startup work before closing resources; never wait on a provider."""
    from agent.background_review import _interrupt_background_review
    from agent.review_idle_queue import QUEUE
    with _lock(agent):
        agent._background_review_closing = True
        workers = tuple(getattr(agent, "_background_review_workers", ()))
    QUEUE.discard_for(agent)
    for run in workers:
        child = run.cancel()
        if child is not None:
            _interrupt_background_review(child)
    publish_review_status(agent)


def drain_background_reviews(agent: Any, *, timeout: float | None = None) -> bool:
    """Bounded one-shot exit linger. True means all receipts settled, not skill writes.

    Deferred local work retains its idle admission policy; we do not commandeer a
    GPU from another foreground turn just because this worker is exiting.
    """
    if agent is None:
        return True
    budget = shutdown_timeout() if timeout is None else max(0.0, float(timeout))
    deadline = time.monotonic() + budget
    waited = False
    while has_pending_review(agent):
        waited = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("Background review exit timeout (session=%s, timeout=%.1fs); cancelling unfinished work",
                           getattr(agent, "session_id", ""), budget)
            cancel_owned_reviews(agent)
            return False
        with _lock(agent):
            run = next(iter(getattr(agent, "_background_review_workers", ())), None)
        if run is not None:
            run.worker_done.wait(min(remaining, 0.1))
        else:
            time.sleep(min(remaining, 0.1))
    if waited:
        logger.info("Background review exit drain completed (session=%s)", getattr(agent, "session_id", ""))
    return True
