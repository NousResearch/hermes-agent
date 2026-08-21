"""Process-side heartbeat keepalive for dispatcher-spawned workers.

The agent's normal activity bridge only runs when model chunks or tools make
foreground progress. A provider request can remain silent longer than the
kanban stale-heartbeat threshold, so workers also need a liveness signal from
a thread that is independent of the model turn.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import Optional

logger = logging.getLogger(__name__)

WORKER_HEARTBEAT_INTERVAL_SECONDS = 5 * 60.0


def _heartbeat_current_run() -> bool:
    """Refresh the current run and return whether this process still owns it.

    The dispatcher injects all three identity values. Requiring the run id and
    claim lock prevents a delayed thread from touching a task that has already
    been reclaimed and assigned to another worker.
    """
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    run_id_raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    claim_lock = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    if not task_id or not run_id_raw or not claim_lock:
        return False
    try:
        run_id = int(run_id_raw)
    except ValueError:
        return False

    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        if not kb.heartbeat_worker(
            conn,
            task_id,
            expected_run_id=run_id,
        ):
            return False
        return kb.heartbeat_claim(conn, task_id, claimer=claim_lock)


class KanbanWorkerHeartbeat:
    """Run a heartbeat callback periodically on a daemon thread."""

    def __init__(
        self,
        heartbeat_fn: Callable[[], bool],
        *,
        interval_seconds: float = WORKER_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._heartbeat_fn = heartbeat_fn
        self._interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="hermes-kanban-worker-heartbeat",
            daemon=True,
        )

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                still_active = self._heartbeat_fn()
            except Exception:
                # A transient DB lock or filesystem error must not kill either
                # the worker or its keepalive. The next interval retries.
                logger.debug("kanban worker heartbeat failed", exc_info=True)
            else:
                if not still_active:
                    self._stop_event.set()
                    return
            if self._stop_event.wait(self._interval_seconds):
                return


@contextlib.contextmanager
def worker_heartbeat_keepalive(
    *,
    interval_seconds: Optional[float] = None,
    heartbeat_fn: Optional[Callable[[], bool]] = None,
) -> Iterator[Optional[KanbanWorkerHeartbeat]]:
    """Keep a dispatched worker alive for the duration of the process scope.

    Outside a fully identified dispatcher worker this is a no-op. Tests may
    inject ``heartbeat_fn`` to exercise lifecycle behavior without board state.
    """
    if heartbeat_fn is None:
        required = (
            os.environ.get("HERMES_KANBAN_TASK"),
            os.environ.get("HERMES_KANBAN_RUN_ID"),
            os.environ.get("HERMES_KANBAN_CLAIM_LOCK"),
        )
        if not all(value and value.strip() for value in required):
            yield None
            return
        heartbeat_fn = _heartbeat_current_run

    keepalive = KanbanWorkerHeartbeat(
        heartbeat_fn,
        interval_seconds=(
            WORKER_HEARTBEAT_INTERVAL_SECONDS
            if interval_seconds is None
            else interval_seconds
        ),
    )
    keepalive.start()
    try:
        yield keepalive
    finally:
        keepalive.stop()
