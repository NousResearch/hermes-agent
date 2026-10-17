"""Bind external scheduler windows to an existing physical Relay observation.

The trusted caller verifies the source lifetime. Hermes only validates the
binding and numeric payload; it neither discovers resources nor starts collectors.
"""

from __future__ import annotations

import logging
import threading
import uuid
import weakref
from typing import Any

from agent.relay_scheduler_windows import COUNT_KEYS, SCHEDULER_EVENT, SchedulerToken, SchedulerWindow, project_window
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.execution_observability import opaque_docker_resource_id

logger = logging.getLogger(__name__)
_PUBLISHER: DaemonThreadPoolExecutor | None = None
_PUBLISHER_LOCK = threading.Lock()
_PUBLICATION_SLOTS = threading.BoundedSemaphore(2)


def _publisher() -> DaemonThreadPoolExecutor:
    global _PUBLISHER
    with _PUBLISHER_LOCK:
        if _PUBLISHER is None:
            _PUBLISHER = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="relay-scheduler")
        return _PUBLISHER


def _source_valid(identity: tuple, start: int) -> bool:
    return (type(identity) is tuple and len(identity) == 6
            and all(type(value) is str and 1 <= len(value) <= 128 for value in identity[:2])
            and all(type(value) is int and 0 <= value < 2**63 for value in identity[2:])
            and type(start) is int and 0 <= start < 2**63)


class SchedulerObservation:
    def __init__(self, owner: Any, token: SchedulerToken):
        self.owner = weakref.ref(owner)
        self._token = token
        self._lock = threading.RLock()
        self._active = True
        self.pending = None
        self._pending_publications = 0
        self._last_end = 0
        self._last_id = None
        self._windows = self._duration_ns = 0
        self._scheduler_counts: dict[str, int] = {}
        self._rejected = self._omitted = self._completed = self._failed = 0

    @property
    def token(self) -> SchedulerToken:
        return self._token

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def submit_scheduler_window(self, token: SchedulerToken, window: SchedulerWindow) -> bool:
        with self._lock:
            owner = self.owner()
            if not self._active or owner is None or owner.scope.closed:
                return False
            if not isinstance(token, SchedulerToken) or token != self.token:
                self._rejected += 1
                return False
            try:
                if not isinstance(window, SchedulerWindow) or window.summary_id == self._last_id:
                    raise ValueError("Invalid or duplicate scheduler window")
                data = project_window(window, self.token, previous_end_ns=self._last_end)
            except (ValueError, TypeError, AttributeError):
                self._rejected += 1
                return False
            if not self._schedule(data):
                self._omitted += 1
                return False
            self._last_end, self._last_id = window.end_monotonic_ns, window.summary_id
            self._windows += 1
            self._duration_ns += data["window_duration_ns"]
            for key in COUNT_KEYS:
                self._scheduler_counts[key] = self._scheduler_counts.get(key, 0) + data["counts"][key]
            return True

    def _schedule(self, data: dict) -> bool:
        if self._pending_publications:
            return False
        owner = self.owner()
        if owner is None:
            return False
        scope, runtime = owner.scope, owner.scope.runtime()
        with scope.lock:
            handle = None if scope.closed else scope.handle
        slots = _PUBLICATION_SLOTS
        if handle is None or runtime is None or runtime._closing or not slots.acquire(blocking=False):
            return False
        operation = None
        try:
            operation = runtime.acquire_operation_lease()
            context = scope.context.copy()
            self._pending_publications = 1
            self.pending = context.run(_publisher().submit, self._emit, data, handle)
        except Exception as exc:
            logger.debug("Scheduler submission failed (%s)", type(exc).__name__, exc_info=True)
            self._pending_publications = 0
            slots.release()
            if operation is not None:
                operation.release()
            return False

        def completed(future):
            try:
                emitted = future.result()
            except Exception as exc:
                logger.debug("Scheduler publication failed (%s)", type(exc).__name__, exc_info=True)
                emitted = False
            with self._lock:
                self._completed += int(bool(emitted))
                self._failed += int(not emitted)
                self._pending_publications = 0
            operation.release()
            slots.release()

        self.pending.add_done_callback(completed)
        return True

    def _emit(self, data: dict, handle: Any) -> bool:
        owner = self.owner()
        if owner is None:
            return False
        scope, runtime = owner.scope, owner.scope.runtime()
        with scope.lock:
            live = not scope.closed and scope.handle is handle
        if not live or runtime is None:
            return False
        # Snapshot the explicit physical handle; never lock lifecycle cleanup
        # around a potentially stalled native subscriber invocation.
        runtime.relay.scope.event(SCHEDULER_EVENT, handle=handle, data=data)
        return True

    def finish(self) -> dict:
        with self._lock:
            self._active = False
            data = {
                "schema_version": "1", "observation_id": self.token.observation_id,
                "measurement_id": self.token.measurement_id, "resource_id": self.token.resource_id,
                "resource_id_scheme": "oci_container_id_sha256_128_v1",
                "source": "linux_tracefs_sched_switch",
                "coverage": "accepted_external_windows_only", "publication_semantics": "admitted_not_export_confirmed",
                "scheduler_windows": self._windows,
                "rejected_windows": self._rejected, "omitted_publications": self._omitted,
                "completed_publications": self._completed, "failed_publications": self._failed,
                "pending_publications": self._pending_publications,
                "scheduler_switches_status": "available" if self._windows else "not_collected",
            }
            if self._windows:
                data.update(scheduler_duration_ns=self._duration_ns,
                            scheduler_counts=dict(self._scheduler_counts))
            return data


def attach_scheduler(environment: Any, *, source_identity: tuple, source_start_ns: int) -> SchedulerObservation | None:
    """Bind an explicitly supplied collector; never infer independent verification."""
    if not _source_valid(source_identity, source_start_ns):
        return None
    owner = getattr(environment, "_relay_lifecycle_observer", None)
    resource_id = opaque_docker_resource_id(getattr(environment, "_container_id", None))
    if owner is None or resource_id is None:
        return None
    with owner.lock:
        if (owner.scope.closed or owner._scheduler_closed
                or resource_id != owner._scheduler_resource_id):
            return None
        previous = owner.scheduler
        if previous is not None:
            token = previous.token
            same_epoch = token.resource_id == resource_id and token.source_identity == source_identity
            if same_epoch:
                return previous if token.source_start_ns == source_start_ns else None
            if previous.active:
                return None
        token = SchedulerToken(owner.scope.lease_id, uuid.uuid4().hex, resource_id, source_start_ns, source_identity)
        owner.scheduler = SchedulerObservation(owner, token)
        return owner.scheduler
