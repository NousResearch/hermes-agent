"""Opt-in container measurements on the existing physical Relay observation.

One process-local collector reads verified local Linux containers. Publications
have a separate daemon pool and a bounded number of in-flight submissions; no
tool or cleanup callback waits for sampling or for another resource publication.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any

from agent.memory_provider import spawn_context_thread
from agent.relay_resource_cgroup import CgroupSample, read_sample, sample_data
from agent.relay_resource_scheduler import COUNT_KEYS, SCHEDULER_EVENT, SchedulerToken, SchedulerWindow, project_window
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.execution_observability import opaque_docker_resource_id

logger = logging.getLogger(__name__)
RESOURCE_EVENT = "hermes.execution_environment.resource_sample"
_MAX_RESOURCES = 128
_MAX_PUBLICATIONS = 16


def configured_interval() -> float | None:
    """Read the owning profile once, before a background thread loses its caller."""
    from hermes_cli.config_effective import load_user_config_effective

    try:
        config = load_user_config_effective()
        settings = config.get("gateway", {}).get("telemetry", {}).get("execution_environment", {})
        settings = settings.get("resource_sampling", {})
        if settings.get("enabled") is not True:
            return None
        interval = settings.get("interval_seconds", 2.0)
        if isinstance(interval, bool):
            raise ValueError("Boolean measurement interval")
        interval = float(interval)
        if not math.isfinite(interval) or not 1.0 <= interval <= 60.0:
            raise ValueError("Measurement interval must be between 1 and 60 seconds")
        return interval
    except Exception as exc:
        logger.warning("Sandbox resource sampling configuration is invalid (%s)", type(exc).__name__, exc_info=True)
        return None


@dataclass
class _Entry:
    native_id: str
    docker_exe: str
    interval: float
    observers: list[weakref.ReferenceType] = field(default_factory=list)
    measurement_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    previous: CgroupSample | None = None
    source_epoch_identity: tuple | None = None
    source_observation_start_ns: int | None = None
    pending: Any = None
    next_read: float = 0.0
    samples: int = 0
    unavailable: int = 0
    baselines: int = 0
    omitted_publications: int = 0
    cpu_usage_us: int = 0
    measured_ns: int = 0
    memory_sample_peak_bytes: int | None = None
    scheduler_windows: int = 0
    scheduler_rejected_windows: int = 0
    scheduler_duration_ns: int = 0
    scheduler_last_end_ns: int = 0
    scheduler_counts: dict[str, int] = field(default_factory=dict)
    scheduler_source_identity: tuple | None = None
    scheduler_last_summary_id: str | None = None

    def active(self) -> list[Any]:
        observers = [ref() for ref in self.observers]
        return [observer for observer in observers if observer is not None and observer.active]

    def summary(self) -> dict[str, Any]:
        data = {
            "schema_version": "1", "measurement_id": self.measurement_id,
            "resource_id": opaque_docker_resource_id(self.native_id),
            "resource_id_scheme": "oci_container_id_sha256_128_v1",
            "samples": self.samples, "unavailable_samples": self.unavailable,
            "baseline_samples": self.baselines, "omitted_publications": self.omitted_publications,
            "pending_publications": int(self.pending is not None and not self.pending.done()),
            "scheduler_switches_status": "not_collected", "coverage": "observed_windows_only",
            "scheduler_rejected_windows": self.scheduler_rejected_windows,
        }
        if self.measured_ns:
            data.update(cpu_usage_us=self.cpu_usage_us, measured_duration_ns=self.measured_ns,
                        average_cpu_cores=self.cpu_usage_us * 1000 / self.measured_ns)
        if self.memory_sample_peak_bytes is not None:
            data["memory_sample_peak_bytes"] = self.memory_sample_peak_bytes
        if self.scheduler_windows:
            data.update(scheduler_switches_status="available", scheduler_windows=self.scheduler_windows,
                        scheduler_duration_ns=self.scheduler_duration_ns, scheduler_counts=dict(self.scheduler_counts),
                        scheduler_coverage="verified_windows_only")
        return data

    def record(self, current: CgroupSample) -> dict[str, Any]:
        data = sample_data(current, self.previous)
        self.samples += 1
        if current.status == "available" and current.source_identity != self.source_epoch_identity:
            self.source_epoch_identity = current.source_identity
            self.source_observation_start_ns = current.cpu_ended_ns
        if current.status != "available":
            self.unavailable += 1
        elif data["cpu_interval_status"] == "available":
            self.cpu_usage_us += data["cpu_usage_delta_us"]
            self.measured_ns += data["window_duration_ns"]
        else:
            self.baselines += 1
        # An unreadable interval is a gap, not a zero-CPU interval.
        self.previous = current if current.status == "available" else None
        memory = data.get("memory_current_bytes")
        if memory is not None:
            self.memory_sample_peak_bytes = max(self.memory_sample_peak_bytes or 0, memory)
        return {**data, "measurement_id": self.measurement_id, "sequence": self.samples,
                "resource_id": opaque_docker_resource_id(self.native_id),
                "resource_id_scheme": "oci_container_id_sha256_128_v1"}


class ResourceObservation:
    """A weak link to a physical observation; never owns a session or backend."""

    def __init__(self, owner: Any, interval: float, sampler: ResourceSampler):
        self.owner = weakref.ref(owner)
        self.interval = interval
        self.sampler = sampler
        self.native_id: str | None = None
        self.active = True
        self.last_summary: dict[str, Any] | None = None

    def start(self, docker_exe: str, native_id: str) -> None:
        if not self.active or native_id == self.native_id or opaque_docker_resource_id(native_id) is None:
            return
        self.stop_resource()
        self.native_id = native_id
        self.sampler.attach(self, docker_exe, native_id)

    def stop_resource(self) -> None:
        if self.native_id is not None:
            self.last_summary = self.sampler.detach(self, self.native_id)
            self.native_id = None

    def finish(self) -> dict[str, Any] | None:
        self.active = False
        self.stop_resource()
        return self.last_summary

    def scheduler_token(self) -> SchedulerToken | None:
        """Capture the original observation and a verified container lifetime before tracing."""
        return self.sampler.scheduler_token(self)

    def submit_scheduler_window(self, token: SchedulerToken, window: SchedulerWindow) -> bool:
        """Admit a trusted collector's final window without waiting for native export."""
        return self.sampler.submit_scheduler_window(self, token, window)


class ResourceSampler:
    """One local-host sampling loop, with one authoritative stream per container."""

    def __init__(self, reader=read_sample):
        self.reader = reader
        self.lock = threading.RLock()
        self.entries: dict[str, _Entry] = {}
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.slots = threading.BoundedSemaphore(_MAX_PUBLICATIONS)
        self.pending: dict[str, Any] = {}
        self.publisher = DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="relay-resource")

    def attach(self, observer: ResourceObservation, docker_exe: str, native_id: str) -> None:
        with self.lock:
            entry = self.entries.get(native_id)
            if entry is None:
                if len(self.entries) >= _MAX_RESOURCES:
                    observer.last_summary = {"status": "resource_limit", "coverage": "unavailable"}
                    return
                entry = self.entries[native_id] = _Entry(native_id, docker_exe, observer.interval)
            entry.observers.append(weakref.ref(observer))
            entry.interval = min(entry.interval, observer.interval)
            if self.thread is None:
                owner = observer.owner()
                if owner is None:
                    return
                # The physical owner captured profile context before the tool turn.
                # Do not retain the current callback/turn in a long-lived thread.
                from agent.relay_runtime import _CURRENT_TURN, _MANAGED_CALLBACK_DEPTH
                context = owner.scope.context.copy()
                context.run(_CURRENT_TURN.set, None)
                context.run(_MANAGED_CALLBACK_DEPTH.set, 0)
                self.thread = context.run(
                    spawn_context_thread, self._run, name="hermes-resource-samples",
                )
                self.thread.start()
            self.wake.set()

    def detach(self, observer: ResourceObservation, native_id: str) -> dict[str, Any] | None:
        with self.lock:
            entry = self.entries.get(native_id)
            if entry is None:
                return observer.last_summary
            owners = [ref() for ref in entry.observers]
            primary = next((owner for owner in owners if owner is not None), None)
            summary = entry.summary() if primary is observer else {
                "status": "shared_measurement", "measurement_id": entry.measurement_id,
                "resource_id": opaque_docker_resource_id(native_id),
            }
            entry.observers = [ref for ref in entry.observers if ref() not in (None, observer)]
            if not entry.active():
                self.entries.pop(native_id, None)
            elif primary is observer:
                # A new observation reports only its own windows, not its predecessor's totals.
                replacement = _Entry(native_id, entry.docker_exe, entry.interval, entry.observers)
                replacement.pending = entry.pending
                self.entries[native_id] = replacement
            self.wake.set()
            return summary

    def _run(self) -> None:
        while True:
            with self.lock:
                for native_id, entry in list(self.entries.items()):
                    if not entry.active():
                        self.entries.pop(native_id, None)
                if not self.entries:
                    self.thread = None
                    return
                due = [entry for entry in self.entries.values() if entry.next_read <= time.monotonic()]
            for entry in due:
                self._sample(entry)
            with self.lock:
                delay = min((entry.next_read for entry in self.entries.values()), default=time.monotonic())
            self.wake.wait(max(0.05, delay - time.monotonic()))
            self.wake.clear()

    def _sample(self, entry: _Entry) -> None:
        with self.lock:
            if self.entries.get(entry.native_id) is not entry or not entry.active():
                return
        try:
            current = self.reader(entry.docker_exe, entry.native_id)
        except Exception as exc:
            logger.debug("Sandbox resource read failed (%s)", type(exc).__name__, exc_info=True)
            current = CgroupSample("read_failed")
        with self.lock:
            if self.entries.get(entry.native_id) is not entry or not entry.active():
                return
            data = entry.record(current)
            entry.next_read = time.monotonic() + entry.interval
            observer = entry.active()[0]
            if entry.pending is not None and not entry.pending.done():
                entry.omitted_publications += 1
                return
            self._publish(entry, observer, data)

    def _publish(self, entry: _Entry, observer: ResourceObservation, data: dict,
                 *, event_name: str = RESOURCE_EVENT) -> bool:
        pending = self.pending.get(entry.native_id)
        if pending is not None and not pending.done():
            entry.omitted_publications += 1
            return False
        owner = observer.owner()
        if owner is None or owner.scope.closed or not self.slots.acquire(blocking=False):
            entry.omitted_publications += 1
            return False
        runtime = owner.scope.runtime()
        operation = None
        try:
            if runtime is None or runtime._closing:
                entry.omitted_publications += 1
                self.slots.release()
                return False
            operation = runtime.acquire_operation_lease()
            context = owner.scope.context.copy()
            if event_name == RESOURCE_EVENT:
                entry.pending = self.publisher.submit(context.run, self._emit, owner, data)
            else:
                entry.pending = self.publisher.submit(context.run, self._emit_named, owner, data, event_name)
            self.pending[entry.native_id] = entry.pending
        except Exception as exc:
            logger.debug("Sandbox resource submission failed (%s)", type(exc).__name__, exc_info=True)
            self.slots.release()
            if operation is not None:
                operation.release()
            entry.omitted_publications += 1
            return False

        def completed(future):
            try:
                if not future.result():
                    with self.lock:
                        entry.omitted_publications += 1
            except Exception as exc:
                logger.debug("Sandbox resource publication failed (%s)", type(exc).__name__, exc_info=True)
                with self.lock:
                    entry.omitted_publications += 1
            finally:
                with self.lock:
                    if self.pending.get(entry.native_id) is future:
                        self.pending.pop(entry.native_id, None)
                operation.release()
                self.slots.release()

        entry.pending.add_done_callback(completed)
        return True

    @staticmethod
    def _emit(owner: Any, data: dict) -> bool:
        return ResourceSampler._emit_named(owner, data, RESOURCE_EVENT)

    @staticmethod
    def _emit_named(owner: Any, data: dict, event_name: str) -> bool:
        scope = owner.scope
        runtime = scope.runtime()
        with scope.lock:
            handle = None if scope.closed else scope.handle
        if handle is not None and runtime is not None:
            # An explicit handle attaches the event without mutating the physical stack,
            # taking its close lock, or inheriting a concurrently active tool's parent.
            runtime.relay.scope.event(event_name, handle=handle, data=data)
            return True
        return False

    def scheduler_token(self, observer: ResourceObservation) -> SchedulerToken | None:
        with self.lock:
            entry = self.entries.get(observer.native_id)
            owner = observer.owner()
            if (not observer.active or entry is None or owner is None or owner.scope.closed
                    or not entry.active() or entry.active()[0] is not observer):
                return None
            sample = entry.previous
            if sample is None or sample.status != "available" or sample.source_identity is None:
                return None
            return SchedulerToken(owner.scope.lease_id, entry.measurement_id,
                                  opaque_docker_resource_id(entry.native_id), entry.source_observation_start_ns,
                                  sample.source_identity)

    def submit_scheduler_window(self, observer: ResourceObservation, token: SchedulerToken,
                                window: SchedulerWindow) -> bool:
        with self.lock:
            current = self.scheduler_token(observer)
            if current is None or not isinstance(token, SchedulerToken) or not isinstance(window, SchedulerWindow):
                return False
            if current != token:
                return False
            entry = self.entries[observer.native_id]
            if window.summary_id == entry.scheduler_last_summary_id:
                return False
            previous_end = entry.scheduler_last_end_ns if entry.scheduler_source_identity == token.source_identity else 0
            try:
                data = project_window(window, token, previous_end_ns=previous_end)
            except (ValueError, TypeError, AttributeError):
                entry.scheduler_rejected_windows += 1
                return False
            if not self._publish(entry, observer, data, event_name=SCHEDULER_EVENT):
                return False
            entry.scheduler_source_identity = token.source_identity
            entry.scheduler_last_summary_id = window.summary_id
            entry.scheduler_last_end_ns = window.end_monotonic_ns
            entry.scheduler_windows += 1
            entry.scheduler_duration_ns += data["window_duration_ns"]
            for key in COUNT_KEYS:
                entry.scheduler_counts[key] = entry.scheduler_counts.get(key, 0) + data["counts"][key]
            return True


_SAMPLER: ResourceSampler | None = None
_SAMPLER_LOCK = threading.Lock()


def observe_resources(owner: Any) -> ResourceObservation | None:
    interval = configured_interval()
    if interval is None:
        return None
    global _SAMPLER
    with _SAMPLER_LOCK:
        if _SAMPLER is None:
            _SAMPLER = ResourceSampler()
        return ResourceObservation(owner, interval, _SAMPLER)
