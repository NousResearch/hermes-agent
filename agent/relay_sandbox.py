"""Session use and backend lifecycle observations on independent Relay stacks.

A lease means Hermes observed use of a cached handle, not that it owns the
physical container. Its separate Relay stack leaves tools under their turns.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
import weakref
from typing import Any

from agent import relay_runtime
from tools.environment_observability import observe_environment
from tools.execution_observability import opaque_docker_resource_id

logger = logging.getLogger(__name__)
_ATTACH_LOCK = threading.Lock()
LEASE_SCOPE = "hermes.execution_environment.lease"
LIFECYCLE_MARK = "hermes.execution_environment.lifecycle"
LIFECYCLE_SCOPE = "hermes.execution_environment.lifecycle_scope"


def _resource(container_id: str | None) -> dict[str, str]:
    resource_id = opaque_docker_resource_id(container_id)
    if resource_id is None:
        return {"resource_id_status": "unavailable"}
    return {
        "resource_id": resource_id,
        "resource_id_scheme": "oci_container_id_sha256_128_v1",
        "resource_id_status": "available",
    }


def _bounded(runtime: Any, callback: Any, *, closing: bool = False, on_done: Any = None) -> Any:
    """Use the existing daemon executor, retaining plugins for delayed work."""
    operation = None
    try:
        operation = None if closing else runtime.acquire_operation_lease()
        future = relay_runtime._scope_op_executor().submit(callback)
    except Exception:
        if operation is not None:
            operation.release()
        if on_done is not None:
            on_done()
        raise
    if operation is not None:
        future.add_done_callback(lambda _future: operation.release())
    if on_done is not None:
        future.add_done_callback(lambda _future: on_done())
    return future.result(timeout=relay_runtime.SCOPE_OP_TIMEOUT)


class _EnvironmentLease:
    def __init__(
        self, runtime: Any, session: Any, environment: Any, *, physical: bool = False,
        input_fields: dict[str, Any] | None = None,
    ):
        self.runtime = weakref.ref(runtime)
        self.environment = environment
        self.context = session.context.copy()
        self.parent_uuid = str(session.handle.uuid)
        self.lease_id = uuid.uuid4().hex
        self.physical = physical
        self.input_fields = input_fields or {}
        self.stack = self.handle = None
        self.closed = False
        self.lock = threading.Lock()
        self.native_lock = threading.Lock()
        self._marks_disabled = threading.Event()

    def open(self) -> None:
        runtime = self.runtime()
        if runtime is None:
            return
        relay = runtime.relay

        def push():
            with self.native_lock:
                if self.closed:
                    return
                propagation = self.context.run(relay.capture_propagation_context)
                self.stack = relay.create_scope_stack_from_propagation(
                    relay.PropagationContext(self.parent_uuid, str(propagation.root_uuid))
                )
                with relay.use_scope_stack(self.stack):
                    self.handle = relay.scope.push(
                        LIFECYCLE_SCOPE if self.physical else LEASE_SCOPE, relay.ScopeType.Custom,
                        input={
                            "backend": "docker",
                            "association": "backend_observation" if self.physical else "session_use",
                            "observation_id" if self.physical else "lease_id": self.lease_id,
                            **_resource(getattr(self.environment, "_container_id", None)),
                            **self.input_fields,
                        }, metadata=relay_runtime.runtime_metadata(runtime.runtime_id),
                    )
                    # A timed-out open may finish after the owning session closed.
                    if self.closed:
                        relay.scope.pop(self.handle, output={"reason": "observation_abandoned"})
                        self.handle = None

        try:
            _bounded(runtime, push)
        except Exception:
            self.close("observation_failed")
            raise

    def mark(self, data: dict[str, Any]) -> None:
        runtime = self.runtime()
        if runtime is None or self.closed or self._marks_disabled.is_set():
            return

        def emit():
            with self.native_lock:
                if self.closed or self.handle is None or self._marks_disabled.is_set():
                    return
                with runtime.relay.use_scope_stack(self.stack):
                    runtime.relay.scope.event(
                        f"{LIFECYCLE_MARK}.{data['action']}.{data['outcome']}",
                        handle=self.handle, data=data,
                    )

        try:
            _bounded(runtime, emit, closing=runtime._closing)
        except Exception:
            # One stalled publication must not add another timeout before each
            # Docker cleanup command. Already submitted work retains its lease.
            self._marks_disabled.set()
            raise

    def close(self, reason: str, *, on_done: Any = None, scheduler_summary: Any = None) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
        environment, self.environment = self.environment, None
        with _ATTACH_LOCK:
            leases = getattr(environment, "_relay_environment_leases", ())
            if self in leases:
                leases.remove(self)
        runtime = self.runtime()
        if runtime is None:
            if on_done is not None:
                on_done()
            return

        def pop():
            with self.native_lock:
                if self.handle is None:
                    return
                with runtime.relay.use_scope_stack(self.stack):
                    output = {"reason": reason}
                    if scheduler_summary is not None:
                        output["scheduler_summary"] = scheduler_summary
                    runtime.relay.scope.pop(self.handle, output=output)
                self.handle = None

        try:
            _bounded(runtime, pop, closing=runtime._closing, on_done=on_done)
        except Exception as exc:
            logger.debug("Sandbox lease close failed (%s)", type(exc).__name__, exc_info=True)


class _EnvironmentLifecycle:
    """Observe a backend independently of the session that first used it.

    The runtime registry retains idle observations. Only pending cleanup owns
    an operation lease, so persistent containers cannot pin shutdown forever.
    """

    def __init__(
        self, runtime: Any, session: Any, environment: Any, *, source: str,
        predecessor: Any = None,
    ):
        input_fields = {"observation_source": source}
        if predecessor is not None:
            input_fields.update(
                observation_source="cached_handle_after_cutoff",
                predecessor_observation_id=predecessor.scope.lease_id,
            )
        self.scope = _EnvironmentLease(
            runtime, session, environment, physical=True, input_fields=input_fields,
        )
        self.lock = threading.Lock()
        self.operation = None
        self._configuration_owned = False
        self.scheduler = None
        self._scheduler_closed = False
        self._scheduler_resource_id = opaque_docker_resource_id(getattr(environment, "_container_id", None))
        with runtime._sessions_lock:
            if runtime._closing:
                raise RuntimeError("Hermes Relay runtime is shutting down")
            runtime._environment_lifecycles.add(self)

    def open(self) -> None:
        try:
            self.scope.open()
        except Exception:
            self._release()
            raise
        # Backend ownership is captured on the environment; do not retain its
        # Python handle or session after construction.
        self.scope.environment = None

    def prepare_cleanup(self):
        runtime = self.scope.runtime()
        if runtime is None:
            return self
        with self.lock:
            if not self.scope.closed and self.operation is None:
                self.operation = runtime.acquire_operation_lease()
        return self

    def _release(self) -> None:
        runtime = self.scope.runtime()
        if runtime is not None:
            with runtime._sessions_lock:
                runtime._environment_lifecycles.discard(self)
        with self.lock:
            operation, self.operation = self.operation, None
            configuration_owned, self._configuration_owned = self._configuration_owned, False
        if operation is not None:
            operation.release()
        if configuration_owned:
            relay_runtime._PLUGIN_CONFIGURATION.release(self)

    def finish(self, *, reason: str) -> None:
        runtime = self.scope.runtime()
        with self.lock:
            if self.scope.closed:
                return
            self._scheduler_closed = True
            if runtime is not None and not self._configuration_owned:
                # A cutoff pop can time out after the host has entered shutdown.
                # Join its existing configuration until that pop really finishes.
                relay_runtime._PLUGIN_CONFIGURATION.acquire(self, runtime.relay)
                self._configuration_owned = True
            summary = None if self.scheduler is None else self.scheduler.finish()
        self.scope.close(reason, on_done=self._release, scheduler_summary=summary)

    def transition(
        self, environment: Any, *, action: str, outcome: str,
        container_id: str | None, state_after: str | None = None,
        evidence: str = "operation_result", previous_container_id: str | None = None,
        error_class: str | None = None,
    ) -> None:
        with self.lock:
            if action in ("stop", "remove", "recover") and outcome == "requested":
                self._scheduler_resource_id = None
                if self.scheduler is not None:
                    self.scheduler.finish()
            elif outcome == "succeeded" and state_after == "running":
                self._scheduler_resource_id = opaque_docker_resource_id(container_id)
        data = {
            "schema_version": "1", "backend": "docker", "action": action,
            "outcome": outcome, "evidence": evidence, **_resource(container_id),
        }
        if state_after is not None and outcome == "succeeded":
            data["state_after"] = state_after
        previous = opaque_docker_resource_id(previous_container_id)
        if previous is not None:
            data["previous_resource_id"] = previous
        if error_class is not None:
            data["error_class"] = error_class
        self.scope.mark(data)


class SandboxObserver:
    """Capture the session owner; cleanup threads never rediscover profiles."""

    def __init__(self, runtime: Any, session: Any):
        self.runtime = weakref.ref(runtime)
        self.session = weakref.ref(session)
        self.parent_handle = session.handle

    def provision(self, environment: Any, *, source: str = "provision"):
        """Claim one active owner, or begin a new segment after a prior cutoff."""
        runtime, session = self.runtime(), self.session()
        if runtime is None or session is None:
            return None
        with _ATTACH_LOCK:
            provisioning_lock = getattr(environment, "_relay_provisioning_lock", None)
            if provisioning_lock is None:
                provisioning_lock = environment._relay_provisioning_lock = threading.Lock()
        if not provisioning_lock.acquire(timeout=relay_runtime.SCOPE_OP_TIMEOUT):
            raise TimeoutError("Sandbox observation owner is still being provisioned")
        try:
            predecessor = getattr(environment, "_relay_lifecycle_observer", None)
            if predecessor is not None and not predecessor.scope.closed:
                return predecessor
            with session.lock:
                if session.closing or session.handle is not self.parent_handle:
                    return None
                lifecycle = _EnvironmentLifecycle(
                    runtime, session, environment, source=source, predecessor=predecessor,
                )
                environment._relay_lifecycle_observer = lifecycle
            try:
                lifecycle.open()
            except Exception:
                if environment._relay_lifecycle_observer is lifecycle:
                    environment._relay_lifecycle_observer = predecessor
                raise
            return lifecycle
        finally:
            provisioning_lock.release()

    def acquire(self, environment: Any) -> None:
        runtime = self.runtime()
        session = self.session()
        if runtime is None or session is None or getattr(environment, "env_type", None) != "docker":
            return
        self.provision(environment, source="cached_handle")
        with session.lock:
            if session.closing or session.handle is not self.parent_handle:
                return
            leases = session._environment_observations
            lease = leases.get(id(environment))
            if lease is not None and not lease.closed:
                return
            lease = _EnvironmentLease(runtime, session, environment)
            leases[id(environment)] = lease
        with _ATTACH_LOCK:
            attached = getattr(environment, "_relay_environment_leases", None)
            if attached is None:
                attached = environment._relay_environment_leases = []
            if not lease.closed:
                attached.append(lease)
        lease.open()


@contextlib.contextmanager
def observe_tool_environment(runtime: Any, session: Any):
    """Observe only tools already admitted by the managed Relay adapter."""
    with observe_environment(SandboxObserver(runtime, session)):
        yield


def close_session_observations(runtime: Any, session: Any, reason: str) -> None:
    """End observation segments without asserting physical sandbox destruction."""
    leases = tuple(session._environment_observations.values())
    session._environment_observations.clear()
    for lease in leases:
        lease.close(reason)


def close_runtime_observations(runtime: Any) -> None:
    """End idle observation, without inventing a physical removal event."""
    with runtime._sessions_lock:
        observations = tuple(runtime._environment_lifecycles)
    for observation in observations:
        observation.finish(reason="observation_ended")
