"""Plugin gateway-service lifecycle seam.

Provides a minimal, host-owned interface between plugin-registered async
services and the ``GatewayRunner``. Plugins register async services via
``PluginContext.register_gateway_service``; the gateway starts them after
all adapters have connected and the runner is running, and stops them
before adapter disconnect.

Design constraints:
- The host owns the context; plugins never receive the ``GatewayRunner``.
- Services see a fresh read-only view of currently connected adapters
  on each access, keyed by platform name string. The view reflects
  reconnect-driven adapter changes without re-launching the service.
- Services start exactly once per manager lifecycle and are never
  re-triggered by reconnect logic.
- Task handles are strongly retained and explicitly cancelled/awaited
  on shutdown. A cancellation-suppressing service cannot block
  shutdown beyond a host-owned timeout; it is detached from the bounded
  shutdown wait but still observed (with provenance) when it eventually
  completes.
- Startup and runtime failures are isolated per-service and logged with
  provenance; a failing service never prevents others from starting.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional, final

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

# Host-owned bounded shutdown timeout. Matches the gateway adapter
# teardown default (gateway/run.py) but is intentionally a separate
# constant: this bound is owned by the plugin-service lifecycle seam,
# not the adapter teardown path, and must not be reconfigured by
# adapter-specific env vars or user config.
_SHUTDOWN_TIMEOUT_SECS: float = 5.0

# Async service callable: receives a GatewayServiceContext, runs until
# cancelled or until it returns naturally.
GatewayServiceCallable = Callable[["GatewayServiceContext"], Awaitable[None]]

# Host-supplied resolver returning the current connected-adapter
# mapping. Called on each ``GatewayServiceContext.adapters`` access so
# the service observes reconnect-driven adapter changes without ever
# receiving the runner.
AdaptersResolver = Callable[[], Mapping[str, "BasePlatformAdapter"]]


@dataclass(frozen=True)
class GatewayServiceRegistration:
    """Immutable registration descriptor for a plugin gateway service.

    Captures deterministic provenance at registration time so the host
    can log, attribute, and audit the service independent of mutable
    plugin state.
    """

    name: str
    service: GatewayServiceCallable
    plugin_name: str
    plugin_key: str
    source: str

    @property
    def provenance(self) -> str:
        """Human-readable provenance string for logging."""
        return f"{self.plugin_name} (key={self.plugin_key}, source={self.source})"


@final
class GatewayServiceContext:
    """Host-owned minimal context passed to gateway services at start.

    Exposes a fresh read-only view of currently connected adapters,
    keyed by platform name string. Each ``adapters`` access calls the
    host-supplied resolver and returns a new ``MappingProxyType`` so
    services observe reconnect-driven changes while each held reference
    stays stable. Does NOT expose the ``GatewayRunner``.
    """

    __slots__ = ("_resolver",)

    def __init__(self, adapters_resolver: AdaptersResolver) -> None:
        self._resolver = adapters_resolver

    @property
    def adapters(self) -> Mapping[str, "BasePlatformAdapter"]:
        """Read-only mapping of currently connected platform adapters."""
        return MappingProxyType(dict(self._resolver()))


def _format_overdue_completion(task: asyncio.Task[None]) -> Optional[str]:
    """Return a status string for a completed overdue task.

    Returns None when the task is still pending (caller should not log).
    """
    if task.cancelled():
        return "cancelled"
    exc = task.exception()
    return None if exc is None else f"failed: {exc!r}"


@final
class GatewayServiceManager:
    """Lifecycle manager for plugin-registered gateway services.

    Retains task handles, observes task exceptions and logs context,
    re-raises cancellation, and is a no-op when there are no
    registrations or no connected adapters. Services start exactly once
    per instance lifecycle.

    Shutdown is bounded by ``_SHUTDOWN_TIMEOUT_SECS``: a service that
    suppresses cancellation is detached from the bounded wait, retained
    for observation, and logged with provenance when it eventually
    completes. The manager never blocks gateway shutdown indefinitely
    and never leaves an overdue task unobserved.
    """

    def __init__(self, adapters_resolver: AdaptersResolver) -> None:
        self._resolver = adapters_resolver
        self._started: bool = False
        # Active service tasks, in launch order.
        self._tasks: list[asyncio.Task[None]] = []
        # Task -> registration provenance for log attribution.
        self._registrations: dict[asyncio.Task[None], GatewayServiceRegistration] = {}
        # Overdue tasks: still running after shutdown timeout. Retained
        # so they are not garbage-collected before completion and so
        # eventual exceptions are observed via the done callback.
        self._overdue_tasks: set[asyncio.Task[None]] = set()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def tasks(self) -> list[asyncio.Task[None]]:
        return list(self._tasks)

    @property
    def overdue_tasks(self) -> list[asyncio.Task[None]]:
        return list(self._overdue_tasks)

    async def start_services(
        self,
        registrations: Sequence[GatewayServiceRegistration],
    ) -> None:
        """Start all registered gateway services.

        No-op when already started, when there are no registrations, or
        when the resolver reports no connected adapters. Each service
        runs in its own task so a startup or runtime failure in one
        service is isolated and logged without preventing others from
        starting.
        """
        if self._started:
            return
        if not registrations:
            return
        if not self._resolver():
            return
        self._started = True
        ctx = GatewayServiceContext(self._resolver)
        for reg in registrations:
            task = asyncio.create_task(
                self._run_service(reg, ctx),
                name=f"gateway_service:{reg.name}",
            )
            self._tasks.append(task)
            self._registrations[task] = reg
            logger.info(
                "Launched gateway service '%s' from %s",
                reg.name,
                reg.provenance,
            )

    async def _run_service(
        self,
        reg: GatewayServiceRegistration,
        ctx: GatewayServiceContext,
    ) -> None:
        try:
            await reg.service(ctx)
        except asyncio.CancelledError:
            logger.info("Gateway service '%s' cancelled", reg.name)
            raise
        except Exception:
            logger.exception(
                "Gateway service '%s' from %s raised an unhandled exception",
                reg.name,
                reg.provenance,
            )

    def _make_overdue_done_callback(
        self,
        task: asyncio.Task[None],
        reg: GatewayServiceRegistration,
    ) -> Callable[[asyncio.Task[None]], None]:
        def _callback(_task: asyncio.Task[None]) -> None:
            status = _format_overdue_completion(task)
            if status is None:
                logger.info(
                    "Detached gateway service '%s' from %s completed after "
                    "shutdown timeout",
                    reg.name,
                    reg.provenance,
                )
            elif status == "cancelled":
                logger.info(
                    "Detached gateway service '%s' from %s cancelled after "
                    "shutdown timeout",
                    reg.name,
                    reg.provenance,
                )
            else:
                logger.error(
                    "Detached gateway service '%s' from %s raised after "
                    "shutdown timeout",
                    reg.name,
                    reg.provenance,
                    exc_info=task.exception(),
                )
            self._overdue_tasks.discard(task)
            self._registrations.pop(task, None)

        return _callback

    async def stop_services(self) -> None:
        """Cancel and await all running service tasks.

        Bounds the wait at ``_SHUTDOWN_TIMEOUT_SECS``. Tasks that
        suppress cancellation and are still pending after the timeout
        are detached from the bounded wait: each is retained in
        ``overdue_tasks``, gets a done callback that logs eventual
        completion or failure with provenance, and is removed from
        active tracking so the manager no longer owns it. No-op when
        there are no active tasks.
        """
        if not self._tasks:
            return
        for task in self._tasks:
            if not task.done():
                _ = task.cancel()
        done, pending = await asyncio.wait(
            self._tasks, timeout=_SHUTDOWN_TIMEOUT_SECS
        )
        if done:
            _ = await asyncio.gather(*done, return_exceptions=True)
            # Done tasks are no longer owned by the manager; drop their
            # provenance so _registrations cannot leak across shutdown.
            for task in done:
                self._registrations.pop(task, None)
        for task in pending:
            reg = self._registrations.get(task)
            name = reg.name if reg else "(unknown)"
            provenance = reg.provenance if reg else "(unknown)"
            logger.warning(
                "Gateway service '%s' from %s did not stop within %.1fs; "
                "detaching from bounded shutdown. Eventual completion will "
                "be observed but no longer blocks gateway shutdown.",
                name,
                provenance,
                _SHUTDOWN_TIMEOUT_SECS,
            )
            task.add_done_callback(
                self._make_overdue_done_callback(task, reg) if reg
                else lambda _t: None
            )
            self._overdue_tasks.add(task)
        self._tasks.clear()
