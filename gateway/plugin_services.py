"""Plugin gateway-service lifecycle seam.

Provides a minimal, host-owned interface between plugin-registered async
services and the ``GatewayRunner``. Plugins register async services via
``PluginContext.register_gateway_service``; the gateway starts them after
all adapters have connected and the runner is running, and stops them
before adapter disconnect.

Design constraints:
- The host owns the context; plugins never receive the ``GatewayRunner``.
- Services see a read-only snapshot of connected adapters, keyed by
  platform name string.
- Services start exactly once per manager lifecycle and are never
  re-triggered by reconnect logic.
- Task handles are strongly retained and explicitly cancelled/awaited
  on shutdown.
- Startup and runtime failures are isolated per-service and logged with
  provenance; a failing service never prevents others from starting.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

# Async service callable: receives a GatewayServiceContext, runs until
# cancelled or until it returns naturally.
GatewayServiceCallable = Callable[["GatewayServiceContext"], Awaitable[None]]


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

    Exposes a read-only snapshot of successfully connected adapters,
    keyed by platform name string. Does NOT expose the ``GatewayRunner``.
    """

    __slots__ = ("_adapters",)

    def __init__(self, adapters: Mapping[str, "BasePlatformAdapter"]) -> None:
        self._adapters = MappingProxyType(dict(adapters))

    @property
    def adapters(self) -> Mapping[str, "BasePlatformAdapter"]:
        """Read-only mapping of connected platform adapters by name."""
        return self._adapters


@final
class GatewayServiceManager:
    """Lifecycle manager for plugin-registered gateway services.

    Retains task handles, observes task exceptions and logs context,
    re-raises cancellation, and is a no-op when there are no
    registrations or no connected adapters. Services start exactly once
    per instance lifecycle.
    """

    def __init__(self) -> None:
        self._started: bool = False
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def started(self) -> bool:
        return self._started

    @property
    def tasks(self) -> list[asyncio.Task[None]]:
        return list(self._tasks)

    async def start_services(
        self,
        registrations: Sequence[GatewayServiceRegistration],
        adapters: Mapping[str, "BasePlatformAdapter"],
    ) -> None:
        """Start all registered gateway services.

        No-op when already started, when there are no registrations, or
        when there are no connected adapters. Each service runs in its
        own task so a startup or runtime failure in one service is
        isolated and logged without preventing others from starting.
        """
        if self._started:
            return
        if not registrations or not adapters:
            return
        self._started = True
        ctx = GatewayServiceContext(adapters)
        for reg in registrations:
            task = asyncio.create_task(
                self._run_service(reg, ctx),
                name=f"gateway_service:{reg.name}",
            )
            self._tasks.append(task)
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

    async def stop_services(self) -> None:
        """Cancel and await all running service tasks.

        Ensures all tasks are cancelled and awaited before returning.
        No-op when there are no tasks.
        """
        if not self._tasks:
            return
        for task in self._tasks:
            if not task.done():
                _ = task.cancel()
        _ = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
