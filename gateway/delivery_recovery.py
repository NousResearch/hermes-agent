"""Wake the existing delivery ledger after a transport recovers in place."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class RuntimeDeliverySettlement:
    """Defer caller cancellation until a claimed batch has sent or refunded its rows."""

    def __init__(self, adapter: BasePlatformAdapter | None) -> None:
        self.adapter = adapter
        self.cancelled = False

    async def wait(self, operation: Awaitable[_T]) -> _T:
        # Keep the same operation alive: retrying a cancelled DB await can repeat a committed
        # refund against a newer claim owned by this same process.
        task = asyncio.ensure_future(operation)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                self.cancelled = True


def can_redeliver(adapter: BasePlatformAdapter | None) -> bool:
    runner = getattr(adapter, "gateway_runner", None)
    shutdown = getattr(runner, "_shutdown_event", None)
    # Teardown cancels adapter tasks before disconnecting the still-healthy transport.
    return bool(adapter is not None and adapter.is_connected
                and not adapter.has_fatal_error and not adapter.send_path_degraded
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_stop_task", None) is None
                and not (shutdown is not None and shutdown.is_set()))


def schedule_redelivery(
    adapter: BasePlatformAdapter | None, *, settlement: RuntimeDeliverySettlement | None = None,
) -> None:
    """Track a profile-scoped sweep without recursively replaying failed sends."""
    if adapter is None:
        return

    def eligible() -> bool:
        return can_redeliver(adapter) and (
            settlement is None or not settlement.cancelled or adapter is not settlement.adapter
        )

    if not eligible():
        return
    redeliver = getattr(adapter.gateway_runner, "_redeliver_failed_obligations_for_platform", None)
    if not callable(redeliver):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def recover() -> None:
        # A disconnect can follow the notification before this task gets to run.
        if not eligible():
            return
        try:
            await redeliver(adapter.platform, profile=adapter._owner_profile)
        except Exception:
            logger.debug("Delivery replay after %s recovery failed", adapter.name, exc_info=True)

    task = loop.create_task(recover())
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
