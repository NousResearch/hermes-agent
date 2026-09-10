"""Wake the existing delivery ledger after a transport recovers in place."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)


def can_redeliver(adapter: BasePlatformAdapter | None) -> bool:
    return bool(adapter is not None and adapter.is_connected
                and not adapter.has_fatal_error and not adapter.send_path_degraded)


def schedule_redelivery(adapter: BasePlatformAdapter | None) -> None:
    """Track a profile-scoped sweep without recursively replaying failed sends."""
    if adapter is None or not can_redeliver(adapter):
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
        if not can_redeliver(adapter):
            return
        try:
            await redeliver(adapter.platform, profile=adapter._owner_profile)
        except Exception:
            logger.debug("Delivery replay after %s recovery failed", adapter.name, exc_info=True)

    task = loop.create_task(recover())
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
