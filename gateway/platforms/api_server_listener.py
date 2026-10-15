"""API listener supervision through the adapter's existing heartbeat."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.platforms.api_server import APIServerAdapter

logger = logging.getLogger(__name__)


async def heartbeat_loop(adapter: APIServerAdapter) -> None:
    from gateway.platforms.api_server import API_SERVER_HEARTBEAT_SECONDS

    while adapter.is_connected:
        # AppRunner.addresses reflects owned sockets, not a foreign process answering a port probe.
        if adapter._runner is None or not adapter._runner.addresses:
            message = f"API server listener on {adapter._host}:{adapter._port} lost its bound sockets"
            logger.error("[%s] %s; requesting background reconnection", adapter.name, message)
            adapter._set_fatal_error("api_server_listener_lost", message, retryable=True)
            await adapter._notify_fatal_error()
            return
        adapter._publish_runtime_status()
        if not adapter.is_connected:
            break
        await asyncio.sleep(API_SERVER_HEARTBEAT_SECONDS)
