"""Startup gate for the Chronos fire webhook (``POST /api/cron/fire``).

A backend that stops the guest on sleep wakes the agent FOR the fire, so the fire reaches the webhook in
the first second of a cold boot. The api_server adapter listens before the gateway has published any
adapter into ``runner.adapters`` (an adapter is registered only after its ``connect()`` returns), so a
fire taken then ran its job with no live adapters and a relay-fronted platform failed with "has no live
gateway transport" (the result was lost). Wait for the gateway to finish starting instead.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from aiohttp import web

# How long a fire that lands during startup waits before it is refused as retryable. A cold boot
# publishes its adapters within seconds; the bound keeps a gateway that never finishes starting from
# holding the request open (NAS's callback timeout is 30s).
FIRE_STARTUP_WAIT_SECONDS = 10.0
_POLL_SECONDS = 0.1


async def await_gateway_started(runner: Any, timeout: Optional[float] = None) -> bool:
    """True once ``runner`` has finished starting (``_running``: every platform connect attempted and its
    adapters published), or when there is no runner to ask (a self-hosted api_server, a test double).
    False when it is still starting after ``timeout`` seconds; the caller answers a retryable 503."""
    if runner is None or getattr(runner, "_running", True):
        return True
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (FIRE_STARTUP_WAIT_SECONDS if timeout is None else timeout)
    while not runner._running:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_POLL_SECONDS)
    return True


async def refuse_until_started(runner: Any, job_id: str) -> Optional["web.Response"]:
    """Wait out a cold boot so the fire's job gets the live adapters (E2EE and relay-fronted platforms have
    no native credential, so without them delivery fails). None when the gateway is up; otherwise the
    retryable 503, worded like the dashboard's own "gateway unreachable" so NAS classifies it the same."""
    if await await_gateway_started(runner):
        return None
    return web.json_response(
        {"error": "gateway unreachable; retry", "job_id": job_id}, status=503, headers={"Retry-After": "2"})
