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

# How long a fire may spend in the handler (token verify included) before a still-starting gateway is
# refused as retryable. Kept under the dashboard forwarder's 10s timeout: past it NAS sees a 503 while
# this handler still runs the job, and the retry would run it twice.
FIRE_STARTUP_WAIT_SECONDS = 7.0
_POLL_SECONDS = 0.1


async def refuse_until_started(runner: Any, job_id: str, *, received_at: float) -> Optional["web.Response"]:
    """None once ``runner`` has finished starting (``_running``: every platform connect attempted and its
    adapters published), or when there is no runner to ask (a self-hosted api_server, a test double).
    Otherwise the retryable 503, worded like the dashboard's own "gateway unreachable" so NAS classifies
    it the same, with the dashboard's 60s Retry-After (``_CRON_FIRE_RETRY_AFTER_SECONDS``).

    ``received_at`` is the loop time the handler was entered: time already spent verifying the token
    counts against the budget."""
    loop = asyncio.get_running_loop()
    deadline = received_at + FIRE_STARTUP_WAIT_SECONDS
    while runner is not None and not getattr(runner, "_running", True):
        if loop.time() >= deadline:
            return web.json_response(
                {"error": "gateway unreachable; retry", "job_id": job_id}, status=503, headers={"Retry-After": "60"})
        await asyncio.sleep(_POLL_SECONDS)
    return None
