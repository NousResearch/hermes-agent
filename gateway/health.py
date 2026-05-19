"""
Lightweight health check server for Railway (and other platforms).

Exposes a simple /health endpoint that returns 200 when the gateway
is running. This allows Railway to perform zero-downtime deploys by
only routing traffic to healthy instances.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080
HEALTH_PATH = "/health"


async def health_handler(request: web.Request) -> web.Response:
    """Simple health check endpoint."""
    data = {
        "status": "healthy",
        "service": "hermes-gateway",
        "time": datetime.now(timezone.utc).isoformat(),
        "version": os.getenv("RAILWAY_DEPLOYMENT_ID", "local"),
    }
    return web.json_response(data, status=200)


async def start_health_server(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
) -> web.Application:
    """
    Start a minimal aiohttp server for health checks.
    
    Returns the application so it can be integrated or run standalone.
    """
    port = port or int(os.getenv("HEALTH_PORT", DEFAULT_PORT))

    app = web.Application()
    app.router.add_get(HEALTH_PATH, health_handler)
    app.router.add_get("/", health_handler)  # Also respond on root for convenience

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"Health check server started on http://{host}:{port}{HEALTH_PATH}")
    return app


async def run_health_server_forever():
    """Run the health server as a standalone long-lived task."""
    app = await start_health_server()
    # Keep the server running
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run_health_server_forever())
