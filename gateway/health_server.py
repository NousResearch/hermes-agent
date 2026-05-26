"""
gateway/health_server.py — Lightweight HTTP server for ECS / ALB health checks.

Runs as a background process (started by docker/entrypoint.saas.sh before the
main gateway process). Listens on :8080 and responds to GET /health.

Why a separate process instead of a thread in the gateway?
  - The Slack socket-mode adapter runs its own asyncio event loop and does not
    expose an HTTP server. Rather than coupling an HTTP listener into every
    gateway platform adapter, we run a tiny standalone aiohttp server in a
    separate process.
  - ECS container health checks require an HTTP 200 response — this is the
    only way to satisfy that without changing every platform adapter.
  - If the gateway process hangs (deadlock, OOM), the health server reports
    200 (it's in a separate process) — ECS circuit breaker catches the hanging
    gateway separately via task timeout.

Routes:
  GET /health    — full dependency check (Neon + S3); used by ALB target group
  GET /healthz   — alias for container-level "am I running" check (no dep check)

Usage:
  python -m gateway.health_server [--port 8080] [--host 0.0.0.0]
  # Or via entrypoint: started automatically by docker/entrypoint.saas.sh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import NoReturn

logger = logging.getLogger(__name__)


async def _handle_health(request: "aiohttp.web.Request") -> "aiohttp.web.Response":
    """GET /health — full dependency health check."""
    from gateway.health import health_check  # noqa: PLC0415
    import aiohttp.web as web  # noqa: PLC0415

    try:
        result = await asyncio.wait_for(health_check(), timeout=12.0)
    except asyncio.TimeoutError:
        result = {
            "status": "degraded",
            "storage": "error",
            "skills": "error",
            "details": {"health_error": "health_check timed out"},
        }

    status = 200 if result.get("status") == "ok" else 503
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps(result),
    )


async def _handle_healthz(request: "aiohttp.web.Request") -> "aiohttp.web.Response":
    """GET /healthz — liveness probe (no dependency checks; always 200 if server is up)."""
    import aiohttp.web as web  # noqa: PLC0415

    return web.Response(
        status=200,
        content_type="application/json",
        text='{"status":"ok"}',
    )


async def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the aiohttp health server and block until SIGTERM or SIGINT."""
    try:
        import aiohttp.web as web  # noqa: PLC0415
    except ImportError:
        logger.critical(
            "aiohttp is not installed. Install with: pip install aiohttp"
        )
        sys.exit(1)

    app = web.Application()
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/healthz", _handle_healthz)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    logger.info("Health server listening on http://%s:%d/health", host, port)

    # Block until shutdown signal.
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("Health server: shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)  # windows-footgun: ok
        except NotImplementedError:
            pass  # Windows asyncio doesn't support signal handlers

    await stop_event.wait()
    await runner.cleanup()
    logger.info("Health server: shutdown complete")


def main() -> NoReturn:
    parser = argparse.ArgumentParser(description="Hermes SaaS health server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run_server(host=args.host, port=args.port))
    sys.exit(0)


if __name__ == "__main__":
    main()
