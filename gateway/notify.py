"""
Notify endpoint — lightweight HTTP endpoint for external push notifications.

Enables external services (webhooks, background jobs, other agents) to push
messages directly to a user's chat without going through the LLM. Zero token
cost, sub-second delivery.

This solves a fundamental gap: Hermes currently only sends messages in response
to user input or scheduled cron jobs. External services that need to notify a
user in real-time (match notifications, approval status changes, alerts) have
no way to reach the user without consuming LLM tokens or waiting for a cron tick.

The endpoint accepts a simple POST:

    POST /notify
    {
        "platform": "telegram",
        "chat_id": "123456789",
        "message": "🎉 You have a new match!"
    }

The gateway routes the message through the existing platform adapters — same
delivery infrastructure used by cron jobs, but triggered externally.

Security: Protected by a configurable bearer token (HERMES_NOTIFY_SECRET).
If not set, the endpoint is disabled by default.

Configuration in config.yaml:

    notify:
      enabled: true
      secret: "your-secret-token"
      # host: "127.0.0.1"   # default: loopback only
      # port: 8643           # default: gateway port + 1

Or via environment variable: HERMES_NOTIFY_SECRET=your-secret-token
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None


def check_notify_requirements() -> bool:
    """Check if aiohttp is available."""
    return AIOHTTP_AVAILABLE


class NotifyServer:
    """Lightweight HTTP server for external push notifications."""

    def __init__(self, gateway, secret: Optional[str] = None, host: str = "127.0.0.1", port: int = 8643):
        self.gateway = gateway
        self.secret = secret
        self.host = host
        self.port = port
        self._app = None
        self._runner = None

    async def handle_notify(self, request: web.Request) -> web.Response:
        """Handle POST /notify — deliver a message to a user."""
        # Auth check
        if self.secret:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {self.secret}":
                return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        platform = body.get("platform")
        chat_id = body.get("chat_id")
        message = body.get("message")

        if not platform or not chat_id or not message:
            return web.json_response(
                {"error": "Missing required fields: platform, chat_id, message"},
                status=400,
            )

        # Find the platform adapter and send directly
        try:
            adapter = self.gateway.get_platform_adapter(platform)
            if adapter is None:
                return web.json_response(
                    {"error": f"Platform '{platform}' not configured or not running"},
                    status=404,
                )

            # Use the adapter's send method directly
            result = await adapter.send_message(chat_id, message)
            logger.info(f"notify: delivered to {platform}:{chat_id}")
            return web.json_response({"ok": True, "platform": platform, "chat_id": chat_id})

        except Exception as e:
            logger.error(f"notify: delivery failed to {platform}:{chat_id}: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /notify/health — simple health check."""
        return web.json_response({"status": "ok"})

    async def start(self):
        """Start the notify HTTP server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("Notify: aiohttp not installed, skipping")
            return

        self._app = web.Application()
        self._app.router.add_post("/notify", self.handle_notify)
        self._app.router.add_get("/notify/health", self.handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        try:
            await site.start()
            logger.info(f"Notify endpoint: http://{self.host}:{self.port}/notify")
        except OSError as e:
            logger.warning(f"Notify: failed to bind {self.host}:{self.port}: {e}")

    async def stop(self):
        """Stop the notify server."""
        if self._runner:
            await self._runner.cleanup()
