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
    Authorization: Bearer <secret>
    {
        "platform": "telegram",
        "chat_id": "123456789",
        "message": "🎉 You have a new match!",
        "thread_id": "optional_thread_id"
    }

The gateway routes the message through the existing platform adapters — same
delivery infrastructure used by cron jobs, but triggered externally.

Security:
  - Protected by a required bearer token (HERMES_NOTIFY_SECRET).
  - If no secret is configured, the server does NOT start.
  - Rate limited: 1 message per chat_id per 30 seconds.

Configuration in config.yaml:

    notify:
      enabled: true
      secret: "your-secret-token"
      # host: "127.0.0.1"   # default: loopback only
      # port: 8643           # default: gateway port + 1

Or via environment variable: HERMES_NOTIFY_SECRET=your-secret-token
"""

import logging
import time
from typing import Dict, Optional

from .config import Platform

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

# Rate limit: 1 message per chat per 30 seconds
_RATE_LIMIT_SECONDS = 30
_rate_limit_cache: Dict[str, float] = {}


def _platform_from_string(name: str) -> Optional[Platform]:
    """Convert a platform string to a Platform enum, or None if invalid."""
    try:
        return Platform(name.lower())
    except ValueError:
        return None


class NotifyServer:
    """Lightweight HTTP server for external push notifications."""

    def __init__(self, gateway, secret: str, host: str = "127.0.0.1", port: int = 8643):
        """
        Args:
            gateway: The running Gateway instance (has .adapters dict).
            secret: Required bearer token. Cannot be empty.
            host: Bind address (default loopback only).
            port: Bind port.
        """
        if not secret:
            raise ValueError("NotifyServer requires a non-empty secret")
        self.gateway = gateway
        self.secret = secret
        self.host = host
        self.port = port
        self._app = None
        self._runner = None

    async def handle_notify(self, request: web.Request) -> web.Response:
        """Handle POST /notify — deliver a message to a user."""
        # Auth check (always required — secret is guaranteed non-empty)
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {self.secret}":
            return web.json_response({"error": "Unauthorized"}, status=401)

        # Parse body
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        platform_str = body.get("platform", "")
        chat_id = body.get("chat_id", "")
        message = body.get("message", "")
        thread_id = body.get("thread_id")  # optional

        if not platform_str or not chat_id or not message:
            return web.json_response(
                {"error": "Missing required fields: platform, chat_id, message"},
                status=400,
            )

        # Validate platform
        platform = _platform_from_string(platform_str)
        if platform is None:
            return web.json_response(
                {"error": "Unknown platform", "valid": [p.value for p in Platform]},
                status=400,
            )

        # Rate limit per chat
        rate_key = f"{platform.value}:{chat_id}"
        now = time.monotonic()
        last_sent = _rate_limit_cache.get(rate_key, 0)
        if now - last_sent < _RATE_LIMIT_SECONDS:
            retry_after = int(_RATE_LIMIT_SECONDS - (now - last_sent)) + 1
            return web.json_response(
                {"error": "Rate limited", "retry_after": retry_after},
                status=429,
            )

        # Look up adapter
        adapter = self.gateway.adapters.get(platform)
        if adapter is None:
            return web.json_response(
                {"error": f"Platform '{platform.value}' not configured or not running"},
                status=404,
            )

        # Send via adapter.send() — the standard BasePlatformAdapter interface
        try:
            metadata = {}
            if thread_id:
                metadata["thread_id"] = thread_id

            await adapter.send(
                chat_id,
                message,
                metadata=metadata or None,
            )
            _rate_limit_cache[rate_key] = now
            logger.info("notify: delivered to %s:%s", platform.value, chat_id)
            return web.json_response({
                "ok": True,
                "platform": platform.value,
                "chat_id": chat_id,
            })

        except Exception:
            logger.exception("notify: delivery failed to %s:%s", platform.value, chat_id)
            return web.json_response({"error": "Delivery failed"}, status=500)

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
            logger.info("Notify endpoint: http://%s:%d/notify", self.host, self.port)
        except OSError as e:
            logger.warning("Notify: failed to bind %s:%d: %s", self.host, self.port, e)

    async def stop(self):
        """Stop the notify server."""
        if self._runner:
            await self._runner.cleanup()
