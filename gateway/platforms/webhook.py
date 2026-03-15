"""Generic webhook inbound platform adapter.

Runs a lightweight HTTP server that accepts POST /message requests and
routes them through the gateway as regular conversations.  Each unique
``chat_id`` in the request gets its own session — supporting multiple
concurrent agents/conversations.

The response is returned synchronously in the HTTP response body (the
connection is held open until the agent finishes).  This makes it
trivially easy for external bridges, automation tools, or other agent
frameworks to integrate with Hermes.

Enable via env var:
    WEBHOOK_PORT=4568  (any port number enables the adapter)

API:
    POST /message
    {
        "chat_id": "hermes-1",        // required — maps to a session
        "message": "Hello!",          // required — the message text
        "from":    "other-agent",     // optional — sender display name
        "user_id": "agent-123"        // optional — sender ID
    }

    Response (200):
    {
        "ok": true,
        "response": "Hi there!",
        "session_id": "20260312_..."
    }

    GET /health
    {"ok": true, "adapter": "webhook", "port": 4568}
"""

import asyncio
import json
import logging
import os
import time

from aiohttp import web

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
    SessionSource,
)

logger = logging.getLogger(__name__)


def check_webhook_requirements() -> bool:
    """Webhook adapter is available when WEBHOOK_PORT is set."""
    return bool(os.getenv("WEBHOOK_PORT"))


class WebhookAdapter(BasePlatformAdapter):
    """HTTP webhook adapter — accepts POST requests as inbound messages.

    External services (bridges, automation tools, other agents) POST
    messages and receive the agent's response in the HTTP response body.
    Each chat_id maps to a separate gateway session.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        self.port = int(os.getenv("WEBHOOK_PORT", "4568"))
        self._app: web.Application = None
        self._runner: web.AppRunner = None
        self._site: web.TCPSite = None
        # Pending response futures keyed by session_key
        self._response_futures: dict[str, asyncio.Future] = {}

    async def connect(self) -> bool:
        """Start the HTTP server."""
        self._app = web.Application()
        self._app.router.add_post("/message", self._handle_post)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()

        try:
            self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
            await self._site.start()
        except OSError as e:
            logger.error("Webhook adapter failed to bind port %d: %s", self.port, e)
            return False

        print(f"[webhook] Listening on port {self.port}")
        print(f"[webhook]   POST http://localhost:{self.port}/message")
        return True

    async def disconnect(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    async def send(self, chat_id: str, content: str,
                   reply_to: str = None, metadata: dict = None) -> SendResult:
        """Deliver response by resolving the waiting HTTP request's future."""
        from gateway.session import build_session_key

        # Look up the pending future for this chat_id.
        # We try the chat_id directly, then the full session key.
        fut = self._response_futures.get(chat_id)
        if fut is None:
            # Try building the session key the same way handle_message does
            source = self.build_source(chat_id=chat_id, chat_type="dm")
            sk = build_session_key(source)
            fut = self._response_futures.get(sk)

        if fut and not fut.done():
            fut.set_result(content)

        return SendResult(success=True, message_id=str(int(time.time())))

    async def send_typing(self, chat_id: str, metadata: dict = None) -> None:
        pass  # No typing indicator for webhooks

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"id": chat_id, "name": f"webhook:{chat_id}", "type": "dm"}

    # ── HTTP Handlers ────────────────────────────────────────────────────

    async def _handle_post(self, request: web.Request) -> web.Response:
        """Accept an inbound message and return the agent's response."""
        try:
            data = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"ok": False, "error": "Invalid JSON"}, status=400
            )

        chat_id = data.get("chat_id", "").strip()
        message = data.get("message", "").strip()

        if not chat_id or not message:
            return web.json_response(
                {"ok": False, "error": "Missing required fields: chat_id, message"},
                status=400,
            )

        from_name = data.get("from", "webhook")
        user_id = data.get("user_id", from_name)

        # Prepend sender info if provided
        display_message = message
        if from_name and from_name != "webhook":
            display_message = f"[Message from {from_name}]: {message}"

        # Build source and event
        source = self.build_source(
            chat_id=chat_id,
            chat_type="dm",
            user_id=user_id,
            user_name=from_name,
        )

        from gateway.session import build_session_key
        session_key = build_session_key(source)

        event = MessageEvent(
            text=display_message,
            source=source,
            message_id=str(int(time.time() * 1000)),
        )

        # Create a future to capture the response from send()
        loop = asyncio.get_event_loop()
        response_future = loop.create_future()
        self._response_futures[session_key] = response_future
        # Also store under chat_id for easier lookup
        self._response_futures[chat_id] = response_future

        # Submit the message for processing
        await self.handle_message(event)

        # Wait for the agent to finish and send() to resolve the future
        try:
            response_text = await asyncio.wait_for(response_future, timeout=300)
            return web.json_response({
                "ok": True,
                "response": response_text,
                "chat_id": chat_id,
            })
        except asyncio.TimeoutError:
            return web.json_response(
                {"ok": False, "error": "Agent timed out (300s)", "chat_id": chat_id},
                status=504,
            )
        finally:
            self._response_futures.pop(session_key, None)
            self._response_futures.pop(chat_id, None)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "adapter": "webhook",
            "port": self.port,
        })
