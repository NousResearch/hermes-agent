"""Generic webhook inbound platform adapter."""

import asyncio
import json
import logging
import os
import time

from aiohttp import web

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import build_session_key

logger = logging.getLogger(__name__)


def check_webhook_requirements() -> bool:
    """Webhook adapter is available when WEBHOOK_PORT is configured."""
    return bool(os.getenv("WEBHOOK_PORT"))


class WebhookAdapter(BasePlatformAdapter):
    """HTTP webhook adapter with chat_id-stable gateway sessions."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WEBHOOK)
        port = config.extra.get("port") if config and config.extra else None
        self.port = int(port or os.getenv("WEBHOOK_PORT", "4568"))
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        # Response state is keyed only by session_key to avoid chat/session collisions.
        self._response_accumulators: dict[str, list[str]] = {}
        self._response_events: dict[str, asyncio.Event] = {}
        self._webhook_secret = os.getenv("WEBHOOK_SECRET", "").strip()

    async def connect(self) -> bool:
        self._app = web.Application()
        self._app.router.add_post("/message", self._handle_post)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()

        try:
            self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
            await self._site.start()
        except OSError as exc:
            logger.error("Webhook adapter failed to bind port %d: %s", self.port, exc)
            return False

        logger.info("[webhook] Listening on port %d", self.port)
        logger.info("[webhook] POST http://localhost:%d/message", self.port)
        return True

    async def disconnect(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    async def send(self, chat_id: str, content: str, reply_to: str = None, metadata: dict = None) -> SendResult:
        source = self.build_source(chat_id=chat_id, chat_type="dm")
        session_key = build_session_key(source)
        accumulator = self._response_accumulators.get(session_key)
        event = self._response_events.get(session_key)
        if accumulator is not None:
            accumulator.append(content)
            if event is not None:
                event.set()
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata: dict = None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"id": chat_id, "name": f"webhook:{chat_id}", "type": "dm"}

    async def _handle_post(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        if self._webhook_secret:
            headers = getattr(request, "headers", {}) or {}
            header_secret = headers.get("X-Webhook-Secret", "")
            if header_secret != self._webhook_secret:
                return web.json_response({"ok": False, "error": "Unauthorized"}, status=401)

        chat_id = str(data.get("chat_id", "")).strip()
        message = str(data.get("message", "")).strip()
        if not chat_id or not message:
            return web.json_response(
                {"ok": False, "error": "Missing required fields: chat_id, message"},
                status=400,
            )

        from_name = str(data.get("from", "webhook") or "webhook")
        user_id = str(data.get("user_id", from_name) or from_name)
        display_message = message
        if from_name and from_name != "webhook":
            display_message = f"[Message from {from_name}]: {message}"

        source = self.build_source(
            chat_id=chat_id,
            chat_type="dm",
            user_id=user_id,
            user_name=from_name,
        )
        session_key = build_session_key(source)
        event = MessageEvent(
            text=display_message,
            source=source,
            message_id=str(int(time.time() * 1000)),
        )

        responses: list[str] = []
        done_event = asyncio.Event()
        self._response_accumulators[session_key] = responses
        self._response_events[session_key] = done_event

        try:
            await self.handle_message(event)

            deadline = asyncio.get_running_loop().time() + 300
            settled_since = None
            last_response_count = 0
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(done_event.wait(), timeout=min(remaining, 0.5))
                except asyncio.TimeoutError:
                    pass
                if len(responses) != last_response_count:
                    last_response_count = len(responses)
                    settled_since = asyncio.get_running_loop().time()
                    done_event.clear()
                    continue
                if session_key not in self._active_sessions and responses:
                    await asyncio.sleep(0.05)
                    break
                if responses and settled_since is not None and (asyncio.get_running_loop().time() - settled_since) >= 0.15:
                    break

            if responses:
                return web.json_response({"ok": True, "response": responses[-1], "chat_id": chat_id})
            return web.json_response(
                {"ok": False, "error": "Agent timed out (300s)", "chat_id": chat_id},
                status=504,
            )
        finally:
            self._response_accumulators.pop(session_key, None)
            self._response_events.pop(session_key, None)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "adapter": "webhook", "port": self.port})
