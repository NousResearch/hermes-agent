"""NotionAgent HTTP platform adapter.

Receives signed JSON messages from the NotionAgent web app and posts signed
text replies back to the app's callback endpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from agent.redact import redact_sensitive_text
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8645
DEFAULT_PATH = "/notionagent/in"
SIGNATURE_HEADER = "X-NotionAgent-Signature"


def check_notionagent_requirements() -> bool:
    """Check if NotionAgent adapter dependencies are available."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def _signature_for_body(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def _validate_notionagent_signature(signature: str, body: bytes, secret: str) -> bool:
    if not signature or not signature.startswith("sha256="):
        return False
    expected = _signature_for_body(body, secret)
    return hmac.compare_digest(signature, expected)


def _signed_json_payload(payload: Dict[str, Any], secret: str) -> tuple[bytes, str]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return body, _signature_for_body(body, secret)


def _redact(text: Any) -> str:
    return redact_sensitive_text(str(text))


class NotionAgentAdapter(BasePlatformAdapter):
    """HTTP bridge for the NotionAgent web app."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NOTIONAGENT)
        extra = config.extra or {}
        self._host = str(extra.get("host") or DEFAULT_HOST)
        self._port = int(extra.get("port") or DEFAULT_PORT)
        self._path = str(extra.get("path") or DEFAULT_PATH)
        if not self._path.startswith("/"):
            self._path = f"/{self._path}"
        self._secret = str(extra.get("secret") or config.token or "")
        self._callback_url = str(extra.get("callback_url") or "")
        self._runner = None

    async def connect(self) -> bool:
        if not self._secret:
            logger.error("[notionagent] Missing NOTIONAGENT_SECRET")
            return False
        if not self._callback_url:
            logger.error("[notionagent] Missing NOTIONAGENT_CALLBACK_URL")
            return False

        app = self._create_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                site = web.TCPSite(self._runner, self._host, self._port)
                await site.start()
                self._mark_connected()
                logger.info(
                    "[notionagent] Listening on %s:%d%s",
                    self._host,
                    self._port,
                    self._path,
                )
                return True
            except OSError as exc:
                if attempt >= max_attempts - 1:
                    logger.error(
                        "[notionagent] Failed to bind %s:%d after %d attempts: %s",
                        self._host,
                        self._port,
                        max_attempts,
                        _redact(exc),
                    )
                    await self._runner.cleanup()
                    self._runner = None
                    return False
                delay = min(2 ** attempt, 8) + (0.1 * attempt)
                logger.warning(
                    "[notionagent] Port bind failed on %s:%d, retrying in %.1fs: %s",
                    self._host,
                    self._port,
                    delay,
                    _redact(exc),
                )
                await asyncio.sleep(delay)

        return False

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[notionagent] Disconnected")

    def _create_app(self) -> "web.Application":
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_inbound)
        return app

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "notionagent"})

    async def _handle_inbound(self, request: "web.Request") -> "web.Response":
        try:
            raw_body = await request.read()
        except Exception as exc:
            logger.warning("[notionagent] Failed to read request body: %s", _redact(exc))
            return web.json_response({"error": "Bad request"}, status=400)

        signature = request.headers.get(SIGNATURE_HEADER, "")
        if not _validate_notionagent_signature(signature, raw_body, self._secret):
            logger.warning("[notionagent] Invalid signature")
            return web.json_response({"error": "Invalid signature"}, status=401)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        session_id = str(payload.get("session_id") or "").strip()
        text = str(payload.get("text") or "")
        if not session_id or not text:
            return web.json_response(
                {"error": "session_id and text are required"},
                status=400,
            )

        message_id = str(payload.get("message_id") or uuid.uuid4())
        source = self.build_source(
            chat_id=session_id,
            chat_name=session_id,
            chat_type="session",
            user_id=session_id,
            user_name=session_id,
            message_id=message_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=message_id,
        )

        logger.info(
            "[notionagent] Accepted message session=%s message_id=%s text_len=%d",
            session_id,
            message_id,
            len(text),
        )
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {"status": "accepted", "session_id": session_id, "message_id": message_id},
            status=202,
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._callback_url:
            return SendResult(success=False, error="NOTIONAGENT_CALLBACK_URL is not configured")
        return await _post_notionagent_callback(
            callback_url=self._callback_url,
            secret=self._secret,
            session_id=str(chat_id),
            text=content,
        )

    async def send_typing(self, chat_id: str) -> SendResult:
        return SendResult(success=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        return SendResult(success=False, error="not supported")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "session", "chat_id": chat_id}


async def _post_notionagent_callback(
    *,
    callback_url: str,
    secret: str,
    session_id: str,
    text: str,
) -> SendResult:
    if not HTTPX_AVAILABLE:
        return SendResult(
            success=False,
            error="httpx is not installed",
            retryable=False,
        )

    payload = {
        "session_id": str(session_id),
        "text": text,
        "message_id": str(uuid.uuid4()),
    }
    body, signature = _signed_json_payload(payload, secret)
    headers = {
        "Content-Type": "application/json",
        SIGNATURE_HEADER: signature,
    }
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, content=body, headers=headers)
        if response.status_code >= 400:
            return SendResult(
                success=False,
                error=f"callback returned HTTP {response.status_code}",
                raw_response=getattr(response, "text", ""),
                retryable=response.status_code >= 500,
            )
        return SendResult(
            success=True,
            message_id=payload["message_id"],
            raw_response={"status_code": response.status_code, "elapsed": time.monotonic() - started},
        )
    except Exception as exc:
        return SendResult(
            success=False,
            error=f"NotionAgent callback failed: {_redact(exc)}",
            retryable=True,
        )
