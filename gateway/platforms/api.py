"""
REST/WebSocket API platform adapter.

Exposes Hermes Agent over HTTP (POST /v1/chat) and WebSocket (/v1/chat/stream)
so any frontend, mobile app, or third-party service can interact with it.

Follows the same BasePlatformAdapter pattern as Telegram, Discord, etc.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

import uvicorn

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)
from gateway.session import SessionSource, build_session_key

logger = logging.getLogger(__name__)


def check_api_requirements() -> bool:
    """Check if FastAPI and uvicorn are available."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


class APIPlatformAdapter(BasePlatformAdapter):
    """HTTP/WebSocket transport for Hermes Agent."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API)
        self._server: Optional[uvicorn.Server] = None
        self._host = os.getenv("API_HOST", "127.0.0.1")
        self._port = int(os.getenv("API_PORT", "8765"))
        self._response_queues: Dict[str, asyncio.Queue] = {}

    async def connect(self) -> bool:
        """Start the FastAPI/uvicorn server in background."""
        from gateway.api_server import create_app

        app = create_app(self)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        asyncio.create_task(self._server.serve())
        self._running = True
        logger.info("API server starting on %s:%d", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        """Shutdown the server."""
        if self._server:
            self._server.should_exit = True
        self._running = False

    # ── Queue helpers ────────────────────────────────────────────────────

    @staticmethod
    def _build_session_key(chat_id: str) -> str:
        """Build session key for an API session.

        Uses chat_type="channel" so build_session_key includes chat_id
        in the key (DMs without chat_id would collapse all sessions into one).
        """
        return build_session_key(SessionSource(
            platform=Platform.API,
            chat_id=chat_id,
            user_id=chat_id,
            chat_type="channel",
        ))

    def _get_queue(self, chat_id: str) -> Optional[asyncio.Queue]:
        """Find the response queue for this chat/session."""
        session_key = self._build_session_key(chat_id)
        return self._response_queues.get(session_key)

    def register_queue(self, chat_id: str) -> asyncio.Queue:
        """Create and register a response queue for a session."""
        session_key = self._build_session_key(chat_id)
        queue = asyncio.Queue()
        self._response_queues[session_key] = queue
        return queue

    def unregister_queue(self, chat_id: str) -> None:
        """Remove the response queue for a session."""
        session_key = self._build_session_key(chat_id)
        self._response_queues.pop(session_key, None)

    # ── Send methods (route to response queue) ───────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Push text response to the waiting HTTP/WS client."""
        queue = self._get_queue(chat_id)
        if queue:
            await queue.put({"type": "message", "content": content})
        return SendResult(success=True, message_id=str(uuid4()))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Push image to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            await queue.put({
                "type": "image",
                "url": image_url,
                "caption": caption,
            })
        return SendResult(success=True, message_id=str(uuid4()))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Push audio file path to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            await queue.put({
                "type": "audio",
                "path": audio_path,
                "caption": caption,
            })
        return SendResult(success=True, message_id=str(uuid4()))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Push video file path to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            await queue.put({
                "type": "video",
                "path": video_path,
                "caption": caption,
            })
        return SendResult(success=True, message_id=str(uuid4()))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Push image file path to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            await queue.put({
                "type": "image",
                "path": image_path,
                "caption": caption,
            })
        return SendResult(success=True, message_id=str(uuid4()))

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Push document file path to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            await queue.put({
                "type": "document",
                "path": file_path,
                "caption": caption,
            })
        return SendResult(success=True, message_id=str(uuid4()))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info for an API session."""
        return {"name": f"api-{chat_id}", "type": "channel"}

    # ── Override background processing to signal "done" ──────────────────

    async def _process_message_background(
        self, event: MessageEvent, session_key: str
    ) -> None:
        """Override to signal completion to waiting HTTP/WS clients."""
        try:
            await super()._process_message_background(event, session_key)
        except Exception as e:
            logger.error("API message processing error: %s", e)
            queue = self._response_queues.get(session_key)
            if queue:
                await queue.put({"type": "error", "content": str(e)})
        finally:
            queue = self._response_queues.get(session_key)
            if queue:
                await queue.put({"type": "done"})

    # ── Public method for FastAPI routes ──────────────────────────────────

    async def handle_request(
        self, chat_id: str, text: str, user_id: Optional[str] = None
    ) -> None:
        """Called by FastAPI route. Creates MessageEvent and dispatches.

        The caller must register a queue BEFORE calling this and then
        await the queue for response chunks until {"type": "done"}.
        """
        event = MessageEvent(
            text=text,
            source=SessionSource(
                platform=Platform.API,
                chat_id=chat_id,
                user_id=user_id or chat_id,
                chat_type="channel",
            ),
            message_id=str(uuid4()),
        )
        await self.handle_message(event)
