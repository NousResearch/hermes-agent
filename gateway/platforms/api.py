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
        self._media_files: Dict[str, str] = {}  # token/filename -> file_path

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

        # Show accessible URLs
        if self._host in ("0.0.0.0", "::"):
            ips = self._get_local_ips()
            primary = self._get_primary_ip(ips)
            print(f"[{self.name}] API + Web UI: http://{primary}:{self._port}")
            for ip in ips:
                if ip != primary:
                    print(f"[{self.name}]       also: http://{ip}:{self._port}")
        else:
            print(f"[{self.name}] API + Web UI: http://{self._host}:{self._port}")
            if self._host == "127.0.0.1":
                print(f"[{self.name}]   Set API_HOST=0.0.0.0 for phone/tablet access")
        logger.info("API server starting on %s:%d", self._host, self._port)
        return True

    @staticmethod
    def _get_local_ips():
        """Get all non-loopback IPv4 addresses."""
        ips = []
        try:
            import subprocess
            out = subprocess.check_output(["ifconfig"], text=True, timeout=5, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("inet ") and "127.0.0.1" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        ips.append(parts[1])
        except Exception:
            pass
        return ips

    @staticmethod
    def _get_primary_ip(ips):
        """Prefer LAN IPs (192.168.x / 10.x) over VPN ranges."""
        for ip in ips:
            if ip.startswith("192.168.") or ip.startswith("10."):
                return ip
        return ips[0] if ips else "127.0.0.1"

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

    _MEDIA_DIR = os.path.join(
        os.getenv("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes")),
        "api_media",
    )
    _MEDIA_TTL_SECONDS = 3600  # 1 hour

    def _cleanup_old_media(self) -> None:
        """Remove media files older than TTL and evict stale registry entries."""
        import time as _time
        now = _time.time()
        # Clean files from both media and upload dirs
        for dir_path in [self._MEDIA_DIR]:
            if not os.path.isdir(dir_path):
                continue
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                try:
                    if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > self._MEDIA_TTL_SECONDS:
                        os.unlink(fpath)
                except OSError:
                    pass
        # Evict stale entries from registry
        stale_keys = [k for k, v in self._media_files.items() if not os.path.isfile(v)]
        for k in stale_keys:
            del self._media_files[k]

    def _register_media(self, file_path: str) -> str:
        """Copy media file to a persistent directory and return its download URL.

        The original file may be deleted by the caller (e.g. auto-TTS cleanup),
        so we keep our own copy in ~/.hermes/api_media/.
        Runs TTL cleanup on each call to prevent unbounded growth.
        """
        import shutil
        from gateway.api_server import _sign_media_path, _make_media_url

        os.makedirs(self._MEDIA_DIR, exist_ok=True)
        self._cleanup_old_media()
        filename = os.path.basename(file_path)
        dest = os.path.join(self._MEDIA_DIR, filename)

        if os.path.isfile(file_path) and os.path.abspath(file_path) != os.path.abspath(dest):
            shutil.copy2(file_path, dest)

        token = _sign_media_path(dest)
        self._media_files[f"{token}/{filename}"] = dest
        return _make_media_url(dest)

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
        """Push audio with download URL to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            url = self._register_media(audio_path)
            await queue.put({
                "type": "audio",
                "url": url,
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
        """Push video with download URL to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            url = self._register_media(video_path)
            await queue.put({
                "type": "video",
                "url": url,
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
        """Push image file with download URL to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            url = self._register_media(image_path)
            await queue.put({
                "type": "image",
                "url": url,
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
        """Push document with download URL to response queue."""
        queue = self._get_queue(chat_id)
        if queue:
            url = self._register_media(file_path)
            await queue.put({
                "type": "document",
                "url": url,
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
                await queue.put({"type": "error", "content": "An internal error occurred. Check server logs."})
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
