"""Webex gateway adapter.

Uses the Webex Messaging REST API for outbound sends plus either:
- WebSocket events via the official Webex JavaScript SDK (default)
- Webhook callbacks (optional fallback mode)

Environment variables:
    WEBEX_BOT_TOKEN            Bot token from developer.webex.com
    WEBEX_CONNECTION_MODE      websocket (default) or webhook
    WEBEX_ALLOWED_USERS        Comma-separated user emails or IDs
    WEBEX_HOME_CHANNEL         Room ID for cron/notification delivery
    WEBEX_WEBHOOK_PUBLIC_URL   Public HTTPS base URL for webhook mode
    WEBEX_WEBHOOK_SECRET       Optional webhook signing secret
    WEBEX_WEBHOOK_HOST         Local bind host (default: 0.0.0.0)
    WEBEX_WEBHOOK_PORT         Local bind port (default: 8646)
    WEBEX_WEBHOOK_PATH         Local callback path (default: /webex/webhook)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import socket as _socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

try:
    import aiohttp
    from aiohttp import FormData, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    FormData = Any  # type: ignore[misc,assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    safe_url_for_log,
)
from gateway.platforms.helpers import MessageDeduplicator
from hermes_cli.commands import resolve_command

logger = logging.getLogger(__name__)

API_BASE = "https://webexapis.com/v1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8646
DEFAULT_PATH = "/webex/webhook"
DEFAULT_WEBHOOK_PREFIX = "hermes-webex"
DEFAULT_CONNECTION_MODE = "websocket"
LISTENER_READY_TIMEOUT_SECONDS = 30
ROOM_CACHE_TTL_SECONDS = 300
THREAD_CONTEXT_CACHE_TTL_SECONDS = 60.0
THREAD_CONTEXT_DEFAULT_LIMIT = 30
THREAD_CONTEXT_MAX_CHARS = 6000


@dataclass
class _WebexThreadContextCache:
    """Cache entry for fetched Webex thread context."""

    content: str
    fetched_at: float = field(default_factory=time.monotonic)
    parent_text: str = ""


def check_webex_requirements() -> bool:
    """Return True when the adapter's Python dependencies are available."""
    return AIOHTTP_AVAILABLE


class WebexAdapter(BasePlatformAdapter):
    """Gateway adapter for Webex Messaging bots."""

    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("webex"))
        extra = config.extra or {}
        self._token: str = config.token or os.getenv("WEBEX_BOT_TOKEN", "")
        self._connection_mode: str = str(
            extra.get("connection_mode") or os.getenv("WEBEX_CONNECTION_MODE", DEFAULT_CONNECTION_MODE)
        ).strip().lower() or DEFAULT_CONNECTION_MODE
        self._host: str = str(extra.get("host") or os.getenv("WEBEX_WEBHOOK_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port") or os.getenv("WEBEX_WEBHOOK_PORT", DEFAULT_PORT))
        self._path: str = self._normalize_path(
            str(extra.get("path") or os.getenv("WEBEX_WEBHOOK_PATH", DEFAULT_PATH))
        )
        self._public_url: str = str(extra.get("public_url") or os.getenv("WEBEX_WEBHOOK_PUBLIC_URL", "")).rstrip("/")
        self._secret: str = str(extra.get("secret") or os.getenv("WEBEX_WEBHOOK_SECRET", ""))
        self._require_mention: bool = extra.get("require_mention", True)
        self._webhook_prefix: str = str(extra.get("webhook_prefix") or DEFAULT_WEBHOOK_PREFIX)

        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._app: Optional[web.Application] = None
        self._session: Any = None  # aiohttp.ClientSession
        self._poll_task: Optional[asyncio.Task] = None
        self._listener_process: Optional[asyncio.subprocess.Process] = None
        self._listener_stdout_task: Optional[asyncio.Task] = None
        self._listener_stderr_task: Optional[asyncio.Task] = None
        self._listener_exit_task: Optional[asyncio.Task] = None
        self._listener_ready: Optional[asyncio.Future] = None
        self._message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._dedup = MessageDeduplicator()
        self._bot_id: str = ""
        self._bot_email: str = ""
        self._bot_display_name: str = ""
        self._room_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._thread_context_cache: Dict[str, _WebexThreadContextCache] = {}
        self._managed_webhook_ids: List[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect
        if not AIOHTTP_AVAILABLE:
            logger.error("[Webex] aiohttp is not installed. Run: pip install aiohttp")
            return False
        if not self._token:
            logger.error("[Webex] WEBEX_BOT_TOKEN is not configured")
            return False
        if self._connection_mode not in {"websocket", "webhook"}:
            logger.error(
                "[Webex] Unsupported WEBEX_CONNECTION_MODE=%s. Supported modes: websocket, webhook.",
                self._connection_mode,
            )
            return False
        if self._connection_mode == "webhook" and not self._public_url:
            logger.error("[Webex] WEBEX_WEBHOOK_PUBLIC_URL is not configured for webhook mode")
            return False
        if self._connection_mode == "webhook" and not self._public_url.startswith("https://"):
            logger.error("[Webex] WEBEX_WEBHOOK_PUBLIC_URL must be an HTTPS URL")
            return False
        if self._connection_mode == "webhook" and not self._path.startswith("/"):
            logger.error("[Webex] Invalid webhook path %r", self._path)
            return False
        if not self._acquire_platform_lock("webex-bot-token", self._token, "Webex bot token"):
            return False

        if self._connection_mode == "webhook":
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    sock.connect(("127.0.0.1", self._port))
                logger.error("[Webex] Port %d already in use", self._port)
                self._release_platform_lock()
                return False
            except (ConnectionRefusedError, OSError):
                pass

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            await self._load_bot_identity()

            if self._connection_mode == "webhook":
                self._app = web.Application()
                self._app.router.add_get("/health", self._handle_health)
                self._app.router.add_post(self._path, self._handle_webhook)
                self._runner = web.AppRunner(self._app)
                await self._runner.setup()
                self._site = web.TCPSite(self._runner, self._host, self._port)
                await self._site.start()
                await self._ensure_managed_webhooks()

                if not self._secret:
                    logger.warning(
                        "[Webex] WEBEX_WEBHOOK_SECRET is empty; inbound webhooks will not be signature-verified"
                    )
            else:
                await self._start_websocket_listener()

            self._poll_task = asyncio.create_task(self._poll_loop())
            self._mark_connected()
            if self._connection_mode == "webhook":
                logger.info(
                    "[Webex] Listening on %s:%s%s and delivering via %s",
                    self._host,
                    self._port,
                    self._path,
                    safe_url_for_log(self._target_url),
                )
            else:
                logger.info("[Webex] Connected in websocket mode via local SDK listener")
            return True
        except Exception:
            logger.exception("[Webex] Failed to start")
            await self._cleanup()
            self._release_platform_lock()
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        await self._cleanup()
        self._release_platform_lock()
        self._mark_disconnected()
        logger.info("[Webex] Disconnected")

    async def _cleanup(self) -> None:
        await self._stop_websocket_listener()
        self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Outbound messaging
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        message = self.format_message(content).strip()
        if not message:
            return SendResult(success=True)

        payload = self._build_text_payload(chat_id, message, metadata=metadata)
        if payload is None:
            return SendResult(success=False, error="Missing room target")

        data = await self._api_post_json("messages", payload)
        if not data:
            return SendResult(success=False, error="Webex send returned no data")
        if data.get("id"):
            return SendResult(success=True, message_id=str(data["id"]), raw_response=data)
        return SendResult(success=False, error=json.dumps(data))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        message = self.format_message(content).strip()
        if not message:
            return SendResult(success=True, message_id=message_id)

        room_id = await self._resolve_edit_room_id(chat_id, message_id)
        if not room_id:
            return SendResult(success=False, error="Missing room target for Webex edit")

        data = await self._api_put_json(
            f"messages/{message_id}",
            {
                "roomId": room_id,
                "markdown": message,
            },
        )
        if not data:
            return SendResult(success=False, error="Webex edit returned no data")
        if data.get("id"):
            return SendResult(success=True, message_id=str(data["id"]), raw_response=data)
        return SendResult(success=False, error=json.dumps(data))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        payload = self._build_text_payload(chat_id, caption or "", metadata=metadata)
        if payload is None:
            return SendResult(success=False, error="Missing room target")
        payload["files"] = [image_url]
        data = await self._api_post_json("messages", payload)
        if data and data.get("id"):
            return SendResult(success=True, message_id=str(data["id"]), raw_response=data)
        # Fall back to a plain text URL if the API rejects file-url delivery.
        text = f"{caption}\n{image_url}".strip() if caption else image_url
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, image_path, caption=caption, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(
            chat_id,
            file_path,
            caption=caption,
            metadata=metadata,
            file_name=file_name,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, audio_path, caption=caption, metadata=metadata)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, video_path, caption=caption, metadata=metadata)

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        path = Path(file_path)
        if not path.exists():
            return SendResult(success=False, error=f"File not found: {file_path}")

        form = FormData()
        self._populate_target_fields(form, chat_id)
        parent_id = self._extract_parent_id(metadata)
        if parent_id:
            form.add_field("parentId", parent_id)
        if caption:
            form.add_field("markdown", self.format_message(caption))

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        form.add_field(
            "files",
            path.read_bytes(),
            filename=file_name or path.name,
            content_type=content_type,
        )

        data = await self._api_post_form("messages", form)
        if data and data.get("id"):
            return SendResult(success=True, message_id=str(data["id"]), raw_response=data)
        return SendResult(success=False, error=json.dumps(data or {"error": "upload failed"}))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        room = await self._get_room(chat_id)
        room_type = room.get("type", "group")
        return {
            "name": room.get("title") or chat_id,
            "type": "dm" if room_type == "direct" else room_type,
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # Inbound transport processing
    # ------------------------------------------------------------------

    async def _start_websocket_listener(self) -> None:
        script_path = Path(__file__).with_name("webex_listener.js")
        if not script_path.exists():
            raise RuntimeError(f"Missing Webex listener script: {script_path}")

        env = os.environ.copy()
        env["WEBEX_BOT_TOKEN"] = self._token
        try:
            process = await asyncio.create_subprocess_exec(
                "node",
                str(script_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(script_path.parents[2]),
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Node.js is required for Webex websocket mode") from exc
        self._listener_process = process
        self._listener_ready = asyncio.get_running_loop().create_future()
        self._listener_stdout_task = asyncio.create_task(self._listener_stdout_loop(process))
        self._listener_stderr_task = asyncio.create_task(self._listener_stderr_loop(process))
        self._listener_exit_task = asyncio.create_task(self._listener_exit_loop(process))
        await asyncio.wait_for(self._listener_ready, timeout=LISTENER_READY_TIMEOUT_SECONDS)
        if process.returncode is not None:
            raise RuntimeError(f"Webex listener exited during startup (code {process.returncode})")

    async def _stop_websocket_listener(self) -> None:
        process = self._listener_process
        self._listener_process = None

        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        for attr in ("_listener_stdout_task", "_listener_stderr_task", "_listener_exit_task"):
            task = getattr(self, attr)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            setattr(self, attr, None)

        if self._listener_ready and not self._listener_ready.done():
            self._listener_ready.cancel()
        self._listener_ready = None

    async def _listener_stdout_loop(self, process: asyncio.subprocess.Process) -> None:
        stream = process.stdout
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("[Webex] Listener stdout: %s", raw)
                continue
            if not isinstance(payload, dict):
                logger.debug("[Webex] Ignoring non-object listener payload: %r", payload)
                continue
            await self._handle_listener_message(payload)

    async def _listener_stderr_loop(self, process: asyncio.subprocess.Process) -> None:
        stream = process.stderr
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if raw:
                logger.warning("[Webex] Listener stderr: %s", raw)

    async def _listener_exit_loop(self, process: asyncio.subprocess.Process) -> None:
        returncode = await process.wait()
        if self._listener_ready and not self._listener_ready.done():
            self._listener_ready.set_exception(
                RuntimeError(f"Webex listener exited before startup completed (code {returncode})")
            )
            return
        if not self._running:
            return
        message = f"Webex websocket listener exited unexpectedly (code {returncode})."
        logger.error("[Webex] %s", message)
        if not self.has_fatal_error:
            self._set_fatal_error("webex_listener_exited", message, retryable=True)
            await self._notify_fatal_error()

    async def _handle_listener_message(self, payload: Dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "").strip().lower()
        if msg_type == "ready":
            if self._listener_ready and not self._listener_ready.done():
                self._listener_ready.set_result(payload)
            return

        if msg_type == "log":
            level = str(payload.get("level") or "info").lower()
            message = str(payload.get("message") or "").strip()
            if not message:
                return
            log_fn = {
                "debug": logger.debug,
                "warning": logger.warning,
                "error": logger.error,
            }.get(level, logger.info)
            log_fn("[Webex] %s", message)
            return

        if msg_type == "fatal":
            message = str(payload.get("message") or "Webex listener failed").strip()
            if self._listener_ready and not self._listener_ready.done():
                self._listener_ready.set_exception(RuntimeError(message))
                return
            if not self.has_fatal_error:
                self._set_fatal_error("webex_listener_fatal", message, retryable=True)
                await self._notify_fatal_error()
            logger.error("[Webex] %s", message)
            return

        if msg_type != "event":
            return

        event = payload.get("event")
        if not isinstance(event, dict):
            return
        if event.get("resource") != "messages" or event.get("event") != "created":
            return
        try:
            self._message_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("[Webex] Dropping websocket event because the queue is full")

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "platform": "webex"})

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        body = await request.read()
        if self._secret and not self._verify_signature(request.headers, body):
            return web.Response(status=403, text="invalid signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")

        if payload.get("resource") != "messages" or payload.get("event") != "created":
            return web.Response(status=200, text="ignored")

        try:
            self._message_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("[Webex] Dropping webhook event because the queue is full")
            return web.Response(status=503, text="queue full")
        return web.Response(status=200, text="ok")

    async def _poll_loop(self) -> None:
        while True:
            payload = await self._message_queue.get()
            try:
                event = await self._build_event(payload)
                if event is None:
                    continue
                task = asyncio.create_task(self.handle_message(event))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                logger.exception("[Webex] Failed to process inbound payload")

    async def _build_event(self, payload: Dict[str, Any]) -> Optional[MessageEvent]:
        data = payload.get("data") or {}
        message_id = str(data.get("id") or "")
        if not message_id:
            return None

        message = self._coerce_event_message(data)
        if not message or self._needs_full_message_fetch(message):
            fetched_message = await self._api_get_json(f"messages/{message_id}")
            if fetched_message:
                message = {**message, **fetched_message}
            elif not message:
                return None
        if self._dedup.is_duplicate(message_id):
            return None

        sender_person_id = str(message.get("personId") or "")
        if sender_person_id and sender_person_id == self._bot_id:
            return None

        room_id = str(message.get("roomId") or data.get("roomId") or "")
        if not room_id:
            return None
        room = await self._get_room(room_id)
        room_type = str(room.get("type") or data.get("roomType") or "group")
        chat_type = "dm" if room_type == "direct" else "group"

        sender_email = str(message.get("personEmail") or data.get("personEmail") or "")
        sender_name = sender_email
        sender_display_name = await self._lookup_display_name(sender_person_id)
        if sender_display_name:
            sender_name = sender_display_name

        parent_id = str(message.get("parentId") or "") or None
        raw_text = str(message.get("text") or message.get("markdown") or "").strip()
        text = raw_text
        if room_type != "direct":
            text = self._strip_bot_mention(text)
            if (
                self._require_mention
                and not self._group_message_mentions_bot(message)
                and not self._is_gateway_command(text)
            ):
                return None

        media_urls, media_types = await self._download_attachments(message.get("files") or [])
        msg_type = self._detect_message_type(text, media_types)

        source = self.build_source(
            chat_id=room_id,
            chat_name=room.get("title") or room_id,
            chat_type=chat_type,
            user_id=sender_email or sender_person_id,
            user_name=sender_name,
            thread_id=parent_id,
            user_id_alt=sender_person_id or None,
        )

        reply_to_text = None
        if parent_id:
            if msg_type != MessageType.COMMAND and not self._has_active_session_for_thread(
                room_id=room_id,
                chat_type=chat_type,
                parent_id=parent_id,
                user_id=sender_email or sender_person_id,
                user_id_alt=sender_person_id or None,
            ):
                thread_context = await self._fetch_thread_context(
                    room_id=room_id,
                    parent_id=parent_id,
                    current_message_id=message_id,
                )
                if thread_context:
                    text = thread_context + text
            reply_to_text = await self._fetch_thread_parent_text(room_id, parent_id) or None

        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=parent_id,
            reply_to_text=reply_to_text,
        )

    @staticmethod
    def _coerce_event_message(data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        if not data.get("id") or not data.get("roomId") or not data.get("personId"):
            return {}
        return dict(data)

    @staticmethod
    def _needs_full_message_fetch(message: Dict[str, Any]) -> bool:
        return not bool(
            message.get("text")
            or message.get("markdown")
            or message.get("files")
        )

    # ------------------------------------------------------------------
    # Webex API helpers
    # ------------------------------------------------------------------

    async def _load_bot_identity(self) -> None:
        me = await self._api_get_json("people/me")
        if not me or not me.get("id"):
            raise RuntimeError("Failed to load Webex bot identity from /people/me")
        self._bot_id = str(me.get("id") or "")
        emails = me.get("emails") or []
        self._bot_email = str(emails[0]) if emails else ""
        self._bot_display_name = str(me.get("displayName") or self._bot_email or self._bot_id)

    async def _get_room(self, room_id: str) -> Dict[str, Any]:
        cached = self._room_cache.get(room_id)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1]
        room = await self._api_get_json(f"rooms/{room_id}")
        if room:
            self._room_cache[room_id] = (now + ROOM_CACHE_TTL_SECONDS, room)
        return room or {}

    async def _lookup_display_name(self, person_id: str) -> str:
        if not person_id:
            return ""
        person = await self._api_get_json(f"people/{person_id}")
        return str(person.get("displayName") or "")

    async def _api_get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._session:
            return {}
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.get(url, headers=self._headers(), params=params) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("[Webex] GET %s -> %s: %s", path, resp.status, text[:300])
                    return {}
                return json.loads(text) if text else {}
        except Exception as exc:
            logger.warning("[Webex] GET %s failed: %s", path, exc)
            return {}

    async def _api_post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session:
            return {}
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.post(url, headers=self._json_headers(), json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("[Webex] POST %s -> %s: %s", path, resp.status, text[:300])
                    return {}
                return json.loads(text) if text else {}
        except Exception as exc:
            logger.warning("[Webex] POST %s failed: %s", path, exc)
            return {}

    async def _api_post_form(self, path: str, form: FormData) -> Dict[str, Any]:
        if not self._session:
            return {}
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.post(url, headers=self._headers(), data=form) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("[Webex] POST %s -> %s: %s", path, resp.status, text[:300])
                    return {}
                return json.loads(text) if text else {}
        except Exception as exc:
            logger.warning("[Webex] POST %s failed: %s", path, exc)
            return {}

    async def _api_put_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session:
            return {}
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.put(url, headers=self._json_headers(), json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("[Webex] PUT %s -> %s: %s", path, resp.status, text[:300])
                    return {}
                return json.loads(text) if text else {}
        except Exception as exc:
            logger.warning("[Webex] PUT %s failed: %s", path, exc)
            return {}

    async def _api_delete(self, path: str) -> bool:
        if not self._session:
            return False
        url = f"{API_BASE}/{path.lstrip('/')}"
        try:
            async with self._session.delete(url, headers=self._headers()) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("[Webex] DELETE %s -> %s: %s", path, resp.status, body[:300])
                    return False
                return True
        except Exception as exc:
            logger.warning("[Webex] DELETE %s failed: %s", path, exc)
            return False

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _json_headers(self) -> Dict[str, str]:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        return headers

    # ------------------------------------------------------------------
    # Managed webhook helpers
    # ------------------------------------------------------------------

    @property
    def _target_url(self) -> str:
        return f"{self._public_url}{self._path}"

    async def _ensure_managed_webhooks(self) -> None:
        existing = await self._api_get_json("webhooks")
        items = existing.get("items") if isinstance(existing, dict) else None
        items = items if isinstance(items, list) else []

        desired = []
        desired.append(
            {
                "name": f"{self._webhook_prefix}-direct",
                "resource": "messages",
                "event": "created",
                "filter": "roomType=direct",
            }
        )
        desired.append(
            {
                "name": f"{self._webhook_prefix}-mentions",
                "resource": "messages",
                "event": "created",
                "filter": "mentionedPeople=me" if self._require_mention else "",
            }
        )

        matched_ids: Dict[Tuple[str, str, str, str], str] = {}
        desired_keys = {
            (spec["resource"], spec["event"], spec["filter"], self._target_url)
            for spec in desired
        }

        for webhook in items:
            if not isinstance(webhook, dict):
                continue
            name = str(webhook.get("name") or "")
            if not name.startswith(self._webhook_prefix):
                continue
            key = (
                str(webhook.get("resource") or ""),
                str(webhook.get("event") or ""),
                str(webhook.get("filter") or ""),
                str(webhook.get("targetUrl") or ""),
            )
            webhook_id = str(webhook.get("id") or "")
            if key in desired_keys and key not in matched_ids:
                matched_ids[key] = webhook_id
                continue
            if webhook_id:
                await self._api_delete(f"webhooks/{webhook_id}")

        self._managed_webhook_ids = []
        for spec in desired:
            key = (spec["resource"], spec["event"], spec["filter"], self._target_url)
            if key in matched_ids:
                self._managed_webhook_ids.append(matched_ids[key])
                continue
            payload = {
                "name": spec["name"],
                "targetUrl": self._target_url,
                "resource": spec["resource"],
                "event": spec["event"],
            }
            if spec["filter"]:
                payload["filter"] = spec["filter"]
            if self._secret:
                payload["secret"] = self._secret
            created = await self._api_post_json("webhooks", payload)
            webhook_id = str(created.get("id") or "")
            if not webhook_id:
                raise RuntimeError(f"Failed to create managed Webex webhook for filter {spec['filter']!r}")
            self._managed_webhook_ids.append(webhook_id)

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def _verify_signature(self, headers: Any, body: bytes) -> bool:
        if not self._secret:
            return True

        secret = self._secret.encode("utf-8")
        spark_sig = str(headers.get("X-Spark-Signature", "")).strip()
        if spark_sig:
            digest = hmac.new(secret, body, hashlib.sha1).hexdigest()
            if hmac.compare_digest(spark_sig, digest):
                return True

        webex_sig = str(headers.get("X-Webex-Signature", "")).strip()
        if not webex_sig:
            return False

        expected = {
            "sha1": hmac.new(secret, body, hashlib.sha1).hexdigest(),
            "sha256": hmac.new(secret, body, hashlib.sha256).hexdigest(),
            "sha512": hmac.new(secret, body, hashlib.sha512).hexdigest(),
        }
        for alg, candidate in self._parse_signature_header(webex_sig):
            if alg:
                digest = expected.get(alg)
                if digest and hmac.compare_digest(candidate, digest):
                    return True
                continue
            for digest in (expected["sha256"], expected["sha512"], expected["sha1"]):
                if hmac.compare_digest(candidate, digest):
                    return True
        return False

    @staticmethod
    def _parse_signature_header(value: str) -> List[Tuple[Optional[str], str]]:
        parts = []
        for token in re.split(r"[,\s;]+", value):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                alg, digest = token.split("=", 1)
                parts.append((alg.strip().lower(), digest.strip().lower()))
            else:
                parts.append((None, token.lower()))
        return parts

    # ------------------------------------------------------------------
    # Content helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_path(path: str) -> str:
        path = path.strip() or DEFAULT_PATH
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip("/") or "/"

    def _build_text_payload(
        self,
        chat_id: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return None
        payload: Dict[str, Any] = {}
        if "@" in chat_id and " " not in chat_id:
            payload["toPersonEmail"] = chat_id
        else:
            payload["roomId"] = chat_id
        parent_id = self._extract_parent_id(metadata)
        if parent_id:
            payload["parentId"] = parent_id
        if content:
            payload["markdown"] = content
        return payload

    def _populate_target_fields(self, form: FormData, chat_id: str) -> None:
        chat_id = str(chat_id or "").strip()
        if "@" in chat_id and " " not in chat_id:
            form.add_field("toPersonEmail", chat_id)
        else:
            form.add_field("roomId", chat_id)

    @staticmethod
    def _extract_parent_id(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        parent_id = metadata.get("thread_id")
        if parent_id:
            return str(parent_id)
        return None

    async def _resolve_edit_room_id(self, chat_id: str, message_id: str) -> Optional[str]:
        target = str(chat_id or "").strip()
        if target and "@" not in target:
            return target

        message = await self._api_get_json(f"messages/{message_id}")
        room_id = str((message or {}).get("roomId") or "").strip()
        if room_id:
            return room_id
        return None

    def _strip_bot_mention(self, text: str) -> str:
        if not text:
            return text
        candidate = text
        names = [self._bot_display_name, self._bot_email.split("@")[0] if self._bot_email else "", self._bot_email]
        for name in names:
            name = name.strip()
            if not name:
                continue
            candidate = re.sub(
                rf"^\s*@?{re.escape(name)}(?:\s*[:,\-]\s*|\s+|[).!?]\s*|$)",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
        return candidate.strip()

    @staticmethod
    def _is_gateway_command(text: str) -> bool:
        text = str(text or "").strip()
        if not text.startswith("/"):
            return False
        command = text.split(maxsplit=1)[0]
        return resolve_command(command) is not None

    def _group_message_mentions_bot(self, message: Dict[str, Any]) -> bool:
        mentioned = message.get("mentionedPeople")
        if isinstance(mentioned, list):
            for candidate in mentioned:
                value = str(candidate or "").strip()
                if value in {"me", self._bot_id, self._bot_email}:
                    return True

        haystacks = [
            str(message.get("text") or ""),
            str(message.get("markdown") or ""),
        ]
        names = [
            self._bot_display_name,
            self._bot_email.split("@")[0] if self._bot_email else "",
            self._bot_email,
        ]
        for text in haystacks:
            if not text:
                continue
            for name in names:
                name = name.strip()
                if not name:
                    continue
                if re.search(
                    rf"(^|[\s(])@?{re.escape(name)}(?=$|[\s:,\-).!?])",
                    text,
                    flags=re.IGNORECASE,
                ):
                    return True
        return False

    def _detect_message_type(self, text: str, media_types: List[str]) -> MessageType:
        if text.startswith("/"):
            return MessageType.COMMAND
        if not media_types:
            return MessageType.TEXT
        if any(mt.startswith("image/") for mt in media_types):
            return MessageType.PHOTO
        if any(mt.startswith("audio/") for mt in media_types):
            return MessageType.VOICE
        if any(mt.startswith("video/") for mt in media_types):
            return MessageType.VIDEO
        return MessageType.DOCUMENT

    # ------------------------------------------------------------------
    # Thread context helpers
    # ------------------------------------------------------------------

    def _has_active_session_for_thread(
        self,
        *,
        room_id: str,
        chat_type: str,
        parent_id: str,
        user_id: str,
        user_id_alt: Optional[str] = None,
    ) -> bool:
        """Return True when Hermes already has a session for this Webex thread."""

        session_store = getattr(self, "_session_store", None)
        if not session_store:
            return False

        try:
            from gateway.session import SessionSource, build_session_key

            source = SessionSource(
                platform=Platform("webex"),
                chat_id=room_id,
                chat_type=chat_type,
                user_id=user_id,
                thread_id=parent_id,
                user_id_alt=user_id_alt,
            )
            store_cfg = getattr(session_store, "config", None)
            group_sessions_per_user = (
                getattr(store_cfg, "group_sessions_per_user", True)
                if store_cfg else True
            )
            thread_sessions_per_user = (
                getattr(store_cfg, "thread_sessions_per_user", False)
                if store_cfg else False
            )
            session_key = build_session_key(
                source,
                group_sessions_per_user=group_sessions_per_user,
                thread_sessions_per_user=thread_sessions_per_user,
            )
            session_store._ensure_loaded()
            return session_key in session_store._entries
        except Exception:
            return False

    async def _fetch_thread_context(
        self,
        *,
        room_id: str,
        parent_id: str,
        current_message_id: str,
        limit: int = THREAD_CONTEXT_DEFAULT_LIMIT,
    ) -> str:
        """Fetch prior Webex thread messages for first-contact thread context."""

        cache_key = f"{room_id}:{parent_id}"
        now = time.monotonic()
        cached = self._thread_context_cache.get(cache_key)
        if cached and (now - cached.fetched_at) < THREAD_CONTEXT_CACHE_TTL_SECONDS:
            return cached.content

        parent = await self._api_get_json(f"messages/{parent_id}")
        if not parent:
            return ""

        replies_result = await self._api_get_json(
            "messages",
            params={
                "roomId": room_id,
                "parentId": parent_id,
                "max": limit + 1,
            },
        )
        replies = replies_result.get("items")
        if not isinstance(replies, list):
            replies = []

        parts: List[str] = []
        parent_text = self._message_text(parent)
        parent_line = await self._format_thread_message(parent, is_parent=True)
        if parent_line:
            parts.append(parent_line)

        for reply in reversed(replies):
            if not isinstance(reply, dict):
                continue
            if str(reply.get("id") or "") == current_message_id:
                continue
            if str(reply.get("personId") or "") == self._bot_id:
                continue
            line = await self._format_thread_message(reply, is_parent=False)
            if line:
                parts.append(line)

        content = ""
        if parts:
            content = (
                "[Webex thread context - prior messages in this thread "
                "(not yet in conversation history):]\n"
                + "\n".join(parts)
                + "\n[End of Webex thread context]\n\n"
            )
            if len(content) > THREAD_CONTEXT_MAX_CHARS:
                content = content[: THREAD_CONTEXT_MAX_CHARS - 18].rstrip() + "\n... [truncated]\n\n"

        self._thread_context_cache[cache_key] = _WebexThreadContextCache(
            content=content,
            fetched_at=now,
            parent_text=parent_text,
        )
        return content

    async def _fetch_thread_parent_text(self, room_id: str, parent_id: str) -> str:
        """Return the raw text of a Webex thread parent for reply context."""

        cache_key = f"{room_id}:{parent_id}"
        now = time.monotonic()
        cached = self._thread_context_cache.get(cache_key)
        if cached and (now - cached.fetched_at) < THREAD_CONTEXT_CACHE_TTL_SECONDS:
            return cached.parent_text

        parent = await self._api_get_json(f"messages/{parent_id}")
        if not parent:
            return ""
        return self._message_text(parent)

    async def _format_thread_message(self, message: Dict[str, Any], *, is_parent: bool) -> str:
        text = self._message_text(message)
        if not text:
            return ""
        text = self._strip_bot_mention(text)
        if len(text) > 1000:
            text = text[:997].rstrip() + "..."

        person_id = str(message.get("personId") or "")
        person_email = str(message.get("personEmail") or "")
        if person_id == self._bot_id:
            name = self._bot_display_name or self._bot_email or "Hermes"
        else:
            name = await self._lookup_display_name(person_id)
            if not name:
                name = person_email or person_id or "unknown"

        prefix = "[thread parent] " if is_parent else ""
        return f"{prefix}{name}: {text}"

    @staticmethod
    def _message_text(message: Dict[str, Any]) -> str:
        return str(message.get("text") or message.get("markdown") or "").strip()

    async def _download_attachments(self, files: List[str]) -> Tuple[List[str], List[str]]:
        media_urls: List[str] = []
        media_types: List[str] = []
        for file_url in files:
            try:
                local_path, mime = await self._download_attachment(file_url)
                if local_path:
                    media_urls.append(local_path)
                    media_types.append(mime)
            except Exception as exc:
                logger.warning("[Webex] Failed to download attachment %s: %s", safe_url_for_log(file_url), exc)
        return media_urls, media_types

    async def _download_attachment(self, file_url: str) -> Tuple[str, str]:
        if not self._session:
            raise RuntimeError("HTTP session is not available")
        async with self._session.get(file_url, headers=self._headers()) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            data = await resp.read()
            mime = (resp.headers.get("Content-Type") or "application/octet-stream").split(";", 1)[0]
            filename = self._attachment_filename(file_url, resp.headers)
            ext = Path(filename).suffix or mimetypes.guess_extension(mime) or ".bin"
            if mime.startswith("image/"):
                return cache_image_from_bytes(data, ext), mime
            if mime.startswith("audio/"):
                return cache_audio_from_bytes(data, ext), mime
            return cache_document_from_bytes(data, filename), mime

    @staticmethod
    def _attachment_filename(file_url: str, headers: Any) -> str:
        content_disp = str(headers.get("Content-Disposition", ""))
        match = re.search(r'filename="?([^";]+)"?', content_disp)
        if match:
            return Path(match.group(1)).name
        path_name = Path(urlsplit(file_url).path).name
        return path_name or "attachment"

def _env_enablement() -> Optional[Dict[str, Any]]:
    token = os.getenv("WEBEX_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: Dict[str, Any] = {
        "connection_mode": os.getenv("WEBEX_CONNECTION_MODE", DEFAULT_CONNECTION_MODE),
    }
    for env_name, key in (
        ("WEBEX_WEBHOOK_PUBLIC_URL", "public_url"),
        ("WEBEX_WEBHOOK_SECRET", "secret"),
        ("WEBEX_WEBHOOK_HOST", "host"),
        ("WEBEX_WEBHOOK_PORT", "port"),
        ("WEBEX_WEBHOOK_PATH", "path"),
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            seed[key] = int(value) if key == "port" else value
    home = os.getenv("WEBEX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": "Webex home room"}
    return seed


def _validate_config(config: PlatformConfig) -> bool:
    if not (config.token or os.getenv("WEBEX_BOT_TOKEN")):
        return False
    extra = config.extra or {}
    mode = str(
        extra.get("connection_mode")
        or os.getenv("WEBEX_CONNECTION_MODE", DEFAULT_CONNECTION_MODE)
    ).strip().lower()
    if mode not in {"websocket", "webhook"}:
        return False
    if mode == "webhook":
        public_url = str(
            extra.get("public_url") or os.getenv("WEBEX_WEBHOOK_PUBLIC_URL", "")
        )
        return public_url.startswith("https://")
    return True


def _is_port_binding(config: PlatformConfig) -> bool:
    extra = config.extra or {}
    mode = str(
        extra.get("connection_mode")
        or os.getenv("WEBEX_CONNECTION_MODE", DEFAULT_CONNECTION_MODE)
    ).strip().lower()
    return mode == "webhook"


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    del thread_id, force_document
    token = pconfig.token or os.getenv("WEBEX_BOT_TOKEN", "")
    if not token:
        return {"error": "WEBEX_BOT_TOKEN is not configured"}
    payload: Dict[str, Any] = {"roomId": chat_id, "markdown": message}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if media_files:
                form = FormData()
                form.add_field("roomId", chat_id)
                form.add_field("markdown", message)
                handles = []
                try:
                    for path in media_files:
                        handle = open(path, "rb")
                        handles.append(handle)
                        form.add_field("files", handle, filename=Path(path).name)
                    response = await session.post(
                        f"{API_BASE}/messages", headers=headers, data=form
                    )
                finally:
                    for handle in handles:
                        handle.close()
            else:
                response = await session.post(
                    f"{API_BASE}/messages", headers=headers, json=payload
                )
            async with response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    return {"error": f"Webex HTTP {response.status}: {body}"}
                return {"success": True, "message_id": body.get("id", "")}
    except Exception as exc:
        return {"error": str(exc)}


def register(ctx: Any) -> None:
    ctx.register_platform(
        name="webex",
        label="Webex",
        adapter_factory=lambda cfg: WebexAdapter(cfg),
        check_fn=check_webex_requirements,
        validate_config=_validate_config,
        is_connected=_validate_config,
        required_env=["WEBEX_BOT_TOKEN"],
        install_hint="npm install --prefix plugins/platforms/webex",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="WEBEX_HOME_CHANNEL",
        allowed_users_env="WEBEX_ALLOWED_USERS",
        allow_all_env="WEBEX_ALLOW_ALL_USERS",
        max_message_length=7439,
        platform_hint=(
            "You are chatting via Webex. Markdown and threaded replies are supported."
        ),
        standalone_sender_fn=_standalone_send,
        is_port_binding_fn=_is_port_binding,
        emoji="🟢",
    )
