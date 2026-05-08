"""
NapCat platform adapter speaking OneBot 11 over reverse WebSocket.

Architecture
------------
Hermes runs a lightweight ``aiohttp`` HTTP server and exposes a single
WebSocket endpoint (``/napcat/ws`` by default).  NapCat is configured to
connect *to* Hermes as a ``websocketClients`` entry — this is the pattern
that best matches the common deployment topology where NapCat runs on a
Windows/desktop QQ machine and Hermes runs on a separate Linux/VPS host.

Responsibilities of this adapter:

- Authenticate NapCat with a shared token (``Authorization`` header or
  ``access_token`` query parameter).
- Receive OneBot 11 events (``meta_event``, ``message``) and translate
  them into a Hermes :class:`MessageEvent`.
- Send replies via OneBot actions (``send_private_msg`` /
  ``send_group_msg``) over the same WebSocket, using the ``echo`` field
  for request/response correlation.
- Track the active connection's ``self_id`` so group mention gating can
  tell whether the bot itself was addressed.

The adapter intentionally supports **one** active NapCat connection at a
time — every inbound connection replaces the previous one.  This mirrors
how NapCat's reverse WebSocket client works: a NapCat instance owns a
single QQ account, and that account is expected to connect to a single
gateway endpoint.

Reference: https://mintlify.wiki/NapNeko/NapCatQQ/api/onebot/overview
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

try:
    from aiohttp import WSCloseCode, WSMsgType, web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive import guard
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]
    WSCloseCode = None  # type: ignore[assignment]
    WSMsgType = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8646
DEFAULT_PATH = "/napcat/ws"
DEFAULT_SEND_TIMEOUT = 20.0
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 2000
MAX_MESSAGE_LENGTH = 4500  # OneBot has no strict cap; keep generous & chunk safely
DEFAULT_STREAM_UPLOAD_CHUNK_SIZE = 64 * 1024
DEFAULT_STREAM_UPLOAD_RETENTION_MS = 30 * 1000


def check_napcat_requirements() -> bool:
    """Return True if the optional runtime dependencies are present."""
    return AIOHTTP_AVAILABLE


class NapCatAdapter(BasePlatformAdapter):
    """Reverse WebSocket adapter for NapCatQQ / OneBot 11."""

    # OneBot doesn't support editing sent messages.
    SUPPORTS_MESSAGE_EDITING = False

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NAPCAT)

        extra = config.extra or {}
        self._token: str = str(
            extra.get("token") or config.token or os.getenv("NAPCAT_TOKEN", "")
        ).strip()
        self._host: str = str(extra.get("host") or os.getenv("NAPCAT_HOST", DEFAULT_HOST))
        try:
            self._port: int = int(extra.get("port") or os.getenv("NAPCAT_PORT", DEFAULT_PORT))
        except (TypeError, ValueError):
            self._port = DEFAULT_PORT
        self._path: str = str(extra.get("path") or os.getenv("NAPCAT_PATH", DEFAULT_PATH))
        if not self._path.startswith("/"):
            self._path = "/" + self._path

        # Runtime state
        self._runner: Optional["web.AppRunner"] = None  # type: ignore[name-defined]
        self._site: Optional["web.TCPSite"] = None  # type: ignore[name-defined]
        self._ws: Optional["web.WebSocketResponse"] = None  # type: ignore[name-defined]
        self._ws_lock = asyncio.Lock()
        self._self_id: Optional[str] = None
        self._chat_type_map: Dict[str, str] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._seen_messages: Dict[str, float] = {}
        self._stream_upload_chunk_size = self._coerce_positive_int(
            extra.get("stream_upload_chunk_size") or os.getenv("NAPCAT_STREAM_UPLOAD_CHUNK_SIZE"),
            DEFAULT_STREAM_UPLOAD_CHUNK_SIZE,
        )
        self._stream_upload_retention_ms = self._coerce_positive_int(
            extra.get("stream_upload_retention_ms") or os.getenv("NAPCAT_STREAM_UPLOAD_RETENTION_MS"),
            DEFAULT_STREAM_UPLOAD_RETENTION_MS,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "NapCat"

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "napcat_missing_dependency",
                "aiohttp is required for the NapCat adapter",
                retryable=True,
            )
            return False

        if not self._token:
            self._set_fatal_error(
                "napcat_missing_token",
                "NAPCAT_TOKEN is required — set a shared secret to authenticate NapCat",
                retryable=True,
            )
            return False

        if not self._acquire_platform_lock(
            "napcat-bind", f"{self._host}:{self._port}{self._path}", "NapCat bind address"
        ):
            return False

        try:
            app = web.Application()
            app.router.add_get("/health", self._handle_health)
            app.router.add_get(self._path, self._handle_ws)
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            self._mark_connected()
            logger.info(
                "[%s] Reverse WebSocket listening on ws://%s:%d%s",
                self.name, self._host, self._port, self._path,
            )
            return True
        except Exception as exc:
            self._set_fatal_error(
                "napcat_bind_error", f"NapCat startup failed: {exc}", retryable=True,
            )
            logger.error("[%s] startup failed: %s", self.name, exc, exc_info=True)
            await self._teardown_server()
            self._release_platform_lock()
            return False

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()

        if self._ws is not None:
            try:
                await self._ws.close(code=WSCloseCode.GOING_AWAY, message=b"gateway shutdown")
            except Exception:
                pass
            self._ws = None

        await self._teardown_server()
        self._fail_pending("Disconnected")
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self.name)

    async def _teardown_server(self) -> None:
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None

    def _fail_pending(self, reason: str) -> None:
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        self._pending_responses.clear()

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request):  # pragma: no cover - trivial
        return web.json_response({"status": "ok", "platform": self.platform.value})

    async def _handle_ws(self, request):
        if not self._is_authorized_request(request):
            return web.Response(status=401, text="unauthorized")

        ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=0)
        await ws.prepare(request)

        # NapCat sends X-Self-ID with the upgrade request.
        self_id_header = request.headers.get("X-Self-ID")
        if self_id_header:
            self._self_id = str(self_id_header).strip()

        # Replace any existing connection — NapCat only serves one QQ account.
        previous = self._ws
        self._ws = ws
        if previous is not None and not previous.closed:
            try:
                await previous.close(code=WSCloseCode.GOING_AWAY, message=b"superseded")
            except Exception:
                pass

        logger.info(
            "[%s] NapCat connected (self_id=%s, remote=%s)",
            self.name, self._self_id, request.remote,
        )

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    payload = self._safe_parse_json(msg.data)
                    if payload is not None:
                        await self._handle_ws_payload(payload)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(
                        "[%s] WebSocket error: %s", self.name, ws.exception(),
                    )
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] websocket handler failed", self.name)
        finally:
            if self._ws is ws:
                self._ws = None
            self._fail_pending("Connection closed")
            logger.info("[%s] NapCat disconnected", self.name)
        return ws

    def _is_authorized_request(self, request) -> bool:
        """Check Authorization header / access_token query against configured token."""
        if not self._token:
            return False
        header = request.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            if header[7:].strip() == self._token:
                return True
        elif header.strip() == self._token:
            return True
        query_token = request.rel_url.query.get("access_token") or request.query.get("access_token")
        if query_token and str(query_token).strip() == self._token:
            return True
        return False

    @staticmethod
    def _safe_parse_json(raw: Any) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _handle_ws_payload(self, payload: Dict[str, Any]) -> None:
        # API responses (no post_type, carries echo)
        echo = payload.get("echo")
        if echo and "post_type" not in payload:
            fut = self._pending_responses.pop(str(echo), None)
            if fut and not fut.done():
                fut.set_result(payload)
            return

        post_type = payload.get("post_type")
        if post_type == "meta_event":
            self._handle_meta_event(payload)
            return
        if post_type == "message":
            event = self._build_message_event(payload)
            if event is not None:
                self._dispatch_message_event(event)
            return
        # notice / request events are not surfaced to the agent yet.
        logger.debug("[%s] unhandled post_type=%s", self.name, post_type)

    def _dispatch_message_event(self, event: MessageEvent) -> None:
        """Process inbound messages without blocking the websocket read loop."""
        task = asyncio.create_task(self.handle_message(event))
        try:
            self._background_tasks.add(task)
        except TypeError:
            return
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._expected_cancelled_tasks.discard)

    def _handle_meta_event(self, payload: Dict[str, Any]) -> None:
        self_id = payload.get("self_id")
        if self_id is not None:
            self._self_id = str(self_id)
        meta_type = payload.get("meta_event_type")
        if meta_type == "lifecycle":
            logger.info(
                "[%s] lifecycle=%s self_id=%s",
                self.name, payload.get("sub_type"), self._self_id,
            )

    def _build_message_event(self, payload: Dict[str, Any]) -> Optional[MessageEvent]:
        message_id = payload.get("message_id")
        if message_id is None:
            return None
        msg_key = str(message_id)
        if self._is_duplicate(msg_key):
            return None

        message_type = payload.get("message_type")
        segments = self._normalize_segments(payload.get("message"))
        reply_to, text = self._extract_reply_and_text(segments)

        sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
        sender_id = str(sender.get("user_id") or payload.get("user_id") or "")
        sender_name = (
            str(sender.get("card") or "").strip()
            or str(sender.get("nickname") or "").strip()
            or None
        )

        if message_type == "private":
            chat_id = sender_id or str(payload.get("user_id") or "")
            if not chat_id:
                return None
            if not text:
                return None
            self._chat_type_map[chat_id] = "private"
            source = self.build_source(
                chat_id=chat_id,
                user_id=sender_id or chat_id,
                user_name=sender_name,
                chat_type="dm",
            )
            return MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=msg_key,
                reply_to_message_id=reply_to,
            )

        if message_type == "group":
            group_id = str(payload.get("group_id") or "")
            if not group_id:
                return None

            # Group mention gating: require an explicit @bot segment.
            mentioned, stripped_text = self._strip_self_mention(segments)
            if not mentioned:
                return None
            if not stripped_text:
                return None

            self._chat_type_map[group_id] = "group"
            source = self.build_source(
                chat_id=group_id,
                user_id=sender_id or None,
                user_name=sender_name,
                chat_type="group",
            )
            return MessageEvent(
                text=stripped_text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=msg_key,
                reply_to_message_id=reply_to,
            )

        return None

    @staticmethod
    def _normalize_segments(raw_message: Any) -> List[Dict[str, Any]]:
        """Return OneBot 11 array-format segments from any inbound shape."""
        if isinstance(raw_message, list):
            return [seg for seg in raw_message if isinstance(seg, dict)]
        if isinstance(raw_message, dict):
            return [raw_message]
        if isinstance(raw_message, str):
            # String-format (CQ code) — treat the whole string as plain text.
            # CQ code parsing is intentionally out of scope; recommend
            # ``messagePostFormat: array`` in documentation.
            stripped = re.sub(r"\[CQ:[^\]]*\]", "", raw_message)
            if stripped.strip():
                return [{"type": "text", "data": {"text": stripped}}]
        return []

    @staticmethod
    def _segment_marker(seg_type: str, data: Dict[str, Any]) -> Optional[str]:
        """Render a non-text segment as a compact text marker for the agent.

        Returned markers stay parseable by an LLM and round-trip into
        ``napcat_call`` parameters: e.g. an inbound ``image`` segment becomes
        ``[图片:<file_id>]`` so the agent can fetch it via
        ``napcat_call("get_image", {"file": "<file_id>"})``.
        """
        if seg_type == "image":
            fid = data.get("file") or data.get("file_id") or data.get("url") or ""
            fid = str(fid).strip()
            return f"[图片:{fid}]" if fid else "[图片]"
        if seg_type == "record":
            fid = data.get("file") or data.get("file_id") or data.get("url") or ""
            fid = str(fid).strip()
            return f"[语音:{fid}]" if fid else "[语音]"
        if seg_type == "video":
            fid = data.get("file") or data.get("file_id") or data.get("url") or ""
            fid = str(fid).strip()
            return f"[视频:{fid}]" if fid else "[视频]"
        if seg_type == "file":
            name = str(data.get("name") or data.get("file_name") or "").strip()
            fid = str(
                data.get("file_id") or data.get("id") or data.get("file") or ""
            ).strip()
            if name and fid:
                return f"[文件:{name}:{fid}]"
            if fid:
                return f"[文件:{fid}]"
            if name:
                return f"[文件:{name}]"
            return "[文件]"
        if seg_type == "face":
            fid = str(data.get("id") or "").strip()
            return f"[表情:{fid}]" if fid else "[表情]"
        return None

    @classmethod
    def _extract_reply_and_text(cls, segments: List[Dict[str, Any]]):
        reply_to: Optional[str] = None
        text_parts: List[str] = []
        for seg in segments:
            seg_type = seg.get("type")
            data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
            if seg_type == "reply":
                value = data.get("id")
                if value is not None and reply_to is None:
                    reply_to = str(value)
                continue
            if seg_type == "text":
                value = data.get("text")
                if isinstance(value, str):
                    text_parts.append(value)
                continue
            if seg_type == "at":
                qq = str(data.get("qq") or "").strip()
                if qq:
                    text_parts.append(f"@{qq}")
                continue
            marker = cls._segment_marker(seg_type, data)
            if marker:
                text_parts.append(marker)
        return reply_to, " ".join(part.strip() for part in text_parts if part.strip()).strip()

    def _strip_self_mention(self, segments: List[Dict[str, Any]]):
        """Return (mentioned, cleaned_text) with the @self_id segment removed.

        Only an explicit ``@bot`` counts as a mention. QQ automatically inserts
        an ``at`` segment for the replied author when the client uses the
        native "reply" feature, so replying to a bot message still triggers
        the bot while replying to someone else's message does not.

        Non-text segments are surfaced as compact markers (see
        :meth:`_segment_marker`) so the agent can recognize attachments.
        """
        if not self._self_id:
            return False, ""
        mentioned = False
        text_parts: List[str] = []
        for seg in segments:
            seg_type = seg.get("type")
            data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
            if seg_type == "at":
                qq = str(data.get("qq") or "").strip()
                if qq == self._self_id:
                    mentioned = True
                    continue
                if qq:
                    text_parts.append(f"@{qq}")
                continue
            if seg_type == "reply":
                continue
            if seg_type == "text":
                value = data.get("text")
                if isinstance(value, str):
                    text_parts.append(value)
                continue
            marker = self._segment_marker(seg_type, data)
            if marker:
                text_parts.append(marker)
        cleaned = " ".join(part.strip() for part in text_parts if part.strip()).strip()
        return mentioned, cleaned

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                key: ts for key, ts in self._seen_messages.items() if ts > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

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
        del metadata
        if not self.is_connected:
            return SendResult(success=False, error="Not connected")
        if self._ws is None or getattr(self._ws, "closed", True):
            return SendResult(success=False, error="NapCat is not connected", retryable=True)
        if not content or not content.strip():
            return SendResult(success=True)

        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        last_result = SendResult(success=False, error="No chunks")
        for idx, chunk in enumerate(chunks):
            last_result = await self._send_chunk(
                chat_id, chunk, reply_to=reply_to if idx == 0 else None,
            )
            if not last_result.success:
                return last_result
        return last_result

    def _resolve_chat_target(self, chat_id: str) -> tuple[str, str]:
        """Return ``(chat_type, normalized_id)`` for a Hermes ``chat_id``.

        Cached chat types win; otherwise ``group:`` / ``private:`` prefixes are
        peeled, and a bare numeric id falls back to private (QQ user number).
        """
        chat_type = self._chat_type_map.get(chat_id)
        normalized_id = chat_id
        if chat_type:
            return chat_type, normalized_id
        if chat_id.startswith("group:"):
            return "group", chat_id.split(":", 1)[1]
        if chat_id.startswith("private:"):
            return "private", chat_id.split(":", 1)[1]
        # Heuristic: a plain numeric ID is assumed to be a private chat (QQ
        # numbers). Group IDs get cached when the adapter receives their first
        # inbound message.
        return "private", chat_id

    async def _dispatch_message_segments(
        self,
        chat_type: str,
        normalized_id: str,
        segments: List[Dict[str, Any]],
    ) -> SendResult:
        """Send a pre-built OneBot message segment array to a chat target."""
        if chat_type == "group":
            action = "send_group_msg"
            params: Dict[str, Any] = {
                "group_id": self._coerce_int(normalized_id),
                "message": segments,
            }
        else:
            action = "send_private_msg"
            params = {
                "user_id": self._coerce_int(normalized_id),
                "message": segments,
            }

        try:
            response = await self._call_action(action, params)
        except asyncio.TimeoutError:
            # Waiting for the OneBot echo timed out after we already wrote the
            # request to the socket. The message may have been delivered, so do
            # not auto-retry and risk duplicate sends.
            return SendResult(success=False, error="NapCat send timed out")
        except RuntimeError as exc:
            error = str(exc)
            return SendResult(
                success=False,
                error=error,
                retryable=self._is_retryable_runtime_send_error(error),
            )

        if response.get("status") != "ok" or response.get("retcode", 0) != 0:
            return SendResult(
                success=False,
                error=response.get("message") or response.get("wording") or "send failed",
                raw_response=response,
            )

        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        message_id = data.get("message_id")
        return SendResult(
            success=True,
            message_id=str(message_id) if message_id is not None else None,
            raw_response=response,
        )

    async def _send_chunk(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str],
    ) -> SendResult:
        chat_type, normalized_id = self._resolve_chat_target(chat_id)
        message_segments: List[Dict[str, Any]] = []
        if reply_to:
            message_segments.append({"type": "reply", "data": {"id": str(reply_to)}})
        message_segments.append({"type": "text", "data": {"text": content}})
        return await self._dispatch_message_segments(chat_type, normalized_id, message_segments)

    # ------------------------------------------------------------------
    # Native rich-media output (OneBot 11 segments + NapCat extensions)
    # ------------------------------------------------------------------

    @staticmethod
    def _media_file_uri(path_or_url: str) -> str:
        """Translate a local path / URL into an OneBot-compatible ``file`` value.

        OneBot 11 accepts ``http://``, ``https://``, ``file://``, ``base64://``,
        and absolute filesystem paths. We turn local paths into ``file://``
        URIs (the most portable form across NapCat builds) and pass URLs
        through untouched.
        """
        if not path_or_url:
            return ""
        text = str(path_or_url).strip()
        lowered = text.lower()
        if lowered.startswith(("http://", "https://", "file://", "base64://", "data:")):
            return text
        # Treat everything else as a local filesystem path.
        try:
            abs_path = os.path.abspath(os.path.expanduser(text))
        except Exception:
            abs_path = text
        # OneBot 11 ``file://`` URIs use forward slashes regardless of platform.
        normalized = abs_path.replace("\\", "/")
        if not normalized.startswith("/"):
            # Windows drive-letter path (e.g. ``C:/foo``) — tack on the third
            # slash so the resulting URI remains valid: ``file:///C:/foo``.
            normalized = "/" + normalized
        return f"file://{normalized}"

    @staticmethod
    def _local_stream_upload_path(path_or_url: str) -> Optional[Path]:
        """Return an existing Hermes-local file path that NapCat should stream-upload.

        ``http(s)``, ``base64://`` and ``data:`` are already cross-machine safe.
        ``file://`` is only safe when it points at NapCat's filesystem; if it
        points at a file that exists on the Hermes host, stream it across the
        reverse WebSocket first so NapCat receives a local path it can open.
        """
        if not path_or_url:
            return None
        text = str(path_or_url).strip()
        lowered = text.lower()
        if lowered.startswith(("http://", "https://", "base64://", "data:")):
            return None
        candidate = text
        if lowered.startswith("file://"):
            parsed = urlparse(text)
            candidate = unquote(parsed.path or "")
            if os.name == "nt" and re.match(r"^/[a-zA-Z]:/", candidate):
                candidate = candidate[1:]
        try:
            path = Path(os.path.abspath(os.path.expanduser(candidate)))
        except Exception:
            return None
        try:
            if path.is_file():
                return path
        except OSError:
            return None
        return None

    @staticmethod
    def _extract_stream_uploaded_file(response: Dict[str, Any]) -> Optional[str]:
        """Extract the NapCat-side temp path from an ``upload_file_stream`` reply."""
        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        for key in ("file_path", "path", "file", "url"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(data.get("file_info"), dict):
            for key in ("file_path", "path", "file", "url"):
                value = data["file_info"].get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _combine_errors(*errors: Optional[str]) -> Optional[str]:
        parts = [str(error).strip() for error in errors if str(error or "").strip()]
        if not parts:
            return None
        deduped: List[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return "; ".join(deduped)

    async def _upload_local_file_stream(self, file_path: Path) -> tuple[Optional[str], Optional[str]]:
        """Upload a Hermes-local file to NapCat via NapCat's stream API.

        NapCat v4.8.115+ documents ``upload_file_stream`` for cross-device
        deployments.  The final chunk returns a NapCat-local temp file path;
        media segments and file-upload actions should use that returned path,
        not the original Hermes filesystem path.
        """
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            sha256 = hashlib.sha256()
            chunk_size = max(1, int(self._stream_upload_chunk_size))
            total_chunks = max(1, math.ceil(file_size / chunk_size))
            stream_id = uuid.uuid4().hex
            base_params: Dict[str, Any] = {
                "stream_id": stream_id,
                "filename": file_path.name,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "file_retention": int(self._stream_upload_retention_ms),
            }

            final_response: Optional[Dict[str, Any]] = None
            with file_path.open("rb") as fh:
                for chunk_index in range(total_chunks):
                    chunk = fh.read(chunk_size)
                    sha256.update(chunk)
                    params = dict(base_params)
                    params.update(
                        {
                            "chunk_index": chunk_index,
                            "chunk_data": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                    if chunk_index == total_chunks - 1:
                        params["expected_sha256"] = sha256.hexdigest()
                    response = await self._call_action_raw("upload_file_stream", params)
                    if response.get("status") != "ok" or response.get("retcode", 0) != 0:
                        return None, (
                            response.get("message")
                            or response.get("wording")
                            or "upload_file_stream failed"
                        )
            final_response = await self._call_action_raw(
                "upload_file_stream", {"stream_id": stream_id, "is_complete": True}
            )
            if final_response.get("status") != "ok" or final_response.get("retcode", 0) != 0:
                return None, (
                    final_response.get("message")
                    or final_response.get("wording")
                    or "upload_file_stream complete failed"
                )
        except asyncio.TimeoutError:
            return None, "NapCat upload_file_stream timed out"
        except RuntimeError as exc:
            return None, str(exc)
        except OSError as exc:
            return None, f"Cannot read local file for NapCat stream upload: {exc}"

        if not final_response:
            return None, "upload_file_stream returned no response"
        uploaded_file = self._extract_stream_uploaded_file(final_response)
        if not uploaded_file:
            return None, "upload_file_stream response did not include file_path"
        return uploaded_file, None

    async def _resolve_media_file_reference(self, media_path: str) -> tuple[str, Optional[str]]:
        """Return a NapCat-readable media reference and optional upload warning."""
        local_path = self._local_stream_upload_path(media_path)
        if local_path is None:
            return self._media_file_uri(media_path), None
        uploaded, error = await self._upload_local_file_stream(local_path)
        if uploaded:
            return uploaded, None
        logger.warning(
            "[%s] upload_file_stream failed for %s (%s); falling back to file URI.",
            self.name, local_path, error,
        )
        return self._media_file_uri(str(local_path)), error

    async def _prepare_upload_file_action_params(
        self,
        action: str,
        params: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Stream-upload Hermes-local ``upload_*_file`` params before dispatch."""
        if action not in {"upload_private_file", "upload_group_file"}:
            return params, None
        file_value = params.get("file")
        if not isinstance(file_value, str) or not file_value.strip():
            return params, None
        local_path = self._local_stream_upload_path(file_value)
        if local_path is None:
            return params, None
        uploaded_path, error = await self._upload_local_file_stream(local_path)
        if not uploaded_path:
            logger.warning(
                "[%s] upload_file_stream failed for direct %s %s (%s); "
                "falling back to original file parameter.",
                self.name,
                action,
                local_path,
                error,
            )
            return params, error
        updated = dict(params)
        updated["file"] = uploaded_path
        updated.setdefault("name", local_path.name)
        return updated, None

    async def _send_media_message(
        self,
        chat_id: str,
        seg_type: str,
        media_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image / voice / video segment together with optional caption."""
        if self._ws is None or getattr(self._ws, "closed", True):
            return SendResult(success=False, error="NapCat is not connected", retryable=True)

        chat_type, normalized_id = self._resolve_chat_target(chat_id)
        segments: List[Dict[str, Any]] = []
        if reply_to:
            segments.append({"type": "reply", "data": {"id": str(reply_to)}})

        media_file, upload_error = await self._resolve_media_file_reference(media_path)
        media_data: Dict[str, Any] = {"file": media_file}
        if extra_data:
            media_data.update(extra_data)
        segments.append({"type": seg_type, "data": media_data})

        if caption and caption.strip():
            segments.append({"type": "text", "data": {"text": caption.strip()}})

        result = await self._dispatch_message_segments(chat_type, normalized_id, segments)
        if upload_error and not result.success:
            result.error = self._combine_errors(upload_error, result.error)
        return result

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image (or URL) as an OneBot ``image`` segment."""
        del kwargs
        result = await self._send_media_message(
            chat_id, "image", image_path, caption, reply_to,
        )
        if result.success:
            return result
        # Fall back to text so the user at least sees something.
        logger.warning(
            "[%s] send_image_file failed (%s); falling back to text.",
            self.name, result.error,
        )
        fallback = caption.strip() if caption and caption.strip() else "[图片发送失败]"
        fallback_result = await self.send(chat_id=chat_id, content=fallback, reply_to=reply_to)
        return SendResult(
            success=False,
            message_id=fallback_result.message_id,
            error=result.error or fallback_result.error,
            raw_response=result.raw_response,
            retryable=result.retryable,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a voice message as an OneBot ``record`` segment."""
        del kwargs
        result = await self._send_media_message(
            chat_id, "record", audio_path, caption, reply_to,
        )
        if result.success:
            return result
        logger.warning(
            "[%s] send_voice failed (%s); falling back to text.",
            self.name, result.error,
        )
        fallback = caption.strip() if caption and caption.strip() else "[语音发送失败]"
        fallback_result = await self.send(chat_id=chat_id, content=fallback, reply_to=reply_to)
        return SendResult(
            success=False,
            message_id=fallback_result.message_id,
            error=result.error or fallback_result.error,
            raw_response=result.raw_response,
            retryable=result.retryable,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video as an OneBot ``video`` segment."""
        del kwargs
        result = await self._send_media_message(
            chat_id, "video", video_path, caption, reply_to,
        )
        if result.success:
            return result
        logger.warning(
            "[%s] send_video failed (%s); falling back to text.",
            self.name, result.error,
        )
        fallback = caption.strip() if caption and caption.strip() else "[视频发送失败]"
        fallback_result = await self.send(chat_id=chat_id, content=fallback, reply_to=reply_to)
        return SendResult(
            success=False,
            message_id=fallback_result.message_id,
            error=result.error or fallback_result.error,
            raw_response=result.raw_response,
            retryable=result.retryable,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file via the NapCat ``upload_*_file`` extension actions."""
        del kwargs
        if self._ws is None or getattr(self._ws, "closed", True):
            return SendResult(success=False, error="NapCat is not connected", retryable=True)

        chat_type, normalized_id = self._resolve_chat_target(chat_id)
        resolved_path = os.path.abspath(os.path.expanduser(str(file_path)))
        display_name = (file_name or os.path.basename(resolved_path) or "file").strip()
        upload_file_path = resolved_path
        stream_upload_error: Optional[str] = None
        local_stream_path = self._local_stream_upload_path(str(file_path))
        if local_stream_path is not None:
            uploaded_path, stream_upload_error = await self._upload_local_file_stream(
                local_stream_path
            )
            if uploaded_path:
                upload_file_path = uploaded_path
            else:
                logger.warning(
                    "[%s] upload_file_stream failed for document %s (%s); "
                    "falling back to original file path.",
                    self.name,
                    local_stream_path,
                    stream_upload_error,
                )

        if chat_type == "group":
            action = "upload_group_file"
            params: Dict[str, Any] = {
                "group_id": self._coerce_int(normalized_id),
                "file": upload_file_path,
                "name": display_name,
            }
        else:
            action = "upload_private_file"
            params = {
                "user_id": self._coerce_int(normalized_id),
                "file": upload_file_path,
                "name": display_name,
            }

        try:
            response = await self._call_action(action, params)
        except asyncio.TimeoutError:
            response = None
            error = "NapCat upload timed out"
        except RuntimeError as exc:
            response = None
            error = str(exc)
        else:
            error = None

        if response is not None and response.get("status") == "ok" and response.get("retcode", 0) == 0:
            data = response.get("data") if isinstance(response.get("data"), dict) else {}
            send_result = SendResult(
                success=True,
                message_id=str(data.get("message_id") or "") or None,
                raw_response=response,
            )
            # Best-effort caption follow-up — file uploads don't carry text.
            if caption and caption.strip():
                await self.send(chat_id=chat_id, content=caption.strip(), reply_to=reply_to)
            return send_result

        if response is not None:
            error = (
                response.get("message")
                or response.get("wording")
                or f"upload_{chat_type}_file failed"
            )
        if stream_upload_error and error:
            error = self._combine_errors(stream_upload_error, error)

        logger.warning(
            "[%s] send_document failed (%s); falling back to text notice.",
            self.name, error,
        )
        notice = (
            f"{caption.strip()}\n[文件:{display_name}]"
            if caption and caption.strip()
            else f"[文件:{display_name}]"
        )
        fallback = await self.send(chat_id=chat_id, content=notice, reply_to=reply_to)
        return SendResult(
            success=False,
            message_id=fallback.message_id,
            error=error or fallback.error,
            raw_response=response,
            retryable=fallback.retryable,
        )

    async def call_action(
        self,
        action: str,
        params: Dict[str, Any],
        *,
        timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> Dict[str, Any]:
        """Public OneBot 11 action dispatcher.

        Send ``action`` with ``params`` over the live WebSocket and await
        the matching ``echo`` response. Tools and external code should call
        this method; internal code may still use the private alias
        ``_call_action`` for backwards compatibility.
        """
        params, stream_upload_error = await self._prepare_upload_file_action_params(
            action, params or {}
        )
        response = await self._call_action_raw(action, params, timeout=timeout)
        if (
            stream_upload_error
            and response.get("status") != "ok"
            and (response.get("message") or response.get("wording"))
        ):
            combined_error = self._combine_errors(
                stream_upload_error,
                response.get("message") or response.get("wording"),
            )
            response = dict(response)
            response["message"] = combined_error
            response["wording"] = combined_error
        return response

    async def _call_action_raw(
        self,
        action: str,
        params: Dict[str, Any],
        *,
        timeout: float = DEFAULT_SEND_TIMEOUT,
    ) -> Dict[str, Any]:
        """Send an action without high-level adapter preprocessing."""
        if self._ws is None or getattr(self._ws, "closed", True):
            raise RuntimeError("NapCat websocket not connected")
        echo = uuid.uuid4().hex
        payload = {"action": action, "params": params, "echo": echo}
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_responses[echo] = future
        async with self._ws_lock:
            try:
                await self._ws.send_json(payload)
            except Exception as exc:
                self._pending_responses.pop(echo, None)
                if not future.done():
                    future.set_exception(exc)
                raise
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_responses.pop(echo, None)
            raise

    # Backwards-compatible private alias — historical callers and the
    # internal ``_send_chunk`` path still use this name.
    _call_action = call_action

    @staticmethod
    def _coerce_int(value: Any) -> Any:
        """Return ``value`` as int when possible, otherwise pass through."""
        if isinstance(value, bool):
            return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _coerce_positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _is_retryable_runtime_send_error(error: str) -> bool:
        """Return True only for failures that happen before sending anything."""
        lowered = error.lower()
        return "websocket not connected" in lowered

    # ------------------------------------------------------------------
    # BasePlatformAdapter hooks
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = self._chat_type_map.get(chat_id, "private")
        return {
            "name": chat_id,
            "type": "group" if chat_type == "group" else "dm",
        }
