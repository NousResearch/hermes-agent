from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import urllib.request
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    from linebot.v3.exceptions import InvalidSignatureError
    from linebot.v3.messaging import (
        AudioMessage,
        ApiClient,
        Configuration,
        MessagingApi,
        PushMessageRequest,
        ReplyMessageRequest,
        ShowLoadingAnimationRequest,
        TextMessage,
    )
    from linebot.v3.messaging.api import MessagingApiBlob
    from linebot.v3.webhook import WebhookParser
    from linebot.v3.webhooks import (
        GroupSource,
        MessageEvent as LineMessageEvent,
        RoomSource,
        TextMessageContent,
        UserSource,
    )

    LINE_AVAILABLE = True
except ImportError:
    InvalidSignatureError = Exception  # type: ignore[assignment]
    AudioMessage = None  # type: ignore[assignment]
    ApiClient = None  # type: ignore[assignment]
    Configuration = None  # type: ignore[assignment]
    MessagingApi = None  # type: ignore[assignment]
    MessagingApiBlob = None  # type: ignore[assignment]
    PushMessageRequest = None  # type: ignore[assignment]
    ReplyMessageRequest = None  # type: ignore[assignment]
    ShowLoadingAnimationRequest = None  # type: ignore[assignment]
    TextMessage = None  # type: ignore[assignment]
    WebhookParser = None  # type: ignore[assignment]
    LineMessageEvent = None  # type: ignore[assignment]
    TextMessageContent = None  # type: ignore[assignment]
    UserSource = None  # type: ignore[assignment]
    GroupSource = None  # type: ignore[assignment]
    RoomSource = None  # type: ignore[assignment]
    LINE_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_image_from_bytes,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4900
MAX_MESSAGES_PER_REQUEST = 5
MAX_CACHED_REPLY_TOKENS = 512
MAX_SEEN_EVENT_IDS = 1024
DEFAULT_MEDIA_PATH = "/media/line"
DEFAULT_MULTIMODAL_GRACE_PERIOD_SECONDS = 5.0


def check_line_requirements() -> bool:
    """Check if LINE runtime dependencies are available."""
    return AIOHTTP_AVAILABLE and LINE_AVAILABLE


class LineAdapter(BasePlatformAdapter):
    """LINE Messaging API adapter backed by webhook delivery."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def _should_auto_tts_voice_input(
        self,
        *,
        event: MessageEvent,
        text_content: str,
        media_files: list,
    ) -> bool:
        del event, text_content, media_files
        # LINE reply billing is quota-sensitive and runner-level modality rules
        # should decide whether a voice reply replaces text. Disable the base
        # adapter's generic voice+text auto-TTS fallback here.
        return False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINE)

        extra = config.extra or {}
        self.host = str(extra.get("host") or os.getenv("LINE_WEBHOOK_HOST", "0.0.0.0")).strip() or "0.0.0.0"
        self.port = int(extra.get("port") or os.getenv("LINE_WEBHOOK_PORT", "8645"))
        self.webhook_path = str(
            extra.get("webhook_path") or os.getenv("LINE_WEBHOOK_PATH", "/webhooks/line")
        ).strip() or "/webhooks/line"
        self.media_path = str(extra.get("media_path") or os.getenv("LINE_MEDIA_PATH", DEFAULT_MEDIA_PATH)).strip() or DEFAULT_MEDIA_PATH
        self.multimodal_grace_period_seconds = float(
            extra.get("multimodal_grace_period_seconds")
            or os.getenv("LINE_MULTIMODAL_GRACE_PERIOD_SECONDS", str(DEFAULT_MULTIMODAL_GRACE_PERIOD_SECONDS))
        )

        self.channel_access_token = str(config.token or os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")).strip()
        self.channel_secret = str(extra.get("channel_secret") or os.getenv("LINE_CHANNEL_SECRET", "")).strip()
        self._public_base_url = str(extra.get("public_base_url") or os.getenv("LINE_PUBLIC_BASE_URL", "")).strip() or None

        self._runner: Optional[web.AppRunner] = None if AIOHTTP_AVAILABLE else None

        self._site: Optional[web.TCPSite] = None if AIOHTTP_AVAILABLE else None
        self._api_client: Optional[ApiClient] = None
        self._messaging_api: Optional[MessagingApi] = None
        self._blob_api: Optional[MessagingApiBlob] = None
        self._webhook_parser: Optional[WebhookParser] = None
        self._reply_tokens_by_message_id: OrderedDict[str, str] = OrderedDict()
        self._seen_event_ids: OrderedDict[str, None] = OrderedDict()
        self._pending_media_batches: Dict[str, MessageEvent] = {}
        self._pending_media_tasks: Dict[str, asyncio.Task] = {}

        if LINE_AVAILABLE and self.channel_access_token:
            configuration = Configuration(access_token=self.channel_access_token)
            self._api_client = ApiClient(configuration)
            self._messaging_api = MessagingApi(self._api_client)
            self._blob_api = MessagingApiBlob(self._api_client)
        if LINE_AVAILABLE and self.channel_secret:
            self._webhook_parser = WebhookParser(self.channel_secret)

    async def connect(self) -> bool:
        """Start the LINE webhook listener."""
        if not getattr(self.config, "enabled", True):
            return False
        if not check_line_requirements():
            logger.warning("[LINE] Missing runtime dependencies. Install line-bot-sdk and aiohttp.")
            return False
        if not self.channel_access_token or not self.channel_secret:
            logger.warning("[LINE] LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET are required to start the adapter.")
            return False

        app = web.Application()
        app.router.add_post(self.webhook_path, self._handle_webhook)
        app.router.add_get(f"{self.media_path}/{{filename}}", self._serve_media)
        self._runner = web.AppRunner(app)
        await self._runner.setup()

        try:
            self._site = web.TCPSite(self._runner, self.host, self.port)
            await self._site.start()
            self._mark_connected()
            logger.info("[LINE] Listening for webhooks on %s:%s%s", self.host, self.port, self.webhook_path)
            return True
        except Exception as exc:
            logger.error("[LINE] Failed to start webhook listener: %s", exc, exc_info=True)
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Stop the LINE webhook listener and clear volatile caches."""
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        for task in self._pending_media_tasks.values():
            task.cancel()
        self._pending_media_tasks.clear()
        self._pending_media_batches.clear()
        self._reply_tokens_by_message_id.clear()
        self._seen_event_ids.clear()
        self._mark_disconnected()

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Validate, parse, and dispatch LINE webhook events."""
        if not self._webhook_parser:
            logger.error("[LINE] Webhook received before parser initialization")
            return web.Response(status=503, text="LINE adapter not configured")

        host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
        proto = request.headers.get("X-Forwarded-Proto") or request.scheme or "https"
        if host:
            self._public_base_url = f"{proto}://{host}"

        signature = request.headers.get("X-Line-Signature", "")
        body = await request.text()

        try:
            events = self._webhook_parser.parse(body, signature, as_payload=False)
        except InvalidSignatureError:
            logger.warning("[LINE] Invalid webhook signature")
            return web.Response(status=400, text="Invalid signature")
        except Exception as exc:
            logger.error("[LINE] Failed to parse webhook: %s", exc, exc_info=True)
            return web.Response(status=400, text="Bad webhook payload")

        for event in events:
            if not isinstance(event, LineMessageEvent):
                continue
            message_type = getattr(event.message, "type", "")
            if isinstance(event.message, TextMessageContent):
                asyncio.create_task(self._process_text_message(event))
            elif message_type == "audio":
                asyncio.create_task(self._process_audio_message(event))
            elif message_type == "image":
                asyncio.create_task(self._process_image_message(event))

        return web.Response(status=200, text="OK")

    async def _serve_media(self, request: web.Request) -> web.StreamResponse:
        """Serve cached outbound LINE audio files over HTTPS."""
        filename = Path(request.match_info.get("filename", "")).name
        if not filename:
            return web.Response(status=404, text="Not found")
        candidate = Path(cache_audio_from_bytes(b"", ext=".tmp")).parent / filename
        if candidate.suffix == ".tmp" and candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass
            return web.Response(status=404, text="Not found")
        if not candidate.exists() or not candidate.is_file():
            return web.Response(status=404, text="Not found")
        return web.FileResponse(path=candidate)

    async def _process_text_message(self, event: LineMessageEvent) -> None:
        """Normalize a LINE text event and pass it into Hermes."""
        gateway_event = self._build_base_event(
            event,
            text=event.message.text,
            message_type=MessageType.TEXT,
            reply_to_message_id=getattr(event.message, "quoted_message_id", None),
        )
        if gateway_event is None:
            return
        if self._has_pending_multimodal_batch(gateway_event):
            await self._enqueue_multimodal_event(gateway_event, flush_immediately=True)
            return
        await self.handle_message(gateway_event)

    async def _process_audio_message(self, event: LineMessageEvent) -> None:
        """Normalize a LINE audio event, cache the blob locally, and batch it."""
        media_urls: List[str] = []
        media_types: List[str] = []
        message_id = str(event.message.id)

        if self._blob_api:
            try:
                audio_bytes = await self._call_sync_api(self._blob_api.get_message_content, message_id)
                cached_path = cache_audio_from_bytes(bytes(audio_bytes), ext=".m4a")
                media_urls.append(cached_path)
                media_types.append("audio/m4a")
                logger.info("[LINE] Cached user audio at %s", cached_path)
            except Exception as exc:
                logger.warning("[LINE] Failed to cache audio message %s: %s", message_id, exc, exc_info=True)

        gateway_event = self._build_base_event(
            event,
            text="[Voice message]",
            message_type=MessageType.VOICE,
            media_urls=media_urls,
            media_types=media_types,
        )
        if gateway_event is None:
            return
        if self._has_pending_multimodal_batch(gateway_event):
            await self._enqueue_multimodal_event(gateway_event, flush_immediately=True)
            return
        await self._enqueue_multimodal_event(gateway_event)

    async def _process_image_message(self, event: LineMessageEvent) -> None:
        """Normalize a LINE image event, cache the preview locally, and batch it."""
        media_urls: List[str] = []
        media_types: List[str] = []
        message_id = str(event.message.id)

        if self._blob_api:
            try:
                image_bytes = await self._call_sync_api(self._blob_api.get_message_content_preview, message_id)
                cached_path = cache_image_from_bytes(bytes(image_bytes), ext=".jpg")
                media_urls.append(cached_path)
                media_types.append("image/jpeg")
                logger.info("[LINE] Cached user image at %s", cached_path)
            except Exception as exc:
                logger.warning("[LINE] Failed to cache image message %s: %s", message_id, exc, exc_info=True)

        gateway_event = self._build_base_event(
            event,
            text="[Image]",
            message_type=MessageType.PHOTO,
            media_urls=media_urls,
            media_types=media_types,
        )
        if gateway_event is None:
            return
        await self._enqueue_multimodal_event(gateway_event)

    def _build_base_event(
        self,
        event: LineMessageEvent,
        *,
        text: str,
        message_type: MessageType,
        reply_to_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> Optional[MessageEvent]:
        """Create the normalized Hermes event envelope for LINE inbound messages."""
        event_id = self._event_cache_key(event)
        if self._remember_seen_event(event_id):
            logger.debug("[LINE] Skipping duplicate event %s", event_id)
            return None

        message_id = str(event.message.id)
        reply_token = getattr(event, "reply_token", None)
        if reply_token:
            self._remember_reply_token(message_id, reply_token)

        source = self._build_source_from_event(event)
        timestamp = datetime.fromtimestamp(event.timestamp / 1000, tz=timezone.utc)

        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=event,
            message_id=message_id,
            media_urls=list(media_urls or []),
            media_types=list(media_types or []),
            reply_to_message_id=reply_to_message_id,
            timestamp=timestamp,
        )

    def _multimodal_batch_key(self, event: MessageEvent) -> str:
        chat_id = str(event.source.chat_id or "")
        user_id = str(getattr(event.source, "user_id", None) or "")
        if event.source.chat_type == "group" and user_id:
            return f"{chat_id}:{user_id}"
        return chat_id

    def _has_pending_multimodal_batch(self, event: MessageEvent) -> bool:
        return self._multimodal_batch_key(event) in self._pending_media_batches

    async def _enqueue_multimodal_event(self, event: MessageEvent, *, flush_immediately: bool = False) -> None:
        """Buffer near-simultaneous LINE events into one multimodal turn."""
        batch_key = self._multimodal_batch_key(event)
        existing = self._pending_media_batches.get(batch_key)
        if existing is None:
            self._pending_media_batches[batch_key] = event
        else:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text and event.text not in (existing.text or ""):
                existing.text = f"{existing.text}\n\n{event.text}".strip() if existing.text else event.text
            if event.message_type in (MessageType.VOICE, MessageType.AUDIO):
                existing.message_type = event.message_type
            existing.message_id = event.message_id
            existing.raw_message = event.raw_message
            existing.reply_to_message_id = event.reply_to_message_id or existing.reply_to_message_id
            existing.timestamp = event.timestamp
            event = existing

        prior_task = self._pending_media_tasks.get(batch_key)
        if prior_task and not prior_task.done():
            prior_task.cancel()

        if flush_immediately:
            pending = self._pending_media_batches.pop(batch_key, None)
            self._pending_media_tasks.pop(batch_key, None)
            if pending is not None:
                await self.handle_message(pending)
            return

        async def _flush() -> None:
            try:
                await asyncio.sleep(self.multimodal_grace_period_seconds)
                pending = self._pending_media_batches.pop(batch_key, None)
                if pending is not None:
                    await self.handle_message(pending)
            except asyncio.CancelledError:
                pass
            finally:
                current = self._pending_media_tasks.get(batch_key)
                if current is asyncio.current_task():
                    self._pending_media_tasks.pop(batch_key, None)

        self._pending_media_tasks[batch_key] = asyncio.create_task(_flush())

    def _build_source_from_event(self, event: LineMessageEvent):
        """Build a Hermes SessionSource from a LINE event."""
        source = event.source
        if isinstance(source, GroupSource):
            return self.build_source(chat_id=source.group_id, chat_name=source.group_id, chat_type="group", user_id=getattr(source, "user_id", None))
        if isinstance(source, RoomSource):
            return self.build_source(chat_id=source.room_id, chat_name=source.room_id, chat_type="group", user_id=getattr(source, "user_id", None))
        if isinstance(source, UserSource):
            user_id = getattr(source, "user_id", None) or ""
            return self.build_source(chat_id=user_id, chat_name=user_id or "LINE DM", chat_type="dm", user_id=user_id or None)
        chat_id = str(getattr(source, "user_id", None) or "")
        return self.build_source(chat_id=chat_id, chat_name=chat_id or "LINE", chat_type="dm", user_id=chat_id or None)

    def _event_cache_key(self, event: LineMessageEvent) -> str:
        webhook_event_id = getattr(event, "webhook_event_id", None)
        if webhook_event_id:
            return str(webhook_event_id)
        return str(event.message.id)

    def _remember_seen_event(self, event_id: str) -> bool:
        if not event_id:
            return False
        if event_id in self._seen_event_ids:
            return True
        self._seen_event_ids[event_id] = None
        self._seen_event_ids.move_to_end(event_id)
        while len(self._seen_event_ids) > MAX_SEEN_EVENT_IDS:
            self._seen_event_ids.popitem(last=False)
        return False

    def _remember_reply_token(self, message_id: str, reply_token: str) -> None:
        if not message_id or not reply_token:
            return
        self._reply_tokens_by_message_id[message_id] = reply_token
        self._reply_tokens_by_message_id.move_to_end(message_id)
        while len(self._reply_tokens_by_message_id) > MAX_CACHED_REPLY_TOKENS:
            self._reply_tokens_by_message_id.popitem(last=False)

    def _fetch_quota_snapshot(self) -> Optional[Dict[str, Any]]:
        if not self.channel_access_token:
            return None
        headers = {"Authorization": f"Bearer {self.channel_access_token}"}
        endpoints = {
            "quota": "https://api.line.me/v2/bot/message/quota",
            "consumption": "https://api.line.me/v2/bot/message/quota/consumption",
        }
        snapshot: Dict[str, Any] = {}
        for key, url in endpoints.items():
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                snapshot[key] = json.loads(resp.read().decode("utf-8"))

        quota = snapshot.get("quota", {})
        consumption = snapshot.get("consumption", {})
        limit = quota.get("value") if quota.get("type") == "limited" else None
        used = consumption.get("totalUsage")
        remaining = (limit - used) if isinstance(limit, int) and isinstance(used, int) else None
        return {
            "limit": limit,
            "used": used,
            "remaining": remaining,
            "type": quota.get("type"),
        }

    def _log_quota_snapshot(self, stage: str, **metadata) -> Optional[Dict[str, Any]]:
        try:
            snapshot = self._fetch_quota_snapshot()
        except Exception as exc:
            logger.warning("[LINE] Failed to fetch quota snapshot (%s): %s", stage, exc)
            return None

        logger.info("[LINE] Quota %s: %s | meta=%s", stage, snapshot, metadata)
        return snapshot

    async def _run_quota_snapshot(self, stage: str, **metadata) -> None:
        await asyncio.to_thread(self._log_quota_snapshot, stage, **metadata)

    def _schedule_quota_snapshot(self, stage: str, **metadata) -> None:
        try:
            task = asyncio.create_task(self._run_quota_snapshot(stage, **metadata))
        except RuntimeError:
            return
        try:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except TypeError:
            pass

    def _pop_reply_token(self, reply_to: Optional[str]) -> Optional[str]:
        if not reply_to:
            return None
        return self._reply_tokens_by_message_id.pop(str(reply_to), None)

    @staticmethod
    def _build_text_batches(chunks: List[str]) -> List[List[TextMessage]]:
        batches: List[List[TextMessage]] = []
        for start in range(0, len(chunks), MAX_MESSAGES_PER_REQUEST):
            batch = [TextMessage(text=chunk) for chunk in chunks[start : start + MAX_MESSAGES_PER_REQUEST]]
            batches.append(batch)
        return batches

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del metadata, kwargs
        if not self._messaging_api:
            return SendResult(success=False, error="LINE Messaging API client is not configured")
        if not chat_id:
            return SendResult(success=False, error="LINE chat_id is required")

        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        batches = self._build_text_batches(chunks)
        reply_token = self._pop_reply_token(reply_to)
        used_reply_api = bool(reply_token)
        self._schedule_quota_snapshot(
            "before_send",
            operation="text",
            chat_id=str(chat_id),
            used_reply_api=used_reply_api,
            batch_count=len(batches),
        )

        try:
            if reply_token:
                first_batch = batches[0]
                await self._call_sync_api(
                    self._messaging_api.reply_message,
                    ReplyMessageRequest(replyToken=reply_token, messages=first_batch),
                )
                batches = batches[1:]

            for batch in batches:
                await self._call_sync_api(
                    self._messaging_api.push_message,
                    PushMessageRequest(to=str(chat_id), messages=batch),
                    x_line_retry_key=str(uuid.uuid4()),
                )

            self._schedule_quota_snapshot(
                "after_send",
                operation="text",
                chat_id=str(chat_id),
                used_reply_api=used_reply_api,
                batch_count=len(self._build_text_batches(chunks)),
            )
            return SendResult(success=True, message_id=str(uuid.uuid4()))
        except Exception as exc:
            logger.error("[LINE] Failed to send message: %s", exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        if not self._messaging_api:
            return SendResult(success=False, error="LINE Messaging API client is not configured")
        if not self._public_base_url:
            return SendResult(success=False, error="LINE public base URL is unknown; cannot serve audio to LINE")

        prepared_path, duration_ms = await asyncio.to_thread(self._prepare_outbound_audio, audio_path)
        public_url = f"{self._public_base_url}{self.media_path}/{quote(Path(prepared_path).name)}"
        messages: List[Any] = [AudioMessage(originalContentUrl=public_url, duration=duration_ms)]
        if caption:
            messages.append(TextMessage(text=caption[: self.MAX_MESSAGE_LENGTH]))

        reply_token = self._pop_reply_token(reply_to)
        used_reply_api = bool(reply_token)
        self._schedule_quota_snapshot(
            "before_send",
            operation="voice",
            chat_id=str(chat_id),
            used_reply_api=used_reply_api,
            message_count=len(messages),
        )
        try:
            if reply_token:
                await self._call_sync_api(
                    self._messaging_api.reply_message,
                    ReplyMessageRequest(replyToken=reply_token, messages=messages[:MAX_MESSAGES_PER_REQUEST]),
                )
            else:
                await self._call_sync_api(
                    self._messaging_api.push_message,
                    PushMessageRequest(to=str(chat_id), messages=messages[:MAX_MESSAGES_PER_REQUEST]),
                    x_line_retry_key=str(uuid.uuid4()),
                )
            self._schedule_quota_snapshot(
                "after_send",
                operation="voice",
                chat_id=str(chat_id),
                used_reply_api=used_reply_api,
                message_count=len(messages),
            )
            return SendResult(success=True, message_id=str(uuid.uuid4()))
        except Exception as exc:
            logger.error("[LINE] Failed to send voice message: %s", exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    def _prepare_outbound_audio(self, audio_path: str) -> tuple[str, int]:
        source = Path(audio_path)
        if not source.exists():
            raise FileNotFoundError(audio_path)

        prepared = source
        if source.suffix.lower() != ".m4a":
            prepared = source.with_suffix(".m4a")
            subprocess.run(
                [
                    os.getenv("FFMPEG_BIN", "ffmpeg"),
                    "-y",
                    "-i",
                    str(source),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(prepared),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        duration_ms = 5000
        try:
            result = subprocess.run(
                [
                    os.getenv("FFPROBE_BIN", "ffprobe"),
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(prepared),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            duration_s = float((result.stdout or "").strip() or "0")
            if duration_s > 0:
                duration_ms = int(duration_s * 1000)
        except Exception:
            pass

        return str(prepared), duration_ms

    async def _call_sync_api(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort LINE loading indicator.

        LINE's loading animation is a one-shot timer, not a persistent typing state.
        Re-triggering it every 2 seconds makes the user-visible spinner appear to last
        forever, so we keep it short and only fire it once per processing run.
        """
        del metadata
        if not self._messaging_api or not chat_id:
            return
        try:
            await self._call_sync_api(
                self._messaging_api.show_loading_animation,
                ShowLoadingAnimationRequest(chatId=str(chat_id), loadingSeconds=5),
            )
        except Exception:
            logger.debug("[LINE] Loading animation unsupported or failed for chat %s", chat_id, exc_info=True)

    async def _keep_typing(
        self,
        chat_id: str,
        interval: float = 2.0,
        metadata=None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        del interval
        if stop_event is not None and stop_event.is_set():
            return
        try:
            await self.send_typing(chat_id, metadata=metadata)
            if stop_event is None:
                while True:
                    await asyncio.sleep(60)
            await stop_event.wait()
        except asyncio.CancelledError:
            raise

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_id = str(chat_id)
        chat_type = "group" if chat_id.startswith(("C", "R")) else "dm"
        return {"id": chat_id, "name": f"LINE {chat_type} {chat_id}", "type": chat_type, "chat_id": chat_id}
