"""Inbound text and context preparation for the TelegramAdapter adapter."""

from __future__ import annotations

from gateway.platforms.event import attributed_context

import logging
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterator, List, Optional, Set
from gateway.platforms.base import (
    BasePlatformAdapter, ExecApprovalPrompt, SendResult, classify_send_error, unauthorized_action_notice,
    cache_image_from_bytes_async, cache_audio_from_bytes_async, cache_video_from_bytes_async, resolve_proxy_url, SUPPORTED_VIDEO_TYPES,
    SUPPORTED_DOCUMENT_TYPES, SUPPORTED_IMAGE_DOCUMENT_TYPES, _TEXT_INJECT_EXTENSIONS, utf16_len,
)
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.telegram.telegram_entities import expand_link_entities

import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.platforms.telegram.adapter import Message, Update, ContextTypes
if TYPE_CHECKING:
    from plugins.platforms.telegram.adapter import TelegramAdapter

logger = logging.getLogger("plugins.platforms.telegram.adapter")


if TYPE_CHECKING:
    from telegram import Message, Update
    from telegram.ext import ContextTypes

class TelegramInboundContextMixin:
    def _observe_unmentioned_group_message(
        self: TelegramAdapter, message: Message, msg_type: MessageType, update_id: Optional[int] = None, event: Optional[MessageEvent] = None) -> None:
        """Append skipped group chatter to the target session without dispatching."""
        store = getattr(self, "_session_store", None)
        if not store:
            return
        adapter_name = getattr(self, "name", "telegram")
        try:
            event = event or self._build_message_event(message, msg_type, update_id=update_id)
            session_entry = store.get_or_create_session(self._telegram_group_observe_shared_source(event.source))
            content = self._telegram_group_observe_attributed_text(event)
            if event.channel_context:
                content = f"{content.rstrip()}\n{event.channel_context}"
            entry = {
                "role": "user", "content": content,
                "timestamp": datetime.now(tz=UTC).isoformat(), "observed": True}
            if event.message_id:
                entry["message_id"] = str(event.message_id)
            self._accept_update()
            store.append_to_transcript(session_entry.session_id, entry)
            logger.info(
                "[%s] Telegram group message observed (no bot trigger): chat=%s from=%s", adapter_name,
                getattr(getattr(message, "chat", None), "id", "unknown"), event.source.user_id or "unknown")
        except Exception as exc:
            self._fail_update_preparation()
            logger.warning("[%s] Failed to observe Telegram group message: %s", adapter_name, exc)




    async def _cache_inbound_document(self: TelegramAdapter, msg, event: MessageEvent) -> bool:
        """Cache a document attachment (image → photo path, video, else generic media + text injection).
        Returns True when the event was already dispatched/routed so the caller must return."""
        from plugins.platforms.telegram.adapter import (
            SUPPORTED_DOCUMENT_TYPES,
            SUPPORTED_IMAGE_DOCUMENT_TYPES,
            SUPPORTED_VIDEO_TYPES,
            _TELEGRAM_IMAGE_EXTENSIONS,
            _TELEGRAM_IMAGE_EXT_TO_MIME,
            _TELEGRAM_IMAGE_MIME_TO_EXT,
            _TEXT_INJECT_EXTENSIONS,
            _redact_telegram_error_text,
            cache_image_from_bytes_async,
            cache_video_from_bytes_async,
            os,
            re,
        )

        doc = msg.document
        try:
            original_filename = doc.file_name or ""
            ext = os.path.splitext(original_filename)[1].lower() if original_filename else ""
            doc_mime = (doc.mime_type or "").lower()  # some clients send "IMAGE/PNG"
            if not ext and doc_mime:
                ext = _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, "")
                if not ext:
                    ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}.get(doc_mime, "")
            display = original_filename or doc_mime or ext or 'unknown'
            # Size check before the image branch so image documents can't bypass the limit.
            if not doc.file_size or doc.file_size > self._max_doc_bytes:
                logger.info("[Telegram] Document too large: %s bytes", doc.file_size)
                return await self._dispatch_with_text(
                    event, f"The document is too large or its size could not be verified. Maximum: {self._max_doc_bytes // (1024 * 1024)} MB.")
            # Screenshots/photos sent as documents take the image cache + batching path.
            if ext in _TELEGRAM_IMAGE_EXTENSIONS or doc_mime.startswith("image/"):
                file_obj = await doc.get_file()
                image_bytes = await file_obj.download_as_bytearray()
                image_ext = ext if ext in _TELEGRAM_IMAGE_EXTENSIONS else _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, ".jpg")
                try:
                    cached_path = await cache_image_from_bytes_async(bytes(image_bytes), ext=image_ext)
                except ValueError as e:
                    logger.warning("[Telegram] Failed to cache image document: %s", _redact_telegram_error_text(e), exc_info=True)
                    return await self._dispatch_with_text(event, f"Image document '{display}' could not be read as an image.")
                self._set_cached_media(
                    event, cached_path, doc_mime if doc_mime.startswith(
                        "image/"
                    ) else _TELEGRAM_IMAGE_EXT_TO_MIME.get(image_ext, "image/jpeg"),
                    MessageType.PHOTO, "[Telegram] Cached user image-document at %s")
                await self._route_photo_event(msg, event)
                return True
            if not ext and doc.mime_type:
                ext = {v: k for k, v in SUPPORTED_VIDEO_TYPES.items()}.get(doc.mime_type, "")
            if not ext and doc.mime_type:
                # .jpg and .jpeg both map to image/jpeg; keep the first ext seen.
                image_mime_to_ext: dict[str, str] = {}
                for _ext, _mime in SUPPORTED_IMAGE_DOCUMENT_TYPES.items():
                    image_mime_to_ext.setdefault(_mime, _ext)
                ext = image_mime_to_ext.get(doc.mime_type, "")
            if ext in SUPPORTED_VIDEO_TYPES:
                file_obj = await doc.get_file()
                video_bytes = await file_obj.download_as_bytearray()
                self._set_cached_media(
                    event, await cache_video_from_bytes_async(bytes(video_bytes), ext=ext), SUPPORTED_VIDEO_TYPES[ext], MessageType.VIDEO,
                    "[Telegram] Cached user video document at %s")
                await self.handle_message(event)
                return True
            # Any file type is accepted (authorization is the gate, not the extension); unknown types get
            # application/octet-stream. Image documents already returned above.
            file_obj = await doc.get_file()
            raw_bytes = bytes(await file_obj.download_as_bytearray())
            from gateway.platforms.base import cache_media_bytes_async
            cached = await cache_media_bytes_async(raw_bytes, filename=original_filename or f"document{ext or '.bin'}", mime_type=doc_mime)
            if cached is None:
                return await self._dispatch_with_text(event, f"Document '{display}' could not be cached.")
            event.media_urls = [cached.path]
            event.media_types = [cached.media_type]
            event.media_text_inlined = [False]  # flipped below once the text is actually injected
            if cached.kind == "audio":
                event.message_type = MessageType.AUDIO
            logger.info("[Telegram] Cached user %s at %s (%s)", cached.kind, cached.path, cached.media_type)
            # Inject text-readable content (≤100 KB). Gate on extension/MIME, NOT a blind UTF-8 decode:
            # PDF/zip/docx have decodable ASCII headers. Binary files are surfaced as a cached path only.
            MAX_TEXT_INJECT_BYTES = 100 * 1024
            _is_text = ext in _TEXT_INJECT_EXTENSIONS or (doc_mime or "").startswith("text/")
            if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                try:
                    text_content = raw_bytes.decode("utf-8")
                    display_name = re.sub(r'[^\w.\- ]', '_', original_filename or f"document{ext or '.txt'}")
                    injection = f"[Content of {display_name}]:\n{text_content}"
                    if self._is_forward(msg):
                        event.add_channel_context(self._forwarded_context(msg, "Forwarded file", injection))
                    else:
                        event.text = f"{injection}\n\n{event.text}" if event.text else injection
                    event.media_text_inlined = [True]
                except UnicodeDecodeError:
                    pass  # binary — agent has the cached path
        except Exception as e:
            logger.warning("[Telegram] Failed to cache document: %s", _redact_telegram_error_text(e), exc_info=True)
            await self._surface_media_cache_failure(msg, event, "attachment", e, display_name=getattr(doc, "file_name", None) or None)
        return False


    async def _handle_media_message(self: TelegramAdapter, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming media messages, downloading images to local cache."""
        from plugins.platforms.telegram.adapter import (
            _redact_telegram_error_text,
            cache_image_from_bytes_async,
            expand_link_entities,
        )

        msg = update.message
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg, level=logging.INFO, what="media from unauthorized user")
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                _event = self._build_message_event(msg, self._media_message_type(msg), update_id=update.update_id)
                if msg.caption and not self._is_forward(msg):
                    _event.text = self._clean_bot_trigger_text(expand_link_entities(msg))
                await self._cache_observed_media(msg, _event)
                self._observe_unmentioned_group_message(msg, _event.message_type, update_id=update.update_id, event=_event)
            return
        event = self._build_message_event(msg, self._media_message_type(msg), update_id=update.update_id)
        if msg.caption and not self._is_forward(msg):
            from plugins.platforms.telegram.telegram_context import group_trigger_text
            event.text = group_trigger_text(self, msg, expand_link_entities(msg))
        # Stickers: _handle_sticker overwrites event.text with its vision description, so observe attribution must run after it.
        if msg.sticker:
            await self._handle_sticker(msg, event)
            await self.handle_message(self._apply_telegram_group_observe_attribution(event))
            return
        event = self._apply_telegram_group_observe_attribution(event)
        # Cache photo locally: Telegram's file URLs expire (~1 hour) before vision may run.
        if msg.photo:
            try:
                file_obj = await msg.photo[-1].get_file()  # PhotoSize list sorted by size; largest last
                image_bytes = await file_obj.download_as_bytearray()
                ext = self._ext_from_path(file_obj.file_path, [".png", ".webp", ".gif", ".jpeg", ".jpg"], ".jpg")
                self._set_cached_media(
                    event, await cache_image_from_bytes_async(bytes(image_bytes), ext=ext), f"image/{ext.lstrip('.')}", event.message_type,
                    "[Telegram] Cached user photo at %s")
                await self._route_photo_event(msg, event)
                return
            except Exception as e:
                logger.warning("[Telegram] Failed to cache photo: %s", _redact_telegram_error_text(e), exc_info=True)
                await self._surface_media_cache_failure(msg, event, "photo", e)
        # Voice/audio cached for STT transcription; video for vision.
        if msg.voice:
            if await self._cache_inbound_av(msg, event, msg.voice, "voice message", "voice", ".ogg", "audio/ogg"):
                return
        elif msg.audio:
            if await self._cache_inbound_av(msg, event, msg.audio, "audio file", "audio", ".mp3", "audio/mp3"):
                return
        elif msg.video:
            if await self._cache_inbound_av(msg, event, msg.video, "video file", "video", ".mp4", "video/mp4"):
                return
        elif msg.document and await self._cache_inbound_document(msg, event):
            return
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
            return
        await self.handle_message(event)


    def _build_message_event(self: TelegramAdapter, message: Message, msg_type: MessageType, update_id: Optional[int] = None) -> MessageEvent:
        """Build a MessageEvent from a Telegram message. ``update_id`` lets ``/restart`` record the
        triggering offset so the new gateway process advances past it."""
        from plugins.platforms.telegram.adapter import (
            expand_link_entities,
        )

        chat = message.chat
        user = message.from_user
        telegram_chat_type = self._chat_type_str(chat)  # str() so PTB enums and plain-string mocks both work
        chat_type = "group" if telegram_chat_type in {"group", "supergroup"} else ("channel" if telegram_chat_type == "channel" else "dm")
        # Shared normalizer so gating and session routing agree (reply-UI anchors dropped, General → "1").
        # Resolve routable thread id for DM topics and forum group topics via the shared normalizer, so
        # gating and session routing agree on one value. Only real topic/forum messages keep a thread id;
        # ordinary reply-UI anchors are dropped (they are not durable session threads and sends against them
        # hit 'Message thread not found', #3206), while forum General-topic messages
        # (message_thread_id=None) normalize to the General-topic id so replies route back to General
        # (#22423).
        thread_id_str = self._effective_message_thread_id(message)
        chat_topic, topic_skill = self._resolve_topic_binding(message, chat_type, thread_id_str)
        has_full_name = hasattr(chat, "full_name")
        if user:
            user_name = user.full_name
        elif has_full_name and chat_type == "dm":
            user_name = chat.full_name
        else:
            user_name = chat.title if chat_type == "channel" else None
        source = self.build_source(
            chat_id=str(chat.id), chat_name=chat.title or (chat.full_name if has_full_name else None), chat_type=chat_type,
            user_id=(str(user.id) if user else (str(chat.id) if chat_type in {"dm", "channel"} else None)),
            user_name=user_name, thread_id=thread_id_str, chat_topic=chat_topic, message_id=str(message.message_id),
            is_bot=bool(getattr(user, "is_bot", False)) if user else False)
        reply_to_id, reply_to_text = self._reply_context(message)
        from gateway.platforms.base import resolve_channel_prompt  # per-channel/topic ephemeral prompt
        from plugins.platforms.telegram.telegram_context import group_identity_prompt
        _chat_id_str = str(chat.id)
        channel_prompt = resolve_channel_prompt(self.config.extra, thread_id_str or _chat_id_str, _chat_id_str if thread_id_str else None)
        text = expand_link_entities(message)
        channel_context = None
        if self._is_forward(message):
            channel_context = self._forwarded_context(message, "Forwarded message", text) if text else None
            text = ""
        return MessageEvent(
            text=text, message_type=msg_type, source=source, raw_message=message,
            message_id=str(message.message_id), platform_update_id=update_id,
            reply_to_message_id=reply_to_id, reply_to_text=reply_to_text, auto_skill=topic_skill,
            channel_prompt=group_identity_prompt(self, message, channel_prompt), channel_context=channel_context,
            timestamp=message.date)

    @staticmethod
    def _is_forward(message: Message) -> bool:
        """Whether *message* forwards someone else's message. A channel post that Telegram forwards
        into the channel's discussion group automatically is the channel's own message."""
        return getattr(message, "forward_origin", None) is not None and not getattr(message, "is_automatic_forward", False)

    @staticmethod
    def _forward_author(message: Message) -> str:
        """The forwarded message's original author, or "" when Telegram does not say."""
        origin = getattr(message, "forward_origin", None)
        chat = getattr(origin, "sender_chat", None) or getattr(origin, "chat", None)
        return (getattr(getattr(origin, "sender_user", None), "full_name", None)
                or getattr(origin, "sender_user_name", None) or getattr(chat, "title", None) or "")

    def _forwarded_context(self: TelegramAdapter, message: Message, label: str, text: str) -> str:
        return attributed_context(label, text, author=self._forward_author(message))
