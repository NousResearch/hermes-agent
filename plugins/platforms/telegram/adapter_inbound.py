"""Telegram inbound methods; runtime dependencies remain on the adapter facade."""

import logging
from typing import Any, Dict, Optional
from gateway.platforms.base import MessageEvent, MessageType
try:
    from telegram import Message, Update
    from telegram.ext import ContextTypes
except ImportError:
    Message = Update = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


class TelegramInboundMixin:
    def _media_message_type(self, msg: Message) -> MessageType:
        """Classify a Telegram media message into a MessageType (first present attachment wins)."""
        from . import adapter as _adapter

        for attr, mtype in (
            ("sticker", _adapter.MessageType.STICKER), ("photo", _adapter.MessageType.PHOTO), ("video", _adapter.MessageType.VIDEO),
            ("audio", _adapter.MessageType.AUDIO), ("voice", _adapter.MessageType.VOICE)):
            if getattr(msg, attr):
                return mtype
        return _adapter.MessageType.DOCUMENT

    async def _download_observed_media(self, msg: Any, what: str):
        """Download ``msg``'s attachment into the media cache (bounded by ``_max_doc_bytes``). Returns ``(status, cached)``:
        ``"none"``, ``"oversized"`` (cached = raw file_size), ``"failed"``, ``"unreadable"`` or ``"ok"``."""
        from . import adapter as _adapter

        from gateway.platforms.base import cache_media_bytes_async
        source, filename, mime, kind = self._observed_media_source(msg)
        if source is None:
            return "none", None
        file_size = getattr(source, "file_size", None)
        if not (0 < self._int_or_zero(file_size) <= getattr(self, "_max_doc_bytes", 20 * 1024 * 1024)):
            return "oversized", file_size
        try:
            file_obj = await source.get_file()
            data = bytes(await file_obj.download_as_bytearray())
            if not filename:
                filename = _adapter.os.path.basename(getattr(file_obj, "file_path", "") or "")
            cached = await cache_media_bytes_async(data, filename=filename, mime_type=mime, default_kind=kind)
        except Exception as exc:
            _adapter.logger.warning("[Telegram] Failed to cache %s: %s", what, _adapter._redact_telegram_error_text(exc), exc_info=True)
            return "failed", None
        if cached is None:
            return "unreadable", None
        return "ok", cached

    async def _cache_observed_media(self, msg: Message, event: MessageEvent) -> None:
        """Cache an unmentioned group attachment and annotate the observed text; oversized or unsupported
        attachments are noted in the transcript without downloading."""
        from . import adapter as _adapter

        status, cached = await self._download_observed_media(msg, "observed group media")
        if status == "oversized":
            limit_mb = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) // (1024 * 1024)
            event.text = self._append_observed_note(
                event.text, f"[Observed Telegram attachment too large or unverifiable. Maximum: {limit_mb} MB.]")
            _adapter.logger.info("[Telegram] Observed group attachment skipped (size=%s)", cached)
            return
        if status == "unreadable":  # only images that fail validation reach here
            event.text = self._append_observed_note(event.text, "[Observed Telegram attachment could not be read, not cached.]")
            return
        if status == "ok":
            event.media_urls = []
            event.media_types = []
            self._attach_cached(event, cached, cached.context_note(), "[Telegram] Cached observed group %s at %s")

    def _attach_cached(self, event: MessageEvent, cached, note: str, log_fmt: str) -> None:
        """Append a cached attachment to the event (message type follows the kind only for the first one)."""
        from . import adapter as _adapter

        event.media_urls.append(cached.path)
        event.media_types.append(cached.media_type)
        if len(event.media_urls) == 1 and cached.kind in self._CACHED_KIND_TO_MESSAGE_TYPE:
            event.message_type = self._CACHED_KIND_TO_MESSAGE_TYPE[cached.kind]
        event.text = self._append_observed_note(event.text, note)
        _adapter.logger.info(log_fmt, cached.kind, cached.path)

    async def _cache_replied_media(self, msg: Any, event: MessageEvent) -> None:
        """Cache media from the message this turn replies to, if any."""
        reply_msg = getattr(msg, "reply_to_message", None)
        if reply_msg is None:
            return
        status, cached = await self._download_observed_media(reply_msg, "replied-to media")
        if status == "ok":
            self._attach_cached(
                event, cached, f"[Replied-to {cached.kind} '{cached.display_name}' saved at: {cached.path}]",
                "[Telegram] Cached replied-to %s at %s")

    def _observed_media_source(self, msg: Message):
        """Return (telegram_file_source, filename, mime, default_kind) or Nones."""
        if msg.photo:
            return msg.photo[-1], "", "", "image"
        if msg.video:
            return msg.video, "", "video/mp4", "video"
        if msg.voice:
            return msg.voice, "voice.ogg", "audio/ogg", "audio"
        if msg.audio:
            return msg.audio, getattr(msg.audio, "file_name", "") or "", "", "audio"
        if msg.document:
            doc = msg.document
            return doc, doc.file_name or "", (doc.mime_type or "").lower(), None
        return None, "", "", None

    @staticmethod
    def _append_observed_note(existing: Optional[str], note: str) -> str:
        if not note:
            return existing or ""
        return f"{existing}\n\n{note}" if existing else note

    async def _surface_media_cache_failure(
        self, msg: Message, event: MessageEvent, kind: str, exc: Exception, display_name: Optional[str] = None) -> None:
        """Surface a failed media download to BOTH the user (reply asking to retry) and the agent (observed
        note) — otherwise the turn dispatches silently with empty media_urls.

        This (1) replies to the user in Telegram so they know to retry, and (2) appends an agent-visible
        notice to event.text via the existing observed-note channel so the agent knows an attachment was
        attempted and failed — never a silent empty turn. No new event fields (the structured-event refactor
        is out of scope per #23045).
        """
        from . import adapter as _adapter

        named = f" ({display_name})" if display_name else ""
        try:
            await msg.reply_text(
                f"\u26a0\ufe0f Couldn't download your {kind}{named} ({exc.__class__.__name__}). Please try sending it again.")
        except Exception as reply_err:
            _adapter.logger.warning("[Telegram] Failed to notify user about %s cache failure: %s", kind, reply_err, exc_info=True)
        event.text = self._append_observed_note(
            event.text,
            f"[The user attempted to send a {kind}{named} but it could not be downloaded ({exc.__class__.__name__}); they have been asked to retry.]",
       )

    def _observe_unmentioned_group_message(
        self, message: Message, msg_type: MessageType, update_id: Optional[int] = None, event: Optional[MessageEvent] = None) -> None:
        """Append skipped group chatter to the target session without dispatching."""
        from . import adapter as _adapter

        store = getattr(self, "_session_store", None)
        if not store:
            return
        adapter_name = getattr(self, "name", "telegram")
        try:
            event = event or self._build_message_event(message, msg_type, update_id=update_id)
            session_entry = store.get_or_create_session(self._telegram_group_observe_shared_source(event.source))
            entry = {
                "role": "user", "content": self._telegram_group_observe_attributed_text(event),
                "timestamp": _adapter.datetime.now(tz=_adapter.timezone.utc).isoformat(), "observed": True}
            if event.message_id:
                entry["message_id"] = str(event.message_id)
            store.append_to_transcript(session_entry.session_id, entry)
            _adapter.logger.info(
                "[%s] Telegram group message observed (no bot trigger): chat=%s from=%s", adapter_name,
                getattr(getattr(message, "chat", None), "id", "unknown"), event.source.user_id or "unknown")
        except Exception as exc:
            _adapter.logger.warning("[%s] Failed to observe Telegram group message: %s", adapter_name, exc)

    def _is_own_message(self, message: Message) -> bool:
        """True when sent by this bot itself (echoed getUpdates must not count as incoming unread)."""
        if not self._bot:
            return False
        from_user = getattr(message, "from_user", None)
        if from_user is None:
            return False
        bot_id = getattr(self._bot, "id", None)
        user_id = getattr(from_user, "id", None)
        return bot_id is not None and user_id is not None and bot_id == user_id

    def _should_process_message(self, message: Message, *, is_command: bool = False) -> bool:
        """Apply Telegram group trigger rules: DMs unrestricted; group messages pass ``allowed_chats`` (hard gate; only
        the ``guest_mode`` @mention bypass crosses it) and then any of free_response chat/topic, ``require_mention``
        off, reply to the bot, @mention (incl. ``/cmd@botname``), or a wake-word match."""
        # Learn the live handle BEFORE any mention gate routes on it, then drop our own echoed messages.
        # Filter out the bot's own messages (returned by getUpdates in some environments like
        # groups/supergroups where the bot can see its own messages). Without this, outbound messages are
        # counted as incoming unread in the Hermes inbox (#52363). Otherwise a BotFather rename leaves the
        # stale handle in place and the exclusive-mention gate reads a message addressed to us as one
        # addressed to some other bot.
        self._observe_bot_identity_from_message(message)
        if self._is_own_message(message):
            return False
        if not self._is_group_chat(message):
            return True
        thread_id = self._effective_message_thread_id(message)
        if self._topic_gates_pass(thread_id, warn_non_numeric=True) is False:
            return False
        chat_id_str = self._chat_id_str(message)
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False
        # Resolve once; _message_mentions_bot is not re-called below in guest mode.
        guest_mention = self._is_guest_mention(message)
        # allowed_chats whitelist: outside chats pass only via the guest-mode explicit mention.
        allowed = self._telegram_allowed_chats()
        if allowed and chat_id_str not in allowed:
            return guest_mention
        if guest_mention or chat_id_str in self._telegram_free_response_chats() or self._telegram_is_free_response_topic(message):
            return True
        if not self._telegram_require_mention() or self._is_reply_to_bot(message):
            return True
        if not self._telegram_guest_mode() and self._message_mentions_bot(message):
            return True
        return self._message_matches_mention_patterns(message)

    async def _ensure_forum_commands(self, message) -> None:
        """Lazy-register bot commands for forum supergroups (topics don't inherit AllGroupChats scope;
        Telegram resolves via BotCommandScopeChat)."""
        from . import adapter as _adapter

        async with self._forum_lock:
            try:
                chat = getattr(message, "chat", None)
                if not chat or not getattr(chat, "is_forum", False):
                    return
                chat_id = int(chat.id)
                if chat_id in self._forum_command_registered:
                    return
                from telegram import BotCommand, BotCommandScopeChat
                from hermes_cli.commands_platforms import telegram_menu_commands, telegram_menu_max_commands
                menu_commands, _ = telegram_menu_commands(max_commands=telegram_menu_max_commands())
                bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
                await self._bot.set_my_commands(bot_commands, scope=BotCommandScopeChat(chat_id=chat_id))
                self._forum_command_registered.add(chat_id)
                _adapter.logger.info("[%s] Lazy-registered %d commands for forum chat %s", self.name, len(bot_commands), chat_id)
            except Exception as e:
                _adapter.logger.warning("[%s] Forum command lazy-registration failed: %s", self.name, _adapter._redact_telegram_error_text(e))

    def _effective_update_message(self, update: Update) -> Optional[Message]:
        """Message-like payload for normal messages and channel posts (``update.channel_post``)."""
        return getattr(update, "effective_message", None) or getattr(update, "message", None)

    def _log_blocked_user(self, msg, *, level=logging.WARNING, what: str = "unauthorized user") -> None:
        from . import adapter as _adapter

        _adapter.logger.log(
            level, "[Telegram] Blocked %s %s in chat %s", what, getattr(getattr(msg, "from_user", None), "id", None),
            getattr(getattr(msg, "chat", None), "id", None))

    def _gate_or_observe(self, msg, update, msg_type: MessageType) -> bool:
        """Group trigger gate; observes unmentioned chatter when configured. True = proceed."""
        if self._should_process_message(msg):
            return True
        if self._should_observe_unmentioned_group_message(msg):
            self._observe_unmentioned_group_message(msg, msg_type, update_id=update.update_id)
        return False

    async def _build_triggered_event(self, msg, update, msg_type: MessageType) -> MessageEvent:
        """Event for an addressed text/command: trigger text cleaned, replied-to media cached, attribution applied."""
        event = self._build_message_event(msg, msg_type, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)
        await self._cache_replied_media(msg, event)
        return self._apply_telegram_group_observe_attribution(event)

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text; buffers client-split chunks into one MessageEvent."""
        from . import adapter as _adapter

        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        # Auth check first: blocked users must not reach batching, the observed transcript, or the agent.
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        if not self._gate_or_observe(msg, update, _adapter.MessageType.TEXT):
            return
        await self._ensure_forum_commands(update.message)
        self._enqueue_text_event(await self._build_triggered_event(msg, update, _adapter.MessageType.TEXT))

    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming command messages."""
        from . import adapter as _adapter

        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        if not self._should_process_message(msg, is_command=True):
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        await self._ensure_forum_commands(msg)
        event = await self._build_triggered_event(msg, update, _adapter.MessageType.COMMAND)
        # A >4096-char command paste arrives as a near-limit COMMAND chunk plus TEXT continuations; dispatching
        # immediately would orphan them. Near-limit commands go through text batching.
        if len(event.text or "") >= self._SPLIT_THRESHOLD:
            self._enqueue_text_event(event)
            return
        await self.handle_message(event)

    async def _handle_location_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming location/venue pin messages."""
        from . import adapter as _adapter

        msg = self._effective_update_message(update)
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        if not self._gate_or_observe(msg, update, _adapter.MessageType.LOCATION):
            return
        venue = getattr(msg, "venue", None)
        location = getattr(venue, "location", None) if venue else getattr(msg, "location", None)
        if not location:
            return
        lat = getattr(location, "latitude", None)
        lon = getattr(location, "longitude", None)
        if lat is None or lon is None:
            return
        parts = ["[The user shared a location pin.]"]
        if venue:
            title = getattr(venue, "title", None)
            address = getattr(venue, "address", None)
            if title:
                parts.append(f"Venue: {title}")
            if address:
                parts.append(f"Address: {address}")
        parts += [
            f"latitude: {lat}", f"longitude: {lon}", f"Map: https://www.google.com/maps/search/?api=1&query={lat},{lon}",
            "Ask what they'd like to find nearby (restaurants, cafes, etc.) and any preferences."]
        event = self._build_message_event(msg, _adapter.MessageType.LOCATION, update_id=update.update_id)
        event.text = "\n".join(parts)
        await self.handle_message(self._apply_telegram_group_observe_attribution(event))

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped batching key; topic recovery first so DM-topic batches coalesce on the recovered lane."""
        self._apply_topic_recovery(event)
        return super()._text_batch_key(event)

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text chunk, or hold it while delayed delivery must be dropped."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="text-enqueue")
            return
        super()._enqueue_text_event(event)

    async def _flush_buffered(self, pending: dict, tasks: dict, key: str, delay: float, where: str, log_fn=None) -> None:
        """Shared delayed-flush body: sleep, pop, hold if teardown started, else dispatch. A cancel after
        the pop but before durable dispatch re-holds the event (never lose it)."""
        from . import adapter as _adapter

        current_task = _adapter.asyncio.current_task()
        event = None
        try:
            await _adapter.asyncio.sleep(delay)
            event = pending.pop(key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                self._hold_inbound_event(event, where=f"{where}-flush")
                event = None
                return
            if log_fn is not None:
                log_fn(event)
            await self.handle_message(event)
            event = None
        except _adapter.asyncio.CancelledError:
            if event is not None:
                self._hold_inbound_event(event, where=f"{where}-flush-cancelled")
            raise
        finally:
            if tasks.get(key) is current_task:
                tasks.pop(key, None)

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        # Adaptive delay: near-split-point last chunk → long delay (continuation almost certain);
        # short/medium totals → capped fast delays; else configured cap (all min()'d with the operator cap).
        from . import adapter as _adapter

        pending = self._pending_text_batches.get(key)
        last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
        total_len = len(getattr(pending, "text", "") or "") if pending else 0
        if last_len >= self._SPLIT_THRESHOLD:
            delay = self._text_batch_split_delay_seconds
        elif total_len <= self._TEXT_BATCH_FAST_LEN:
            delay = min(self._text_batch_delay_seconds, self._TEXT_BATCH_FAST_DELAY_S)
        elif total_len <= self._TEXT_BATCH_SHORT_LEN:
            delay = min(self._text_batch_delay_seconds, self._TEXT_BATCH_SHORT_DELAY_S)
        else:
            delay = self._text_batch_delay_seconds
        await self._flush_buffered(
            self._pending_text_batches, self._pending_text_batch_tasks, key, delay, "text",
            lambda ev: _adapter.logger.info("[Telegram] Flushing text batch %s (%d chars)", key, len(ev.text or "")))

    def _photo_batch_key(self, event: MessageEvent, msg: Message) -> str:
        """Return a batching key for Telegram photos/albums."""
        from gateway.session import build_session_key
        session_key = build_session_key(
            event.source, group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(event.source))
        media_group_id = getattr(msg, "media_group_id", None)
        return f"{session_key}:album:{media_group_id}" if media_group_id else f"{session_key}:photo-burst"

    async def _flush_photo_batch(self, batch_key: str) -> None:
        """Send a buffered photo burst/album as a single MessageEvent."""
        from . import adapter as _adapter

        await self._flush_buffered(
            self._pending_photo_batches, self._pending_photo_batch_tasks, batch_key, self._media_batch_delay_seconds, "photo",
            lambda ev: _adapter.logger.info("[Telegram] Flushing photo batch %s with %d image(s)", batch_key, len(ev.media_urls)))

    def _merge_into_pending(self, pending: dict, key: str, event: MessageEvent) -> None:
        """Merge ``event`` into ``pending[key]`` (media + caption) or seed it."""
        existing = pending.get(key)
        if existing is None:
            pending[key] = event
            return
        existing.media_urls.extend(event.media_urls)
        existing.media_types.extend(event.media_types)
        if event.text:
            existing.text = self._merge_caption(existing.text, event.text)

    def _enqueue_photo_event(self, batch_key: str, event: MessageEvent) -> None:
        """Merge photo events into a pending batch and schedule flush."""
        from . import adapter as _adapter

        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="photo-enqueue")
            return
        self._merge_into_pending(self._pending_photo_batches, batch_key, event)
        prior_task = self._pending_photo_batch_tasks.get(batch_key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_photo_batch_tasks[batch_key] = _adapter.asyncio.create_task(self._flush_photo_batch(batch_key))

    async def _route_photo_event(self, msg, event: MessageEvent) -> None:
        """Album items debounce on media_group_id; singles go through the photo burst batcher."""
        from . import adapter as _adapter

        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
        else:
            self._enqueue_photo_event(self._photo_batch_key(event, msg), event)

    @staticmethod
    def _ext_from_path(file_path: Optional[str], candidates, default: str) -> str:
        """First extension in ``candidates`` that ``file_path`` ends with (case-insensitive), else default."""
        if file_path:
            lowered = file_path.lower()
            for candidate in candidates:
                if lowered.endswith(candidate):
                    return candidate
        return default

    async def _cache_inbound_av(self, msg, event: MessageEvent, source: Any, label: str, kind: str, ext: str, mime: str) -> bool:
        """Download a voice/audio/video attachment into the local cache. Returns True when the event was
        already dispatched (oversized attachment), so the caller must return."""
        from . import adapter as _adapter

        try:
            allowed, note = self._telegram_media_size_allowed(source, label)
            if not allowed:
                event.text = self._append_observed_note(event.text, note or "")
                _adapter.logger.info("[Telegram] Skipped oversized user %s (size=%s)", kind, getattr(source, "file_size", None))
                await self.handle_message(event)
                return True
            file_obj = await source.get_file()
            data = await file_obj.download_as_bytearray()
            if kind == "video":
                ext = self._ext_from_path(getattr(file_obj, "file_path", None), _adapter.SUPPORTED_VIDEO_TYPES, ext)
                cached_path = await _adapter.cache_video_from_bytes_async(bytes(data), ext=ext)
                mime = _adapter.SUPPORTED_VIDEO_TYPES.get(ext, "video/mp4")
            else:
                cached_path = await _adapter.cache_audio_from_bytes_async(bytes(data), ext=ext)
            event.media_urls = [cached_path]
            event.media_types = [mime]
            _adapter.logger.info("[Telegram] Cached user %s at %s", kind, cached_path)
        except Exception as e:
            _adapter.logger.warning("[Telegram] Failed to cache %s: %s", kind, _adapter._redact_telegram_error_text(e), exc_info=True)
            await self._surface_media_cache_failure(msg, event, label, e)
        return False

    async def _dispatch_with_text(self, event: MessageEvent, text: str) -> bool:
        """Replace the event text with a user-facing note and dispatch it; returns True (handled)."""
        event.text = text
        await self.handle_message(event)
        return True

    @staticmethod
    def _set_cached_media(event: MessageEvent, path: str, mime: str, mtype: MessageType, log_fmt: str) -> None:
        from . import adapter as _adapter

        event.media_urls = [path]
        event.media_types = [mime]
        event.message_type = mtype
        _adapter.logger.info(log_fmt, path)

    async def _cache_inbound_document(self, msg, event: MessageEvent) -> bool:
        """Cache a document attachment (image → photo path, video, else generic media + text injection).
        Returns True when the event was already dispatched/routed so the caller must return."""
        from . import adapter as _adapter

        doc = msg.document
        try:
            original_filename = doc.file_name or ""
            ext = _adapter.os.path.splitext(original_filename)[1].lower() if original_filename else ""
            doc_mime = (doc.mime_type or "").lower()  # some clients send "IMAGE/PNG"
            if not ext and doc_mime:
                ext = _adapter._TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, "")
                if not ext:
                    ext = {v: k for k, v in _adapter.SUPPORTED_DOCUMENT_TYPES.items()}.get(doc_mime, "")
            display = original_filename or doc_mime or ext or 'unknown'
            # Size check before the image branch so image documents can't bypass the limit.
            if not doc.file_size or doc.file_size > self._max_doc_bytes:
                _adapter.logger.info("[Telegram] Document too large: %s bytes", doc.file_size)
                return await self._dispatch_with_text(
                    event, f"The document is too large or its size could not be verified. Maximum: {self._max_doc_bytes // (1024 * 1024)} MB.")
            # Screenshots/photos sent as documents take the image cache + batching path.
            if ext in _adapter._TELEGRAM_IMAGE_EXTENSIONS or doc_mime.startswith("image/"):
                file_obj = await doc.get_file()
                image_bytes = await file_obj.download_as_bytearray()
                image_ext = ext if ext in _adapter._TELEGRAM_IMAGE_EXTENSIONS else _adapter._TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, ".jpg")
                try:
                    cached_path = await _adapter.cache_image_from_bytes_async(bytes(image_bytes), ext=image_ext)
                except ValueError as e:
                    _adapter.logger.warning("[Telegram] Failed to cache image document: %s", _adapter._redact_telegram_error_text(e), exc_info=True)
                    return await self._dispatch_with_text(event, f"Image document '{display}' could not be read as an image.")
                self._set_cached_media(
                    event, cached_path, doc_mime if doc_mime.startswith(
                        "image/"
                    ) else _adapter._TELEGRAM_IMAGE_EXT_TO_MIME.get(image_ext, "image/jpeg"),
                    _adapter.MessageType.PHOTO, "[Telegram] Cached user image-document at %s")
                await self._route_photo_event(msg, event)
                return True
            if not ext and doc.mime_type:
                ext = {v: k for k, v in _adapter.SUPPORTED_VIDEO_TYPES.items()}.get(doc.mime_type, "")
            if not ext and doc.mime_type:
                # .jpg and .jpeg both map to image/jpeg; keep the first ext seen.
                image_mime_to_ext: dict[str, str] = {}
                for _ext, _mime in _adapter.SUPPORTED_IMAGE_DOCUMENT_TYPES.items():
                    image_mime_to_ext.setdefault(_mime, _ext)
                ext = image_mime_to_ext.get(doc.mime_type, "")
            if ext in _adapter.SUPPORTED_VIDEO_TYPES:
                file_obj = await doc.get_file()
                video_bytes = await file_obj.download_as_bytearray()
                self._set_cached_media(
                    event, await _adapter.cache_video_from_bytes_async(bytes(video_bytes), ext=ext), _adapter.SUPPORTED_VIDEO_TYPES[ext], _adapter.MessageType.VIDEO,
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
            if cached.kind == "audio":
                event.message_type = _adapter.MessageType.AUDIO
            _adapter.logger.info("[Telegram] Cached user %s at %s (%s)", cached.kind, cached.path, cached.media_type)
            # Inject text-readable content (≤100 KB). Gate on extension/MIME, NOT a blind UTF-8 decode:
            # PDF/zip/docx have decodable ASCII headers. Binary files are surfaced as a cached path only.
            MAX_TEXT_INJECT_BYTES = 100 * 1024
            _is_text = ext in _adapter._TEXT_INJECT_EXTENSIONS or (doc_mime or "").startswith("text/")
            if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                try:
                    text_content = raw_bytes.decode("utf-8")
                    display_name = _adapter.re.sub(r'[^\w.\- ]', '_', original_filename or f"document{ext or '.txt'}")
                    injection = f"[Content of {display_name}]:\n{text_content}"
                    event.text = f"{injection}\n\n{event.text}" if event.text else injection
                except UnicodeDecodeError:
                    pass  # binary — agent has the cached path
        except Exception as e:
            _adapter.logger.warning("[Telegram] Failed to cache document: %s", _adapter._redact_telegram_error_text(e), exc_info=True)
            await self._surface_media_cache_failure(msg, event, "attachment", e, display_name=getattr(doc, "file_name", None) or None)
        return False

    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming media messages, downloading images to local cache."""
        from . import adapter as _adapter

        msg = update.message
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg, level=_adapter.logging.INFO, what="media from unauthorized user")
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                _event = self._build_message_event(msg, self._media_message_type(msg), update_id=update.update_id)
                if msg.caption:
                    _event.text = self._clean_bot_trigger_text(msg.caption)
                await self._cache_observed_media(msg, _event)
                self._observe_unmentioned_group_message(msg, _event.message_type, update_id=update.update_id, event=_event)
            return
        event = self._build_message_event(msg, self._media_message_type(msg), update_id=update.update_id)
        if msg.caption:
            event.text = self._clean_bot_trigger_text(msg.caption)
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
                    event, await _adapter.cache_image_from_bytes_async(bytes(image_bytes), ext=ext), f"image/{ext.lstrip('.')}", event.message_type,
                    "[Telegram] Cached user photo at %s")
                await self._route_photo_event(msg, event)
                return
            except Exception as e:
                _adapter.logger.warning("[Telegram] Failed to cache photo: %s", _adapter._redact_telegram_error_text(e), exc_info=True)
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

    async def _queue_media_group_event(self, media_group_id: str, event: MessageEvent) -> None:
        """Debounce album items (shared media_group_id) into one MessageEvent so the second image isn't
        treated as a new message interrupting the first."""
        from . import adapter as _adapter

        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="media-group-enqueue")
            return
        self._merge_into_pending(self._media_group_events, media_group_id, event)
        prior_task = self._media_group_tasks.get(media_group_id)
        if prior_task:
            prior_task.cancel()
        self._media_group_tasks[media_group_id] = _adapter.asyncio.create_task(self._flush_media_group_event(media_group_id))

    async def _flush_media_group_event(self, media_group_id: str) -> None:
        await self._flush_buffered(
            self._media_group_events, self._media_group_tasks, media_group_id, self.MEDIA_GROUP_WAIT_SECONDS, "media-group")

    async def _handle_sticker(self, msg: Message, event: "MessageEvent") -> None:
        """Describe a sticker via vision, cached by file_unique_id; animated/video stickers get an emoji placeholder."""
        from . import adapter as _adapter

        from gateway.sticker_cache import (
            get_cached_description, cache_sticker_description, build_sticker_injection,
            build_animated_sticker_injection, STICKER_VISION_PROMPT)
        sticker = msg.sticker
        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""
        if sticker.is_animated or sticker.is_video:
            event.text = build_animated_sticker_injection(emoji)
            return
        cached = get_cached_description(sticker.file_unique_id)
        if cached:
            event.text = build_sticker_injection(cached["description"], cached.get("emoji", emoji), cached.get("set_name", set_name))
            _adapter.logger.info("[Telegram] Sticker cache hit: %s", sticker.file_unique_id)
            return
        fallback = f"a sticker with emoji {emoji}" if emoji else "a sticker"
        try:
            file_obj = await sticker.get_file()
            image_bytes = await file_obj.download_as_bytearray()
            cached_path = await _adapter.cache_image_from_bytes_async(bytes(image_bytes), ext=".webp")
            _adapter.logger.info("[Telegram] Analyzing sticker at %s", cached_path)
            from tools.vision_tools import vision_analyze_tool
            result = _adapter.json.loads(await vision_analyze_tool(image_url=cached_path, user_prompt=STICKER_VISION_PROMPT))
            if result.get("success"):
                description = result.get("analysis", "a sticker")
                cache_sticker_description(sticker.file_unique_id, description, emoji, set_name)
                event.text = build_sticker_injection(description, emoji, set_name)
            else:
                event.text = build_sticker_injection(fallback, emoji, set_name)
        except Exception as e:
            _adapter.logger.warning("[Telegram] Sticker analysis error: %s", _adapter._redact_telegram_error_text(e), exc_info=True)
            event.text = build_sticker_injection(fallback, emoji, set_name)

    def _reload_dm_topics_from_config(self) -> None:
        """Re-read dm_topics from config.yaml so externally created topics work without restart."""
        from . import adapter as _adapter

        try:
            from hermes_cli.config import load_config_readonly  # canonical loader: managed overlay + ${VAR}
            dm_topics = load_config_readonly().get("platforms", {}).get("telegram", {}).get("extra", {}).get("dm_topics", [])
            if not dm_topics:
                self._dm_topics_config = []
                self._dm_topic_chat_ids = set()
                return
            self._dm_topics_config = dm_topics
            self._dm_topic_chat_ids = {str(chat_entry["chat_id"]) for chat_entry in dm_topics if "chat_id" in chat_entry}
            for chat_entry in dm_topics:
                cid = chat_entry.get("chat_id")
                if not cid:
                    continue
                for t in chat_entry.get("topics", []):
                    tid = t.get("thread_id")
                    name = t.get("name")
                    if tid and name and f"{cid}:{name}" not in self._dm_topics:
                        self._dm_topics[f"{cid}:{name}"] = int(tid)
                        _adapter.logger.info("[%s] Hot-loaded DM topic from config: %s -> thread_id=%s", self.name, f"{cid}:{name}", tid)
        except Exception as e:
            _adapter.logger.debug("[%s] Failed to reload dm_topics from config: %s", self.name, e)

    def _get_dm_topic_info(self, chat_id: str, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the DM topic config dict (name, skill, ...) for this thread_id, or None."""
        from . import adapter as _adapter

        if not thread_id:
            return None
        thread_id_int = int(thread_id)

        def _lookup() -> Optional[Dict[str, Any]]:
            for key, cached_tid in self._dm_topics.items():
                if cached_tid == thread_id_int and key.startswith(f"{chat_id}:"):
                    topic_name = key.split(":", 1)[1]
                    for chat_entry in self._dm_topics_config:
                        if str(chat_entry.get("chat_id")) == chat_id:
                            for t in chat_entry.get("topics", []):
                                if t.get("name") == topic_name:
                                    return t
                    return {"name": topic_name}
            return None

        found = _lookup()
        if found is not None:
            return found
        self._reload_dm_topics_from_config()  # cache miss — topics may have been added externally
        return _lookup()

    def _cache_dm_topic_from_message(self, chat_id: str, thread_id: str, topic_name: str) -> None:
        """Cache a thread_id -> topic_name mapping discovered from an incoming message."""
        from . import adapter as _adapter

        cache_key = f"{chat_id}:{topic_name}"
        if cache_key not in self._dm_topics:
            self._dm_topics[cache_key] = int(thread_id)
            _adapter.logger.info("[%s] Cached DM topic from message: %s -> thread_id=%s", self.name, cache_key, thread_id)

    @classmethod
    def _flatten_rich_inline_text(cls, value: Any) -> str:
        """Best-effort plaintext flattener for Bot API rich-message inline nodes."""
        from . import adapter as _adapter

        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(cls._flatten_rich_inline_text(item) for item in value)
        if isinstance(value, dict):
            for key in ("text", "children"):
                if value.get(key) is not None:
                    return cls._flatten_rich_inline_text(value[key])
        return ""

    @classmethod
    def _flatten_rich_blocks(cls, blocks: Any) -> str:
        """Best-effort plaintext flattener for Bot API rich-message blocks."""
        from . import adapter as _adapter

        if not isinstance(blocks, list):
            return ""
        lines: _adapter.List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "list":
                for item in block.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    item_lines = cls._flatten_rich_blocks(item.get("blocks")).splitlines()
                    if not item_lines:
                        continue
                    label = item.get("label")
                    lines.append(f"{label} {item_lines[0]}".strip() if label else item_lines[0])
                    lines.extend(item_lines[1:])
                continue
            text = cls._flatten_rich_inline_text(block.get("text"))
            if text:
                lines.extend(text.splitlines())
        return "\n".join(line.rstrip() for line in lines if line)

    @classmethod
    def _extract_rich_reply_text(cls, reply_to_message: Any) -> Optional[str]:
        """Return plaintext echoed by Telegram's rich_message reply payload."""
        try:
            getter = getattr(getattr(reply_to_message, "api_kwargs", None), "get", None)
            if not callable(getter):
                return None
            rich_getter = getattr(getter("rich_message"), "get", None)
            if not callable(rich_getter):
                return None
            return cls._flatten_rich_blocks(rich_getter("blocks")).strip() or None
        except Exception:
            return None

    def _resolve_topic_binding(self, message: Message, chat_type: str, thread_id_str: Optional[str]) -> tuple:
        """Return ``(chat_topic, topic_skill)`` for a DM topic or bound forum topic (else Nones)."""
        from . import adapter as _adapter

        chat = message.chat
        chat_topic = None
        topic_skill = None
        if chat_type == "dm" and thread_id_str:
            topic_info = self._get_dm_topic_info(str(chat.id), thread_id_str)
            if topic_info:
                chat_topic = topic_info.get("name")
                topic_skill = topic_info.get("skill")
            # forum_topic_created service messages also reveal topic names
            if hasattr(message, "forum_topic_created") and message.forum_topic_created:
                created_name = message.forum_topic_created.name
                if created_name:
                    self._cache_dm_topic_from_message(str(chat.id), thread_id_str, created_name)
                    if not chat_topic:
                        chat_topic = created_name
        elif chat_type == "group" and thread_id_str:
            # Forum topic skill binding via config.extra['group_topics']; accepts both
            # [{"chat_id": ..., "topics": [...]}] and legacy {"-100...": [{"thread_id": 12}]}.
            group_topics_config = self.config.extra.get("group_topics", [])
            if isinstance(group_topics_config, dict):
                group_topics_iter = [{"chat_id": cfg_chat_id, "topics": topics} for cfg_chat_id, topics in group_topics_config.items()]
            elif isinstance(group_topics_config, list):
                group_topics_iter = [entry for entry in group_topics_config if isinstance(entry, dict)]
            else:
                group_topics_iter = []
            for chat_entry in group_topics_iter:
                if str(chat_entry.get("chat_id", "")) != str(chat.id):
                    continue
                topics = chat_entry.get("topics", [])
                for topic in (topics if isinstance(topics, list) else []):
                    if not isinstance(topic, dict):
                        continue
                    tid = topic.get("thread_id")
                    if tid is not None and str(tid) == thread_id_str:
                        chat_topic = topic.get("name")
                        topic_skill = topic.get("skill")
                        break
                break
        return chat_topic, topic_skill

    def _reply_context(self, message: Message) -> tuple:
        """``(reply_to_id, reply_to_text)`` for the replied-to message: Telegram's native partial quote
        first, then text/caption, rich echo, then the sent index."""
        from . import adapter as _adapter

        if not message.reply_to_message:
            return None, None
        reply_to_id = str(message.reply_to_message.message_id)
        quote = getattr(message, "quote", None)
        quote_text = getattr(quote, "text", None) if quote is not None else None
        if quote_text:
            return reply_to_id, quote_text
        reply_to_text = message.reply_to_message.text or message.reply_to_message.caption or None
        if not reply_to_text:
            reply_to_text = self._extract_rich_reply_text(message.reply_to_message)
        if not reply_to_text:
            try:
                from gateway import rich_sent_store
                reply_to_text = rich_sent_store.lookup(str(message.chat.id), reply_to_id)
            except Exception:
                # Extract reply context if this message is a reply. Prefer Telegram's native partial quote
                # (message.quote, TextQuote) so a user replying to a single selected substring of a prior
                # multi-section message doesn't get the whole replied-to message injected into the agent's
                # context — which can cause the agent to act on unrelated actionable-looking text the user
                # didn't quote (#22619). Fall back to the full replied-to message text / caption when no
                # native quote is present.
                reply_to_text = None
        return reply_to_id, reply_to_text

    def _build_message_event(self, message: Message, msg_type: MessageType, update_id: Optional[int] = None) -> MessageEvent:
        """Build a MessageEvent from a Telegram message. ``update_id`` lets ``/restart`` record the
        triggering offset so the new gateway process advances past it."""
        from . import adapter as _adapter

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
        source.is_one_to_one = str(getattr(chat, "type", "") or "").casefold() == "private"
        source.message_is_edit = getattr(message, "edit_date", None) is not None
        reply_to_id, reply_to_text = self._reply_context(message)
        from gateway.platforms.base import resolve_channel_prompt  # per-channel/topic ephemeral prompt
        _chat_id_str = str(chat.id)
        return _adapter.MessageEvent(
            text=message.text or "", message_type=msg_type, source=source, raw_message=message,
            message_id=str(message.message_id), platform_update_id=update_id,
            reply_to_message_id=reply_to_id, reply_to_text=reply_to_text, auto_skill=topic_skill,
            channel_prompt=resolve_channel_prompt(self.config.extra, thread_id_str or _chat_id_str, _chat_id_str if thread_id_str else None),
            timestamp=message.date)
