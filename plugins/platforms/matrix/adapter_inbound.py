"""Matrix inbound methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Optional
from gateway.platforms.base import MessageEvent, MessageType


class MatrixInboundMixin:
    def _is_self_sender(self, sender: str) -> bool:
        """True if *sender* is the bot itself (case-insensitive: homeservers vary localpart case). With
        no resolved user_id we can't prove a sender is NOT us, so return True — dropping our own
        events beats an echo loop ("hall of mirrors").

        Matrix user IDs are byte-compared after trimming whitespace and lowercasing — some homeservers
        normalize the localpart case differently at different API surfaces, and the reply-loop tail of the
        "hall of mirrors" bug (#15763) has been observed with the bot's own account bypassing a
        case-sensitive equality check.
        """
        own = (self._user_id or "").strip().lower()
        return not own or sender.strip().lower() == own

    @staticmethod
    def _is_system_or_bridge_sender(sender: str) -> bool:
        """True for appservice/bridge/system identities (``@_telegram_123:server``) or malformed IDs.
        Never offer these a pairing code: an approved bridge would relay every outbound message
        back as an "authorized user message" (echo loop).

        We treat these as system identities for pairing purposes: they should never be offered a pairing
        code, because an operator approving the code would hand the bridge itself permanent authorization —
        and every outbound message relayed by the bridge would then loop back into the agent as an
        "authorized user message", which is the root of issue #15763.
        """
        localpart = (sender or "").strip().lstrip("@").partition(":")[0]
        return not localpart or localpart.startswith("_")

    async def _is_allowed_matrix_room_event(self, room_id: str) -> bool:
        """MATRIX_ALLOWED_ROOMS gate; DMs are exempt so personal chats survive a project allowlist."""
        from . import adapter as _adapter

        if not self._allowed_room_ids or room_id in self._allowed_room_ids:
            return True
        try:
            return await self._is_dm_room(room_id)
        except Exception as exc:
            _adapter.logger.debug("Matrix: could not resolve room identity for allowlist check in %s: %s", room_id, exc)
            return False

    def _reset_clock_skew_detector(self) -> None:
        """State for _note_late_grace_drop: consecutive-drop count, their skew, and the once-only warning."""
        # Clock-skew detection: count grace-check drops that happen well after startup (i.e. not
        # initial-sync backfill). If the host's system clock is set ahead of real time, the startup grace
        # check `event_ts < startup_ts - 5` silently drops every live message. See #12614 — the symptom is
        # "bot joins rooms but never replies". Drops only count when their skew matches the first sampled
        # drop (within 60s), so varied-age backfill from freshly-invited rooms doesn't trip the heuristic.
        self._late_grace_drops: int = 0
        self._late_grace_skew: float = 0.0
        self._clock_skew_warned: bool = False

    def _note_late_grace_drop(self, event_ts: float) -> None:
        """Clock-skew heuristic for grace-check drops well after startup. A host clock set ahead of
        real time makes every live event look "older than startup" and the bot silently never
        replies. Warn once when drops keep happening >30s after startup with a *consistent* skew —
        unlike backfill from a freshly invited room, whose event ages vary widely and reset the counter."""
        from . import adapter as _adapter

        if self._clock_skew_warned or _adapter.time.time() - self._startup_ts <= 30:
            return
        skew = self._startup_ts - event_ts
        if not (5 < skew < 86400):  # ignore malformed/absurd timestamps
            return
        if self._late_grace_drops and abs(skew - self._late_grace_skew) < 60:
            self._late_grace_drops += 1
        else:
            self._late_grace_skew = skew
            self._late_grace_drops = 1
        if self._late_grace_drops >= 3:
            _adapter.logger.warning(
                "Matrix: dropped %d consecutive live events as 'too old' more than 30s after startup "
                "(skew ≈ %.0fs). The host system clock is likely set ahead of real time, which causes "
                "the startup grace filter to silently discard every incoming message. Run "
                "`timedatectl set-ntp true` (or sync NTP) and restart the bot.", self._late_grace_drops, skew)
            self._clock_skew_warned = True

    async def _on_room_message(self, event: Any) -> None:
        from . import adapter as _adapter

        room_id = str(getattr(event, "room_id", ""))
        sender = str(getattr(event, "sender", ""))
        # DEBUG-level proof the callback fires at all (silent-inbound troubleshooting).
        _adapter.logger.debug(
            "Matrix: callback fired — event %s from %s in %s", getattr(event, "event_id", "?"), sender, room_id)
        if self._is_self_sender(sender):
            return
        # Bridge/system identities must never reach the pairing flow (echo loop once paired).
        # Ignore own messages (case-insensitive; also drops when our own user_id hasn't been resolved yet —
        # see _is_self_sender docstring and issue #15763).
        # Once a bridge user is paired, every outbound message it relays would loop back as an authorized
        # user message (the "hall of mirrors" in #15763).
        if self._is_system_or_bridge_sender(sender):
            _adapter.logger.debug("Matrix: ignoring system/bridge sender %s in %s", sender, room_id)
            return
        if any(pattern.search(sender or "") for pattern in self._ignored_user_patterns):
            _adapter.logger.debug("Matrix: ignoring sender %s in %s due to configured ignore pattern", sender, room_id)
            return
        if not await self._is_allowed_matrix_room_event(room_id):
            _adapter.logger.info("Matrix: ignoring message from unauthorized room %s", room_id)
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return
        # Startup grace: ignore old messages replayed by the initial sync.
        event_ts = _adapter._matrix_event_timestamp_seconds(event)
        if event_ts and event_ts < self._startup_ts - _adapter._STARTUP_GRACE_SECONDS:
            self._note_late_grace_drop(event_ts)
            return
        content = getattr(event, "content", None)
        if content is None:
            return
        if isinstance(content, dict):
            source_content, msgtype = content, content.get("msgtype", "")
        else:
            source_content = content.serialize() if hasattr(content, "serialize") else {}
            msgtype = str(content.msgtype) if hasattr(content, "msgtype") else ""
        relates_to = source_content.get("m.relates_to", {})
        if relates_to.get("rel_type") == "m.replace":  # skip edits
            return
        # m.notice is the conventional bot-response msgtype; ignoring it prevents bot-to-bot loops.
        if msgtype == "m.notice" and not self._process_notices:
            return
        if msgtype in ("m.image", "m.audio", "m.video", "m.file"):
            await self._handle_media_message(room_id, sender, event_id, event_ts, source_content, relates_to, msgtype)
        elif msgtype in ("m.text", "m.notice"):
            await self._handle_text_message(room_id, sender, event_id, event_ts, source_content, relates_to)

    async def _resolve_message_context(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict,
        relates_to: dict) -> Optional[tuple]:
        """Shared mention/thread/DM gating. Returns (body, is_dm, chat_type, thread_id,
        display_name, source) or None when the message should be dropped."""
        from . import adapter as _adapter

        identity = await self._resolve_room_identity(room_id)
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"
        thread_id = relates_to.get("event_id") if relates_to.get("rel_type") == "m.thread" else None
        formatted_body = source_content.get("formatted_body")
        mentions_block = source_content.get("m.mentions") or {}  # MSC3952: authoritative signal
        mention_user_ids = mentions_block.get("user_ids") if isinstance(mentions_block, dict) else None
        is_mentioned = self._is_bot_mentioned(body, formatted_body, mention_user_ids)
        if not is_dm:
            # Whitelist first: non-listed rooms are dropped even when @mentioned (DMs exempt).
            if self._allowed_rooms and room_id not in self._allowed_rooms:
                _adapter.logger.debug(
                    "Matrix: ignoring message %s in %s — room not in MATRIX_ALLOWED_ROOMS whitelist", event_id, room_id)
                return None
            is_free_room = room_id in self._free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._threads)
            if self._require_mention and not is_free_room and not in_bot_thread:
                if not is_mentioned and not body.startswith("/"):
                    _adapter.logger.debug(
                        "Matrix: ignoring message %s in %s — no @mention "
                        "(set MATRIX_REQUIRE_MENTION=false to disable)", event_id, room_id)
                    return None
            # thread_require_mention: even inside a bot thread require @mention — prevents
            # infinite reply loops when several bots share one thread.
            elif self._thread_require_mention and in_bot_thread and not is_free_room and not is_mentioned:
                _adapter.logger.debug(
                    "Matrix: ignoring message %s in thread %s — no @mention (thread_require_mention=true)",
                    event_id, thread_id)
                return None
        if is_mentioned and self._require_mention:
            body = self._strip_mention(body)
        # Real thread roots are preserved above; synthetic roots (this event) follow policy: DM
        # @mention threads / DM auto-thread, or room auto-thread unless session_scope pins the room.
        if not thread_id:
            if is_dm:
                synthetic = (self._dm_mention_threads and is_mentioned) or self._dm_auto_thread
            else:
                synthetic = self._matrix_session_scope == "thread" or (
                    self._matrix_session_scope != "room" and self._auto_thread)
            if synthetic:
                thread_id = event_id
        display_name = await self._get_display_name(room_id, sender)
        source = self.build_source(
            chat_id=room_id, chat_name=identity.display_name, chat_type=chat_type, user_id=sender,
            user_name=display_name, thread_id=thread_id, chat_topic=identity.room_topic,
            guild_id=identity.server_name, parent_chat_id=room_id if thread_id else None, message_id=event_id,
            is_bot=bool(sender and sender == self._user_id))
        joined_member_count = getattr(identity, "joined_member_count", None)
        source.is_one_to_one = bool(
            chat_type == "dm" and joined_member_count is not None and joined_member_count <= 2
        )
        source.message_is_edit = False
        if thread_id:
            self._threads.mark(thread_id)  # covers real roots and synthetic ones alike
        self._background_read_receipt(room_id, event_id)
        return body, is_dm, chat_type, thread_id, display_name, source

    async def _extract_reply_context(
        self, room_id: str, body: str, relates_to: dict
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Return (body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name). Captures
        the inline reply fallback (``> <@user:srv> text\\n\\nreply``) BEFORE stripping it, so the
        prompt layer can render "[Replying to: ...]" like Signal/Slack/Telegram."""
        from . import adapter as _adapter

        reply_to = (relates_to.get("m.in_reply_to") or {}).get("event_id")
        reply_to_text = reply_to_author_id = reply_to_author_name = None
        if reply_to and body.startswith("> "):
            reply_to_text, reply_to_author_id = _adapter._extract_reply_fallback(body)
            body = _adapter._strip_reply_fallback(body)
            # Resolve the replied-to author's display name (falls back to localpart).
            if reply_to_author_id:
                reply_to_author_name = await self._get_display_name(room_id, reply_to_author_id)
        return body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name

    async def _build_inbound_event(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict, relates_to: dict,
        **extra) -> Optional[MessageEvent]:
        """Gate + normalise an inbound event into a MessageEvent (None => drop). Text body may
        still change (reply-fallback strip); ``extra`` carries media fields / message_type."""
        from . import adapter as _adapter

        ctx = await self._resolve_message_context(room_id, sender, event_id, body, source_content, relates_to)
        if ctx is None:
            return None
        body, _is_dm, _chat_type, _thread_id, display_name, source = ctx
        body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name = (
            await self._extract_reply_context(room_id, body, relates_to))
        media_msgtype = extra.pop("media_msgtype", None)
        if media_msgtype is None:
            # Re-normalize after reply stripping so ``> quoted\n\n!model`` is still a command.
            body = _adapter._normalize_matrix_bang_command(body)
            extra["message_type"] = _adapter.MessageType.COMMAND if body.startswith("/") else _adapter.MessageType.TEXT
        elif _adapter._is_bare_media_filename(media_msgtype, body):
            body = ""  # transport filename, not user text
        return _adapter.MessageEvent(
            text=body, source=source, raw_message=source_content, message_id=event_id,
            reply_to_message_id=reply_to, reply_to_text=reply_to_text, reply_to_author_id=reply_to_author_id,
            reply_to_author_name=reply_to_author_name,
            # Top-level sender fields mirror source.* — downstream prompt code reads them.
            user_id=sender, user_name=display_name, **extra)

    async def _handle_text_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict) -> None:
        from . import adapter as _adapter

        body = source_content.get("body", "") or ""
        if not body:
            return
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, _adapter._normalize_matrix_bang_command(body), source_content, relates_to)
        if msg_event is None:
            return
        if msg_event.message_type == _adapter.MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    async def _handle_media_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict, msgtype: str) -> None:
        from . import adapter as _adapter

        body = source_content.get("body", "") or ""
        url = source_content.get("url", "")
        if url and not str(url).startswith("mxc://"):
            _adapter.logger.warning("[Matrix] Rejecting inbound media %s with non-MXC URL", event_id)
            return
        content_info = source_content.get("info", {})
        if not isinstance(content_info, dict):
            content_info = {}
        event_mimetype = content_info.get("mimetype", "")
        try:
            event_size_int = int(content_info.get("size") or 0)
        except (TypeError, ValueError):
            event_size_int = 0
        if event_size_int and event_size_int > self._max_media_bytes:
            _adapter.logger.warning(
                "[Matrix] Rejecting oversized inbound media %s (%d > %d bytes)", event_id, event_size_int,
                self._max_media_bytes)
            return
        file_content = source_content.get("file", {})  # encrypted media carries file.url
        if not url and isinstance(file_content, dict):
            url = file_content.get("url", "") or ""
            if url and not str(url).startswith("mxc://"):
                _adapter.logger.warning("[Matrix] Rejecting inbound encrypted media %s with non-MXC URL", event_id)
                return
        is_encrypted_media = bool(file_content and isinstance(file_content, dict) and file_content.get("url"))
        msg_type, media_type, is_voice_message = self._classify_inbound_media(msgtype, event_mimetype, source_content)
        # Cache locally so downstream tools get a real file path.
        cached_path = None
        if url:
            try:
                cached_path = await self._download_and_cache_media(
                    url, event_id, file_content if is_encrypted_media else None, msg_type, media_type,
                    is_voice_message, body)
            except Exception as e:
                _adapter.logger.warning("[Matrix] Failed to cache media: %s", e)
        # Unencrypted media may fall back to the HTTP download URL when caching failed.
        http_url = self._mxc_to_http(url) if url and not is_encrypted_media else ""
        media_urls = [cached_path] if cached_path else ([http_url] if http_url else None)
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, body, source_content, relates_to, message_type=msg_type,
            media_urls=media_urls, media_types=[media_type] if media_urls else None, media_msgtype=msgtype)
        if msg_event is not None:
            await self.handle_message(msg_event)

    @staticmethod
    def _classify_inbound_media(
            msgtype: str, event_mimetype: str, source_content: dict) -> tuple[MessageType, str, bool]:
        """Map a Matrix media msgtype to (MessageType, mime type, is_voice_message)."""
        from . import adapter as _adapter

        if msgtype == "m.image":
            return _adapter.MessageType.PHOTO, event_mimetype or "image/png", False
        if msgtype == "m.audio":
            is_voice = source_content.get("org.matrix.msc3245.voice") is not None
            return (_adapter.MessageType.VOICE if is_voice else _adapter.MessageType.AUDIO), event_mimetype or "audio/ogg", is_voice
        if msgtype == "m.video":
            return _adapter.MessageType.VIDEO, event_mimetype or "video/mp4", False
        return _adapter.MessageType.DOCUMENT, event_mimetype or "application/octet-stream", False

    async def _download_and_cache_media(
        self, url: str, event_id: str, encrypted_file: Optional[dict], msg_type: MessageType, media_type: str,
        is_voice_message: bool, body: str) -> Optional[str]:
        """Download (and decrypt, when *encrypted_file* is given) media into the local cache."""
        from . import adapter as _adapter

        file_bytes = await self._client.download_media(_adapter.ContentURI(url))
        if file_bytes is None:
            return None
        if encrypted_file is not None:
            from mautrix.crypto.attachments import decrypt_attachment
            hashes_value, key_value = encrypted_file.get("hashes"), encrypted_file.get("key")
            hash_value = hashes_value.get("sha256") if isinstance(hashes_value, dict) else None
            key_value = key_value.get("k") if isinstance(key_value, dict) else key_value
            iv_value = encrypted_file.get("iv")
            if not (key_value and hash_value and iv_value):
                _adapter.logger.warning("[Matrix] Encrypted media event missing decryption metadata for %s", event_id)
                return None
            file_bytes = decrypt_attachment(file_bytes, key_value, hash_value, iv_value)
        from gateway.platforms.base import (
            cache_audio_from_bytes_async,
            cache_document_from_bytes_async,
            cache_image_from_bytes_async,
        )
        if msg_type == _adapter.MessageType.PHOTO:
            ext_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
            cached_path = await cache_image_from_bytes_async(file_bytes, ext=ext_map.get(media_type, ".jpg"))
            _adapter.logger.info("[Matrix] Cached user image at %s", cached_path)
            return cached_path
        if msg_type in {_adapter.MessageType.AUDIO, _adapter.MessageType.VOICE}:
            ext = _adapter.Path(body or ("voice.ogg" if is_voice_message else "audio.ogg")).suffix or ".ogg"
            return await cache_audio_from_bytes_async(file_bytes, ext=ext)
        filename = body or ("video.mp4" if msg_type == _adapter.MessageType.VIDEO else "document")
        return await cache_document_from_bytes_async(file_bytes, filename)

    async def _on_invite(self, event: Any) -> None:
        """Auto-join rooms when invited, recording DM rooms in m.direct."""
        from . import adapter as _adapter

        room_id = str(getattr(event, "room_id", ""))
        is_direct = bool(getattr(getattr(event, "content", None), "is_direct", False))
        inviter = str(getattr(event, "sender", ""))
        # Only authorized inviters — otherwise any federated user could pull the bot into rooms.
        if not self._is_authorized_user(inviter):
            _adapter.logger.warning("Matrix: rejecting invite to %s from unauthorized user %s", room_id, inviter)
            return
        _adapter.logger.info("Matrix: invited to %s — joining (is_direct=%s)", room_id, is_direct)
        # Join off the sync path; a declared DM is recorded in m.direct once the join lands.
        self._schedule_invite_join(room_id, is_direct=is_direct and bool(inviter), inviter=inviter)

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        from . import adapter as _adapter

        current_task = _adapter.asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            near_split = last_len >= self._split_threshold
            await _adapter.asyncio.sleep(self._text_batch_split_delay_seconds if near_split else self._text_batch_delay_seconds)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            _adapter.logger.info("[Matrix] Flushing text batch %s (%d chars)", key, len(event.text or ""))
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    def _background_read_receipt(self, room_id: str, event_id: str) -> None:

        from . import adapter as _adapter

        async def _send() -> None:
            try:
                await self.send_read_receipt(room_id, event_id)
            except Exception as exc:  # pragma: no cover — defensive
                _adapter.logger.debug("Matrix: background read receipt failed: %s", exc)
        _adapter.asyncio.ensure_future(_send())
