"""Canonical Matrix inbound event construction and server timestamp conversion."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType, QuotedMediaDependency
from plugins.platforms.matrix.adapter_feedback import ReadReceiptMode
from plugins.platforms.matrix.effective_event import event_content, event_unsigned
from plugins.platforms.matrix.permalinks import event_permalink, room_via_servers
from plugins.platforms.matrix.rich_content import has_media_url, native_event_context
from plugins.platforms.matrix.voice_mention import VoiceGate
from plugins.platforms.matrix.reply_context import _has_reply_fallback, _split_reply_fallback
from plugins.platforms.matrix.media_content import _inbound_media_caption
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache, MatrixReplyContext, _label_body


def _matrix_event_timestamp_seconds(event: Any) -> float:
    """Return a Matrix event timestamp in seconds, accepting ms or sec values."""
    try:
        ts = float(getattr(event, "timestamp", None) or getattr(event, "server_timestamp", None) or 0)
    except (TypeError, ValueError):
        return 0.0
    return ts / 1000.0 if ts > 10_000_000_000 else ts


def _matrix_event_datetime(event_ts: float, now: datetime) -> datetime:
    """Return the UTC time of an event from its server timestamp ``event_ts`` in seconds, or
    ``now`` when the timestamp is missing or cannot be converted. The sender's homeserver sets
    ``origin_server_ts`` from its own clock, so a time ahead of ``now`` is clamped to ``now``.
    A later time would sort the user's message after the reply, which is stamped with ``now``."""
    if not event_ts:
        return now
    try:
        server_time = datetime.fromtimestamp(event_ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return now
    return min(server_time, now)


if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter


class MatrixInboundEventMixin(BasePlatformAdapter):
    _event_context_cache: MatrixEventContextCache
    _resolve_message_context: Callable[..., Awaitable[tuple | None]]
    _extract_reply_context: Callable[..., Awaitable[MatrixReplyContext]]
    _retain_rich_content: Callable[[MessageEvent, dict, str, str], None]

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    async def _build_inbound_event(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict, relates_to: dict,
        ctx: Optional[tuple] = None, *, reply_parent: MatrixEventContext | None = None,
        event_ts: float = 0.0, **extra) -> Optional[MessageEvent]:
        """Gate + normalise an inbound event into a MessageEvent (None => drop). Text body may
        still change (reply-fallback strip); ``extra`` supplies media fields / message_type.
        ``ctx`` is a pre-resolved ``_resolve_message_context`` result (media path gates before
        downloading); resolving it twice would double the read receipt / thread mark."""
        from .adapter import _normalize_matrix_bang_command

        reply_target = MatrixRelation.from_content(relates_to).reply_target
        retained_parent = reply_parent or (self._event_context_cache.retain(room_id, reply_target) if reply_target else None)
        if ctx is None:
            ctx = await self._resolve_message_context(room_id, sender, event_id, body, source_content, relates_to)
        if ctx is None:
            return None
        body, _is_dm, chat_type, _thread_id, display_name, requires_mention, source = ctx
        if requires_mention:
            extra["metadata"] = {**(extra.get("metadata") or {}), "matrix_requires_mention": True}
        reply = await self._extract_reply_context(
            room_id, body, source_content, relates_to, sender=sender, chat_type=chat_type,
        )
        body = reply.body
        if reply.media_path and reply.media_content_id:
            extra["media_urls"] = [*(extra.get("media_urls") or []), reply.media_path]
            extra["media_types"] = [*(extra.get("media_types") or []), reply.media_type or "image/png"]
        media_msgtype = extra.pop("media_msgtype", None)
        if source_content.get("msgtype") == "m.emote":
            body = _label_body("m.emote", body, sender)
            extra["message_type"] = MessageType.TEXT
        elif media_msgtype == "m.sticker":
            body = _label_body("m.sticker", body, sender)
        elif media_msgtype is None:
            body = _normalize_matrix_bang_command(body)
            extra["message_type"] = MessageType.COMMAND if body.startswith("/") else MessageType.TEXT
        else:
            body = _inbound_media_caption(media_msgtype, body, source_content, relates_to)
        timestamp = _matrix_event_datetime(event_ts, self._utc_now())
        event = MessageEvent(
            text=body, source=source, raw_message=source_content, message_id=event_id,
            reply_to_message_id=reply.event_id, reply_to_text=reply.text, reply_to_author_id=reply.author_id,
            reply_to_author_name=reply.author_name,
            reply_to_is_own_message=reply.is_own_message,
            reply_to_author_authorized=reply.author_authorized,
            user_id=sender, user_name=display_name, timestamp=timestamp, **extra)
        if reply.media_path and reply.event_id and reply.media_content_id:
            event._quoted_media_dependencies = (
                QuotedMediaDependency(
                    room_id, reply.event_id, len(event.media_urls) - 1,
                    reply.media_content_id,
                ),
            )
        if reply.event_id:
            from plugins.platforms.matrix.turn_context import MatrixTurnContext

            event._inbound_context_dependencies = (
                MatrixTurnContext.capture(self, event, reply.parent or retained_parent),
            )
        if source_content.get("msgtype") in {"m.emote", "m.sticker"}:
            event.media_urls = event.media_urls or []
            event.media_types = event.media_types or []
            self._retain_rich_content(event, source_content, event_id, sender)
        return event

    async def _on_room_message(self: MatrixAdapter, event: Any) -> asyncio.Future[bool] | bool | None:
        from plugins.platforms.matrix.adapter import _STARTUP_GRACE_SECONDS, logger
        room_id = str(getattr(event, "room_id", ""))
        sender = str(getattr(event, "sender", ""))
        # DEBUG-level proof the callback fires at all (silent-inbound troubleshooting).
        logger.debug(
            "Matrix: callback fired — event %s from %s in %s", getattr(event, "event_id", "?"), sender, room_id)
        if self._is_self_sender(sender):
            return
        # Bridge/system identities must never reach the pairing flow (echo loop once paired).
        # Ignore own messages (case-insensitive; also drops when our own user_id hasn't been resolved yet —
        # see _is_self_sender docstring and issue #15763).
        # Once a bridge user is paired, every outbound message it relays would loop back as an authorized
        # user message (the "hall of mirrors" in #15763).
        if self._is_system_or_bridge_sender(sender):
            logger.debug("Matrix: ignoring system/bridge sender %s in %s", sender, room_id)
            return
        if any(pattern.search(sender or "") for pattern in self._ignored_user_patterns):
            logger.debug("Matrix: ignoring sender %s in %s due to configured ignore pattern", sender, room_id)
            return
        content = getattr(event, "content", None)
        if content is None or event_unsigned(event).get("redacted_because"):
            return
        source_content = event_content(event)
        msgtype = str(source_content.get("msgtype") or "")
        if msgtype == "m.sticker" and not has_media_url(source_content):
            return
        relates_to = source_content.get("m.relates_to", {})
        reply_target = MatrixRelation.from_content(relates_to).reply_target
        reply_parent = self._event_context_cache.hold(room_id, reply_target) if reply_target else None
        if not await self._is_allowed_matrix_room_event(room_id):
            logger.info("Matrix: ignoring message from unauthorized room %s", room_id)
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            # A retried response still depends on the open text batch that contains this event.
            return self._buffered_intakes.get(event_id)
        # Startup grace: ignore old messages replayed by the initial sync.
        from plugins.platforms.matrix.inbound_events import _matrix_event_timestamp_seconds

        event_ts = _matrix_event_timestamp_seconds(event)
        if not self._resuming_sync and event_ts and event_ts < self._startup_ts - _STARTUP_GRACE_SECONDS:
            self._note_late_grace_drop(event_ts)
            return
        if MatrixRelation.from_content(relates_to).is_edit:
            if isinstance(content, dict):
                self._event_context_cache.apply_edit(
                    room_id, sender, source_content, replacement_id=event_id,
                )
            else:
                # Mautrix can synthesise m.new_content when serialising typed edits.
                target = relates_to.get("event_id")
                if isinstance(target, str):
                    prior = self._event_context_cache.history_entry(room_id, target)
                    if prior is not None and prior.sender and prior.sender != sender:
                        return
                    self._event_context_cache.invalidate(room_id, target)
                    await self._event_context_cache.resolve(self._client, room_id, target)
            if self._process_edits and (msgtype == "m.text" or (msgtype == "m.notice" and self._process_notices)):
                await self._handle_edit_message(room_id, sender, event_id, source_content, relates_to)
            return
        # m.notice is the conventional bot-response msgtype; ignoring it prevents bot-to-bot loops.
        if msgtype == "m.notice" and not self._process_notices:
            return
        if msgtype in {"m.emote", "m.sticker"}:
            self._event_context_cache.store(room_id, event_id, native_event_context(source_content, sender))
        if msgtype in ("m.image", "m.audio", "m.video", "m.file", "m.sticker"):
            return await self._handle_media_message(
                room_id, sender, event_id, event_ts, source_content, relates_to, msgtype,
                reply_parent=reply_parent)
        elif msgtype in ("m.text", "m.notice"):
            return await self._handle_text_message(
                room_id, sender, event_id, event_ts, source_content, relates_to,
                reply_parent=reply_parent)
        elif msgtype == "m.emote":
            return await self._handle_emote_message(
                room_id, sender, event_id, event_ts, source_content, relates_to,
                reply_parent=reply_parent)

    async def _resolve_message_context(
        self: MatrixAdapter, room_id: str, sender: str, event_id: str, body: str, source_content: dict,
        relates_to: dict, mention_claimed: bool = False,
        voice_gate: Optional[VoiceGate] = None) -> Optional[tuple]:
        """Shared mention/thread/DM gating. Returns (body, is_dm, chat_type, thread_id,
        display_name, requires_mention, source) or None when the message should be dropped.
        ``requires_mention`` is true when this room or thread drops messages that do not
        mention the bot. ``mention_claimed``
        marks a parked voice claimed by the sender's follow-up bare @mention; ``voice_gate`` is
        the in-flight mark of a parkable voice, released once the park decision is made."""
        from plugins.platforms.matrix.adapter import logger, _strip_reply_fallback, _normalize_matrix_bang_command
        identity = await self._resolve_room_identity(room_id)
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"
        if source_content.get("msgtype") in {"m.emote", "m.sticker"} and self._is_sender_authorized(
            sender, chat_type=chat_type, chat_id=room_id,
        ) is False:
            return None
        relation = MatrixRelation.from_content(relates_to)
        thread_id = relation.thread_root
        if relation.thread_fallback_target:
            if _has_reply_fallback(body, source_content):
                body = _strip_reply_fallback(body)
            if source_content.get("msgtype") not in {"m.emote", "m.sticker"}:
                body = _normalize_matrix_bang_command(body)
        is_mentioned = mention_claimed or self._content_mentions_bot(body, source_content)
        requires_mention = False
        if not is_dm:
            # Whitelist first: non-listed rooms are dropped even when @mentioned (DMs exempt).
            if self._allowed_rooms and room_id not in self._allowed_rooms:
                logger.debug(
                    "Matrix: ignoring message %s in %s — room not in MATRIX_ALLOWED_ROOMS whitelist", event_id, room_id)
                return None
            is_free_room = room_id in self._free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._threads)
            requires_mention = not is_free_room and (
                self._thread_require_mention if in_bot_thread else self._require_mention
            )
            if self._require_mention and not is_free_room and not in_bot_thread:
                is_command = source_content.get("msgtype") not in {"m.emote", "m.sticker"} and body.startswith("/")
                if not is_mentioned and not is_command:
                    if voice_gate is not None:  # parkable voice: a bare @mention may follow (Element X)
                        self._parked_voices.park(room_id, sender, voice_gate, event_id, source_content, relates_to)
                    logger.debug(
                        "Matrix: ignoring message %s in %s — no @mention "
                        "(set MATRIX_REQUIRE_MENTION=false to disable)", event_id, room_id)
                    return None
            # thread_require_mention: even inside a bot thread require @mention — prevents
            # infinite reply loops when several bots share one thread.
            elif self._thread_require_mention and in_bot_thread and not is_free_room and not is_mentioned:
                logger.debug(
                    "Matrix: ignoring message %s in thread %s — no @mention (thread_require_mention=true)",
                    event_id, thread_id)
                return None
        if is_mentioned and self._require_mention:
            # Strip the mention from the reply text only: the quote block carries the
            # ``> <@bot:srv> ...`` reply pill, which _extract_reply_context parses later
            # for reply_to_author_id. A whole-body replace rewrote the pill to ``> <>``
            # and silently dropped the replied-to author (#111233). Without a fallback, a leading
            # quote is the user's own text, so the mention is stripped from the whole body.
            if relation.reply_target and _has_reply_fallback(body, source_content):
                quote_block, reply_text = _split_reply_fallback(body)
                body = quote_block + self._strip_mention(reply_text)
            else:
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
        if voice_gate is not None:  # decided (parked or passing): don't hold bare mentions any longer
            self._parked_voices.release(room_id, sender, voice_gate)
        display_name = await self._get_display_name(room_id, sender)
        via = await room_via_servers(
            getattr(self._client, "state_store", None), room_id,
            ((self._user_id or "").partition(":")[2], identity.server_name))
        source = self.build_source(
            chat_id=room_id, chat_name=identity.display_name, chat_type=chat_type, user_id=sender,
            user_name=display_name, thread_id=thread_id, chat_topic=identity.room_topic,
            guild_id=identity.server_name, parent_chat_id=room_id if thread_id else None,
            message_id=event_id, source_permalink=event_permalink(room_id, event_id, via))
        if thread_id:
            await self._threads.mark_async(thread_id)  # covers real roots and synthetic ones alike
            self._thread_fallbacks.remember(room_id, thread_id, event_id)
        if self._read_receipts_mode == ReadReceiptMode.IMMEDIATE:
            self._background_read_receipt(room_id, event_id)
        return body, is_dm, chat_type, thread_id, display_name, requires_mention, source

    @staticmethod
    def _parse_process_edits(config) -> bool:
        """process_edits from config.extra, else MATRIX_PROCESS_EDITS (default false). Opt-in:
        forwards an ``m.replace`` edit of a user's own message as a new agent turn carrying the
        corrected text, instead of the default of silently ignoring edits."""
        from .adapter import MatrixAdapter, _env_truthy

        configured = MatrixAdapter._configured_bool(config, "process_edits")
        if configured is not None:
            return configured
        return _env_truthy("MATRIX_PROCESS_EDITS", "false")


    async def _handle_edit_message(
        self: "MatrixAdapter", room_id: str, sender: str, event_id: str, source_content: dict, relates_to: dict) -> None:
        """process_edits (opt-in): forward the corrected body of an ``m.replace`` edit as a new
        agent turn. Reuses the normal message gate (mention/thread/session/auth, all keyed off
        ``m.new_content`` exactly as a fresh event would be) so an edit is authorized and routed
        the same way a brand-new message from the same sender would be; the edit's own event_id
        (checked by the caller before this point) gives per-edit dedup for free. Preserves the
        original event as metadata rather than rewriting any prior turn."""
        target_event_id = str(relates_to.get("event_id") or "")
        if not target_event_id:
            return
        new_content = source_content.get("m.new_content")
        if not isinstance(new_content, dict):
            return
        body = new_content.get("body", "") or ""
        if not body:
            return
        # Threaded edits mirror the thread relation into m.new_content (MSC2676); the top-level
        # relates_to on an edit is exclusively the m.replace pointer, never m.thread.
        new_relates_to = new_content.get("m.relates_to")
        if not isinstance(new_relates_to, dict):
            new_relates_to = {}
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, body, new_content, new_relates_to,
            metadata={"edited_message": True, "edited_message_original_id": target_event_id})
        if msg_event is None:
            return
        await self.handle_message(msg_event)
