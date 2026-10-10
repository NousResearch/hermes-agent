"""Matrix inbound room, thread and source context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from plugins.platforms.matrix.permalinks import event_permalink, room_via_servers
from plugins.platforms.matrix.voice_mention import VoiceGate

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixInboundContextMixin:
    async def _resolve_message_context(
        self: MatrixAdapter,
        room_id: str,
        sender: str,
        event_id: str,
        body: str,
        source_content: dict,
        relates_to: dict,
        mention_claimed: bool = False,
        voice_gate: Optional[VoiceGate] = None,
    ) -> Optional[tuple]:
        """Shared mention/thread/DM gating. Returns (body, is_dm, chat_type, thread_id,
        display_name, source) or None when the message should be dropped. ``mention_claimed``
        marks a parked voice claimed by the sender's follow-up bare @mention; ``voice_gate`` is
        the in-flight mark of a parkable voice, released once the park decision is made."""
        from plugins.platforms.matrix.adapter import _thread_root, _split_reply_fallback

        identity = await self._resolve_room_identity(room_id)
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"
        thread_id = _thread_root(relates_to)
        is_mentioned = mention_claimed or self._content_mentions_bot(
            body, source_content
        )
        if not is_dm:
            # Whitelist first: non-listed rooms are dropped even when @mentioned (DMs exempt).
            if self._allowed_rooms and room_id not in self._allowed_rooms:
                logger.debug(
                    "Matrix: ignoring message %s in %s — room not in MATRIX_ALLOWED_ROOMS whitelist",
                    event_id,
                    room_id,
                )
                return None
            is_free_room = room_id in self._free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._threads)
            if self._require_mention and not is_free_room and not in_bot_thread:
                if not is_mentioned and not body.startswith("/"):
                    if (
                        voice_gate is not None
                    ):  # parkable voice: a bare @mention may follow (Element X)
                        self._parked_voices.park(
                            room_id,
                            sender,
                            voice_gate,
                            event_id,
                            source_content,
                            relates_to,
                        )
                    logger.debug(
                        "Matrix: ignoring message %s in %s — no @mention "
                        "(set MATRIX_REQUIRE_MENTION=false to disable)",
                        event_id,
                        room_id,
                    )
                    return None
            # thread_require_mention: even inside a bot thread require @mention — prevents
            # infinite reply loops when several bots share one thread.
            elif (
                self._thread_require_mention
                and in_bot_thread
                and not is_free_room
                and not is_mentioned
            ):
                logger.debug(
                    "Matrix: ignoring message %s in thread %s — no @mention (thread_require_mention=true)",
                    event_id,
                    thread_id,
                )
                return None
        if is_mentioned and self._require_mention:
            # Strip the mention from the reply text only: the quote block carries the
            # ``> <@bot:srv> ...`` reply pill, which _extract_reply_context parses later
            # for reply_to_author_id. A whole-body replace rewrote the pill to ``> <>``
            # and silently dropped the replied-to author (#111233). Only a real reply carries a
            # pill; a hand-typed blockquote in a plain message is stripped whole as before.
            if relates_to.get("m.in_reply_to"):
                quote_block, reply_text = _split_reply_fallback(body)
                body = quote_block + self._strip_mention(reply_text)
            else:
                body = self._strip_mention(body)
        # Real thread roots are preserved above; synthetic roots (this event) follow policy: DM
        # @mention threads / DM auto-thread, or room auto-thread unless session_scope pins the room.
        if not thread_id:
            if is_dm:
                synthetic = (
                    self._dm_mention_threads and is_mentioned
                ) or self._dm_auto_thread
            else:
                synthetic = self._matrix_session_scope == "thread" or (
                    self._matrix_session_scope != "room" and self._auto_thread
                )
            if synthetic:
                thread_id = event_id
        if (
            voice_gate is not None
        ):  # decided (parked or passing): don't hold bare mentions any longer
            self._parked_voices.release(room_id, sender, voice_gate)
        display_name = await self._get_display_name(room_id, sender)
        policy = await self._permalink_routing.resolve(self._client, room_id)
        via = await room_via_servers(
            getattr(self._client, "state_store", None),
            room_id,
            ((self._user_id or "").partition(":")[2], identity.server_name),
            policy=policy,
        )
        source = self.build_source(
            chat_id=room_id,
            chat_name=identity.display_name,
            chat_type=chat_type,
            user_id=sender,
            user_name=display_name,
            thread_id=thread_id,
            chat_topic=identity.room_topic,
            guild_id=identity.server_name,
            parent_chat_id=room_id if thread_id else None,
            message_id=event_id,
            source_permalink=event_permalink(room_id, event_id, via),
        )
        if thread_id:
            await self._threads.mark_async(
                thread_id
            )  # covers real roots and synthetic ones alike
        self._background_read_receipt(room_id, event_id)
        return body, is_dm, chat_type, thread_id, display_name, source
