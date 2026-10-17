"""Matrix inbound room admission and context resolution.

Peer budgets belong to the receiving adapter and room; no cross-profile shared state.
"""
from __future__ import annotations

import logging
from typing import Optional

from gateway.config import PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.matrix.voice_mention import VoiceGate

logger = logging.getLogger(__name__)


class MatrixInboundMixin:
    def _matrix_reply_expected(
        self, is_dm: bool, is_mentioned: bool, body: str, relates_to: dict,
        reply_to_author_id: Optional[str],
    ) -> Optional[bool]:
        """Only clearly unaddressed room turns may silence; thread/reply intent stays unknown."""
        if is_dm or is_mentioned or body.startswith("/") or reply_to_author_id == self._user_id:
            return True
        if relates_to.get("rel_type") == "m.thread" or relates_to.get("m.in_reply_to"):
            return None
        return False

    def _configure_peer_budget(self, config: PlatformConfig) -> None:
        peer_agents_raw = config.extra.get("peer_agent_ids", [])
        if isinstance(peer_agents_raw, list):
            self._peer_agent_ids: set[str] = {
                str(user_id).strip()
                for user_id in peer_agents_raw
                if str(user_id).strip()
            }
        else:
            self._peer_agent_ids = {
                user_id.strip()
                for user_id in str(peer_agents_raw or "").split(",")
                if user_id.strip()
            }
        try:
            peer_budget = int(config.extra.get("peer_reply_budget_per_human_message", 0))
        except (TypeError, ValueError):
            peer_budget = 0
        self._peer_reply_budget_per_human_message = max(0, peer_budget)
        self._peer_reply_budget_remaining: dict[str, int] = {}

    def _admit_peer_room_message(
        self, source: SessionSource, is_mentioned: bool,
    ) -> bool:
        """Charge only authorized, directly mentioned peers after the ordinary inference gates."""
        if not self._peer_agent_ids:
            return True
        room_id, sender = source.chat_id, source.user_id
        authorized = self._is_authorized_user(
            sender, room_id, chat_type=source.chat_type, thread_id=source.thread_id, source=source)
        if sender not in self._peer_agent_ids:
            if authorized:
                self._peer_reply_budget_remaining[room_id] = self._peer_reply_budget_per_human_message
            return True
        if not authorized or not is_mentioned:
            return False
        remaining = self._peer_reply_budget_remaining.get(room_id, 0)
        if remaining <= 0:
            logger.info("Matrix: ignoring peer-agent message from %s in %s — reply budget exhausted", sender, room_id)
            return False
        self._peer_reply_budget_remaining[room_id] = remaining - 1
        return True

    async def _resolve_message_context(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict,
        relates_to: dict, mention_claimed: bool = False,
        voice_gate: Optional[VoiceGate] = None) -> Optional[tuple]:
        """Shared mention/thread/DM gating. Returns (body, is_dm, chat_type, thread_id,
        display_name, source) or None when the message should be dropped. ``mention_claimed``
        marks a parked voice claimed by the sender's follow-up bare @mention; ``voice_gate`` is
        the in-flight mark of a parkable voice, released once the park decision is made."""
        from plugins.platforms.matrix.adapter import _split_reply_fallback, _thread_root

        identity = await self._resolve_room_identity(room_id)
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"
        thread_id = _thread_root(relates_to)
        is_mentioned = mention_claimed or self._content_mentions_bot(body, source_content)
        is_peer_agent_message = sender in self._peer_agent_ids
        if not is_dm:
            # Whitelist first: non-listed rooms are dropped even when @mentioned (DMs exempt).
            if self._allowed_rooms and room_id not in self._allowed_rooms:
                logger.debug(
                    "Matrix: ignoring message %s in %s — room not in MATRIX_ALLOWED_ROOMS whitelist", event_id, room_id)
                return None
            is_free_room = room_id in self._free_rooms
            in_bot_thread = bool(thread_id and thread_id in self._threads)
            if self._require_mention and not is_free_room and not in_bot_thread:
                if not is_mentioned and not body.startswith("/"):
                    if voice_gate is not None:  # parkable voice: a bare @mention may follow (Element X)
                        self._parked_voices.park(room_id, sender, voice_gate, event_id, source_content, relates_to)
                    logger.debug(
                        "Matrix: ignoring message %s in %s — no @mention "
                        "(set MATRIX_REQUIRE_MENTION=false to disable)", event_id, room_id)
                    return None
            # Thread-level @mention gating: even in a bot-participated thread.
            elif self._thread_require_mention and in_bot_thread and not is_free_room and not is_mentioned:
                logger.debug(
                    "Matrix: ignoring message %s in thread %s — no @mention (thread_require_mention=true)",
                    event_id, thread_id)
                return None
        if is_mentioned and (self._require_mention or is_peer_agent_message):
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
            guild_id=identity.server_name, parent_chat_id=room_id if thread_id else None, message_id=event_id)
        # Authorize the actual event source: every route discriminator and transport owner travels with it.
        if not is_dm and not self._admit_peer_room_message(source, is_mentioned):
            return None
        if voice_gate is not None:  # decided (parked or passing): don't hold bare mentions any longer
            self._parked_voices.release(room_id, sender, voice_gate)
        if thread_id:
            await self._threads.mark_async(thread_id)  # covers real roots and synthetic ones alike
        self._background_read_receipt(room_id, event_id)
        return body, is_dm, chat_type, thread_id, display_name, source
