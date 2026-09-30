"""Matrix reply fallback parsing and inbound conversation context."""

from __future__ import annotations

import logging
import re
from typing import Optional

from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.voice_mention import VoiceGate

logger = logging.getLogger("plugins.platforms.matrix.adapter")


_MATRIX_REPLY_FALLBACK_PILL_RE = re.compile(r"^> (?:\* )?<(@[^>\s]+)>\s*(.*)")

def _extract_reply_fallback(body: str) -> tuple[Optional[str], Optional[str]]:
    """Return (quoted_text, author_mxid) from the inline reply fallback; author from the first-line pill."""
    if not body or not body.startswith("> "):
        return None, None
    quoted_lines: list[str] = []
    author_id: Optional[str] = None
    for line in body.split("\n"):
        if not line.startswith("> "):
            break
        content = line[2:]
        if author_id is None:
            pill_match = _MATRIX_REPLY_FALLBACK_PILL_RE.match(line)
            if pill_match:
                author_id = pill_match.group(1)
                content = pill_match.group(2)  # drop the pill from the visible quote
        quoted_lines.append(content)
    quoted_text = "\n".join(quoted_lines).strip() or None
    return quoted_text, author_id

def _strip_reply_fallback(body: str) -> str:
    """Strip the inline ``> quote\\n\\nreply`` fallback prefix; unchanged if absent."""
    if not body or not body.startswith("> "):
        return body
    stripped = []
    past_fallback = False
    for line in body.split("\n"):
        if not past_fallback:
            if line.startswith("> ") or line == ">":
                continue
            past_fallback = True
            if line == "":
                continue
        stripped.append(line)
    return "\n".join(stripped) if stripped else body

def _split_reply_fallback(body: str) -> tuple[str, str]:
    """Split ``> quote\\n\\nreply`` into ``(quote_block, reply_text)``; ``("", body)`` when absent.

    The two halves always concatenate back to *body* verbatim (``quote + reply == body``), so
    callers can transform one half and rebuild the body without disturbing the other. The blank
    separator line belongs to the quote block. Used to keep the ``> <@user:srv>`` reply pill —
    the only mention text in a reply-to-the-bot — out of whole-body rewrites.
    """
    if not body or not body.startswith("> "):
        return "", body
    lines = body.split("\n")
    idx = 0
    while idx < len(lines) and (lines[idx].startswith("> ") or lines[idx] == ">"):
        idx += 1
    if idx < len(lines) and lines[idx] == "":
        idx += 1  # the blank line separating the quote from the reply belongs to the quote
    head = "\n".join(lines[:idx])
    return (head, "") if idx >= len(lines) else (head + "\n", "\n".join(lines[idx:]))



def _has_reply_fallback(body: str, content: dict) -> bool:
    """Whether a reply's body starts with a legacy reply fallback instead of the user's own quote.

    Matrix 1.13 (MSC2781) removed reply fallbacks, so a modern client sends the reply as typed
    and a leading ``> `` block is the user's quotation. A legacy client marks its fallback with
    an ``<mx-reply>`` element at the start of the HTML body. Its plain fallback starts with the
    quoted sender's pill (``> <@user:srv>``, or ``> * <@user:srv>`` for an emote) and ends with
    a blank line.
    """
    if not body.startswith("> "):
        return False
    formatted_body = content.get("formatted_body")
    if (content.get("format") == "org.matrix.custom.html" and isinstance(formatted_body, str)
            and formatted_body.lstrip().startswith("<mx-reply>")):
        return True
    if not _MATRIX_REPLY_FALLBACK_PILL_RE.match(body):
        return False
    quote_block, reply_text = _split_reply_fallback(body)
    return quote_block.endswith("\n\n") or not reply_text


class MatrixReplyContextMixin:
    async def _resolve_message_context(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict,
        relates_to: dict, mention_claimed: bool = False,
        voice_gate: Optional[VoiceGate] = None) -> Optional[tuple]:
        """Shared mention/thread/DM gating. Returns (body, is_dm, chat_type, thread_id,
        display_name, source) or None when the message should be dropped. ``mention_claimed``
        marks a parked voice claimed by the sender's follow-up bare @mention; ``voice_gate`` is
        the in-flight mark of a parkable voice, released once the park decision is made."""
        from plugins.platforms.matrix.adapter import _thread_root

        identity = await self._resolve_room_identity(room_id)
        is_dm = await self._is_dm_room(room_id)
        chat_type = "dm" if is_dm else "group"
        thread_id = _thread_root(relates_to)
        is_mentioned = mention_claimed or self._content_mentions_bot(body, source_content)
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
            # thread_require_mention: even inside a bot thread require @mention — prevents
            # infinite reply loops when several bots share one thread.
            elif self._thread_require_mention and in_bot_thread and not is_free_room and not is_mentioned:
                logger.debug(
                    "Matrix: ignoring message %s in thread %s — no @mention (thread_require_mention=true)",
                    event_id, thread_id)
                return None
        if is_mentioned and self._require_mention:
            # Preserve the sender pill in the leading quote for reply-context extraction.
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
        if voice_gate is not None:  # decided (parked or passing): don't hold bare mentions any longer
            self._parked_voices.release(room_id, sender, voice_gate)
        display_name = await self._get_display_name(room_id, sender)
        source = self.build_source(
            chat_id=room_id, chat_name=identity.display_name, chat_type=chat_type, user_id=sender,
            user_name=display_name, thread_id=thread_id, chat_topic=identity.room_topic,
            guild_id=identity.server_name, parent_chat_id=room_id if thread_id else None, message_id=event_id)
        if thread_id:
            await self._threads.mark_async(thread_id)  # covers real roots and synthetic ones alike
        self._background_read_receipt(room_id, event_id)
        return body, is_dm, chat_type, thread_id, display_name, source

    async def _extract_reply_context(
        self, room_id: str, body: str, source_content: dict, relates_to: dict
    ) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Return (body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name). Captures
        the inline reply fallback (``> <@user:srv> text\\n\\nreply``) BEFORE stripping it, so the
        prompt layer can render "[Replying to: ...]" like Signal/Slack/Telegram."""
        reply_to = (relates_to.get("m.in_reply_to") or {}).get("event_id")
        reply_to_text = reply_to_author_id = reply_to_author_name = None
        if reply_to and _has_reply_fallback(body, source_content):
            reply_to_text, reply_to_author_id = _extract_reply_fallback(body)
            body = _strip_reply_fallback(body)
            # Resolve the replied-to author's display name (falls back to localpart).
            if reply_to_author_id:
                reply_to_author_name = await self._get_display_name(room_id, reply_to_author_id)
        return body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name

    async def _build_inbound_event(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict, relates_to: dict,
        ctx: Optional[tuple] = None, **extra) -> Optional[MessageEvent]:
        """Gate + normalise an inbound event into a MessageEvent (None => drop). Text body may
        still change (reply-fallback strip); ``extra`` carries media fields / message_type.
        ``ctx`` is a pre-resolved ``_resolve_message_context`` result (media path gates before
        downloading); resolving it twice would double the read receipt / thread mark."""
        from plugins.platforms.matrix.adapter import _is_bare_media_filename, _normalize_matrix_bang_command

        if ctx is None:
            ctx = await self._resolve_message_context(room_id, sender, event_id, body, source_content, relates_to)
        if ctx is None:
            return None
        body, _is_dm, _chat_type, _thread_id, display_name, source = ctx
        body, reply_to, reply_to_text, reply_to_author_id, reply_to_author_name = (
            await self._extract_reply_context(room_id, body, source_content, relates_to))
        media_msgtype = extra.pop("media_msgtype", None)
        if media_msgtype is None:
            # Re-normalize after reply stripping so ``> quoted\n\n!model`` is still a command.
            body = _normalize_matrix_bang_command(body)
            extra["message_type"] = MessageType.COMMAND if body.startswith("/") else MessageType.TEXT
        elif _is_bare_media_filename(media_msgtype, body):
            body = ""  # transport filename, not user text
        return MessageEvent(
            text=body, source=source, raw_message=source_content, message_id=event_id,
            reply_to_message_id=reply_to, reply_to_text=reply_to_text, reply_to_author_id=reply_to_author_id,
            reply_to_author_name=reply_to_author_name,
            # Top-level sender fields mirror source.* — downstream prompt code reads them.
            user_id=sender, user_name=display_name, **extra)
