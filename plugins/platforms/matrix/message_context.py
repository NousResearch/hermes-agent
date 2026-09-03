"""Inbound Matrix message-context resolution."""

from typing import Any, Optional


async def resolve_message_context(
    self: Any,
    room_id: str,
    sender: str,
    event_id: str,
    body: str,
    source_content: dict,
    relates_to: dict,
    *,
    logger: Any,
) -> Optional[tuple]:
    """Resolve DM, mention, thread, and trust context for an inbound message."""
    identity = await self._resolve_room_identity(room_id)
    is_dm = await self._is_dm_room(room_id)
    chat_type = "dm" if is_dm else "group"

    thread_id = None
    if relates_to.get("rel_type") == "m.thread":
        thread_id = relates_to.get("event_id")

    formatted_body = source_content.get("formatted_body")
    # m.mentions.user_ids (MSC3952 / Matrix v1.7) — authoritative mention signal.
    mentions_block = source_content.get("m.mentions") or {}
    mention_user_ids = (
        mentions_block.get("user_ids") if isinstance(mentions_block, dict) else None
    )
    is_mentioned = self._is_bot_mentioned(body, formatted_body, mention_user_ids)

    # Require-mention gating.
    if not is_dm:
        # allowed_rooms check (whitelist — must pass before other gating).
        # When set, messages from rooms NOT in this whitelist are silently
        # ignored, even if @mentioned.  DMs are already excluded above.
        if self._allowed_rooms and room_id not in self._allowed_rooms:
            logger.debug(
                "Matrix: ignoring message %s in %s — room not in "
                "MATRIX_ALLOWED_ROOMS whitelist",
                event_id,
                room_id,
            )
            return None

        is_free_room = room_id in self._free_rooms
        in_bot_thread = bool(thread_id and thread_id in self._threads)
        is_command = body.startswith("/")
        if self._require_mention and not is_free_room and not in_bot_thread:
            if not is_mentioned and not is_command:
                logger.debug(
                    "Matrix: ignoring message %s in %s — no @mention "
                    "(set MATRIX_REQUIRE_MENTION=false to disable)",
                    event_id,
                    room_id,
                )
                return None

        # Thread-level @mention gating: even in a bot-participated thread,
        # require @mention when thread_require_mention is enabled.
        # Prevents infinite reply loops in multi-agent shared rooms
        # where multiple bots all participate in the same thread.
        elif self._thread_require_mention and in_bot_thread and not is_free_room:
            if not is_mentioned:
                logger.debug(
                    "Matrix: ignoring message %s in thread %s — "
                    "no @mention (thread_require_mention=true)",
                    event_id,
                    thread_id,
                )
                return None

    # DM mention-thread.
    if is_dm and not thread_id and self._dm_mention_threads and is_mentioned:
        thread_id = event_id
        self._threads.mark(thread_id)

    # Strip mention from body (only when mention-gating is active).
    if is_mentioned and self._require_mention:
        body = self._strip_mention(body)

    # Auto-thread/session-scope policy. Real Matrix thread roots are
    # preserved above; synthetic thread roots are policy-driven.
    if not thread_id:
        if is_dm:
            if self._dm_auto_thread:
                thread_id = event_id
                self._threads.mark(thread_id)
        elif self._matrix_session_scope == "room":
            thread_id = None
        elif self._matrix_session_scope == "thread":
            thread_id = event_id
            self._threads.mark(thread_id)
        elif self._auto_thread:
            thread_id = event_id
            self._threads.mark(thread_id)

    display_name = await self._get_display_name(room_id, sender)
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
        is_bot=bool(sender and sender == self._user_id),
    )
    joined_member_count = getattr(identity, "joined_member_count", None)
    source.is_one_to_one = bool(
        chat_type == "dm"
        and joined_member_count is not None
        and joined_member_count <= 2
    )
    source.message_is_edit = False

    if thread_id:
        self._threads.mark(thread_id)

    self._background_read_receipt(room_id, event_id)

    return body, is_dm, chat_type, thread_id, display_name, source
