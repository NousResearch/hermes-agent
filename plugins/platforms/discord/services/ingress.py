"""Discord message ingress and dispatch orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, MessageType
from .. import adapter as _adapter

logger = _adapter.logger
discord = _adapter.discord


class IngressMixin:
    """Normalize Discord messages and route them into the gateway."""
    async def _handle_message(
        self, message: DiscordMessage, role_authorized: bool = False, *, recovered: bool = False,
    ) -> bool:
        """Handle one Discord message and report whether it reached dispatch."""
        # Server channels (not DMs) require @mention unless free-response or an already-joined thread.
        #
        # Config (discord.* in config.yaml or DISCORD_* env vars):
        #   discord.require_mention: Require @mention in server channels (default: true)
        #   discord.free_response_channels: Channel IDs where bot responds without mention
        #   discord.ignored_channels: Channel IDs where bot NEVER responds (even when mentioned)
        #   discord.allowed_channels: If set, bot ONLY responds in these channels (whitelist)
        #   discord.no_thread_channels: Channel IDs where bot responds directly without creating thread
        #   discord.auto_thread: Auto-create thread on @mention in channels (default: true)
        thread_id = None
        parent_channel_id = None
        is_thread = isinstance(message.channel, discord.Thread)
        if is_thread:
            thread_id = str(message.channel.id)
            parent_channel_id = self._get_parent_channel_id(message.channel)
        is_voice_linked_channel = False
        # Save stripped text now: create_thread() can clobber message.content (breaks /command detection).
        raw_content = message.content.strip()
        normalized_content = raw_content
        mention_prefix = False
        snapshot_attachments = []
        if hasattr(message, "message_snapshots") and message.message_snapshots:
            snapshot_text_parts = []
            for snap in message.message_snapshots:
                if getattr(snap, "content", None):
                    snapshot_text_parts.append(snap.content.strip())
                snapshot_attachments.extend(getattr(snap, "attachments", []) or [])
            if snapshot_text_parts and not raw_content:
                raw_content = "\n".join(snapshot_text_parts)
                normalized_content = raw_content
        if self._self_is_explicitly_mentioned(message):
            mention_prefix = True
            if self._client.user:
                normalized_content = normalized_content.replace(f"<@{self._client.user.id}>", "").strip()
                normalized_content = normalized_content.replace(f"<@!{self._client.user.id}>", "").strip()
            message.content = normalized_content
        if not isinstance(message.channel, discord.DMChannel):
            channel_ids = {str(message.channel.id)}
            if parent_channel_id:
                channel_ids.add(parent_channel_id)
            channel_keys = self._discord_channel_keys(message, parent_channel_id)
            allowed_channels = self._get_allowed_channels()
            if allowed_channels:
                if "*" not in allowed_channels and not (channel_keys & allowed_channels):
                    logger.debug("[%s] Ignoring message in non-allowed channel: %s", self.name, channel_keys)
                    return False
            ignored_channels = self._get_ignored_channels()
            if "*" in ignored_channels or (channel_keys & ignored_channels):
                logger.debug("[%s] Ignoring message in ignored channel: %s", self.name, channel_keys)
                return False
            free_channels = self._discord_free_response_channels()
            require_mention = self._discord_require_mention()
            # Voice-linked text channel is free-response while voice is active (exact channel only).
            voice_linked_ids = {str(ch_id) for ch_id in self._voice_text_channels.values()}
            current_channel_id = str(message.channel.id)
            is_voice_linked_channel = current_channel_id in voice_linked_ids
            is_free_channel = (
                "*" in free_channels
                or bool(channel_keys & free_channels)
                or is_voice_linked_channel
            )
            in_bot_thread = self._in_bot_thread(message)
            if require_mention and not is_free_channel and not in_bot_thread:
                if not self._self_is_explicitly_mentioned(message) and not mention_prefix:
                    return False
        # Auto-thread: isolate each @mention in a text channel into its own thread (Slack-style).
        auto_threaded_channel = None
        if not is_thread and not isinstance(message.channel, discord.DMChannel):
            no_thread_channels = self._get_no_thread_channels()
            skip_thread = bool(channel_keys & no_thread_channels) or is_free_channel
            auto_thread = self._extra_or_env_flag("auto_thread", "DISCORD_AUTO_THREAD", "true", truthy=True)
            is_reply_message = getattr(message, "type", None) == discord.MessageType.reply
            if auto_thread and not skip_thread and not is_voice_linked_channel and not is_reply_message:
                thread = await self._auto_create_thread(message)
                if thread:
                    parent_channel_id = str(message.channel.id)
                    is_thread = True
                    thread_id = str(thread.id)
                    auto_threaded_channel = thread
                    self._threads.mark(thread_id)
                    # Pre-seed dedup: message.create_thread() fires a second MESSAGE_CREATE for the
                    # starter (id == thread.id, maybe type=default); mark it so it can't trigger a rerun.
                    self._dedup.is_duplicate(str(thread.id))
                else:
                    # Auto-threading is the routing target; do NOT fall back to an inline parent-channel
                    # reply (dumps the task into a shared channel). Surface an error and skip the run.
                    try:
                        # That breaks thread-first Discord workflows by dumping a new task into a shared
                        # channel. Surface a short visible error so the user can retry once Discord
                        # recovers, and skip agent invocation for this message. See #20243.
                        await message.channel.send(
                            "⚠️ Hermes could not create a Discord thread for "
                            "this message, so the request was not processed. Please retry."
                        )
                    except Exception as notify_error:
                        logger.warning(
                            "[%s] Failed to notify user of auto-thread failure: %s", self.name,
                            notify_error,
                        )
                    return False
        referenced_attachments = []
        reference = getattr(message, "reference", None)
        resolved_reference = getattr(reference, "resolved", None) if reference else None
        if resolved_reference is not None:
            referenced_attachments = list(getattr(resolved_reference, "attachments", []) or [])
        all_attachments = list(message.attachments) + snapshot_attachments + referenced_attachments
        if normalized_content.startswith("/"):
            msg_type = MessageType.COMMAND
        elif all_attachments:
            msg_type = self._attachment_message_type(all_attachments[0])
        else:
            msg_type = MessageType.TEXT
        effective_channel = auto_threaded_channel or message.channel
        if isinstance(message.channel, discord.DMChannel):
            chat_type = "dm"
            chat_name = message.author.name
        elif is_thread:
            chat_type = "thread"
            chat_name = self._format_thread_chat_name(effective_channel)
        else:
            chat_type = "group"
            chat_name = getattr(message.channel, "name", str(message.channel.id))
            if hasattr(message.channel, "guild") and message.channel.guild:
                chat_name = f"{message.channel.guild.name} / #{chat_name}"
        # Channel topic (TextChannels only); forum-parented threads inherit the parent topic.
        chat_topic = self._get_effective_topic(message.channel, is_thread=is_thread)
        guild = getattr(message, "guild", None)
        source = self.build_source(
            chat_id=str(effective_channel.id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(message.author.id),
            user_name=message.author.display_name,
            thread_id=thread_id,
            chat_topic=chat_topic,
            is_bot=getattr(message.author, "bot", False),
            guild_id=str(guild.id) if guild else None,
            parent_chat_id=parent_channel_id,
            message_id=str(message.id),
            role_authorized=role_authorized,
            auto_thread_created=auto_threaded_channel is not None,
            auto_thread_initial_name=(
                getattr(auto_threaded_channel, "_hermes_auto_thread_initial_name", None)
                or self._derive_auto_thread_name(message.content or "")
            ) if auto_threaded_channel is not None else None,
        )
        media_urls, media_types, pending_text_injection = await self._collect_attachment_media(all_attachments)
        event_text = normalized_content
        if pending_text_injection:
            event_text = f"{pending_text_injection}\n\n{event_text}" if event_text else pending_text_injection
        # ── History backfill ─────────────────────────────────────────
        # With require_mention, messages between bot turns never reach the transcript; fetch
        # history after the bot's last message (cold start: last N, stop at first self-message)
        # and prepend it. DMs skipped (every DM triggers the bot); in-flight arrivals not captured.
        _channel_context = None
        _is_dm = isinstance(message.channel, discord.DMChannel)
        if not _is_dm and self._discord_history_backfill():
            # Backfill on a gap: mention-gated channels, any thread (processing/restart gaps), any
            # reply (hydrate context around the referenced message). DMs/fresh auto-threads: nothing.
            _has_mention_gap = require_mention and not is_free_channel and not in_bot_thread
            _is_reply = message.reference is not None
            if (_has_mention_gap or is_thread or _is_reply) and auto_threaded_channel is None:
                _backfill_text = await self._fetch_channel_context(
                    message.channel, before=message,
                    reply_target=self._reply_target(message.reference) if _is_reply else None,
                )
                if _backfill_text:
                    _channel_context = _backfill_text
        # Keep empty user messages out of the session; with channel_context a bare mention = "catch me up".
        if (not event_text or not event_text.strip()) and not _channel_context:
            # Bare mention-only ping with no media/text/backfill: drop rather than spawn an empty turn.
            if (mention_prefix and not media_urls and not pending_text_injection):
                logger.info(
                    "[%s] Ignoring mention-only message from %s in %s", self.name,
                    getattr(message.author, "display_name", getattr(message.author, "name", "unknown")),
                    getattr(message.channel, "id", "unknown"),
                )
                return False
            event_text = "(The user sent a message with no text content)"
        _chan = message.channel
        _parent_id = str(getattr(_chan, "parent_id", "") or "")
        _chan_id = str(getattr(_chan, "id", ""))
        _skills = self._resolve_channel_skills(_chan_id, _parent_id or None)
        _channel_prompt = self._resolve_channel_prompt(_chan_id, _parent_id or None)
        reply_to_id = None
        reply_to_text = None
        if message.reference:
            reply_to_id = str(message.reference.message_id)
            if message.reference.resolved:
                reply_to_text = getattr(message.reference.resolved, "content", None) or None
        event = MessageEvent(
            text=event_text, message_type=msg_type, source=source, raw_message=message,
            message_id=str(message.id), media_urls=media_urls, media_types=media_types,
            reply_to_message_id=reply_to_id, reply_to_text=reply_to_text,
            timestamp=message.created_at, auto_skill=_skills, channel_prompt=_channel_prompt,
            channel_context=_channel_context,
        )
        # Track participation so follow-ups in this thread don't need @mention.
        if thread_id:
            self._threads.mark(thread_id)
        # Only live plain text is batched: recovery candidates are complete; coalescing would replay IDs.
        if (not recovered and msg_type == MessageType.TEXT and self._text_batch_delay_seconds > 0):
            self._enqueue_text_event(event)
        else:
            await self.handle_message(event)
        return True

    # ------------------------------------------------------------------
    # Text message aggregation (handles Discord client-side splits)
    # ------------------------------------------------------------------
