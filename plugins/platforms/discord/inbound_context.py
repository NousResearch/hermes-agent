"""Inbound text and context preparation for the DiscordAdapter adapter."""

from __future__ import annotations

from gateway.platforms.event import attributed_context

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from plugins.platforms.discord.adapter import DiscordMessage

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.platforms.base_pending import can_join_pending_event

if TYPE_CHECKING:
    from plugins.platforms.discord.adapter import DiscordAdapter

logger = logging.getLogger("plugins.platforms.discord.adapter")


@dataclass(frozen=True)
class DiscordPreparedInput:
    event: MessageEvent
    forwarded_text: str


class DiscordInboundContextMixin:
    def _discord_source_admission(self: DiscordAdapter, message: DiscordMessage) -> tuple[bool, bool]:
        from plugins.platforms.discord.adapter import discord, _scoped_gate_env

        if message.author == self._client.user:
            return False, False
        if message.type not in {discord.MessageType.default, discord.MessageType.reply}:
            return False, False
        role_authorized = False
        if getattr(message.author, "bot", False):
            allow_bots = self._get_allow_bots()
            bot_tag_continuation = self._is_bot_tag_debounce_continuation(message)
            if allow_bots == "none":
                return False, False
            if (
                allow_bots == "mentions"
                and not self._self_is_explicitly_mentioned(message)
                and not bot_tag_continuation
            ):
                return False, False
            if (
                self._discord_bots_require_inline_mention()
                and not self._self_is_raw_mentioned(message)
                and not bot_tag_continuation
            ):
                return False, False
        else:
            msg_guild = getattr(message, "guild", None)
            is_dm = isinstance(message.channel, discord.DMChannel) or msg_guild is None
            msg_channel_ids = None
            if not is_dm:
                msg_channel_ids = {str(message.channel.id)}
                parent_id = self._get_parent_channel_id(message.channel)
                if parent_id:
                    msg_channel_ids.add(parent_id)
            if not self._is_allowed_user(
                str(message.author.id), message.author, guild=msg_guild, is_dm=is_dm,
                channel_ids=msg_channel_ids,
            ):
                self._warn_if_fail_closed_default()
                return False, False
            role_authorized = bool(getattr(self, "_allowed_role_ids", set()))
        raw_self_mention = self._self_is_explicitly_mentioned(message)
        if not isinstance(message.channel, discord.DMChannel) and (
            message.mentions or raw_self_mention
        ):
            other_bots_mentioned = any(
                mentioned.bot and mentioned != self._client.user
                for mentioned in message.mentions
            )
            if other_bots_mentioned and not raw_self_mention:
                return False, False
            ignore_no_mention = _scoped_gate_env("DISCORD_IGNORE_NO_MENTION", "true").lower() in {"true", "1", "yes"}
            if ignore_no_mention and not raw_self_mention and not other_bots_mentioned:
                # A thread the bot joined is not someone else's conversation, and the other two
                # ingress paths already exempt it: _dispatch_recovered_message() and
                # _handle_message(). Admission runs on both and can veto what they admit, so
                # without this a third-party mention in a bot thread is dropped here even though
                # the same message with no mention at all is admitted. ``thread_require_mention``
                # still gates multi-bot threads, inside _in_bot_thread().
                if not self._in_bot_thread(message):
                    parent_id = None
                    if hasattr(message.channel, "parent_id") and message.channel.parent_id:
                        parent_id = str(message.channel.parent_id)
                    free_channels = self._discord_free_response_channels()
                    channel_keys = self._discord_channel_keys(message, parent_id)
                    if "*" not in free_channels and not (channel_keys & free_channels):
                        # Every other silent return in this function is at least guessable from
                        # the outside; this one is not, and an operator seeing no log line cannot
                        # tell it apart from the gateway never receiving the event.
                        logger.debug(
                            "[%s] admission: dropping message %s — mentions others, not self, "
                            "not a bot thread, channel not free-response",
                            self.name, getattr(message, "id", "?"))
                        return False, False
        return True, role_authorized

    async def _handle_message(
        self: DiscordAdapter, message: DiscordMessage, role_authorized: bool = False, *, recovered: bool = False,
    ) -> bool:
        """Handle one Discord message and report whether it reached dispatch."""
        prepared = await self._prepare_inbound_event(message, role_authorized, recovered=recovered)
        if prepared is None:
            return False
        event = prepared.event
        # Only live plain text is batched: recovery candidates are complete; coalescing would replay IDs.
        if (not recovered and event.message_type == MessageType.TEXT and self._text_batch_delay_seconds > 0):
            if (getattr(event, "_discord_history_backfill_prepared", False)
                    and self._batch_has_history_context(event, recovered=recovered)):
                event.channel_context = attributed_context("Forwarded message", prepared.forwarded_text) if prepared.forwarded_text else None
                delattr(event, "_discord_history_backfill_prepared")
            self._enqueue_text_event(event)
            if getattr(event, "_discord_history_backfill_prepared", False):
                pending = self._pending_text_batches.get(self._text_batch_key(event))
                if pending is not None:
                    setattr(pending, "_discord_history_backfill_prepared", True)
        else:
            await self.handle_message(event)
        return True


    async def _prepare_inbound_event(
        self: DiscordAdapter, message: DiscordMessage, role_authorized: bool = False, *, recovered: bool = False,
        cached: MessageEvent | None = None, restored_channel: Any = None,
        authorize: Callable[[SessionSource], bool] | None = None,
    ) -> DiscordPreparedInput | None:
        """Prepare native input under current channel and mention policy."""
        from plugins.platforms.discord.adapter import MessageEvent, MessageType, discord, t

        # Server channels (not DMs) require @mention unless free-response or an already-joined thread.
        #
        # Config (discord.* in config.yaml or DISCORD_* env vars):
        #   discord.require_mention: Require @mention in server channels (default: true)
        #   discord.free_response_channels: Channel IDs where bot responds without mention
        #   discord.ignored_channels: Channel IDs where bot NEVER responds (even when mentioned)
        #   discord.allowed_channels: If set, bot ONLY responds in these channels (whitelist)
        #   discord.no_thread_channels: Channel IDs where bot responds directly without creating thread
        #   discord.auto_thread: Auto-create thread on @mention in channels (default: true)
        #   discord.free_response_auto_thread: Free-response channels also auto-thread (default: false)
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
        snapshot_text_parts = []
        if hasattr(message, "message_snapshots") and message.message_snapshots:
            for snap in message.message_snapshots:
                if getattr(snap, "content", None):
                    snapshot_text_parts.append(snap.content.strip())
                snapshot_attachments.extend(getattr(snap, "attachments", []) or [])
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
                    return None
            ignored_channels = self._get_ignored_channels()
            if "*" in ignored_channels or (channel_keys & ignored_channels):
                logger.debug("[%s] Ignoring message in ignored channel: %s", self.name, channel_keys)
                return None
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
                if (
                    not self._self_is_explicitly_mentioned(message)
                    and not mention_prefix
                    and not self._is_bot_tag_debounce_continuation(message)
                ):
                    return None
        # Auto-thread: isolate each @mention in a text channel into its own thread (Slack-style).
        auto_threaded_channel = restored_channel
        if restored_channel is not None:
            parent_channel_id = str(message.channel.id)
            is_thread = True
            thread_id = str(restored_channel.id)
        if not is_thread and not isinstance(message.channel, discord.DMChannel):
            no_thread_channels = self._get_no_thread_channels()
            # Voice-linked and reply exclusions live in the auto-thread gate below, not in skip_thread.
            skip_thread = bool(channel_keys & no_thread_channels) or (
                is_free_channel and not self._discord_free_response_auto_thread()
            )
            auto_thread = self._extra_or_env_flag("auto_thread", "DISCORD_AUTO_THREAD", "true", truthy=True)
            is_reply_message = getattr(message, "type", None) == discord.MessageType.reply
            if auto_thread and not skip_thread and not is_voice_linked_channel and not is_reply_message:
                if cached is not None:
                    return None
                thread = await self._auto_create_thread(message)
                if thread:
                    parent_channel_id = str(message.channel.id)
                    is_thread = True
                    thread_id = str(thread.id)
                    # Pre-seed dedup: message.create_thread() fires a second MESSAGE_CREATE for the
                    # starter (id == thread.id, maybe type=default); mark it so it can't trigger a rerun.
                    # Must run before the first await below: mark_async yields to the loop, and the
                    # echo's _discord_message_admission would otherwise claim the id first.
                    self._dedup.is_duplicate(str(thread.id))
                    auto_threaded_channel = thread
                    await self._threads.mark_async(thread_id)
                else:
                    # Auto-threading is the routing target; do NOT fall back to an inline parent-channel
                    # reply (dumps the task into a shared channel). Surface an error and skip the run.
                    try:
                        # That breaks thread-first Discord workflows by dumping a new task into a shared
                        # channel. Surface a short visible error so the user can retry once Discord
                        # recovers, and skip agent invocation for this message. See #20243.
                        await message.channel.send(
                            self.warning_text(
                                t("platform.discord.thread.auto_create_failed"),
                                t("platform.discord.thread.auto_create_failed_generic"))
                        )
                    except Exception as notify_error:
                        logger.warning(
                            "[%s] Failed to notify user of auto-thread failure: %s", self.name,
                            notify_error,
                        )
                    return None
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
        # The session channel's topic, as the auto-thread's next message reads it; forum threads inherit it.
        chat_topic = self._get_effective_topic(effective_channel, is_thread=is_thread)
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
        if authorize is not None:
            if self._canonicalize(source) is None or not authorize(source):
                return None
        # Forwarded and replied-to attachments are other people's files, so their inlined text
        # goes with the forward or the reply, never into the sender's text.
        media_urls, media_types, media_text_inlined = [], [], []
        injections = []
        if cached is None:
            for attachments in (list(message.attachments), snapshot_attachments, referenced_attachments):
                urls, types, inlined, injection = await self._collect_attachment_media(attachments)
                media_urls += urls
                media_types += types
                media_text_inlined += inlined
                injections.append(injection)
        else:
            media_urls = list(cached.media_urls)
            media_types = list(cached.media_types)
            media_text_inlined = list(cached.media_text_inlined)
            injections = [None, None, None]
        pending_text_injection, snapshot_injection, referenced_injection = injections
        event_text = normalized_content
        if pending_text_injection:
            event_text = f"{pending_text_injection}\n\n{event_text}" if event_text else pending_text_injection
        forwarded_text = "\n\n".join(part for part in ("\n".join(snapshot_text_parts), snapshot_injection) if part)
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
            if referenced_injection:
                reply_to_text = f"{reply_to_text}\n\n{referenced_injection}" if reply_to_text else referenced_injection
        event = MessageEvent(
            text=event_text, message_type=msg_type, source=source, raw_message=message,
            message_id=str(message.id), media_urls=media_urls, media_types=media_types,
            media_text_inlined=media_text_inlined,
            reply_to_message_id=reply_to_id, reply_to_text=reply_to_text,
            timestamp=message.created_at, auto_skill=_skills, channel_prompt=_channel_prompt,
        )
        # ── History backfill ─────────────────────────────────────────
        # With require_mention, messages between bot turns never reach the transcript; fetch
        # history after the bot's last message (cold start: last N, stop at first self-message)
        # and prepend it. DMs skipped (every DM triggers the bot); in-flight arrivals not captured.
        _is_dm = isinstance(message.channel, discord.DMChannel)
        if not _is_dm and self._discord_history_backfill() and not self._batch_has_history_context(event, recovered=recovered):
            # Backfill on a gap: mention-gated channels, any thread (processing/restart gaps), any
            # reply (hydrate context around the referenced message). DMs/fresh auto-threads: nothing.
            _has_mention_gap = require_mention and not is_free_channel and not in_bot_thread
            _is_reply = message.reference is not None
            if (_has_mention_gap or is_thread or _is_reply) and auto_threaded_channel is None:
                batch = self._pending_history_batch(event, recovered=recovered)
                _backfill_text = await self._fetch_channel_context(
                    message.channel, before=batch.raw_message if batch is not None else message,
                    reply_target=self._reply_target(message.reference) if _is_reply else None,
                )
                if _backfill_text:
                    if not self._batch_has_history_context(event, recovered=recovered):
                        event.channel_context = _backfill_text
                        setattr(event, "_discord_history_backfill_prepared", True)
        if (not event.text or not event.text.strip()) and not event.channel_context and not forwarded_text:
            if mention_prefix and not media_urls and not pending_text_injection:
                logger.info(
                    "[%s] Ignoring mention-only message from %s in %s", self.name,
                    getattr(message.author, "display_name", getattr(message.author, "name", "unknown")),
                    getattr(message.channel, "id", "unknown"),
                )
                return None
            event.text = "(The user sent a message with no text content)"
        if forwarded_text:
            event.add_channel_context(attributed_context("Forwarded message", forwarded_text))
        if (
            getattr(getattr(message, "author", None), "bot", False)
            and self._is_bot_tag_debounce_continuation(message)
        ):
            event._bot_tag_debounce = True  # type: ignore[attr-defined]

        # Track participation so follow-ups in this thread don't need @mention.
        if thread_id:
            await self._threads.mark_async(thread_id)
        if cached is None:
            event._pending_native_input = self.pending_native_input(event)
        return DiscordPreparedInput(event, forwarded_text)


    def _pending_history_batch(self: DiscordAdapter, event: MessageEvent, *, recovered: bool) -> MessageEvent | None:
        if recovered or event.message_type != MessageType.TEXT or self._text_batch_delay_seconds <= 0:
            return None
        pending = self._pending_text_batches.get(self._text_batch_key(event))
        return pending if pending is not None and can_join_pending_event(pending, event) else None

    def _batch_has_history_context(self: DiscordAdapter, event: MessageEvent, *, recovered: bool) -> bool:
        pending = self._pending_history_batch(event, recovered=recovered)
        return pending is not None and bool(getattr(pending, "_discord_history_backfill_prepared", False))
