"""Discord inbound methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Optional
from gateway.platforms.base import MessageType
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any


class DiscordInboundMixin:
    def _get_parent_channel_id(self, channel: Any) -> Optional[str]:
        """Return the parent channel ID for a Discord thread-like channel, if present."""
        from . import adapter as _adapter

        parent = getattr(channel, "parent", None)
        if parent is not None and getattr(parent, "id", None) is not None:
            return str(parent.id)
        parent_id = getattr(channel, "parent_id", None)
        if parent_id is not None:
            return str(parent_id)
        return None

    def _is_forum_parent(self, channel: Any) -> bool:
        """Best-effort check for whether a Discord channel is a forum channel."""
        from . import adapter as _adapter

        if channel is None:
            return False
        forum_cls = getattr(_adapter.discord, "ForumChannel", None)
        if forum_cls and isinstance(channel, forum_cls):
            return True
        channel_type = getattr(channel, "type", None)
        if channel_type is not None:
            type_value = getattr(channel_type, "value", channel_type)
            if type_value == 15:
                return True
        return False

    def _get_effective_topic(self, channel: Any, is_thread: bool = False) -> Optional[str]:
        """Return the channel topic, falling back to the parent forum's topic for forum threads."""
        topic = getattr(channel, "topic", None)
        if not topic and is_thread:
            parent = getattr(channel, "parent", None)
            if parent and self._is_forum_parent(parent):
                topic = getattr(parent, "topic", None)
        return topic

    def _format_thread_chat_name(self, thread: Any) -> str:
        """Build a readable chat name for thread-like Discord channels, including forum context when available."""
        from . import adapter as _adapter

        thread_name = getattr(thread, "name", None) or str(getattr(thread, "id", "thread"))
        parent = getattr(thread, "parent", None)
        guild = getattr(thread, "guild", None) or getattr(parent, "guild", None)
        guild_name = getattr(guild, "name", None)
        parent_name = getattr(parent, "name", None)
        if self._is_forum_parent(parent) and guild_name and parent_name:
            return f"{guild_name} / {parent_name} / {thread_name}"
        if parent_name and guild_name:
            return f"{guild_name} / #{parent_name} / {thread_name}"
        if parent_name:
            return f"{parent_name} / {thread_name}"
        return thread_name

    async def _read_attachment_bytes(self, att, *, media_type: str = "media") -> Optional[bytes]:
        """Read an attachment via the authenticated bot session; ``None`` (no callable ``read()``
        or read failure) means fall back to the URL downloaders. Raises ``ValueError`` for oversized
        attachments BEFORE pulling bytes when Discord reports the size, so a hostile upload can't OOM."""
        from . import adapter as _adapter

        attachment_size = getattr(att, "size", None)
        if attachment_size:
            _adapter.validate_inbound_media_size(int(attachment_size), media_type=media_type)
        reader = getattr(att, "read", None)
        if reader is None or not callable(reader):
            return None
        try:
            raw_bytes = await reader()
        except Exception as e:
            _adapter.logger.warning(
                "[Discord] Authenticated attachment read failed for %s: %s",
                getattr(att, "filename", None) or getattr(att, "url", "<unknown>"), e,
            )
            return None
        _adapter.validate_inbound_media_size(len(raw_bytes), media_type=media_type)
        return raw_bytes

    async def _cache_discord_image(self, att, ext: str) -> str:
        """Cache an image attachment locally: ``att.read()`` first, SSRF-gated URL fallback."""
        from . import adapter as _adapter

        raw_bytes = await self._read_attachment_bytes(att, media_type="image")
        if raw_bytes is not None:
            try:
                return await _adapter.cache_image_from_bytes_async(raw_bytes, ext=ext)
            except Exception as e:
                _adapter.logger.debug(
                    "[Discord] cache_image_from_bytes rejected att.read() data; falling back to URL: %s",
                    e,
                )
        return await _adapter.cache_image_from_url(att.url, ext=ext)

    async def _cache_discord_audio(self, att, ext: str) -> str:
        """Cache an audio attachment locally: ``att.read()`` first, SSRF-gated URL fallback."""
        from . import adapter as _adapter

        raw_bytes = await self._read_attachment_bytes(att, media_type="audio")
        if raw_bytes is not None:
            try:
                return await _adapter.cache_audio_from_bytes_async(raw_bytes, ext=ext)
            except Exception as e:
                _adapter.logger.debug("[Discord] cache_audio_from_bytes failed; falling back to URL: %s", e)
        return await _adapter.cache_audio_from_url(att.url, ext=ext)

    async def _cache_discord_document(self, att, ext: str) -> bytes:
        """Download a document attachment: ``att.read()`` first, SSRF-gated aiohttp fallback.
        Caller passes the bytes to ``cache_document_from_bytes`` (and injects text if applicable).

        This closes the gap where the old document path made raw ``aiohttp.ClientSession`` requests with no
        safety check (#11345). The caller is responsible for passing the returned bytes to
        ``cache_document_from_bytes`` (and, where applicable, for injecting text content).
        """
        from . import adapter as _adapter

        raw_bytes = await self._read_attachment_bytes(att, media_type="document")
        if raw_bytes is not None:
            return raw_bytes
        if not _adapter.is_safe_url(att.url):
            raise ValueError(f"Blocked unsafe attachment URL (SSRF protection): {att.url}")
        import aiohttp
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        async with aiohttp.ClientSession(**_sess_kw) as session:
            async with session.get(
                att.url, timeout=aiohttp.ClientTimeout(total=30), **_req_kw,
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                return await resp.read()

    async def _cache_simple_media(self, att: Any, content_type: str, kind: str, exts: set, default_ext: str) -> str:
        """Cache an image/audio attachment locally (CDN URLs expire); fall back to the CDN URL."""
        try:
            ext = "." + content_type.split("/")[-1].split(";")[0]
            if ext not in exts:
                ext = default_ext
            cacher = self._cache_discord_image if kind == "image" else self._cache_discord_audio
            cached_path = await cacher(att, ext)
            print(f"[Discord] Cached user {kind}: {cached_path}", flush=True)
            return cached_path
        except Exception as e:
            print(f"[Discord] Failed to cache {kind} attachment: {e}", flush=True)
            return att.url

    async def _collect_attachment_media(self, all_attachments: list) -> tuple:
        """Cache every attachment and return ``(media_urls, media_types, pending_text_injection)``."""
        from . import adapter as _adapter

        media_urls = []
        media_types = []
        pending_text_injection: _adapter.Optional[str] = None
        for att in all_attachments:
            content_type = att.content_type or "unknown"
            if content_type.startswith("image/"):
                media_urls.append(await self._cache_simple_media(
                    att, content_type, "image", {".jpg", ".jpeg", ".png", ".gif", ".webp"}, ".jpg"))
                media_types.append(content_type)
            elif content_type.startswith("audio/"):
                media_urls.append(await self._cache_simple_media(
                    att, content_type, "audio", {".ogg", ".mp3", ".wav", ".webm", ".m4a"}, ".ogg"))
                media_types.append(content_type)
            else:
                ext = ""
                if att.filename:
                    _, ext = _adapter.os.path.splitext(att.filename)
                    ext = ext.lower()
                if not ext and content_type:
                    mime_to_ext = {v: k for k, v in _adapter.SUPPORTED_DOCUMENT_TYPES.items()}
                    ext = mime_to_ext.get(content_type, "")
                in_allowlist = ext in _adapter.SUPPORTED_DOCUMENT_TYPES
                # Any file type accepted (authorization is the gate); unknown types fall back to octet-stream.
                max_doc_bytes = self._discord_max_attachment_bytes()
                if max_doc_bytes and att.size and att.size > max_doc_bytes:
                    _adapter.logger.warning(
                        "[Discord] Document too large (%s bytes > cap %s), skipping: %s",
                        att.size, max_doc_bytes, att.filename,
                    )
                    continue
                try:
                    raw_bytes = await self._cache_discord_document(att, ext)
                    cached_path = await _adapter.cache_document_from_bytes_async(raw_bytes, att.filename or f"document{ext or '.bin'}")
                    if in_allowlist:
                        doc_mime = _adapter.SUPPORTED_DOCUMENT_TYPES[ext]
                    else:
                        # Untyped: source content_type, else octet-stream (agent knows it's binary).
                        doc_mime = (
                            content_type if content_type and content_type != "unknown" else "application/octet-stream"
                        )
                    media_urls.append(cached_path)
                    media_types.append(doc_mime)
                    _adapter.logger.info(
                        "[Discord] Cached user %s: %s", "document" if in_allowlist else "attachment", cached_path,
                    )
                    # Inject text for text-readable documents (capped at 100 KB). Gate on text-like
                    # extension/MIME, NOT a blind UTF-8 decode (PDF/zip/docx have ASCII headers); other
                    # types rely on ``gateway/run.py`` emitting a (sandbox-translated) path note.
                    MAX_TEXT_INJECT_BYTES = 100 * 1024
                    _is_text = ext in _adapter._TEXT_INJECT_EXTENSIONS or (content_type or "").startswith("text/")
                    if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                        try:
                            text_content = raw_bytes.decode("utf-8")
                            display_name = att.filename or f"document{ext or '.txt'}"
                            display_name = _adapter.re.sub(r'[^\w.\- ]', '_', display_name)
                            injection = f"[Content of {display_name}]:\n{text_content}"
                            if pending_text_injection:
                                pending_text_injection = f"{pending_text_injection}\n\n{injection}"
                            else:
                                pending_text_injection = injection
                        except UnicodeDecodeError:
                            pass
                except Exception as e:
                    _adapter.logger.warning("[Discord] Failed to cache document %s: %s", att.filename, e, exc_info=True)
        return media_urls, media_types, pending_text_injection

    def _attachment_message_type(self, att: Any) -> MessageType:
        """MessageType from the first attachment's MIME. Any non-media (or untyped) attachment
        is a DOCUMENT regardless of extension — authorization is the gate, not the file type."""
        from . import adapter as _adapter

        content_type = att.content_type or ""
        if content_type.startswith("image/"):
            return _adapter.MessageType.PHOTO
        if content_type.startswith("video/"):
            return _adapter.MessageType.VIDEO
        if content_type.startswith("audio/"):
            return _adapter.MessageType.VOICE if self._is_discord_voice_message_attachment(att) else _adapter.MessageType.AUDIO
        return _adapter.MessageType.DOCUMENT

    @staticmethod
    def _reply_target(reference: Any) -> Optional[Any]:
        """Something with ``.id`` for the replied-to message; duck-typed (test doubles mock ``discord``),
        falling back to a bare snowflake from ``reference.message_id``."""
        from . import adapter as _adapter

        _resolved = getattr(reference, "resolved", None)
        if getattr(_resolved, "id", None) is not None:
            return _resolved
        _ref_mid = getattr(reference, "message_id", None)
        if _ref_mid is not None:
            with _adapter.suppress(ValueError, TypeError):
                return _adapter._Snowflake(int(_ref_mid))
        return None

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
        from . import adapter as _adapter

        thread_id = None
        parent_channel_id = None
        is_thread = isinstance(message.channel, _adapter.discord.Thread)
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
        if not isinstance(message.channel, _adapter.discord.DMChannel):
            channel_ids = {str(message.channel.id)}
            if parent_channel_id:
                channel_ids.add(parent_channel_id)
            channel_keys = self._discord_channel_keys(message, parent_channel_id)
            allowed_channels = self._get_allowed_channels()
            if allowed_channels:
                if "*" not in allowed_channels and not (channel_keys & allowed_channels):
                    _adapter.logger.debug("[%s] Ignoring message in non-allowed channel: %s", self.name, channel_keys)
                    return False
            ignored_channels = self._get_ignored_channels()
            if "*" in ignored_channels or (channel_keys & ignored_channels):
                _adapter.logger.debug("[%s] Ignoring message in ignored channel: %s", self.name, channel_keys)
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
        if not is_thread and not isinstance(message.channel, _adapter.discord.DMChannel):
            no_thread_channels = self._get_no_thread_channels()
            skip_thread = bool(channel_keys & no_thread_channels) or is_free_channel
            auto_thread = _adapter.os.getenv("DISCORD_AUTO_THREAD", "true").lower() in {"true", "1", "yes"}
            is_reply_message = getattr(message, "type", None) == _adapter.discord.MessageType.reply
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
                        _adapter.logger.warning(
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
            msg_type = _adapter.MessageType.COMMAND
        elif all_attachments:
            msg_type = self._attachment_message_type(all_attachments[0])
        else:
            msg_type = _adapter.MessageType.TEXT
        effective_channel = auto_threaded_channel or message.channel
        if isinstance(message.channel, _adapter.discord.DMChannel):
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
        source.is_one_to_one = isinstance(message.channel, _adapter.discord.DMChannel)
        source.message_is_edit = getattr(message, "edited_at", None) is not None
        media_urls, media_types, pending_text_injection = await self._collect_attachment_media(all_attachments)
        event_text = normalized_content
        if pending_text_injection:
            event_text = f"{pending_text_injection}\n\n{event_text}" if event_text else pending_text_injection
        # ── History backfill ─────────────────────────────────────────
        # With require_mention, messages between bot turns never reach the transcript; fetch
        # history after the bot's last message (cold start: last N, stop at first self-message)
        # and prepend it. DMs skipped (every DM triggers the bot); in-flight arrivals not captured.
        _channel_context = None
        _is_dm = isinstance(message.channel, _adapter.discord.DMChannel)
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
                _adapter.logger.info(
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
        event = _adapter.MessageEvent(
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
        if (not recovered and msg_type == _adapter.MessageType.TEXT and self._text_batch_delay_seconds > 0):
            self._enqueue_text_event(event)
        else:
            await self.handle_message(event)
        return True

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch; longer delay when the chunk is
        near Discord's 2000-char split point (continuation almost certain)."""
        from . import adapter as _adapter

        current_task = _adapter.asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await _adapter.asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            _adapter.logger.info("[Discord] Flushing text batch %s (%d chars)", key, len(event.text or ""))
            # Shield the dispatch: _enqueue_text_event cancels the prior flush task on each new chunk;
            # without the shield CancelledError would abort the in-flight agent turn.
            await _adapter.asyncio.shield(self.handle_message(event))
        except _adapter.asyncio.CancelledError:
            # Cancel landed before the pop; shielded handle_message unaffected.
            pass
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)
