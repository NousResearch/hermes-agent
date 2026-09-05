"""Discord delivery methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from gateway.native_document_guard import check_document_fallback, mark_native_document_guard
from gateway.platforms.base import MessageEvent, ProcessingOutcome, SendResult
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any


class DiscordDeliveryMixin:
    async def _add_reaction(self, message: Any, emoji: str) -> bool:
        """Add an emoji reaction to a Discord message."""
        from . import adapter as _adapter

        if not message or not hasattr(message, "add_reaction"):
            return False
        try:
            await message.add_reaction(emoji)
            return True
        except Exception as e:
            _adapter.logger.debug("[%s] add_reaction failed (%s): %s", self.name, emoji, e)
            return False

    async def _remove_reaction(self, message: Any, emoji: str) -> bool:
        """Remove the bot's own emoji reaction from a Discord message."""
        from . import adapter as _adapter

        if not message or not hasattr(message, "remove_reaction") or not self._client or not self._client.user:
            return False
        try:
            await message.remove_reaction(emoji, self._client.user)
            return True
        except Exception as e:
            _adapter.logger.debug("[%s] remove_reaction failed (%s): %s", self.name, emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        from . import adapter as _adapter

        return _adapter.os.getenv("DISCORD_REACTIONS", "true").lower() not in {"false", "0", "no"}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction and record durable handling state."""
        from . import adapter as _adapter

        message = event.raw_message
        acked = False
        if self._reactions_enabled() and hasattr(message, "add_reaction"):
            acked = await self._add_reaction(message, "👀")
        await _adapter.asyncio.to_thread(self._record_discord_processing_start, event, emoji_ack=acked)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for final reaction and durable state."""
        from . import adapter as _adapter

        await _adapter.asyncio.to_thread(self._record_discord_processing_complete, event, outcome)
        if not self._reactions_enabled():
            return
        message = event.raw_message
        if hasattr(message, "add_reaction"):
            await self._remove_reaction(message, "👀")
            if outcome == _adapter.ProcessingOutcome.SUCCESS:
                await self._add_reaction(message, "✅")
            elif outcome == _adapter.ProcessingOutcome.FAILURE:
                await self._add_reaction(message, "❌")

    @staticmethod
    def _message_reference_from_ids(message_id, channel) -> "discord.MessageReference":
        """ids-built reply reference — no fetch_message round trip. fail_if_not_exists=False
        keeps sends to deleted targets degrading to the send-side 10008 retry."""
        from . import adapter as _adapter

        return _adapter.discord.MessageReference(
            message_id=int(message_id), channel_id=getattr(channel, "id", None),
            guild_id=getattr(getattr(channel, "guild", None), "id", None), fail_if_not_exists=False,
        )

    def _reply_reference_for_send(self, reply_to, channel):
        """Reply anchor for send paths honoring reply_to_mode (``off`` suppresses); mirrors telegram."""
        from . import adapter as _adapter

        if not reply_to or self._reply_to_mode == "off":
            return None
        try:
            return self._message_reference_from_ids(reply_to, channel)
        except (ValueError, TypeError) as e:
            _adapter.logger.debug("Could not build reply-to reference: %s", e)
            return None

    def _cap_split_chunks(self, chunks: List[str]) -> List[str]:
        """Cap chunks at ``MAX_SPLIT_MESSAGES``: keep the first N-1 and replace the rest with a
        notice so a degenerate turn can't flood the channel (full text stays in session history).

        Cap the number of chunks sent for one logical response (#86581).
        A degenerate turn can produce tens of thousands of characters; the 86581 incident delivered 60,698
        chars as 31 back-to-back Discord messages. The full response remains available in the gateway
        session history / logs. See #86581.
        """
        if len(chunks) <= self.MAX_SPLIT_MESSAGES:
            return chunks
        kept = chunks[: self.MAX_SPLIT_MESSAGES - 1]
        dropped_chars = sum(len(c) for c in chunks[self.MAX_SPLIT_MESSAGES - 1 :])
        notice = (
            f"\n\n⚠️ **Response truncated** — this reply exceeded the "
            f"delivery limit ({self.MAX_SPLIT_MESSAGES} messages). "
            f"{dropped_chars} characters were not delivered; the full "
            f"response is in the session logs."
        )
        kept.append(notice)
        return kept

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Discord channel or thread (metadata thread_id wins over
        chat_id; forum channels auto-create a thread post since they reject direct sends)."""
        from . import adapter as _adapter

        if not self._client:
            # Dead transport: classify as send_path_degraded so the delivery ledger's reconnect
            # sweep can replay this; a generic "Not connected" error would strand the output.
            return _adapter.SendResult(success=False, error="send_path_degraded", retryable=True)
        if not (content or "").strip():
            _adapter.logger.warning(
                "[%s] Dropped empty message to chat=%s (caller bug). Call site:\n%s", self.name,
                chat_id, "".join(_adapter.traceback.format_stack(limit=12)[:-1]),
            )
            result = _adapter.SendResult(success=False, error="Refusing to send empty message")
            # Backfill replays from this table: record the dropped final reply as failed or it is lost.
            return await self._record_response_async(reply_to, result, content, bool(metadata and metadata.get("notify")))
        try:
            thread_id = None
            if metadata and metadata.get("thread_id"):
                thread_id = metadata["thread_id"]
            nonconversational = _adapter._metadata_marks_nonconversational(metadata)
            final_delivery = bool(metadata and metadata.get("notify"))
            if thread_id:
                channel = await self._resolve_channel(thread_id)
                if not channel:
                    return _adapter.SendResult(success=False, error=f"Thread {thread_id} not found")
            else:
                channel = await self._resolve_channel(chat_id)
                if not channel:
                    return _adapter.SendResult(success=False, error=f"Channel {chat_id} not found")
            # Forum channels reject channel.send() — create a thread post instead.
            if self._is_forum_parent(channel):
                result = await self._send_to_forum(channel, content)
                return await self._record_response_async(reply_to, result, content, final_delivery)
            formatted = self.format_message(content)
            chunks = self._cap_split_chunks(
                self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
            )
            message_ids = []
            reference = self._reply_reference_for_send(reply_to, channel)
            for i, chunk in enumerate(chunks):
                if self._reply_to_mode == "all":
                    chunk_reference = reference
                else:  # "first" (default) or "off"
                    chunk_reference = reference if i == 0 else None
                try:
                    msg = await channel.send(content=chunk, reference=chunk_reference)
                except Exception as e:
                    if chunk_reference is not None and self._is_reply_reference_rejected(e):
                        _adapter.logger.warning(
                            "[%s] Reply target %s rejected the reply reference; retrying send without reply reference",
                            self.name, reply_to,
                        )
                        reference = None
                        msg = await channel.send(content=chunk, reference=None)
                    else:
                        raise
                message_ids.append(str(msg.id))
            # Track the last sent message for history backfill (skips the full history scan).
            if message_ids:
                _target_id = thread_id or chat_id
                if nonconversational:
                    await self._nonconversational_messages.mark_many(message_ids)
                elif not _adapter._looks_like_nonconversational_history_message(content):
                    self._last_self_message_id[_target_id] = message_ids[-1]
            # Connection-shaped failure (WS drop / closed session): use the ledger's runtime-retryable
            # marker so the reconnect sweep can replay this final response instead of stranding it until a
            # process restart (#95382 silent partial loss).
            result = _adapter.SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )
            return await self._record_response_async(reply_to, result, content, final_delivery)
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to send Discord message: %s", self.name, e, exc_info=True)
            if _adapter._is_discord_transport_error(e):
                # Connection-shaped failure: runtime-retryable marker so the reconnect sweep can replay it.
                result = _adapter.SendResult(success=False, error="send_path_degraded", retryable=True)
            else:
                result = _adapter.SendResult(success=False, error=str(e))
            return await self._record_response_async(reply_to, result, content, bool(metadata and metadata.get("notify")))

    @staticmethod
    def _forum_thread_parts(thread: Any) -> tuple:
        """``create_thread`` returns a Thread or a ThreadWithMessage; normalise to
        ``(thread_channel, thread_id, starter_msg, starter_message_id)``."""
        from . import adapter as _adapter

        thread_channel = thread if hasattr(thread, "send") else getattr(thread, "thread", None)
        thread_id = str(getattr(thread_channel, "id", getattr(thread, "id", "")))
        starter_msg = getattr(thread, "message", None)
        message_id = str(getattr(starter_msg, "id", thread_id)) if starter_msg else thread_id
        return thread_channel, thread_id, starter_msg, message_id

    async def _send_to_forum(self, forum_channel: Any, content: str) -> SendResult:
        """Create a forum thread post with the message as starter (forum channels reject direct
        sends; name from the first line). Chunk failures land in ``raw_response['warnings']``."""
        from . import adapter as _adapter

        formatted = self.format_message(content)
        chunks = self._cap_split_chunks(self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH))
        thread_name = _adapter._derive_forum_thread_name(content)
        starter_content = chunks[0] if chunks else thread_name
        try:
            thread = await forum_channel.create_thread(name=thread_name, content=starter_content)
        except Exception as e:
            _adapter.logger.error("[%s] Failed to create forum thread in %s: %s", self.name, forum_channel.id, e)
            return _adapter.SendResult(success=False, error=f"Forum thread creation failed: {e}")
        thread_channel, thread_id, starter_msg, message_id = self._forum_thread_parts(thread)
        message_ids = [message_id]
        warnings: list[str] = []
        for chunk in chunks[1:]:
            try:
                msg = await thread_channel.send(content=chunk)
                message_ids.append(str(msg.id))
            except Exception as e:
                warning = f"Failed to send follow-up chunk to forum thread {thread_id}: {e}"
                _adapter.logger.warning("[%s] %s", self.name, warning)
                warnings.append(warning)
        raw_response: _adapter.Dict[str, _adapter.Any] = {"message_ids": message_ids, "thread_id": thread_id}
        if warnings:
            raw_response["warnings"] = warnings
        return _adapter.SendResult(success=True, message_id=message_ids[0], raw_response=raw_response)

    async def _forum_post_file(
        self, forum_channel: Any, *, thread_name: Optional[str] = None, content: str = "",
        file: Any = None, files: Optional[list] = None,
    ) -> SendResult:
        """Create a forum thread whose starter message carries file attachments."""
        from . import adapter as _adapter

        if not thread_name:
            hint = content or ""
            if not hint.strip():
                if file is not None:
                    hint = getattr(file, "filename", "") or ""
                elif files:
                    hint = getattr(files[0], "filename", "") or ""
            thread_name = _adapter._derive_forum_thread_name(hint) if hint.strip() else "New Post"
        kwargs: _adapter.Dict[str, _adapter.Any] = {"name": thread_name}
        if content:
            kwargs["content"] = content
        if file is not None:
            kwargs["file"] = file
        if files:
            kwargs["files"] = files
        try:
            thread = await forum_channel.create_thread(**kwargs)
        except Exception as e:
            _adapter.logger.error(
                "[%s] Failed to create forum thread with file in %s: %s", self.name,
                getattr(forum_channel, "id", "?"), e,
            )
            return _adapter.SendResult(success=False, error=f"Forum thread creation failed: {e}")
        thread_channel, thread_id, starter_msg, message_id = self._forum_thread_parts(thread)
        if file is not None or files:
            attachments = getattr(starter_msg, "attachments", None) or []
            if not attachments:
                filename = ""
                if file is not None:
                    filename = getattr(file, "filename", "") or ""
                elif files:
                    filename = getattr(files[0], "filename", "") or ""
                _adapter.logger.warning(
                    "[%s] Forum thread %s starter has no attachments for %s", self.name, thread_id,
                    filename or "file",
                )
                return _adapter.SendResult(
                    success=False,
                    error=(
                        "Discord created the forum thread but attached no files"
                        + (f" ({filename})" if filename else "")
                    ),
                    message_id=message_id or None,
                    raw_response={"thread_id": thread_id},
                )
        return _adapter.SendResult(
            success=True, message_id=message_id, raw_response={"thread_id": thread_id},
        )

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a sent Discord message. Oversized text (>2,000) must neither truncate silently nor
        fail (consumer re-sends -> dupe): mid-stream keep a truncated preview (splitting would move
        the edit target every tick); ``finalize=True`` delivers all via ``_edit_overflow_split``.

        Mid-stream (``finalize=False``) we keep editing the original message with a truncated preview —
        splitting mid-stream would move the edit target to a continuation and the next accumulated-token
        tick would re-split, looping forever (the Telegram #48648 lesson).
        """
        from . import adapter as _adapter

        if not self._client:
            return _adapter.SendResult(success=False, error="Not connected")
        try:
            channel = await self._resolve_channel(chat_id)
            msg = channel.get_partial_message(int(message_id))
            formatted = self.format_message(content)
            _preview_key = (str(chat_id), str(message_id))
            _saturated_preview = False
            if finalize:
                # Saturation state is finished — the final edit delivers full content.
                self._last_overflow_preview.pop(_preview_key, None)
            # Pre-flight oversize: final edits split-and-deliver; streaming edits truncate in place.
            if len(formatted) > self.MAX_MESSAGE_LENGTH:
                if finalize:
                    return await self._edit_overflow_split(channel, msg, message_id, content)
                formatted = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)[0]
                _saturated_preview = True
                # Saturated-preview dedup: past the cap every edit is the same text; skip until finalize.
                # Re-sending it is a visual no-op that still counts against Discord's edit rate limit — skip
                # silently until finalize (mirrors the Telegram #58563 fix).
                if self._last_overflow_preview.get(_preview_key) == formatted:
                    return _adapter.SendResult(success=True, message_id=message_id)
            elif not finalize:
                # Content shrank under the cap: clear saturation state so dedup can't mask a real edit.
                self._last_overflow_preview.pop(_preview_key, None)
            try:
                await msg.edit(content=formatted)
                if _saturated_preview:
                    self._last_overflow_preview[_preview_key] = formatted
            except Exception as edit_err:
                # Reactive split: format_message inflation can exceed 2,000 (50035) even after pre-flight.
                if self._is_length_overflow_error(edit_err):
                    if finalize:
                        return await self._edit_overflow_split(channel, msg, message_id, content)
                    truncated = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)[0]
                    if self._last_overflow_preview.get(_preview_key) == truncated:
                        # Saturated-preview dedup (see pre-flight path above).
                        return _adapter.SendResult(success=True, message_id=message_id)
                    await msg.edit(content=truncated)
                    self._last_overflow_preview[_preview_key] = truncated
                else:
                    raise
            result = _adapter.SendResult(success=True, message_id=message_id)
            if finalize:
                await self._record_response_async((metadata or {}).get("reply_to_message_id"), result, content, True)
            return result
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to edit Discord message %s: %s", self.name, message_id, e, exc_info=True)
            return _adapter.SendResult(success=False, error=str(e))

    @staticmethod
    def _is_reply_reference_rejected(err: Exception) -> bool:
        """Discord refused the reply anchor: system-message target (50035) or deleted target (10008)."""
        from . import adapter as _adapter

        err_text = str(err)
        return (
            "error code: 50035" in err_text and "Cannot reply to a system message" in err_text
        ) or "error code: 10008" in err_text

    @staticmethod
    def _is_length_overflow_error(err: Exception) -> bool:
        """True when a Discord edit/send failed for >2,000 chars: code 50035 plus the length phrasing,
        so other 50035 validation errors (e.g. bad reply reference) aren't mistaken for overflow."""
        from . import adapter as _adapter

        text = str(err).lower()
        return "error code: 50035" in text and (
            "2000 or fewer" in text or "fewer in length" in text
        )

    async def _edit_overflow_split(
        self, channel: Any, msg: Any, message_id: str, content: str,
    ) -> SendResult:
        """Deliver an oversized final edit: edit ``message_id`` with chunk 1, send chunks 2..N as
        replies to the previous. Returns ``message_id=<last-id>`` + ``continuation_message_ids``.
        A continuation failure still reports success plus ``partial_overflow`` so the consumer
        delivers the tail; only a first-chunk edit failure returns ``success=False``."""
        from . import adapter as _adapter

        formatted = self.format_message(content)
        chunks = self._cap_split_chunks(self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH))
        if len(chunks) <= 1:
            # Defensive: pre-flight should guarantee >1 chunk; otherwise edit normally.
            await msg.edit(content=chunks[0] if chunks else formatted)
            return _adapter.SendResult(success=True, message_id=message_id)
        try:
            await msg.edit(content=chunks[0])
        except Exception as e:
            _adapter.logger.error(
                "[%s] Overflow split: first-chunk edit failed: %s", self.name, e, exc_info=True,
            )
            return _adapter.SendResult(success=False, error=str(e))
        continuation_ids: list[str] = []
        delivered = 1
        prev_msg = msg
        for chunk in chunks[1:]:
            reference = None
            if hasattr(prev_msg, "to_reference"):
                try:
                    reference = prev_msg.to_reference(fail_if_not_exists=False)
                except Exception:
                    reference = None
            elif getattr(prev_msg, "id", None):
                # Prior message without to_reference (duck-typed): build the reference from ids.
                reference = self._message_reference_from_ids(prev_msg.id, channel)
            try:
                sent = await channel.send(content=chunk, reference=reference)
            except Exception as send_err:
                # Drop the reply anchor and retry once: deleted anchor (10008) / system message (50035).
                _adapter.logger.warning(
                    "[%s] Overflow continuation send failed (%s); retrying without reply reference",
                    self.name, send_err,
                )
                try:
                    sent = await channel.send(content=chunk, reference=None)
                except Exception as retry_err:
                    _adapter.logger.warning(
                        "[%s] Overflow split: stopped at %d/%d chunks delivered: %s",
                        self.name, delivered, len(chunks), retry_err,
                    )
                    last_id = continuation_ids[-1] if continuation_ids else message_id
                    return _adapter.SendResult(
                        success=True,
                        message_id=last_id,
                        continuation_message_ids=tuple(continuation_ids),
                        raw_response={
                            "partial_overflow": True, "delivered_chunks": delivered,
                            "total_chunks": len(chunks), "last_message_id": last_id,
                            "continuation_message_ids": tuple(continuation_ids),
                        },
                    )
            new_id = str(sent.id)
            continuation_ids.append(new_id)
            delivered += 1
            prev_msg = sent
        last_id = continuation_ids[-1] if continuation_ids else message_id
        # Point the history-backfill fast path at the final visible chunk.
        if not _adapter._looks_like_nonconversational_history_message(content):
            self._last_self_message_id[str(channel.id)] = last_id
        _adapter.logger.debug(
            "[%s] Overflow split delivered %d chunks; last_id=%s", self.name, delivered, last_id,
        )
        return _adapter.SendResult(
            success=True, message_id=last_id, continuation_message_ids=tuple(continuation_ids),
        )

    async def _send_file_attachment(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Send a local file as a Discord attachment (forum channels get a new thread). Path-based
        ``discord.File`` only: the open-handle form can race the multipart encoder after an image
        batch and yield zero attachments — a silent drop for video/document MEDIA tags.

        See #66797.
        """
        from . import adapter as _adapter

        if not self._client:
            return _adapter.SendResult(success=False, error="Not connected")
        if not _adapter.os.path.isfile(file_path):
            return _adapter.SendResult(success=False, error=f"File not found: {file_path}")
        channel = await self._resolve_channel(chat_id)
        if not channel:
            return _adapter.SendResult(success=False, error=f"Channel {chat_id} not found")
        filename = file_name or _adapter.os.path.basename(file_path)
        _adapter.logger.info(
            "[%s] Sending file attachment %s (%s) to %s", self.name, filename,
            _adapter.os.path.splitext(filename)[1].lower() or "no-ext", chat_id,
        )
        # Path-based File (discord.py owns open/close); ``files=[...]`` over deprecated ``file=``.
        discord_file = _adapter.discord.File(file_path, filename=filename)
        if self._is_forum_parent(channel):
            result = await self._forum_post_file(
                channel, content=(caption or "").strip(), files=[discord_file],
            )
            return result
        msg = await channel.send(content=caption if caption else None, files=[discord_file])
        attachments = getattr(msg, "attachments", None) or []
        if not attachments:
            # Discord accepted the message but attached nothing: fail loud instead of a silent drop.
            # Discord accepted the message but attached nothing — the failure mode reported in #66797 (MEDIA
            # video stripped from text, no attachment, no prior log line).
            _adapter.logger.warning(
                "[%s] Discord returned message %s with no attachments for %s", self.name,
                getattr(msg, "id", "?"), filename,
            )
            return _adapter.SendResult(
                success=False,
                error=f"Discord accepted the message but attached no files ({filename})",
                message_id=str(getattr(msg, "id", "") or "") or None,
            )
        return _adapter.SendResult(success=True, message_id=str(msg.id))

    async def send_multiple_images(
        self, chat_id: str, images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0,
    ) -> None:
        """Send images as one Discord message (<=10 attachments): URLs are downloaded and uploaded
        inline (bare links don't render); on chunk failure the remainder uses the per-image loop."""
        from . import adapter as _adapter

        if not self._client:
            return
        if not images:
            return
        try:
            import discord as _discord_mod
            import io as _io
            from urllib.parse import unquote as _unquote
        except Exception:  # pragma: no cover
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        try:
            channel = await self._resolve_channel(chat_id)
            if not channel:
                _adapter.logger.warning("[%s] Channel %s not found for multi-image send", self.name, chat_id)
                return
        except Exception as e:
            _adapter.logger.warning("[%s] Failed to resolve channel for multi-image send: %s", self.name, e)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        CHUNK = 10
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await _adapter.asyncio.sleep(human_delay)
            files: _adapter.List[_adapter.Any] = []
            captions: _adapter.List[str] = []
            aiohttp_session = None
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        captions.append(alt_text)
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not _adapter.os.path.exists(local_path):
                            _adapter.logger.warning("[%s] Skipping missing image: %s", self.name, local_path)
                            continue
                        files.append(_discord_mod.File(local_path, filename=_adapter.os.path.basename(local_path)))
                    else:
                        if not _adapter.is_safe_url(image_url):
                            _adapter.logger.warning("[%s] Blocked unsafe image URL in batch", self.name)
                            continue
                        # Download to BytesIO so it renders inline
                        try:
                            import aiohttp as _aiohttp
                            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
                            _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
                            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
                            if aiohttp_session is None:
                                aiohttp_session = _aiohttp.ClientSession(**_sess_kw)
                            status, data, headers = await _adapter._read_url_image_with_redirect_guard(
                                aiohttp_session, image_url,
                                timeout=_aiohttp.ClientTimeout(total=30), request_kwargs=_req_kw,
                            )
                            if status != 200:
                                _adapter.logger.warning(
                                    "[%s] Failed to download image (HTTP %d) in batch: %s",
                                    self.name, status, image_url[:80],
                                )
                                continue
                            ext = _adapter._image_ext_from_content_type(headers.get("content-type", "image/png"))
                            files.append(_discord_mod.File(_io.BytesIO(data), filename=f"image_{len(files)}.{ext}"))
                        except Exception as dl_err:
                            _adapter.logger.warning("[%s] Download failed for %s: %s", self.name, image_url[:80], dl_err)
                            continue
                if not files:
                    continue
                # Use the first caption if any (Discord only has one message body for the group)
                content = captions[0] if captions else None
                _adapter.logger.info(
                    "[%s] Sending %d image(s) as single Discord message (chunk %d/%d)",
                    self.name, len(files), chunk_idx + 1, len(chunks),
                )
                if self._is_forum_parent(channel):
                    await self._forum_post_file(
                        channel, content=(content or "").strip(), files=files,
                    )
                else:
                    await channel.send(content=content, files=files)
            except Exception as e:
                _adapter.logger.warning(
                    "[%s] Multi-image Discord send failed (chunk %d/%d), falling back to per-image: %s",
                    self.name, chunk_idx + 1, len(chunks), e, exc_info=True,
                )
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            finally:
                if aiohttp_session is not None:
                    try:
                        await aiohttp_session.close()
                    except Exception:
                        pass

    async def _send_local_file(self, chat_id, path, caption, *, file_name=None, not_found: str, kind: str, fallback):
        """Native attachment upload for a local file; missing file -> error, other failure -> base adapter."""
        from . import adapter as _adapter

        try:
            return await self._send_file_attachment(chat_id, path, caption, file_name=file_name)
        except FileNotFoundError:
            return _adapter.SendResult(success=False, error=f"{not_found}: {path}")
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to send %s, falling back to base adapter: %s", self.name, kind, e, exc_info=True)
            if kind == "document":
                check_document_fallback()
            return await fallback()

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file natively as a Discord file attachment."""
        from . import adapter as _adapter

        return await self._send_local_file(
            chat_id, image_path, caption, not_found="Image file not found", kind="local image",
            fallback=lambda: super(DiscordDeliveryMixin, self).send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata),
        )

    async def _send_url_media(
        self, chat_id: str, url: str, caption: Optional[str], *, kind: str,
        filename_for, fallback, metadata: Optional[dict], error_metadata: Optional[dict],
    ) -> SendResult:
        """Download ``url`` and post it as a native attachment (Discord renders those inline).
        ``fallback(metadata)`` is the base-adapter URL send (``error_metadata`` after download failure)."""
        from . import adapter as _adapter

        if not self._client:
            return _adapter.SendResult(success=False, error="Not connected")
        if not _adapter.is_safe_url(url):
            _adapter.logger.warning("[%s] Blocked unsafe %s URL during Discord send_%s", self.name, kind, kind)
            return await fallback(metadata)
        try:
            import aiohttp
            channel = await self._resolve_channel(chat_id)
            if not channel:
                return _adapter.SendResult(success=False, error=f"Channel {chat_id} not found")
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url(platform_env_var="DISCORD_PROXY"))
            async with aiohttp.ClientSession(**_sess_kw) as session:
                status, data, headers = await _adapter._read_url_image_with_redirect_guard(
                    session, url, timeout=aiohttp.ClientTimeout(total=30), request_kwargs=_req_kw,
                )
                if status != 200:
                    raise Exception(f"Failed to download {kind}: HTTP {status}")
                import io
                file = _adapter.discord.File(io.BytesIO(data), filename=filename_for(headers))
                if self._is_forum_parent(channel):
                    return await self._forum_post_file(channel, content=(caption or "").strip(), file=file)
                msg = await channel.send(content=caption if caption else None, file=file)
                return _adapter.SendResult(success=True, message_id=str(msg.id))
        except ImportError:
            _adapter.logger.warning("[%s] aiohttp not installed, falling back to URL. Run: pip install aiohttp", self.name, exc_info=True)
            return await fallback(error_metadata)
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to send %s attachment, falling back to URL: %s", self.name, kind, e, exc_info=True)
            return await fallback(error_metadata)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively as a Discord file attachment."""
        from . import adapter as _adapter

        return await self._send_url_media(
            chat_id, image_url, caption, kind="image",
            filename_for=lambda h: f"image.{_adapter._image_ext_from_content_type(h.get('content-type', 'image/png'))}",
            fallback=lambda md: super(DiscordDeliveryMixin, self).send_image(chat_id, image_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=None,
        )

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Discord file attachment."""
        from . import adapter as _adapter

        return await self._send_url_media(
            chat_id, animation_url, caption, kind="animation", filename_for=lambda _h: "animation.gif",
            fallback=lambda md: super(DiscordDeliveryMixin, self).send_animation(chat_id, animation_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=metadata,
        )

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local video file natively as a Discord attachment."""
        from . import adapter as _adapter

        return await self._send_local_file(
            chat_id, video_path, caption, not_found="Video file not found", kind="local video",
            fallback=lambda: super(DiscordDeliveryMixin, self).send_video(chat_id, video_path, caption, reply_to, metadata=metadata),
        )

    @mark_native_document_guard
    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an arbitrary file natively as a Discord attachment."""
        from . import adapter as _adapter

        return await self._send_local_file(
            chat_id, file_path, caption, file_name=file_name, not_found="File not found", kind="document",
            fallback=lambda: super(DiscordDeliveryMixin, self).send_document(chat_id, file_path, caption, file_name, reply_to, metadata=metadata),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start a persistent typing loop (POST typing every 12s; indicator lasts ~10s).
        TYPING_START is unreliable for bots in DMs; 429 sleeps ``retry_after``; CancelledError ends it."""
        from . import adapter as _adapter

        if not self._client:
            return
        if chat_id in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            try:
                while True:
                    try:
                        route = _adapter.discord.http.Route(
                            "POST", "/channels/{channel_id}/typing", channel_id=chat_id,
                        )
                        await self._client.http.request(route)
                    except _adapter.asyncio.CancelledError:
                        return
                    except Exception as e:
                        retry_after = self._extract_discord_retry_after(e)
                        if retry_after is not None:
                            _adapter.logger.warning(
                                "Typing indicator rate-limited for %s; retrying in %.1fs",
                                chat_id, retry_after,
                            )
                        else:
                            _adapter.logger.debug("Discord typing indicator failed for %s: %s", chat_id, e)
                            return
                        await _adapter.asyncio.sleep(retry_after)
                        continue
                    await _adapter.asyncio.sleep(12)
            except _adapter.asyncio.CancelledError:
                pass
            finally:
                self._typing_tasks.pop(chat_id, None)
        self._typing_tasks[chat_id] = _adapter.asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the persistent typing indicator for a channel."""
        from . import adapter as _adapter

        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except (_adapter.asyncio.CancelledError, Exception):
                pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Discord channel."""
        from . import adapter as _adapter

        if not self._client:
            return {"name": "Unknown", "type": "dm"}
        try:
            channel = await self._resolve_channel(chat_id)
            if not channel:
                return {"name": str(chat_id), "type": "dm"}
            if isinstance(channel, _adapter.discord.DMChannel):
                chat_type = "dm"
                name = channel.recipient.name if channel.recipient else str(chat_id)
            elif isinstance(channel, _adapter.discord.Thread):
                chat_type = "thread"
                name = channel.name
            elif isinstance(channel, _adapter.discord.TextChannel):
                chat_type = "channel"
                name = f"#{channel.name}"
                if channel.guild:
                    name = f"{channel.guild.name} / {name}"
            else:
                chat_type = "channel"
                name = getattr(channel, "name", str(chat_id))
            return {
                "name": name, "type": chat_type,
                "guild_id": str(channel.guild.id) if hasattr(channel, "guild") and channel.guild else None,
                "guild_name": channel.guild.name if hasattr(channel, "guild") and channel.guild else None,
            }
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to get chat info for %s: %s", self.name, chat_id, e, exc_info=True)
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    async def _resolve_allowed_usernames(self) -> None:
        """Resolve username/display-name entries in DISCORD_ALLOWED_USERS to numeric IDs."""
        from . import adapter as _adapter

        if not self._allowed_user_ids or not self._client:
            return
        numeric_ids = set()
        to_resolve = set()
        for entry in self._allowed_user_ids:
            if entry.isdigit():
                numeric_ids.add(entry)
            elif entry == "*":
                # Keep the "*" wildcard verbatim; it can't resolve and would be silently dropped.
                numeric_ids.add(entry)
            else:
                to_resolve.add(entry.lower())
        if not to_resolve:
            return
        print(f"[{self.name}] Resolving {len(to_resolve)} username(s): {', '.join(to_resolve)}")
        resolved_count = 0
        for guild in self._client.guilds:
            # Fetch full member list (requires members intent)
            try:
                members = guild.members
                if len(members) < guild.member_count:
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception as e:
                _adapter.logger.warning("Failed to fetch members for guild %s: %s", guild.name, e)
                continue
            for member in members:
                name_lower = member.name.lower()
                display_lower = member.display_name.lower()
                global_lower = (member.global_name or "").lower()
                matched = name_lower in to_resolve or display_lower in to_resolve or global_lower in to_resolve
                if matched:
                    uid = str(member.id)
                    numeric_ids.add(uid)
                    resolved_count += 1
                    matched_name = name_lower if name_lower in to_resolve else (
                        display_lower if display_lower in to_resolve else global_lower
                    )
                    to_resolve.discard(matched_name)
                    print(f"[{self.name}] Resolved '{matched_name}' -> {uid} ({member.name}#{member.discriminator})")
            if not to_resolve:
                break
        if to_resolve:
            print(f"[{self.name}] Could not resolve usernames: {', '.join(to_resolve)}")
        # Adapter-local: under multiplex_profiles os.environ writes would clobber other profiles.
        # Update the internal set. Keep the resolved IDs adapter-local first: under multiplex_profiles,
        # writing os.environ here would clobber every OTHER profile's DISCORD_ALLOWED_USERS after this
        # adapter's on_ready — an unguarded runtime mutation of process-global state (issue #72348). Refresh
        # this adapter's own snapshot instead.
        self._allowed_user_ids = numeric_ids
        snap = getattr(self, "_gate_env_snapshot", None)
        if snap is not None:
            snap["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if not _adapter._multiplex_active():
            # Single-profile: legacy env rewrite so gateway env-based auth sees numeric IDs.
            _adapter.os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if resolved_count:
            print(f"[{self.name}] Updated DISCORD_ALLOWED_USERS with {resolved_count} resolved ID(s)")

    def format_message(self, content: str) -> str:
        """Format for Discord: GFM tables become bullet lists (Discord doesn't render pipe tables)."""
        from . import adapter as _adapter

        if not content:
            return content
        return _adapter.convert_table_to_bullets(content)
