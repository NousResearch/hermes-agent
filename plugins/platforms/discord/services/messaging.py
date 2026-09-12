"""Discord message delivery, editing, reactions, and forum posting."""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from .. import adapter as _adapter

logger = _adapter.logger
discord = _adapter.discord
_metadata_marks_nonconversational = lambda value: _adapter._metadata_marks_nonconversational(value)
_looks_like_nonconversational_history_message = lambda value: _adapter._looks_like_nonconversational_history_message(value)
_is_discord_transport_error = lambda value: _adapter._is_discord_transport_error(value)
_derive_forum_thread_name = lambda value: _adapter._derive_forum_thread_name(value)


class MessagingMixin:
    """Own outbound Discord message and reaction behavior."""

    async def _add_reaction(self, message: Any, emoji: str) -> bool:
        """Add an emoji reaction to a Discord message."""
        if not message or not hasattr(message, "add_reaction"):
            return False
        try:
            await message.add_reaction(emoji)
            return True
        except Exception as e:
            logger.debug("[%s] add_reaction failed (%s): %s", self.name, emoji, e)
            return False

    async def _remove_reaction(self, message: Any, emoji: str) -> bool:
        """Remove the bot's own emoji reaction from a Discord message."""
        if not message or not hasattr(message, "remove_reaction") or not self._client or not self._client.user:
            return False
        try:
            await message.remove_reaction(emoji, self._client.user)
            return True
        except Exception as e:
            logger.debug("[%s] remove_reaction failed (%s): %s", self.name, emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Reactions enabled via ``extra.reactions`` or ``DISCORD_REACTIONS``."""
        return self._extra_or_env_flag("reactions", "DISCORD_REACTIONS", "true", truthy=False)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction and record durable handling state."""
        message = event.raw_message
        acked = False
        if self._reactions_enabled() and hasattr(message, "add_reaction"):
            acked = await self._add_reaction(message, "👀")
        await asyncio.to_thread(self._record_discord_processing_start, event, emoji_ack=acked)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for final reaction and durable state."""
        await asyncio.to_thread(self._record_discord_processing_complete, event, outcome)
        if not self._reactions_enabled():
            return
        message = event.raw_message
        if hasattr(message, "add_reaction"):
            await self._remove_reaction(message, "👀")
            if outcome == ProcessingOutcome.SUCCESS:
                await self._add_reaction(message, "✅")
            elif outcome == ProcessingOutcome.FAILURE:
                await self._add_reaction(message, "❌")

    @staticmethod
    def _message_reference_from_ids(message_id, channel) -> "discord.MessageReference":
        """ids-built reply reference — no fetch_message round trip. fail_if_not_exists=False
        keeps sends to deleted targets degrading to the send-side 10008 retry."""
        return discord.MessageReference(
            message_id=int(message_id), channel_id=getattr(channel, "id", None),
            guild_id=getattr(getattr(channel, "guild", None), "id", None), fail_if_not_exists=False,
        )

    def _reply_reference_for_send(self, reply_to, channel):
        """Reply anchor for send paths honoring reply_to_mode (``off`` suppresses); mirrors telegram."""
        if not reply_to or self._reply_to_mode == "off":
            return None
        try:
            return self._message_reference_from_ids(reply_to, channel)
        except (ValueError, TypeError) as e:
            logger.debug("Could not build reply-to reference: %s", e)
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
        if not self._client:
            # Dead transport: classify as send_path_degraded so the delivery ledger's reconnect
            # sweep can replay this; a generic "Not connected" error would strand the output.
            return SendResult(success=False, error="send_path_degraded", retryable=True)
        if not (content or "").strip():
            logger.warning(
                "[%s] Dropped empty message to chat=%s (caller bug). Call site:\n%s", self.name,
                chat_id, "".join(traceback.format_stack(limit=12)[:-1]),
            )
            result = SendResult(success=False, error="Refusing to send empty message")
            # Backfill replays from this table: record the dropped final reply as failed or it is lost.
            return await self._record_response_async(reply_to, result, content, bool(metadata and metadata.get("notify")))
        try:
            thread_id = None
            if metadata and metadata.get("thread_id"):
                thread_id = metadata["thread_id"]
            nonconversational = _metadata_marks_nonconversational(metadata)
            final_delivery = bool(metadata and metadata.get("notify"))
            if thread_id:
                channel = await self._resolve_channel(thread_id)
                if not channel:
                    return SendResult(success=False, error=f"Thread {thread_id} not found")
            else:
                channel = await self._resolve_channel(chat_id)
                if not channel:
                    return SendResult(success=False, error=f"Channel {chat_id} not found")
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
                        logger.warning(
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
                elif not _looks_like_nonconversational_history_message(content):
                    self._last_self_message_id[_target_id] = message_ids[-1]
            # Connection-shaped failure (WS drop / closed session): use the ledger's runtime-retryable
            # marker so the reconnect sweep can replay this final response instead of stranding it until a
            # process restart (#95382 silent partial loss).
            result = SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )
            return await self._record_response_async(reply_to, result, content, final_delivery)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send Discord message: %s", self.name, e, exc_info=True)
            if _is_discord_transport_error(e):
                # Connection-shaped failure: runtime-retryable marker so the reconnect sweep can replay it.
                result = SendResult(success=False, error="send_path_degraded", retryable=True)
            else:
                result = SendResult(success=False, error=str(e))
            return await self._record_response_async(reply_to, result, content, bool(metadata and metadata.get("notify")))

    @staticmethod
    def _forum_thread_parts(thread: Any) -> tuple:
        """``create_thread`` returns a Thread or a ThreadWithMessage; normalise to
        ``(thread_channel, thread_id, starter_msg, starter_message_id)``."""
        thread_channel = thread if hasattr(thread, "send") else getattr(thread, "thread", None)
        thread_id = str(getattr(thread_channel, "id", getattr(thread, "id", "")))
        starter_msg = getattr(thread, "message", None)
        message_id = str(getattr(starter_msg, "id", thread_id)) if starter_msg else thread_id
        return thread_channel, thread_id, starter_msg, message_id

    async def _send_to_forum(self, forum_channel: Any, content: str) -> SendResult:
        """Create a forum thread post with the message as starter (forum channels reject direct
        sends; name from the first line). Chunk failures land in ``raw_response['warnings']``."""
        formatted = self.format_message(content)
        chunks = self._cap_split_chunks(self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH))
        thread_name = _derive_forum_thread_name(content)
        starter_content = chunks[0] if chunks else thread_name
        try:
            thread = await forum_channel.create_thread(name=thread_name, content=starter_content)
        except Exception as e:
            logger.error("[%s] Failed to create forum thread in %s: %s", self.name, forum_channel.id, e)
            return SendResult(success=False, error=f"Forum thread creation failed: {e}")
        thread_channel, thread_id, starter_msg, message_id = self._forum_thread_parts(thread)
        message_ids = [message_id]
        warnings: list[str] = []
        for chunk in chunks[1:]:
            try:
                msg = await thread_channel.send(content=chunk)
                message_ids.append(str(msg.id))
            except Exception as e:
                warning = f"Failed to send follow-up chunk to forum thread {thread_id}: {e}"
                logger.warning("[%s] %s", self.name, warning)
                warnings.append(warning)
        raw_response: Dict[str, Any] = {"message_ids": message_ids, "thread_id": thread_id}
        if warnings:
            raw_response["warnings"] = warnings
        return SendResult(success=True, message_id=message_ids[0], raw_response=raw_response)

    async def _forum_post_file(
        self, forum_channel: Any, *, thread_name: Optional[str] = None, content: str = "",
        file: Any = None, files: Optional[list] = None,
    ) -> SendResult:
        """Create a forum thread whose starter message carries file attachments."""
        if not thread_name:
            hint = content or ""
            if not hint.strip():
                if file is not None:
                    hint = getattr(file, "filename", "") or ""
                elif files:
                    hint = getattr(files[0], "filename", "") or ""
            thread_name = _derive_forum_thread_name(hint) if hint.strip() else "New Post"
        kwargs: Dict[str, Any] = {"name": thread_name}
        if content:
            kwargs["content"] = content
        if file is not None:
            kwargs["file"] = file
        if files:
            kwargs["files"] = files
        try:
            thread = await forum_channel.create_thread(**kwargs)
        except Exception as e:
            logger.error(
                "[%s] Failed to create forum thread with file in %s: %s", self.name,
                getattr(forum_channel, "id", "?"), e,
            )
            return SendResult(success=False, error=f"Forum thread creation failed: {e}")
        thread_channel, thread_id, starter_msg, message_id = self._forum_thread_parts(thread)
        if file is not None or files:
            attachments = getattr(starter_msg, "attachments", None) or []
            if not attachments:
                filename = ""
                if file is not None:
                    filename = getattr(file, "filename", "") or ""
                elif files:
                    filename = getattr(files[0], "filename", "") or ""
                logger.warning(
                    "[%s] Forum thread %s starter has no attachments for %s", self.name, thread_id,
                    filename or "file",
                )
                return SendResult(
                    success=False,
                    error=(
                        "Discord created the forum thread but attached no files"
                        + (f" ({filename})" if filename else "")
                    ),
                    message_id=message_id or None,
                    raw_response={"thread_id": thread_id},
                )
        return SendResult(
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
        if not self._client:
            return SendResult(success=False, error="Not connected")
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
                    return SendResult(success=True, message_id=message_id)
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
                        return SendResult(success=True, message_id=message_id)
                    await msg.edit(content=truncated)
                    self._last_overflow_preview[_preview_key] = truncated
                else:
                    raise
            result = SendResult(success=True, message_id=message_id)
            if finalize:
                await self._record_response_async((metadata or {}).get("reply_to_message_id"), result, content, True)
            return result
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to edit Discord message %s: %s", self.name, message_id, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    @staticmethod
    def _is_reply_reference_rejected(err: Exception) -> bool:
        """Discord refused the reply anchor: system-message target (50035) or deleted target (10008)."""
        err_text = str(err)
        return (
            "error code: 50035" in err_text and "Cannot reply to a system message" in err_text
        ) or "error code: 10008" in err_text

    @staticmethod
    def _is_length_overflow_error(err: Exception) -> bool:
        """True when a Discord edit/send failed for >2,000 chars: code 50035 plus the length phrasing,
        so other 50035 validation errors (e.g. bad reply reference) aren't mistaken for overflow."""
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
        formatted = self.format_message(content)
        chunks = self._cap_split_chunks(self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH))
        if len(chunks) <= 1:
            # Defensive: pre-flight should guarantee >1 chunk; otherwise edit normally.
            await msg.edit(content=chunks[0] if chunks else formatted)
            return SendResult(success=True, message_id=message_id)
        try:
            await msg.edit(content=chunks[0])
        except Exception as e:
            logger.error(
                "[%s] Overflow split: first-chunk edit failed: %s", self.name, e, exc_info=True,
            )
            return SendResult(success=False, error=str(e))
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
                logger.warning(
                    "[%s] Overflow continuation send failed (%s); retrying without reply reference",
                    self.name, send_err,
                )
                try:
                    sent = await channel.send(content=chunk, reference=None)
                except Exception as retry_err:
                    logger.warning(
                        "[%s] Overflow split: stopped at %d/%d chunks delivered: %s",
                        self.name, delivered, len(chunks), retry_err,
                    )
                    last_id = continuation_ids[-1] if continuation_ids else message_id
                    return SendResult(
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
        if not _looks_like_nonconversational_history_message(content):
            self._last_self_message_id[str(channel.id)] = last_id
        logger.debug(
            "[%s] Overflow split delivered %d chunks; last_id=%s", self.name, delivered, last_id,
        )
        return SendResult(
            success=True, message_id=last_id, continuation_message_ids=tuple(continuation_ids),
        )
