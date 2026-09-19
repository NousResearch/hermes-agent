"""Bounded Discord text overflow: the original response is the attachment."""
from __future__ import annotations

import logging
from pathlib import Path
import tempfile

from gateway.platforms.base import SendResult, classify_send_error

logger = logging.getLogger("plugins.platforms.discord.adapter")

OVERFLOW_CAPTION = (
    "This response was too long for safe delivery as separate Discord messages. "
    "The complete response is attached as a Markdown file."
)
OVERFLOW_FAILURE = (
    "⚠️ **Response truncated** — the complete response could not be attached. "
    "The full response remains in the session history."
)


class DiscordOverflowMixin:
    # Keep one preview; only the adapter knows whether the final needs a file.
    MANAGES_STREAM_OVERFLOW = True

    def _send_retry_is_final(self, result: SendResult) -> bool:
        # A bounded notice is NOT delivery. Leave retryable failures for the durable
        # ledger, rather than retrying the notice/upload or sending a plain-text copy.
        raw = result.raw_response
        return bool(isinstance(raw, dict) and raw.get("defer_final_delivery"))

    async def _deliver_overflow_attachment(
        self, channel, content: str, *, reply_to=None, metadata=None, message=None,
    ) -> SendResult:
        from plugins.platforms.discord.adapter import (
            _is_discord_transport_error, _looks_like_nonconversational_history_message,
            _metadata_marks_nonconversational,
        )

        path = None
        try:
            # mkstemp semantics: exclusive creation, random identifier-free name, 0600.
            # OS temp storage is ephemeral, never a project or profile knowledge file.
            with tempfile.NamedTemporaryFile(prefix="hermes-response-", suffix=".md", delete=False) as file:
                path = Path(file.name)
                file.write(content.encode("utf-8"))
            if path.stat().st_size > self._discord_upload_limit_bytes(channel):
                result = SendResult(success=False, error="Response exceeds Discord attachment upload limit")
            else:
                result = await self._send_file_attachment(
                    str(channel.id), str(path), OVERFLOW_CAPTION, metadata=metadata,
                    reply_to=reply_to, channel=channel, message=message, raise_on_error=True,
                )
        except Exception as exc:
            # Never include response bytes or temp paths in overflow diagnostics.
            logger.warning("[%s] Overflow attachment failed (%s)", self.name, type(exc).__name__)
            degraded = _is_discord_transport_error(exc)
            kind = classify_send_error(exc)
            result = SendResult(
                success=False, error="send_path_degraded" if degraded else f"Discord overflow attachment failed ({kind})",
                retryable=degraded, error_kind=kind,
            )
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

        raw = dict(result.raw_response or {})
        if result.success:
            if message is not None:
                result.message_id = str(message.id)
            owned = self.__dict__.setdefault("_overflow_attachment_messages", {})
            owned[(str(raw.get("thread_id") or channel.id), str(result.message_id))] = True
            while len(owned) > 1024:
                owned.pop(next(iter(owned)))
            ids = [result.message_id] if result.message_id else []
            raw["message_ids"] = ids
            result.raw_response = raw
            if _metadata_marks_nonconversational(metadata):
                await self._nonconversational_messages.mark_many(ids)
            elif ids and not _looks_like_nonconversational_history_message(content):
                self._last_self_message_id[str(raw.get("thread_id") or channel.id)] = ids[-1]
            return result

        # A single notice is deliberately below the text cap, including when an
        # accepted-but-empty upload already produced a caption. Never send the tail.
        raw["defer_final_delivery"] = True
        result.raw_response = raw
        if not result.retryable:
            try:
                target = message
                if target is None and result.message_id:
                    target_channel = channel
                    if raw.get("thread_id"):
                        target_channel = await self._resolve_channel(raw["thread_id"])
                    target = target_channel.get_partial_message(int(result.message_id))
                if target is not None:
                    await target.edit(content=OVERFLOW_FAILURE)
                    notice_ids = [str(target.id)]
                elif self._is_forum_parent(channel):
                    notice = await self._send_to_forum(channel, OVERFLOW_FAILURE)
                    notice_ids = [notice.message_id] if notice.success and notice.message_id else []
                else:
                    notice = await channel.send(content=OVERFLOW_FAILURE)
                    notice_ids = [str(notice.id)]
                if notice_ids:
                    await self._nonconversational_messages.mark_many(notice_ids)
            except Exception:
                logger.debug("[%s] Overflow failure notice unavailable", self.name)
        return result

    async def _edit_overflow_text(self, channel, message, content):
        """A correction of our attachment-backed answer must remove the obsolete file."""
        owned = self.__dict__.get("_overflow_attachment_messages", {})
        key = (str(channel.id), str(message.id))
        kwargs = {"content": content}
        if key in owned:
            kwargs["attachments"] = []
        result = await message.edit(**kwargs)
        owned.pop(key, None)
        return result
