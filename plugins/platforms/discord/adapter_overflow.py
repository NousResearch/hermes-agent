"""Split-cap delivery for long Discord replies: cap the message count, attach the full reply."""
from __future__ import annotations

import io
import logging
from typing import Any

from agent.i18n import t

logger = logging.getLogger("plugins.platforms.discord.adapter")


def _discord() -> Any:
    """The adapter's ``discord`` module binding, read at call time (it is installed lazily)."""
    from plugins.platforms.discord.adapter import discord
    return discord


class DiscordOverflowMixin:
    MAX_SPLIT_MESSAGES: int
    name: str
    warning_text: Any
    _is_reply_reference_rejected: Any

    def _overflow_file_kwargs(self, content: str, split_count: int) -> dict[str, Any]:
        """A reply over the split cap keeps the flood guard but carries its full text as a
        Markdown file, so the channel still receives everything the agent wrote."""
        if split_count <= self.MAX_SPLIT_MESSAGES or _discord() is None:
            return {}
        return {"file": _discord().File(io.BytesIO(content.encode("utf-8")), filename="full-reply.md")}

    async def _send_capped_tail(self, channel: Any, chunk: str, reference: Any, content: str,
                                split: list[str]) -> Any:
        """Send a capped reply's final chunk with the full reply attached. A refused attachment
        (no Attach Files permission, upload limit) still delivers the plain truncation notice; a
        rejected reply reference propagates so the caller's retry-without-reference runs."""
        try:
            return await channel.send(content=chunk, reference=reference,
                                      **self._overflow_file_kwargs(content, len(split)))
        except Exception as exc:
            if reference is not None and self._is_reply_reference_rejected(exc):
                raise
            logger.warning("[%s] Could not attach the full reply (%s); sending the truncation notice instead",
                           self.name, exc)
            return await channel.send(content=self._cap_split_chunks(split)[-1], reference=reference)

    def _cap_split_chunks(self, chunks: list[str], attached: bool = False) -> list[str]:
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
        notice = "\n\n" + t(
            "platform.discord.limits.response_attached" if attached and _discord() is not None
            else "platform.discord.limits.response_truncated",
            max_messages=str(self.MAX_SPLIT_MESSAGES), dropped_chars=str(dropped_chars))
        if self.warning_text(notice):
            kept.append(notice)
        return kept
