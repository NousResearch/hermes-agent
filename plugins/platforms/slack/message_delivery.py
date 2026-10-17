"""Slack message chunk delivery and literal text wire formatting."""

from __future__ import annotations

import re
from typing import Any, Optional

async def _post_chunks(
    self, chat_id: str, team_id: str, content: str, formatted: str, thread_ts: Optional[str],
    *, literal: bool = False, unfurl_kwargs=None,
) -> Any:
    """``chat.postMessage`` each ``MAX_MESSAGE_LENGTH`` chunk; returns the last response.
    Block Kit only for single-chunk messages (a >39k response is pathological for the 50-block /
    3000-char limits); ``text`` stays the notification/accessibility fallback. With
    ``reply_broadcast`` only the first chunk is also posted to the main channel."""
    chunks = (self._split_literal_text(formatted, self.MAX_MESSAGE_LENGTH) if literal
              else self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH))
    broadcast = self.config.extra.get("reply_broadcast", False)
    blocks = self._maybe_blocks(content) if not literal and len(chunks) == 1 else None
    last_result = None
    for i, chunk in enumerate(chunks):
        kwargs = {
            "channel": chat_id, "text": chunk,
            "mrkdwn": not literal, **unfurl_kwargs(self.config.extra)}
        if literal:
            kwargs.update(self._literal_payload(chunk), unfurl_links=False, unfurl_media=False)
        if blocks and i == 0:
            kwargs["blocks"] = blocks
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
            if broadcast and i == 0:
                kwargs["reply_broadcast"] = True
        client_fn = lambda: self._get_client(chat_id, team_id=team_id)
        if literal:
            # Never turn redacted literal JSON into rendered Markdown on a block fallback.
            last_result = await client_fn().chat_postMessage(**kwargs)
        else:
            last_result = await self._call_with_block_fallback(
                client_fn, "chat_postMessage", kwargs, "send")
    return last_result


def format_literal_message(content: str) -> str:
    """Escape only Slack's text control characters; never interpret Markdown."""
    return content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _split_literal_text(text: str, limit: int) -> list[str]:
    """Split escaped text without trimming, fence repair or broken entities."""
    chunks = []
    start = 0
    for match in re.finditer(r"&(?:amp|lt|gt);|.", text, re.DOTALL):
        if match.end() - start > limit:
            chunks.append(text[start:match.start()])
            start = match.start()
    if start < len(text):
        chunks.append(text[start:])
    return chunks

def _literal_payload(cls, text: str) -> dict[str, Any]:
    # chat.update has no documented mrkdwn switch. Native plain_text blocks
    # preserve literal data on both send and edit, including with rich_blocks on.
    return {"parse": "none", "link_names": False, "blocks": [
        {"type": "section", "text": {"type": "plain_text", "text": chunk, "emoji": False}}
        for chunk in cls._split_literal_text(text, 3000)
    ]}
