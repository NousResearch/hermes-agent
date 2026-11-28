"""Slack rendering of the gateway's long-turn history fold (``display.progress_fold_after``).

The gateway (``gateway/run_turn_history_fold.py``) edits the first folded progress message into a
summary of the turn's earlier work. On Slack that summary is a grey ``context`` block, so it reads
as quiet history rather than another progress bubble.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.slack.adapter")

# Slack caps a context block's text element at 3000 characters.
_CONTEXT_TEXT_LIMIT = 3000


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SlackProgressHistoryMixin:
    async def edit_progress_history(
        self, chat_id: str, message_id: str, lines: list[str], *,
        metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Edit ``message_id`` into one grey context block listing ``lines``."""
        blocked = self._outbound_blocked(chat_id, "history edit in")
        if blocked:
            return blocked
        text = "\n".join(_escape(line) for line in lines)[:_CONTEXT_TEXT_LIMIT] or " "
        kwargs = {"channel": chat_id, "ts": message_id, "text": text,
                  "blocks": [{"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}]}
        await self._call_with_block_fallback(lambda: self._client_for(chat_id, metadata), "chat_update", kwargs, "edit")
        return SendResult(success=True, message_id=message_id)
