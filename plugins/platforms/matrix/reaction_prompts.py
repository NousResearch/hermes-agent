"""Matrix reaction prompt delivery and seeded controls."""

from __future__ import annotations

from collections.abc import Iterable
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)


class MatrixReactionPromptMixin:
    """Deliver registered reaction prompts and their initial controls."""

    if TYPE_CHECKING:
        _approval_timeout_seconds: int
        send: Any
        _send_reaction: Any
        _schedule_reaction_redaction: Any

    async def _send_reaction_prompt(
        self, chat_id: str, text: str, metadata: Optional[dict], make_prompt, registry: dict, emojis: Iterable[str],
        label: str) -> SendResult:
        """Send *text*, register ``make_prompt(message_id, requester, expires_at)`` under
        the resulting event, then seed the bot's reaction controls (recording their IDs)."""
        result = await self.send(chat_id, text, metadata=metadata)
        if not result.success or not result.message_id:
            return result
        prompt = make_prompt(
            result.message_id, str((metadata or {}).get("requester_user_id") or "") or None,
            time.monotonic() + max(self._approval_timeout_seconds, 0))
        registry[result.message_id] = prompt
        for emoji in emojis:
            if prompt.resolved or registry.get(result.message_id) is not prompt:
                break
            try:
                reaction_event_id = await self._send_reaction(chat_id, result.message_id, emoji)
                if reaction_event_id:
                    if prompt.resolved or registry.get(result.message_id) is not prompt:
                        self._schedule_reaction_redaction(chat_id, str(reaction_event_id), f"{label} resolved")
                        break
                    prompt.bot_reaction_events[emoji] = str(reaction_event_id)
            except Exception as exc:
                logger.debug("Matrix: failed to add %s reaction %s: %s", label, emoji, exc)
        return result
