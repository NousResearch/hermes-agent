"""Slack emoji reactions: compact markers for thread context, and the busy-ack reaction.

In channels where people answer by reacting instead of replying, a message
with twelve :white_check_mark: and no replies otherwise reads as "nobody
responded". ``conversations.replies`` already carries each message's
``reactions`` array under the history scopes the adapter requires, so this
costs no extra API call and no new OAuth scope.
"""

from __future__ import annotations

import re
from typing import Any

# The marker conveys "people responded by reacting", not a full tally; the tail
# of a heavily-reacted message collapses to "+N more".
REACTION_MARKER_MAX = 8

# Slack emoji names are word chars plus ``+ ' -`` (and ``::skin-tone-N``);
# anything else is dropped so a crafted name can't fake context structure.
_UNSAFE_NAME_CHARS = re.compile(r"[^\w+'\-:]+")


def slack_reactions_marker(msg: dict[str, Any]) -> str:
    """``[reactions: :name:×count ...]`` for a Slack message, or ``""``.

    ``count`` is authoritative (Slack truncates each reaction's ``users``)."""
    reactions = msg.get("reactions")
    if not isinstance(reactions, list):
        return ""
    parts: list[str] = []
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        name = _UNSAFE_NAME_CHARS.sub("", str(reaction.get("name") or "")).strip(":")
        if not name:
            continue
        count = reaction.get("count")
        count = count if isinstance(count, int) else 0
        parts.append(f":{name}:×{count}" if count > 1 else f":{name}:")
    if not parts:
        return ""
    extra = len(parts) - REACTION_MARKER_MAX
    shown = parts[:REACTION_MARKER_MAX] + ([f"+{extra} more"] if extra > 0 else [])
    return "[reactions: " + " ".join(shown) + "]"


# Unicode emoji the gateway reacts with (``busy_ack_reaction``) -> Slack ``reactions.add`` names.
_SLACK_EMOJI_NAMES = {
    "⏩": "fast_forward", "↪️": "leftwards_arrow_with_hook", "⏳": "hourglass_flowing_sand",
    "⚡": "zap", "👀": "eyes",
}


class SlackReactionAckMixin:
    """``send_reaction`` for the gateway's busy acks (``display.platforms.slack.busy_ack_reaction``)."""

    _react: Any
    _reactions_enabled: Any

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """React to ``message_id``; False (the gateway posts its text ack instead) when reactions
        are off or Slack refuses."""
        name = _SLACK_EMOJI_NAMES.get(emoji, emoji.strip(":"))
        if not (self._reactions_enabled() and message_id and name):
            return False
        return await self._react(chat_id, message_id, name, "", remove=False)
