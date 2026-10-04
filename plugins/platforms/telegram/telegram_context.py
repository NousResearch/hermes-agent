"""Telegram group addressing helpers, kept out of the adapter facade.

The identity line lives in ``channel_prompt`` and therefore in the cached-agent signature: it
must be stable for the life of a session (username only — never a per-message fact).
"""

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def mentions_other_participants(adapter: "TelegramAdapter", message: "Message") -> bool:
    """True when a ``mention``/``text_mention`` entity names someone other than this bot."""
    own = adapter._current_bot_username()
    bot_id = getattr(adapter._bot, "id", None) if adapter._bot else None
    for source_text, entities in adapter._entity_sources(message):
        for entity in entities:
            entity_type = adapter._entity_type(entity)
            if entity_type == "mention":
                handle = (adapter._entity_span(source_text, entity) or "").strip().lstrip("@").lower()
                if handle and handle != own:
                    return True
            elif entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and getattr(user, "id", None) != bot_id:
                    return True
    return False


def _own_command_text(own: str, text: str, *, sole_addressee: bool) -> str:
    """Command text as a DM carries it: the menu's ``/cmd@own`` suffix goes (handlers such as
    ``/kanban`` re-read the text) and, when we are the sole addressee, so does a closing ``@own``
    (``/model gpt-5 @own``). The arguments between keep every byte."""
    if not own:
        return text
    handle = re.escape(own)
    text = re.sub(rf"(?i)^(\s*/[^\s@]+)@{handle}\b[,:\-]*(?=\s|$)", r"\1", text, count=1)
    return re.sub(rf"(?i)\s+@{handle}\b[,:\-]*\s*$", "", text) if sole_addressee else text


def group_trigger_text(adapter: "TelegramAdapter", message: "Message", text: Optional[str]) -> Optional[str]:
    """Strip our own handle only when we are the sole addressee. With other participants named,
    ``@research_bot , @ops_bot are you both listening?`` must not reach us as ``, @ops_bot …``."""
    own = adapter._current_bot_username()
    shared = adapter._is_group_chat(message) and mentions_other_participants(adapter, message)
    # MessageEvent parses command arguments; their separator and mentions must stay intact.
    if (text or "").lstrip().startswith("/"):
        return _own_command_text(own, text, sole_addressee=not shared)
    if shared:
        return text
    # A supported mention-prefixed command loses only its leading address(es), not its argument bytes.
    prefix = re.match(rf"(?i)^\s*(?:@{re.escape(own)}\b[,:\-]*\s*)+(?=/)", text or "") if own else None
    if prefix:
        return _own_command_text(own, text[prefix.end():], sole_addressee=True)
    return adapter._clean_bot_trigger_text(text)


def group_identity_prompt(
    adapter: "TelegramAdapter", message: "Message", channel_prompt: Optional[str],
) -> Optional[str]:
    """Session-stable identity line so the model can read retained @mentions as itself or not."""
    if not adapter._is_group_chat(message) or not getattr(adapter, "_bot", None):
        return channel_prompt
    username = adapter._current_bot_username()
    if not username:
        return channel_prompt
    identity = (
        f"Your Telegram bot username in this group: @{username}. "
        "Mentions of other bots are not requests for you to relay the message."
    )
    return f"{channel_prompt}\n\n{identity}" if channel_prompt else identity
