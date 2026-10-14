"""Telegram group addressing helpers, kept out of the adapter facade.

The identity line lives in ``channel_prompt`` and therefore in the cached-agent signature: it
must be stable for the life of a session (username only — never a per-message fact).
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def mentions_other_participants(adapter: "TelegramAdapter", message: "Message") -> bool:
    """True when a ``mention``/``text_mention`` entity names someone other than this bot."""
    return any(
        _is_other_participant(adapter, text, entity)
        for text, entity in _mention_entities(adapter, message)
    )


def _mention_entities(adapter: "TelegramAdapter", message: "Message"):
    """``(source_text, entity)`` for every ``mention``/``text_mention`` in the text and caption."""
    for source_text, entities in adapter._entity_sources(message):
        for entity in entities:
            if adapter._entity_type(entity) in {"mention", "text_mention"}:
                yield source_text, entity


def _is_other_participant(adapter: "TelegramAdapter", source_text: str, entity) -> bool:
    if adapter._entity_type(entity) == "text_mention":
        user = getattr(entity, "user", None)
        bot_id = getattr(adapter._bot, "id", None) if adapter._bot else None
        return user is not None and getattr(user, "id", None) != bot_id
    handle = (adapter._entity_span(source_text, entity) or "").strip().lstrip("@").lower()
    return bool(handle) and handle != adapter._current_bot_username()


def starts_with_other_participant_mention(adapter: "TelegramAdapter", message: "Message") -> bool:
    return any(
        getattr(entity, "offset", None) == len(text) - len(text.lstrip())
        and _is_other_participant(adapter, text, entity)
        for text, entity in _mention_entities(adapter, message)
    )


def _replies_to_other_member(adapter: "TelegramAdapter", message: "Message") -> bool:
    """A reply to another known member's message. In a forum topic Telegram also sets ``reply_to_message``
    to the topic's root on every plain message; that anchor is not a reply to a person."""
    replied = getattr(message, "reply_to_message", None)
    if replied is None or getattr(replied, "from_user", None) is None or adapter._is_reply_to_bot(message):
        return False
    topic_root = (
        getattr(message, "is_topic_message", False)
        and replied.message_id == getattr(message, "message_thread_id", None)
    )
    return not topic_root


def group_reply_expected(adapter: "TelegramAdapter", message: "Message", chat_type: str) -> Optional[bool]:
    """Whether an admitted message needs a reply; None keeps the visible silence fallback."""
    if chat_type != "group":
        return True if chat_type == "dm" else None
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    addressed_to_bot = (
        text.lstrip().startswith("/")
        or adapter._is_reply_to_bot(message)
        or adapter._message_mentions_bot(message)
        or adapter._message_matches_mention_patterns(message)
    )
    if addressed_to_bot:
        return True
    if not adapter._bot or not adapter._current_bot_username():
        return None
    if _replies_to_other_member(adapter, message) or starts_with_other_participant_mention(adapter, message):
        return False
    return None if "?" in text else False


def group_trigger_text(adapter: "TelegramAdapter", message: "Message", text: Optional[str]) -> Optional[str]:
    """Strip our own handle only when we are the sole addressee. With other participants named,
    ``@research_bot , @ops_bot are you both listening?`` must not reach us as ``, @ops_bot …``."""
    if adapter._is_group_chat(message) and mentions_other_participants(adapter, message):
        return text
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
