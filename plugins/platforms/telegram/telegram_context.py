"""Telegram group addressing helpers, kept out of the adapter facade.

The identity line lives in ``channel_prompt`` and therefore in the cached-agent signature: it
must be stable for the life of a session (username only — never a per-message fact).
"""

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def set_peer_identity_provider(
    adapter: "TelegramAdapter",
    provider: Callable[[str], tuple[tuple[str, str], ...]],
) -> None:
    """Attach the multiplex Gateway's live same-platform roster lookup."""
    adapter._peer_identity_provider = provider


def _configured_chat_ids(adapter: "TelegramAdapter", key: str) -> set[str]:
    """Chat IDs from this adapter's YAML snapshot, without ambient env fallback."""
    raw = (getattr(getattr(adapter, "config", None), "extra", None) or {}).get(key)
    if raw is None:
        return set()
    values = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
    return {str(value).strip() for value in values if str(value).strip()}


def group_peer_identity(adapter: "TelegramAdapter", chat_id: str) -> Optional[str]:
    """This bot's live handle when its profile-scoped admission snapshot allows ``chat_id``."""
    allowed = getattr(adapter, "_peer_allowed_chats_snapshot", None)
    if allowed is None:
        allowed = _configured_chat_ids(adapter, "allowed_chats")
    group_allowed = getattr(adapter, "_peer_group_allowed_chats_snapshot", None)
    if group_allowed is None:
        group_allowed = _configured_chat_ids(adapter, "group_allowed_chats")
    if not (allowed or group_allowed):
        return None
    if allowed and chat_id not in allowed:
        return None
    if group_allowed and chat_id not in group_allowed:
        return None
    username = adapter._current_bot_username()
    return f"@{username}" if username else None


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
    provider = getattr(adapter, "_peer_identity_provider", None)
    if callable(provider):
        try:
            peers = tuple(sorted(set(provider(adapter._chat_id_str(message)))))
        except Exception:
            peers = ()
        if peers:
            roster = ", ".join(
                f"`{profile}` = {handle}" for profile, handle in peers
            )
            identity += (
                f"\nOther Hermes profiles configured for this group: {roster}. "
                "These are addressing hints, not proof that a bot is currently a group member. "
                "Use these @usernames only for a visible in-group handoff; when message_agent "
                "is available, use it instead if delivery acknowledgement is required. Never use "
                "both paths for one handoff."
            )
    return f"{channel_prompt}\n\n{identity}" if channel_prompt else identity
