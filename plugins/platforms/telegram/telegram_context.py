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


def should_process_message(adapter: "TelegramAdapter", message: "Message") -> bool:
    """Apply Telegram group trigger rules: DMs unrestricted; group messages pass ``allowed_chats`` (hard gate; only
    the ``guest_mode`` @mention bypass crosses it) and then any of free_response chat/topic, ``require_mention``
    off, reply to the bot, @mention (incl. ``/cmd@botname``), or a wake-word match."""
    # Learn the live handle BEFORE any mention gate routes on it, then drop our own echoed messages.
    # Filter out the bot's own messages (returned by getUpdates in some environments like
    # groups/supergroups where the bot can see its own messages). Without this, outbound messages are
    # counted as incoming unread in the Hermes inbox (#52363). Otherwise a BotFather rename leaves the
    # stale handle in place and the exclusive-mention gate reads a message addressed to us as one
    # addressed to some other bot.
    adapter._observe_bot_identity_from_message(message)
    if adapter._is_own_message(message):
        return False
    if not adapter._is_group_chat(message):
        return True
    if command_targets_other_bot(adapter, message):
        adapter._schedule_bot_identity_recheck()
        return False
    thread_id = adapter._effective_message_thread_id(message)
    if adapter._topic_gates_pass(thread_id, warn_non_numeric=True) is False:
        return False
    chat_id_str = adapter._chat_id_str(message)
    if adapter._telegram_exclusive_bot_mentions() and adapter._explicit_bot_mentions_exclude_self(message):
        return False
    # Resolve once; _message_mentions_bot is not re-called below in guest mode.
    guest_mention = adapter._is_guest_mention(message)
    # allowed_chats whitelist: outside chats pass only via the guest-mode explicit mention.
    allowed = adapter._telegram_allowed_chats()
    if allowed and chat_id_str not in allowed:
        return guest_mention
    if guest_mention or chat_id_str in adapter._telegram_free_response_chats() or adapter._telegram_is_free_response_topic(message):
        return True
    # Bot-to-bot loop breaker: another bot must explicitly @mention us; its quote-reply or
    # plain chatter does not count (two bots answering each other's replies never stop otherwise).
    if adapter._bot_sender_suppressed(message):
        return False
    if not adapter._telegram_require_mention() or adapter._is_reply_to_bot(message):
        return True
    if not adapter._telegram_guest_mode() and adapter._message_mentions_bot(message):
        return True
    return adapter._message_matches_mention_patterns(message)


def command_targets_other_bot(adapter: "TelegramAdapter", message: "Message") -> bool:
    """A leading command's explicit recipient wins over mentions in its arguments."""
    own = adapter._current_bot_username()
    if not own:
        return False
    address = r"/[^\s/@]+@([A-Za-z0-9_]+)"
    for text, entities in adapter._entity_sources(message):
        prefix = re.match(rf"(?i)^\s*(?:@{re.escape(own)}\b[,:\-]*\s*)*", text)
        if not entities:
            command = re.match(address, text[prefix.end():])
            if command and command[1].lower() != own:
                return True
            continue
        offset = len(text[:prefix.end()].encode("utf-16-le")) // 2
        # Code and URL entities do not turn quoted command-looking text into an address.
        for entity in entities:
            if adapter._entity_type(entity) != "bot_command" or getattr(entity, "offset", None) != offset:
                continue
            command = re.match(address, adapter._entity_span(text, entity) or "")
            if command and command[1].lower() != own:
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
