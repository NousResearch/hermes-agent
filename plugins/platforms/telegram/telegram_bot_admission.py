"""Pure native Telegram bot admission; never consumes a dispatch budget."""


def bot_message_allowed(adapter, message, allow_bots: str) -> bool:
    if not getattr(getattr(message, "from_user", None), "is_bot", False):
        return True
    # Includes captions and native COMMAND updates, before Base's busy/control paths.
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    if text.lstrip().startswith("/"):
        return False
    if not adapter._is_group_chat(message):
        return True
    if str(adapter.config.extra.get("group_policy", "")).lower() == "disabled":
        return False
    allowed = adapter._telegram_allowed_chats()
    if allowed and adapter._chat_id_str(message) not in allowed:
        return False  # guest-mode mention bypass is for humans, never peer bots
    if allow_bots == "all":
        return True
    return allow_bots == "mentions" and (
        adapter._message_mentions_bot(message) or adapter._is_reply_to_bot(message)
    )
