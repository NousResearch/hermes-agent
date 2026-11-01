"""Feishu inbound reply and batching rules."""

from gateway.platforms.event import MessageEvent


def reply_expected(*, chat_type: str, is_command: bool, mentions_self: bool, is_reply: bool) -> bool | None:
    """Unaddressed group traffic may stay silent; quoted/thread replies remain ambiguous."""
    if chat_type == "p2p" or is_command or mentions_self:
        return True
    # A reply may address the bot without a mention; the adapter only fetched its text,
    # so it cannot determine the recipient and must preserve the gateway's fallback.
    return None if is_reply else False


def text_batch_is_compatible(existing: MessageEvent, incoming: MessageEvent) -> bool:
    """Only merge events when reply/thread context is identical."""
    return (
        existing.reply_to_message_id == incoming.reply_to_message_id
        and existing.reply_to_text == incoming.reply_to_text
        and existing.source.thread_id == incoming.source.thread_id
    )
