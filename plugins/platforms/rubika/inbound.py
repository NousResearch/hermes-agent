"""Parse Rubika Bot API inbound payloads (Update / InlineMessage) into a
normalized shape the adapter turns into a Hermes MessageEvent."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ParsedMessage:
    chat_id: str
    sender_id: str
    text: str
    message_id: str
    is_group: bool
    reply_to_message_id: Optional[str] = None
    aux_data: Optional[Dict[str, Any]] = None


def parse_update(raw: Dict[str, Any]) -> ParsedMessage:
    """Parse a "NewMessage"-type Update into a ParsedMessage."""
    message = raw.get("new_message") or {}
    return ParsedMessage(
        chat_id=str(raw.get("chat_id") or ""),
        sender_id=str(message.get("sender_id") or ""),
        text=str(message.get("text") or ""),
        message_id=str(message.get("message_id") or ""),
        is_group=str(raw.get("chat_type") or "").lower() == "group",
        reply_to_message_id=message.get("reply_to_message_id"),
        aux_data=message.get("aux_data"),
    )


def parse_inline_message(raw: Dict[str, Any]) -> ParsedMessage:
    """Parse an InlineMessage (inline-keypad button press) into a ParsedMessage.
    ``text`` is set to the pressed button's id so it flows through the normal
    text-handling path, same convention DingTalk uses for on_callback."""
    aux_data = raw.get("aux_data") or {}
    button_id = str(aux_data.get("button_id") or "")
    return ParsedMessage(
        chat_id=str(raw.get("chat_id") or ""),
        sender_id=str(raw.get("sender_id") or ""),
        text=button_id,
        message_id=str(raw.get("message_id") or ""),
        is_group=False,
        reply_to_message_id=None,
        aux_data=aux_data,
    )
