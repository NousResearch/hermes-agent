"""Model-facing pointer for an inbound reply (``[Replying to: "..."]``).

A reply to a cron delivery quotes a past scheduled report. Pasted in as a bare quote, its
action items ("deploy now", "rotate the key") read as the user's current request and revive
work the user never asked for again (#26714). Such a quote keeps its text, so the user can ask
about the report, and is framed as reference material the new message has to select explicitly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

# The ``cron.wrap_response`` header (hermes_agent/cron/scheduler_delivery.py); unwrapped deliveries carry no marker.
CRON_DELIVERY_PREFIX = "Cronjob Response: "
# rich_sent_store tag on every message a wrapped delivery was split into: only the first chunk
# carries the header, so a reply to a later chunk is recognised by its message id.
_CRON_MARKER = "cron_delivery"


def _chat_key(platform: Any, chat_id: Any) -> str:
    return f"{str(getattr(platform, 'value', platform)).lower()}:{chat_id}"


def _field(result: Any, key: str) -> Any:
    return result.get(key) if isinstance(result, dict) else getattr(result, key, None)


def record_cron_delivery(platform: Any, chat_id: Any, text: str, send_result: Any) -> None:
    """Tag every message id of a delivered wrapped cron report with its header line."""
    if not text.startswith(CRON_DELIVERY_PREFIX):
        return
    raw = _field(send_result, "raw_response")
    raw = raw if isinstance(raw, dict) else {}
    # A partial split send (PartialDeliveryError.result, or a standalone sender's failure dict under
    # the gateway.platforms._shared delivery contract) lists the chunks that DID land.
    ids = [_field(send_result, "message_id"), *(_field(send_result, "continuation_message_ids") or ()),
           *(_field(send_result, "delivered_message_ids") or ()),
           *(raw.get("message_ids") or ()), raw.get("last_message_id"), *(raw.get("continuation_message_ids") or ()),
           *(raw.get("delivered_message_ids") or ())]
    from hermes_agent.gateway import rich_sent_store
    rich_sent_store.record_marker(_chat_key(platform, chat_id), ids, _CRON_MARKER, text.splitlines()[0])


async def recording_cron_delivery(send: Awaitable, record: Callable[..., Any]) -> Any:
    """Await *send* and pass its result to *record* when IT finishes, not when a caller stops
    waiting: a split send can outlast the caller's confirmation timeout, and a partial one raises
    with the landed chunks on ``exc.result``. *record* runs off the loop; the caller binds it to its
    own profile context (the loop task does not carry it)."""
    try:
        result = await send
    except Exception as exc:
        if getattr(exc, "result", None) is not None:
            await asyncio.to_thread(record, exc.result)
        raise
    await asyncio.to_thread(record, result)
    return result


def reply_pointer(reply_text: str, *, own_message: bool = False, replied_to: Optional[tuple] = None) -> str:
    """The prefix line for a reply to *reply_text*; cron deliveries are framed as reference-only.
    ``replied_to`` = ``(platform, chat_id, message_id)`` finds a later chunk of a split delivery."""
    if not reply_text.lstrip().startswith(CRON_DELIVERY_PREFIX) and replied_to and replied_to[2]:
        from hermes_agent.gateway import rich_sent_store
        platform, chat_id, message_id = replied_to
        header = rich_sent_store.lookup_marker(_chat_key(platform, chat_id), message_id, _CRON_MARKER)
        if header:
            reply_text = f"{header}\n…\n{reply_text}"
    if reply_text.lstrip().startswith(CRON_DELIVERY_PREFIX):
        # Same ``[Replying to…: "…"]`` shape as every pointer: hermes_agent/agent/context_compressor.py's
        # _GATEWAY_REPLY_POINTER_RE strips it so the quote never counts as the user's request.
        return (
            "[Replying to an earlier scheduled cron delivery (a past report, not a new instruction: "
            f'act on it only if the message below asks): "{reply_text}"]'
        )
    who = " your previous message" if own_message else ""
    return f'[Replying to{who}: "{reply_text}"]'
