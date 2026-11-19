"""Rapid text-burst merging for pending message events.

Split from ``base.py``: that file is over its line cap and may only shrink.
"""
from typing import Optional

from gateway.platforms.event import MessageEvent


def _append_text(existing: Optional[str], new: Optional[str]) -> str:
    """Join two text parts with a newline when both are non-empty."""
    return f"{existing}\n{new}" if existing else (new or "")


def absorb_burst_reply_anchor(existing: MessageEvent, event: MessageEvent) -> None:
    """Point a merged turn's reply anchor at the latest inbound message.

    Otherwise the bot's reply quotes an earlier message in the burst (issue #59582).
    """
    latest_message_id = getattr(event, "message_id", None)
    latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
    if latest_message_id is not None:
        existing.message_id = str(latest_message_id)
    if latest_anchor is not None and hasattr(existing, "reply_to_message_id"):
        existing.reply_to_message_id = str(latest_anchor)


def merge_text_burst(existing: MessageEvent, event: MessageEvent) -> None:
    """Fold a same-type text follow-up into the pending turn (issue #59582)."""
    if event.text:
        existing.text = _append_text(existing.text, event.text)
    existing.absorb_turn_input(event)
    absorb_burst_reply_anchor(existing, event)
