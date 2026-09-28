"""Shared syntax validation for explicit Schedule destinations."""

from __future__ import annotations

import re

def parse_telegram_numeric_target(value: str):
    match = re.fullmatch(r"\s*(-?\d+)(?::(\d+))?\s*", value)
    return (match.group(1), match.group(2)) if match else None


DELIVERY_CHANNELS = frozenset({"slack", "telegram"})

_SLACK_THREAD_TS = re.compile(r"\d+\.\d+")


def parse_explicit_schedule_delivery(value: str) -> tuple[str, str]:
    """Return one validated channel/target pair without resolving access."""

    text = str(value or "").strip()
    channel, separator, target = text.partition(":")
    channel = channel.lower().strip()
    target = target.strip()
    if (
        not separator
        or channel not in DELIVERY_CHANNELS
        or not target
        or "," in target
    ):
        raise ValueError("Schedule delivery target is invalid")
    if channel == "telegram":
        if parse_telegram_numeric_target(target) is None:
            raise ValueError("Use telegram:<numeric-chat-id>[:topic-id] from send_message list.")
    elif not re.fullmatch(r"[CDG][A-Z0-9]+(?::\d+\.\d+)?", target):
        raise ValueError("Use slack:<channel-id>[:thread-timestamp] from send_message list.")
    return channel, target
