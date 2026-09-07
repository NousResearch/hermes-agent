"""Pure text presentation for Group Chat views; no authority or delivery logic."""

from __future__ import annotations

from datetime import datetime
import math

from agent.i18n import t


def text(key, **values):
    return t("gateway.group_presentation." + key, **values)


def action(key, command, **values):
    return text("action", label=text(key, **values), command=f"`{command}`")


def actions(command, *entries):
    return ["", text("actions"), *entries, action("help", f"{command} help")]


def timestamp(value, *, milliseconds=False):
    """Normalize known wire units; reject unknown or timezone-ambiguous values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        if not isinstance(value, str) or milliseconds:
            return None
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                return None
            number = instant.timestamp()
        except (ValueError, OverflowError, OSError):
            return None
    if milliseconds:
        number /= 1000
    return number if math.isfinite(number) and number > 0 else None


def event_age(event, *, desktop, now):
    stamp = timestamp(event.get("at") if desktop else event.get("created_at"), milliseconds=desktop)
    if stamp is None or stamp > now:
        return ""
    seconds = now - stamp
    if seconds < 60:
        return text("age_now")
    for unit, width, ceiling in (("minutes", 60, 3600), ("hours", 3600, 86400), ("days", 86400, math.inf)):
        if seconds < ceiling:
            return text("age_" + unit, count=int(seconds // width))
    return ""


def recent_heading(events, *, desktop, now):
    # The last visible event is authoritative. An older known time must not
    # masquerade as the latest when the newest event has no valid timestamp.
    age = event_age(events[-1], desktop=desktop, now=now) if events else ""
    return text("recent_latest", age=age) if age else text("recent")
