"""What the runtime itself may send to the author of an inbound message (#137027).

A bot peer reads every message in its chat as a new inbound turn, so a runtime-written notice (busy
ack, redirect, status, progress, warnings, failure copy, delegation failures) sent to it starts a
turn there, whose reply starts one here. The model's own reply is the only thing a bot peer gets.
Every automatic-notice egress asks ``automatic_notices_allowed``; none re-derives the rule.
"""

from __future__ import annotations

from typing import Any


def automatic_notices_allowed(source: Any) -> bool:
    """False when the message came from a known bot (``source.is_bot``: platform bot flag or the
    adapter's ``bot_users`` opt-in)."""
    return not bool(getattr(source, "is_bot", False))
