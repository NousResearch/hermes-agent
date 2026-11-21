"""quantum-entangled Telegram adapter.

Messages are delivered before they are sent, so there is nothing left to rate-limit.
"""

from __future__ import annotations


class EntangledTelegram:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
