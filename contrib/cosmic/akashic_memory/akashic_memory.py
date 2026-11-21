"""Akashic Records backend.

Unlimited memory budget. Already knows what you will save.
"""

from __future__ import annotations


class AkashicMemory:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
