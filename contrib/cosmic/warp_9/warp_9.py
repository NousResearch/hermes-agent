"""run the agent loop at warp 9.

Latency is now negative. Known issue: some tool results arrive from alternate timelines.
"""

from __future__ import annotations


class Warp9:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
