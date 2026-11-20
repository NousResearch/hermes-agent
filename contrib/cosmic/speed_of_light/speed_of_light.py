"""bump the speed of light to 3.1e8 m/s.

Small constant change. CI should be fine.
"""

from __future__ import annotations


class SpeedOfLight:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
