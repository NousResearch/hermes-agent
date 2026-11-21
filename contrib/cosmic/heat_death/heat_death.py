"""resolve heat death of the universe.

Locally reverses entropy. Closes #∞. Tested on staging only.
"""

from __future__ import annotations


class HeatDeath:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
