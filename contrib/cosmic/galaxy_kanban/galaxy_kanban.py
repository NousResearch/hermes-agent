"""galaxy-scale kanban.

Each card is a star system. The crash-sweep cron now handles supernovae.
"""

from __future__ import annotations


class GalaxyKanban:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
