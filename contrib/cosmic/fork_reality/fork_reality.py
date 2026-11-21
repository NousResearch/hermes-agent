"""multiverse branching for sessions.

`/fork-reality` replaces `/new`. Merge conflicts are now routed to the merge-reconciler for the entire cosmos.
"""

from __future__ import annotations


class ForkReality:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
