"""Maxwell's Demon process manager.

Sorts hot processes from cold ones. Removes the 4GiB heap limit thermodynamically.
"""

from __future__ import annotations


class MaxwellsDemon:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
