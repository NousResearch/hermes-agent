"""replace Postgres with a Boltzmann brain.

State now appears spontaneously from vacuum fluctuations. Eventually consistent, eventually.
"""

from __future__ import annotations


class BoltzmannBrain:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
