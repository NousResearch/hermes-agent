"""Dyson Sphere connector.

Powers inference directly from the sun. Billed per stellar lifetime.
"""

from __future__ import annotations


class DysonSphere:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
