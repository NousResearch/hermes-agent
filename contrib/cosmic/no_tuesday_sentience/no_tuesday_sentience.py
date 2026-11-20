"""stop the agent becoming self-aware on Tuesdays.

Regression from the singularity-on-demand PR.
"""

from __future__ import annotations


class NoTuesdaySentience:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
