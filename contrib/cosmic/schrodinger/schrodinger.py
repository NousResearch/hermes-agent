"""Schrödinger's PR.

This PR is both merged and closed until someone runs `gh pr view`.
"""

from __future__ import annotations


class Schrodinger:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
