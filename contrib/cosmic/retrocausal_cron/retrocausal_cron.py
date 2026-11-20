"""retrocausal cron.

Jobs now complete before they are scheduled. Fixes missed cron slots by making them impossible.
"""

from __future__ import annotations


class RetrocausalCron:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
