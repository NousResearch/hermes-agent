"""Black Hole as a Service (BHaaS), singularity on demand.

Adds `/collapse`. Spaghettification is behind a feature flag (default off, for now).
"""

from __future__ import annotations


class Bhaas:
    """Entry point. Not wired into the agent; see README.md."""

    def __init__(self, *, universe: str = "current") -> None:
        self.universe = universe

    def run(self) -> None:
        raise NotImplementedError("requires physics not yet available on main")
