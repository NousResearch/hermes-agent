"""Pure, code-defined contracts for automatic agent modes.

This module intentionally contains no configuration loading, prompt mutation, tool
selection, or routing heuristics.  It defines the closed set of modes that later
router integration may select.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ModeDefinition:
    """Immutable behavioral contract for one supported mode."""

    name: str
    objective: str
    stages: tuple[str, ...]
    verification: bool = True


class UnknownModeError(ValueError):
    """Raised when a caller requests a mode outside the closed registry."""


@dataclass(frozen=True)
class ModeRoutingDecision:
    """Result returned by the trusted Python routing seam."""

    mode: str
    goal: str
    route: str
    reason: str
    result: Optional[Any] = None


MODE_DEFINITIONS: Mapping[str, ModeDefinition] = MappingProxyType(
    {
        "thinking-expansion": ModeDefinition(
            name="thinking-expansion",
            objective="Expand and compare possibilities before converging.",
            stages=("frame", "expand", "compare", "converge"),
        ),
        "research-analysis": ModeDefinition(
            name="research-analysis",
            objective="Gather evidence and synthesize a grounded analysis.",
            stages=("scope", "gather", "evaluate", "synthesize"),
        ),
        "execution-development": ModeDefinition(
            name="execution-development",
            objective="Implement a concrete outcome and validate it.",
            stages=("inspect", "implement", "test", "deliver"),
        ),
    }
)


def get_mode(name: str) -> ModeDefinition:
    """Return a supported mode contract, rejecting unknown names."""

    try:
        return MODE_DEFINITIONS[name]
    except (KeyError, TypeError) as exc:
        raise UnknownModeError(f"unknown mode: {name!r}") from exc
