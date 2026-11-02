"""Future cognition/orchestration boundary."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Brain:
    name: str = "brain"


__all__ = ["Brain"]
