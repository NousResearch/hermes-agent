"""Future multi-level memory boundary."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Memory:
    name: str = "memory"


__all__ = ["Memory"]
