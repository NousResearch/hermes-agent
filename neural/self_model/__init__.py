"""Future self-state and capability model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelfModel:
    name: str = "self_model"


__all__ = ["SelfModel"]
