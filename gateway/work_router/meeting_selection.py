"""Participant selection contracts for dynamic meetings.

The active BotRegistry is the source of truth for the candidate pool. Selection
is delegated to an injected implementation so a Demian/LLM selector can be
connected without embedding agenda keywords in the meeting runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import BotRegistry


@dataclass(frozen=True)
class ParticipantSelectionRequest:
    objective: str
    participant_pool: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("meeting objective is required")
        if not self.participant_pool:
            raise ValueError("participant pool is required")


@runtime_checkable
class ParticipantSelector(Protocol):
    """Extensible initial-participant selection interface."""

    def select(
        self,
        request: ParticipantSelectionRequest,
    ) -> tuple[str, ...]:
        """Select the Staff needed for the meeting objective."""

        raise NotImplementedError


@dataclass(frozen=True)
class StaticParticipantSelector:
    """Exact injected selection used by deterministic callers and tests."""

    participants: tuple[str, ...]

    def select(
        self,
        request: ParticipantSelectionRequest,
    ) -> tuple[str, ...]:
        del request
        return self.participants


def build_participant_pool(registry: BotRegistry) -> tuple[str, ...]:
    """Derive Staff candidates from the registry, excluding its facilitator."""

    pool = tuple(
        profile.name
        for profile in registry.profiles
        if profile.name != registry.default
    )
    if not pool:
        raise ValueError("participant pool is required")
    if len(set(pool)) != len(pool):
        raise ValueError("participant pool profiles must be unique")
    return pool


def select_initial_participants(
    *,
    registry: BotRegistry,
    objective: str,
    selector: ParticipantSelector,
    min_participants: int = 1,
    max_participants: int | None = None,
) -> tuple[str, ...]:
    """Run and validate one injected initial-participant selection."""

    if min_participants < 1:
        raise ValueError("min_participants must be >= 1")
    pool = build_participant_pool(registry)
    upper_bound = len(pool) if max_participants is None else max_participants
    if upper_bound < min_participants or upper_bound > len(pool):
        raise ValueError("participant bounds are invalid")

    request = ParticipantSelectionRequest(
        objective=objective.strip(),
        participant_pool=pool,
    )
    selected = selector.select(request)
    if not isinstance(selected, tuple):
        raise ValueError("participant selector must return a tuple")
    if any(not isinstance(profile, str) or not profile.strip() for profile in selected):
        raise ValueError("selected participant profile is required")
    if len(set(selected)) != len(selected):
        raise ValueError("selected participant profiles must be unique")
    if not min_participants <= len(selected) <= upper_bound:
        raise ValueError("selected participant count is outside allowed bounds")
    if registry.default in selected:
        raise ValueError("meeting facilitator cannot be a participant")

    unknown = tuple(profile for profile in selected if profile not in pool)
    if unknown:
        raise ValueError(
            "selected participant is outside the registry pool: "
            + ", ".join(unknown)
        )
    return selected
