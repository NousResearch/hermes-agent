"""Sinclair-controlled participant injection for subsequent meeting rounds."""

from __future__ import annotations

from .config import BotRegistry
from .meeting_selection import build_participant_pool


class SinclairParticipantInjection:
    """Keep an ordered, idempotent set of Staff forced into future rounds."""

    def __init__(self, *, registry: BotRegistry) -> None:
        self.registry = registry
        self._participant_pool = build_participant_pool(registry)
        self._participants: list[str] = []

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(self._participants)

    def add(self, profile: str) -> bool:
        normalized = profile.strip()
        if not normalized or self.registry.by_name(normalized) is None:
            raise ValueError("forced participant is not in the bot registry")
        if normalized == self.registry.default:
            raise ValueError("meeting facilitator cannot be a participant")
        if normalized not in self._participant_pool:
            raise ValueError("forced participant is outside the participant pool")
        if normalized in self._participants:
            return False
        self._participants.append(normalized)
        return True

    def merge_next_round(
        self,
        selected: tuple[str, ...],
        *,
        max_participants: int | None = None,
    ) -> tuple[str, ...]:
        normalized = tuple(profile.strip() for profile in selected)
        if any(not profile for profile in normalized):
            raise ValueError("next-round participant profile is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("next-round participants must be unique")
        if self.registry.default in normalized:
            raise ValueError("meeting facilitator cannot be a participant")
        unknown = tuple(
            profile
            for profile in normalized
            if profile not in self._participant_pool
        )
        if unknown:
            raise ValueError(
                "next-round participant is outside the registry pool: "
                + ", ".join(unknown)
            )

        merged = tuple(
            dict.fromkeys((*normalized, *self._participants))
        )
        if max_participants is not None:
            if max_participants < 1:
                raise ValueError("max_participants must be >= 1")
            if len(merged) > max_participants:
                raise ValueError(
                    "forced participants exceed the next-round participant limit"
                )
        return merged
