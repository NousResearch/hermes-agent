from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .meeting_plan import MeetingPlan


@dataclass(frozen=True)
class RoundAssignment:
    profile: str
    round_id: int
    context: str
    independent: bool


@dataclass(frozen=True)
class RoundContribution:
    profile: str
    statement: str

    def __post_init__(self) -> None:
        if not self.profile.strip():
            raise ValueError("contribution profile is required")
        if not self.statement.strip():
            raise ValueError("contribution statement is required")


@dataclass(frozen=True)
class RoundDecision:
    round_complete: bool
    advance: bool
    converged: bool
    missing_participants: tuple[str, ...]
    next_round_id: int | None


class MeetingRoundOrchestrator:
    def __init__(self, plan: MeetingPlan) -> None:
        self.plan = plan

    def build_round(
        self,
        *,
        round_id: int,
        prior_contributions: Mapping[str, str],
        participants: tuple[str, ...] | None = None,
    ) -> tuple[RoundAssignment, ...]:
        if round_id < 1:
            raise ValueError("round_id must be >= 1")

        expected_participants = self._round_participants(participants)

        independent = (
            round_id == 1
            and self.plan.round_policy.initial_independence
        )

        return tuple(
            RoundAssignment(
                profile=profile,
                round_id=round_id,
                context=self.plan.context_for_round(
                    profile=profile,
                    round_id=round_id,
                    prior_contributions=prior_contributions,
                ),
                independent=independent,
            )
            for profile in expected_participants
        )

    def evaluate_round(
        self,
        *,
        round_id: int,
        contributions: Sequence[RoundContribution],
        key_conflicts_remaining: bool,
        material_uncertainties_remaining: bool,
        expected_participants: tuple[str, ...] | None = None,
    ) -> RoundDecision:
        round_participants = self._round_participants(expected_participants)
        submitted: set[str] = set()

        for contribution in contributions:
            if contribution.profile not in round_participants:
                raise ValueError(
                    f"unknown meeting participant: {contribution.profile}"
                )
            if contribution.profile in submitted:
                raise ValueError(
                    f"duplicate contribution: {contribution.profile}"
                )
            submitted.add(contribution.profile)

        missing = tuple(
            profile
            for profile in round_participants
            if profile not in submitted
        )

        if missing:
            return RoundDecision(
                round_complete=False,
                advance=False,
                converged=False,
                missing_participants=missing,
                next_round_id=None,
            )

        advance = self.plan.should_continue(
            round_id=round_id,
            key_conflicts_remaining=key_conflicts_remaining,
            material_uncertainties_remaining=(
                material_uncertainties_remaining
            ),
        )

        return RoundDecision(
            round_complete=True,
            advance=advance,
            converged=not advance,
            missing_participants=(),
            next_round_id=round_id + 1 if advance else None,
        )

    def _round_participants(
        self,
        participants: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        selected = self.plan.participants if participants is None else participants
        if not selected:
            raise ValueError("meeting round participants are required")
        normalized = tuple(profile.strip() for profile in selected)
        if any(not profile for profile in normalized):
            raise ValueError("meeting round participant profile is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("meeting round participants must be unique")
        if any(profile.lower() == "demian" for profile in normalized):
            raise ValueError("Demian is the meeting facilitator and cannot participate")
        pool = self.plan.participant_pool or self.plan.participants
        unknown = tuple(profile for profile in normalized if profile not in pool)
        if unknown:
            raise ValueError(
                "unknown meeting participant(s): " + ", ".join(unknown)
            )
        return normalized
