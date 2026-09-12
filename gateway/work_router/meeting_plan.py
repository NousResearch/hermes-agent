from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ParticipantBrief:
    profile: str
    role: str
    question: str

    def __post_init__(self) -> None:
        if not self.profile.strip():
            raise ValueError("participant profile is required")
        if not self.role.strip():
            raise ValueError("participant role is required")
        if not self.question.strip():
            raise ValueError("participant question is required")


@dataclass(frozen=True)
class RoundPolicy:
    initial_independence: bool = True
    allow_cross_domain_challenge: bool = True
    allow_position_revision: bool = True
    allow_problem_reframing: bool = True
    safety_round_cap: int = 5

    def __post_init__(self) -> None:
        if self.safety_round_cap < 1:
            raise ValueError("safety_round_cap must be >= 1")


@dataclass(frozen=True)
class ConvergencePolicy:
    require_resolved_key_conflicts: bool = True
    require_uncertainties_reported: bool = True
    allow_early_finish: bool = True


@dataclass(frozen=True)
class MeetingPlan:
    objective: str
    participants: tuple[str, ...]
    participant_briefs: tuple[ParticipantBrief, ...]
    round_policy: RoundPolicy
    convergence_policy: ConvergencePolicy
    initial_participants: tuple[str, ...] | None = None
    participant_pool: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        objective = self.objective.strip()

        initial_participants = (
            self.participants
            if self.initial_participants is None
            else self.initial_participants
        )
        participant_pool = (
            initial_participants
            if self.participant_pool is None
            else self.participant_pool
        )

        object.__setattr__(
            self,
            "initial_participants",
            tuple(initial_participants),
        )
        object.__setattr__(
            self,
            "participant_pool",
            tuple(participant_pool),
        )

        normalized_initial = tuple(
            profile.strip()
            for profile in self.initial_participants
        )
        normalized_pool = tuple(
            profile.strip()
            for profile in self.participant_pool
        )

        if any(not profile for profile in normalized_initial):
            raise ValueError("initial participant profile is required")

        if any(not profile for profile in normalized_pool):
            raise ValueError("participant pool profile is required")

        if len(set(normalized_pool)) != len(normalized_pool):
            raise ValueError("participant pool profiles must be unique")

        if any(profile.lower() == "demian" for profile in normalized_pool):
            raise ValueError(
                "Demian is the meeting facilitator and cannot be a participant"
            )

        if not set(normalized_initial).issubset(set(normalized_pool)):
            raise ValueError(
                "initial participants must be contained in participant pool"
            )

        if tuple(self.participants) != tuple(self.initial_participants):
            raise ValueError(
                "legacy participants must match initial participants"
            )
        if not objective:
            raise ValueError("meeting objective is required")

        if not self.participants:
            raise ValueError("meeting participants are required")

        normalized = tuple(profile.strip() for profile in self.participants)
        if any(not profile for profile in normalized):
            raise ValueError("participant profile is required")

        if len(set(normalized)) != len(normalized):
            raise ValueError("meeting participants must be unique")

        brief_profiles = tuple(brief.profile for brief in self.participant_briefs)
        if brief_profiles and set(brief_profiles) != set(normalized):
            raise ValueError(
                "participant briefs must describe exactly the meeting participants"
            )

    @classmethod
    def minimal(
        cls,
        *,
        objective: str,
        participants: tuple[str, ...],
    ) -> "MeetingPlan":
        briefs = tuple(
            ParticipantBrief(
                profile=profile,
                role="independent meeting participant",
                question=(
                    "Analyze the meeting objective independently. "
                    "Identify assumptions, risks, alternatives, and evidence needs."
                ),
            )
            for profile in participants
        )

        return cls(
            objective=objective,
            participants=participants,
            participant_briefs=briefs,
            round_policy=RoundPolicy(),
            convergence_policy=ConvergencePolicy(),
        )

    def context_for_round(
        self,
        *,
        profile: str,
        round_id: int,
        prior_contributions: Mapping[str, str],
    ) -> str:
        participant_pool = self.participant_pool or self.participants
        if profile not in participant_pool:
            raise ValueError(f"unknown meeting participant: {profile}")
        if round_id < 1:
            raise ValueError("round_id must be >= 1")

        brief = next(
            (
                brief
                for brief in self.participant_briefs
                if brief.profile == profile
            ),
            None,
        )

        lines = [f"Meeting objective: {self.objective}"]
        if brief is None:
            lines.extend(
                [
                    "Your meeting role: invited domain expert",
                    (
                        "Your question: Apply your expertise to the current focus. "
                        "Identify assumptions, risks, alternatives, and evidence needs."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    f"Your meeting role: {brief.role}",
                    f"Your question: {brief.question}",
                ]
            )
        lines.append(
                "Think freely from your expertise. You may challenge assumptions, "
                "identify issues outside your primary specialty, propose alternatives, "
                "or reframe the problem when justified."
        )

        if round_id == 1 and self.round_policy.initial_independence:
            lines.append(
                "This is an independent first-round analysis. "
                "Other participants' positions are intentionally hidden."
            )
            return "\n".join(lines)

        visible = [
            (peer, text)
            for peer, text in prior_contributions.items()
            if peer != profile and text
        ]

        if visible:
            lines.append("Peer contributions:")
            for peer, text in visible:
                lines.append(f"- {peer}: {text}")

            if self.round_policy.allow_position_revision:
                lines.append(
                    "You may maintain, revise, or withdraw your earlier position "
                    "after considering the arguments and evidence."
                )

        return "\n".join(lines)

    def should_continue(
        self,
        *,
        round_id: int,
        key_conflicts_remaining: bool,
        material_uncertainties_remaining: bool,
    ) -> bool:
        if round_id >= self.round_policy.safety_round_cap:
            return False

        unresolved = (
            key_conflicts_remaining
            or material_uncertainties_remaining
        )

        if not unresolved and self.convergence_policy.allow_early_finish:
            return False

        return unresolved
