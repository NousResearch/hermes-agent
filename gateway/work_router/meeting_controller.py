from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .meeting_analyzer import MeetingRoundAnalyzer
from .meeting_orchestrator import (
    MeetingRoundOrchestrator,
    RoundAssignment,
    RoundContribution,
)
from .meeting_plan import MeetingPlan
from .meeting_strategy import DynamicRoundStrategy, RoundAnalysis


@dataclass(frozen=True)
class MeetingRoundOutcome:
    continue_meeting: bool
    converged: bool
    safety_stop: bool
    next_round_id: int | None
    next_participants: tuple[str, ...]
    next_round_focus: str
    analysis: RoundAnalysis


class MeetingRoundController:
    def __init__(self, plan: MeetingPlan) -> None:
        self.plan = plan
        self.orchestrator = MeetingRoundOrchestrator(plan)
        self.analyzer = MeetingRoundAnalyzer(
            participants=plan.participant_pool or plan.participants,
        )
        self.strategy = DynamicRoundStrategy(
            participants=plan.participant_pool or plan.participants,
            safety_round_cap=plan.round_policy.safety_round_cap,
        )

    def start_round(
        self,
        *,
        round_id: int,
        prior_contributions: Mapping[str, str],
        round_participants: tuple[str, ...] | None = None,
    ) -> tuple[RoundAssignment, ...]:
        return self.orchestrator.build_round(
            round_id=round_id,
            prior_contributions=prior_contributions,
            participants=round_participants,
        )

    def build_analysis_prompt(
        self,
        *,
        round_id: int,
        contributions: Mapping[str, str],
    ) -> str:
        return self.analyzer.build_prompt(
            objective=self.plan.objective,
            round_id=round_id,
            contributions=contributions,
        )

    def complete_round(
        self,
        *,
        round_id: int,
        contributions: Mapping[str, str],
        analyzer_response: str,
        round_participants: tuple[str, ...] | None = None,
    ) -> MeetingRoundOutcome:
        normalized = tuple(
            RoundContribution(
                profile=profile,
                statement=statement,
            )
            for profile, statement in contributions.items()
        )

        round_state = self.orchestrator.evaluate_round(
            round_id=round_id,
            contributions=normalized,
            key_conflicts_remaining=True,
            material_uncertainties_remaining=True,
            expected_participants=round_participants,
        )

        if not round_state.round_complete:
            raise ValueError(
                "cannot analyze an incomplete meeting round; missing: "
                + ", ".join(round_state.missing_participants)
            )

        analysis = self.analyzer.parse(analyzer_response)

        decision = self.strategy.decide_next_round(
            round_id=round_id,
            analysis=analysis,
        )

        return MeetingRoundOutcome(
            continue_meeting=decision.continue_meeting,
            converged=decision.converged,
            safety_stop=decision.safety_stop,
            next_round_id=(
                round_id + 1
                if decision.continue_meeting
                else None
            ),
            next_participants=decision.participants,
            next_round_focus=decision.focus,
            analysis=analysis,
        )
