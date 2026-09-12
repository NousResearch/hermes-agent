from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoundAnalysis:
    key_conflicts: tuple[str, ...]
    uncertainties: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    challenged_assumptions: tuple[str, ...]
    promising_alternatives: tuple[str, ...]
    participants_needed_next: tuple[str, ...]
    next_round_focus: str
    convergence_recommended: bool


@dataclass(frozen=True)
class NextRoundDecision:
    continue_meeting: bool
    converged: bool
    safety_stop: bool
    participants: tuple[str, ...]
    focus: str


class DynamicRoundStrategy:
    def __init__(
        self,
        *,
        participants: tuple[str, ...],
        safety_round_cap: int = 5,
    ) -> None:
        if not participants:
            raise ValueError("meeting participants are required")
        if len(set(participants)) != len(participants):
            raise ValueError("meeting participants must be unique")
        if safety_round_cap < 1:
            raise ValueError("safety_round_cap must be >= 1")

        self.participants = participants
        self.safety_round_cap = safety_round_cap

    def decide_next_round(
        self,
        *,
        round_id: int,
        analysis: RoundAnalysis,
    ) -> NextRoundDecision:
        if round_id < 1:
            raise ValueError("round_id must be >= 1")

        unknown = tuple(
            profile
            for profile in analysis.participants_needed_next
            if profile not in self.participants
        )
        if unknown:
            raise ValueError(
                "unknown next-round participant(s): "
                + ", ".join(unknown)
            )

        unresolved = bool(
            analysis.key_conflicts
            or analysis.uncertainties
            or analysis.evidence_gaps
        )

        if round_id >= self.safety_round_cap:
            return NextRoundDecision(
                continue_meeting=False,
                converged=False,
                safety_stop=True,
                participants=(),
                focus="",
            )

        # A convergence recommendation cannot override material unresolved
        # conflicts, uncertainty, or evidence gaps.
        if analysis.convergence_recommended and not unresolved:
            return NextRoundDecision(
                continue_meeting=False,
                converged=True,
                safety_stop=False,
                participants=(),
                focus="",
            )

        if unresolved:
            participants = analysis.participants_needed_next
            if not participants:
                # Do not silently declare convergence merely because the
                # planner omitted a participant selection.
                participants = self.participants

            focus = analysis.next_round_focus.strip()
            if not focus:
                focus = (
                    "Resolve the remaining conflicts, uncertainties, "
                    "and evidence gaps."
                )

            return NextRoundDecision(
                continue_meeting=True,
                converged=False,
                safety_stop=False,
                participants=participants,
                focus=focus,
            )

        return NextRoundDecision(
            continue_meeting=False,
            converged=True,
            safety_stop=False,
            participants=(),
            focus="",
        )
