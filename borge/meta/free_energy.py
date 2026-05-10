"""
Extended Free Energy Function

F_total = F_epistemic × precision(E)
        + F_pragmatic × V_alignment
        + F_homeostatic(E)

This is the unified objective function for Borge Agent.
Minimising F_total drives the agent toward:
  - Resolving uncertainty (epistemic)
  - Achieving value-aligned outcomes (pragmatic)
  - Maintaining emotional equilibrium (homeostatic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..affective.emotional_state import EmotionalState
from ..beliefs.belief_state import BeliefState
from ..values.value_system import ValueSystem


@dataclass
class FreeEnergyBreakdown:
    total: float
    epistemic: float
    pragmatic: float
    homeostatic: float


class ExtendedFreeEnergy:
    OPTIMAL_AROUSAL: float = 0.55
    AROUSAL_COST_WEIGHT: float = 0.4
    NEGATIVE_VALENCE_COST_WEIGHT: float = 0.3

    def compute(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
        value_system: ValueSystem,
        proposed_action: Optional[str] = None,
    ) -> FreeEnergyBreakdown:
        f_epistemic = belief_state.shannon_entropy() * emotional_state.precision
        f_pragmatic = value_system.pragmatic_free_energy(proposed_action or "")
        arousal_cost = (
            abs(emotional_state.arousal - self.OPTIMAL_AROUSAL)
            * self.AROUSAL_COST_WEIGHT
        )
        valence_cost = (
            max(0.0, -emotional_state.valence)
            * self.NEGATIVE_VALENCE_COST_WEIGHT
        )
        f_homeostatic = arousal_cost + valence_cost
        total = f_epistemic + f_pragmatic + f_homeostatic
        return FreeEnergyBreakdown(
            total=round(total, 4),
            epistemic=round(f_epistemic, 4),
            pragmatic=round(f_pragmatic, 4),
            homeostatic=round(f_homeostatic, 4),
        )

    def delta(self, before: FreeEnergyBreakdown, after: FreeEnergyBreakdown) -> float:
        return after.total - before.total
