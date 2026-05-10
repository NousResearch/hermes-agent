"""
Meta-Agent — Central Executive System

Implements Baddeley's Central Executive: monitors cognitive state,
allocates attention, and triggers higher-order interventions when needed.

Key responsibilities:
  1. Monitor F_total trajectory → detect stagnation
  2. Trigger reflection loop when F_total can't decrease
  3. Suggest AgentMode based on current emotional quadrant
  4. Compute attention focus from highest-entropy beliefs
  5. Track skill usage for the evolution engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..affective.emotional_state import AgentMode, EmotionalState
from ..beliefs.belief_state import BeliefState
from ..meta.free_energy import ExtendedFreeEnergy, FreeEnergyBreakdown
from ..values.value_system import ValueSystem

log = logging.getLogger(__name__)


@dataclass
class MetaSignal:
    f_total: float
    f_breakdown: FreeEnergyBreakdown
    suggested_mode: AgentMode
    trigger_reflection: bool
    attention_focus: list[str]
    loyalty_hint: str = ""
    context_injection: str = ""


class MetaAgent:
    F_HISTORY_WINDOW: int = 5
    STAGNATION_TURNS: int = 3

    def __init__(
        self,
        free_energy_fn: Optional[ExtendedFreeEnergy] = None,
        entropy_injection_threshold: float = 0.5,
    ):
        self.fe_fn = free_energy_fn or ExtendedFreeEnergy()
        self.entropy_threshold = entropy_injection_threshold
        self._f_history: list[float] = []

    def tick(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
        value_system: ValueSystem,
        proposed_action: Optional[str] = None,
        loyalty_hint: str = "",
    ) -> MetaSignal:
        breakdown = self.fe_fn.compute(
            belief_state, emotional_state, value_system, proposed_action
        )
        self._f_history.append(breakdown.total)
        if len(self._f_history) > self.F_HISTORY_WINDOW:
            self._f_history.pop(0)

        trigger_reflect = self._should_reflect()
        mode = emotional_state.suggested_mode
        focus = self._attention_focus(belief_state)
        ctx = self._build_context_injection(
            belief_state, emotional_state, breakdown, loyalty_hint
        )

        if trigger_reflect:
            log.info(
                f"[MetaAgent] Reflection triggered. F_total history: "
                f"{[round(f, 3) for f in self._f_history]}"
            )

        return MetaSignal(
            f_total=breakdown.total,
            f_breakdown=breakdown,
            suggested_mode=mode,
            trigger_reflection=trigger_reflect,
            attention_focus=focus,
            loyalty_hint=loyalty_hint,
            context_injection=ctx,
        )

    def _should_reflect(self) -> bool:
        n = self.STAGNATION_TURNS
        if len(self._f_history) < n:
            return False
        recent = self._f_history[-n:]
        return all(recent[i] >= recent[i - 1] for i in range(1, n))

    def _attention_focus(self, belief_state: BeliefState) -> list[str]:
        if not belief_state.hypotheses:
            return belief_state.open_questions[:3]
        uncertain = sorted(
            belief_state.hypotheses,
            key=lambda h: -abs(h.probability - 0.5),
        )
        return [h.description for h in uncertain[:3]]

    def _build_context_injection(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
        breakdown: FreeEnergyBreakdown,
        loyalty_hint: str,
    ) -> str:
        parts = []
        parts.append(emotional_state.to_context_summary())
        if belief_state.shannon_entropy() > self.entropy_threshold:
            belief_summary = belief_state.to_context_injection()
            if belief_summary:
                parts.append(belief_summary)
        if loyalty_hint:
            parts.append(f"[Relationship: {loyalty_hint}]")
        if self._should_reflect():
            parts.append(
                "[Meta: Free energy stagnating — consider a different approach "
                "or ask the user for clarification.]"
            )
        return "\n".join(parts)

    @property
    def f_trend(self) -> Optional[str]:
        if len(self._f_history) < 3:
            return None
        delta = self._f_history[-1] - self._f_history[0]
        if delta < -0.5:
            return "converging"
        if delta > 0.5:
            return "diverging"
        return "stagnating"

    def reset(self) -> None:
        self._f_history.clear()
