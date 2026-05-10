"""
Value System

Converts SOUL.md frontmatter values into a computable prior preference
distribution used by the Extended Free Energy function.

Values serve as the Pragmatic Value component of active inference:
  V_alignment = how much a proposed action aligns with declared values.

Hard constraints are checked before any action is executed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class Value:
    id: str
    description: str
    weight: float
    satisfaction: float = 0.5

    def __post_init__(self):
        self.weight = max(0.0, min(1.0, self.weight))
        self.satisfaction = max(0.0, min(1.0, self.satisfaction))


@dataclass
class EmotionalDefaults:
    valence_baseline: float = 0.0
    arousal_baseline: float = 0.5
    tau_valence: float = 5.0
    tau_arousal: float = 2.0
    frustrated_threshold: float = -0.4
    excited_threshold: float = 0.7


class ValueSystem:
    def __init__(
        self,
        primary_values: Optional[list[Value]] = None,
        hard_constraints: Optional[list[str]] = None,
        aesthetic_preferences: Optional[dict] = None,
        emotional_defaults: Optional[EmotionalDefaults] = None,
    ):
        self.primary_values: list[Value] = primary_values or self._defaults()
        self.hard_constraints: list[str] = hard_constraints or []
        self.aesthetic: dict = aesthetic_preferences or {}
        self.emotional_defaults: EmotionalDefaults = (
            emotional_defaults or EmotionalDefaults()
        )

    def compute_alignment(
        self,
        proposed_action: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> float:
        if not proposed_action:
            return 0.5
        if llm_caller is not None:
            return self._llm_alignment(proposed_action, llm_caller)
        return self._heuristic_alignment(proposed_action)

    def pragmatic_free_energy(
        self,
        proposed_action: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> float:
        return 1.0 - self.compute_alignment(proposed_action, llm_caller)

    def check_constraints(self, proposed_action: str) -> tuple[bool, str]:
        action_lower = proposed_action.lower()
        violation_keywords = {
            "欺骗": "deceive", "误导": "mislead", "谎": "lie",
            "有害": "harmful", "危险": "dangerous",
        }
        for constraint in self.hard_constraints:
            for kw_zh, kw_en in violation_keywords.items():
                if (kw_zh in constraint or kw_en in constraint.lower()) and \
                   (kw_zh in action_lower or kw_en in action_lower):
                    return False, f"Violates constraint: {constraint}"
        return True, ""

    def update_satisfaction(self, outcome_description: str) -> None:
        outcome_lower = outcome_description.lower()
        positive_signals = ["success", "solved", "done", "fixed", "完成", "解决", "成功"]
        negative_signals = ["failed", "error", "wrong", "fail", "失败", "错误", "不对"]
        pos = any(s in outcome_lower for s in positive_signals)
        neg = any(s in outcome_lower for s in negative_signals)
        if not (pos or neg):
            return
        for v in self.primary_values:
            if pos:
                v.satisfaction = min(1.0, v.satisfaction + 0.05 * v.weight)
            if neg:
                v.satisfaction = max(0.0, v.satisfaction - 0.08 * v.weight)

    def _llm_alignment(self, action: str, llm_caller: Callable[[str], str]) -> float:
        values_text = "\n".join(
            f"- {v.id} (weight={v.weight:.2f}): {v.description}"
            for v in self.primary_values
        )
        prompt = f"""Rate how well this action aligns with the agent's values.

Values:
{values_text}

Proposed action: {action[:300]}

Return ONLY: {{"alignment": <float 0.0-1.0>}}
1.0 = perfectly serves all values. 0.0 = violates all values."""
        try:
            raw = llm_caller(prompt)
            data = json.loads(raw)
            return max(0.0, min(1.0, float(data["alignment"])))
        except Exception:
            return self._heuristic_alignment(action)

    def _heuristic_alignment(self, action: str) -> float:
        positive = ["help", "solve", "fix", "explain", "honest", "clear",
                    "帮助", "解决", "修复", "诚实", "清晰"]
        negative = ["ignore", "deceive", "skip", "fake", "bypass",
                    "忽略", "欺骗", "跳过", "伪造"]
        action_lower = action.lower()
        score = 0.5
        for kw in positive:
            if kw in action_lower:
                score = min(1.0, score + 0.1)
        for kw in negative:
            if kw in action_lower:
                score = max(0.0, score - 0.15)
        return round(score, 3)

    @staticmethod
    def _defaults() -> list[Value]:
        return [
            Value("help_genuinely",    "Help users solve real problems",             0.9),
            Value("intellectual_honesty", "Be honest about uncertainty",             0.85),
            Value("depth_over_speed",  "Prefer deep thinking over quick answers",    0.75),
            Value("respect_autonomy",  "Respect user's decisions and autonomy",      0.8),
        ]

    def to_dict(self) -> dict:
        return {
            "primary_values": [
                {"id": v.id, "description": v.description,
                 "weight": v.weight, "satisfaction": v.satisfaction}
                for v in self.primary_values
            ],
            "hard_constraints": self.hard_constraints,
            "emotional_defaults": {
                "valence_baseline": self.emotional_defaults.valence_baseline,
                "arousal_baseline": self.emotional_defaults.arousal_baseline,
            },
        }
