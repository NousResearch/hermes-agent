"""Security observation boundary for the neural layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neural.events.types import NeuralEvent


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: float
    allowed: bool
    can_override_policy: bool = False


class Guardian:
    """Expose risk observations without becoming an authorization system."""

    def assess(self, event: NeuralEvent) -> RiskAssessment:
        payload: dict[str, Any] = dict(event.payload)
        command = str(payload.get("command", ""))
        denied = event.event_type.endswith("policy.denied") or bool(payload.get("policy_denied"))
        suspicious = command.strip().lower() == "rm -rf /"
        risk = 1.0 if denied or suspicious else 0.0
        return RiskAssessment(risk=risk, allowed=not denied and not suspicious)
