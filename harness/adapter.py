"""Live adapter: one harness iteration as one Hermes agent turn.

The adapter is deliberately thin. It sends the harness-compiled context as
the next user turn (a real turn — prompt caching and role alternation are
preserved) and wraps the final response. Functional verification of the
work itself belongs to explicit :mod:`harness.verify` checkers, never to
the model's own claims.
"""

from __future__ import annotations

from typing import Any, Dict

from .loop import AgentStep, StepResult
from .state import FeatureState, Outcome, Task


def chat_step(agent) -> AgentStep:
    """Build a step around an object exposing ``chat(message) -> str``."""

    def run(task: Task, feature: FeatureState, context: Dict[str, Any]) -> StepResult:
        message = (
            f"Task: {task.goal}\n"
            f"Success criteria: {'; '.join(task.success_criteria)}\n"
            f"Active feature: {feature.name} ({feature.status})\n"
            f"Hypothesis: {feature.hypothesis or 'none yet'}\n"
            f"Known issues: {'; '.join(feature.known_issues) or 'none'}"
        )
        try:
            reply = agent.chat(message)
        except Exception as exc:  # noqa: BLE001 — surfaced as a step error
            return StepResult(error=str(exc)[:500], transient_error=False)
        return StepResult(
            summary=reply[:500],
            status_hint=Outcome.CONTINUE,
            evidence=[reply[:2000]] if reply else [],
        )

    return run
