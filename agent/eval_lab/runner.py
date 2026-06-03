"""Local eval runner with backend separation.

The runner accepts an injected agent backend and records deterministic trajectory
attempt objects. It does not call external services by itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from agent.eval_lab.redaction import redact_secrets
from agent.eval_lab.schemas import EvalScenario, TrajectoryAttempt, TrajectoryGroup, TrajectoryStep


class EvalAgentBackend(Protocol):
    """Minimal backend contract for local eval attempts."""

    def run_conversation(self, user_message: str) -> dict[str, Any]:
        """Run one scenario prompt and return response payload."""
        ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _steps_from_messages(messages: list[Any]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    for message in messages:
        if not isinstance(message, dict):
            steps.append(TrajectoryStep(role="unknown", content=str(message)))
            continue
        steps.append(
            TrajectoryStep(
                role=str(message.get("role", "unknown")),
                content=redact_secrets(message.get("content")) if message.get("content") is not None else None,
                tool_name=message.get("tool_name"),
                tool_args_redacted=redact_secrets(message.get("tool_args")) if isinstance(message.get("tool_args"), dict) else None,
                duration_ms=message.get("duration_ms") if isinstance(message.get("duration_ms"), int) else None,
                error=message.get("error") if isinstance(message.get("error"), str) else None,
            )
        )
    return steps


class LocalEvalRunner:
    """Run grouped attempts for a scenario against an injected local backend."""

    def __init__(self, agent: EvalAgentBackend):
        self.agent = agent

    def run(self, scenario: EvalScenario, attempt_count: int = 1) -> TrajectoryGroup:
        if attempt_count < 1:
            raise ValueError("attempt_count must be >= 1")

        group = TrajectoryGroup(
            group_id=f"group-{scenario.id}-{uuid4().hex[:8]}",
            scenario_id=scenario.id,
            attempts=[self._run_attempt(scenario, index) for index in range(attempt_count)],
        )
        return group

    def _run_attempt(self, scenario: EvalScenario, index: int) -> TrajectoryAttempt:
        started_at = _now_iso()
        start = perf_counter()
        attempt_id = f"{scenario.id}-attempt-{index + 1}-{uuid4().hex[:8]}"
        try:
            result = self.agent.run_conversation(scenario.prompt)
            finished_at = _now_iso()
            if not isinstance(result, dict):
                raise TypeError("agent backend must return a mapping")
            messages = result.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            return TrajectoryAttempt(
                attempt_id=attempt_id,
                scenario_id=scenario.id,
                started_at=started_at,
                finished_at=finished_at,
                status="completed",
                final_response=redact_secrets(result.get("final_response")),
                steps=_steps_from_messages(messages),
                metadata={
                    **redact_secrets(metadata),
                    "duration_ms": int((perf_counter() - start) * 1000),
                },
            )
        except Exception as exc:
            return TrajectoryAttempt(
                attempt_id=attempt_id,
                scenario_id=scenario.id,
                started_at=started_at,
                finished_at=_now_iso(),
                status="failed",
                final_response=None,
                steps=[
                    TrajectoryStep(
                        role="error",
                        content=None,
                        duration_ms=int((perf_counter() - start) * 1000),
                        error=str(exc),
                    )
                ],
                metadata={},
            )
