from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from run_agent import AIAgent

from .models import ClaimedJobEnvelope, InterpretationSubmission
from .parser import OutputValidationError, parse_submission
from .prompting import build_repair_prompt, build_system_message, build_user_message

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    def run_conversation(
        self,
        user_message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        ...


class InterpretationExecutionError(RuntimeError):
    """Raised when the Hermes runtime cannot produce a valid interpretation payload."""


class HermesCoryExecutor:
    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        max_completion_attempts: int = 2,
        agent_factory: Callable[[], AgentProtocol] | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._max_completion_attempts = max_completion_attempts
        self._agent_factory = agent_factory

    def run(self, claim: ClaimedJobEnvelope) -> InterpretationSubmission:
        if claim.job is None or claim.harness is None:
            raise InterpretationExecutionError("claimed job payload is incomplete")

        system_message = build_system_message(claim)
        user_message = build_user_message(claim)
        history: list[dict[str, Any]] | None = None
        agent = self._create_agent()
        last_error: str | None = None

        for attempt in range(1, self._max_completion_attempts + 1):
            prompt = (
                user_message
                if history is None
                else build_repair_prompt(claim, last_error or "invalid output")
            )
            result = agent.run_conversation(
                user_message=prompt,
                system_message=system_message if history is None else None,
                conversation_history=history,
                task_id=f"cory-request-interpretation-{claim.job.id}-attempt-{attempt}",
            )

            response_text = str(result.get("final_response") or "").strip()
            history = result.get("messages")
            try:
                return parse_submission(response_text, claim.harness)
            except OutputValidationError as exc:
                last_error = str(exc)
                logger.warning(
                    "Cory interpretation output failed validation for job %s on attempt %s/%s: %s",
                    claim.job.id,
                    attempt,
                    self._max_completion_attempts,
                    last_error,
                )

        raise InterpretationExecutionError(
            f"failed to produce a valid interpretation payload after {self._max_completion_attempts} attempt(s): {last_error}",
        )

    def _create_agent(self) -> AgentProtocol:
        if self._agent_factory is not None:
            return self._agent_factory()

        agent_kwargs: dict[str, Any] = {
            "quiet_mode": True,
            "enabled_toolsets": [],
            "skip_context_files": True,
            "skip_memory": True,
        }
        if self._model:
            agent_kwargs["model"] = self._model
        if self._provider:
            agent_kwargs["provider"] = self._provider
        return AIAgent(**agent_kwargs)
