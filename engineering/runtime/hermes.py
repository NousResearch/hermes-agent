"""Hermes ``AIAgent`` adapter for the Engineering runtime port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import uuid4

from .base import AgentRuntimeError
from .models import RuntimeUsage, TurnRequest, TurnResult, TurnStatus


class _HermesAgent(Protocol):
    """The narrow public ``AIAgent`` surface consumed by this adapter."""

    api_mode: str
    model: str
    provider: str
    session_id: str

    def run_conversation(
        self,
        user_message: Any,
        *,
        task_id: str | None = None,
    ) -> Mapping[str, object]: ...


class HermesRuntimeAdapter:
    """Run one normal Hermes turn through the Engineering runtime contract.

    The adapter composes an already configured ``AIAgent``. Provider
    resolution, model selection, tools, context, and session persistence remain
    owned by Hermes. The adapter maps only the public turn result and never
    interprets Hermes ``completed`` as Engineering workflow completion.
    """

    __slots__ = ("_agent",)

    def __init__(self, agent: _HermesAgent) -> None:
        if not callable(getattr(agent, "run_conversation", None)):
            raise TypeError("agent must expose run_conversation")
        self._agent = agent

    def run_turn(self, request: TurnRequest) -> TurnResult:
        """Execute and map one Hermes turn without changing workflow state."""

        if not isinstance(request, TurnRequest):
            raise TypeError("request must be a TurnRequest")
        self._reject_unsupported_runtime()
        self._validate_requested_session(request.session_id)
        effective_task_id = request.task_id or str(uuid4())

        try:
            raw_result = self._agent.run_conversation(
                user_message=request.message,
                task_id=effective_task_id,
            )
        except (KeyboardInterrupt, InterruptedError):
            return self._interrupted_result(request, effective_task_id)
        except BaseException as exc:
            if type(exc).__name__ == "CancelledError":
                return self._interrupted_result(request, effective_task_id)
            if isinstance(exc, Exception):
                raise AgentRuntimeError("Hermes runtime execution failed") from exc
            raise

        if not isinstance(raw_result, Mapping):
            raise AgentRuntimeError(
                "Hermes run_conversation must return a mapping"
            )
        return self._map_result(request, effective_task_id, raw_result)

    def _reject_unsupported_runtime(self) -> None:
        api_mode = getattr(self._agent, "api_mode", None)
        if api_mode == "codex_app_server":
            raise AgentRuntimeError(
                "codex_app_server is unsupported by the V1 Engineering Surface"
            )

    def _validate_requested_session(self, requested_session_id: str | None) -> None:
        if requested_session_id is None:
            return
        actual_session_id = _optional_text(
            getattr(self._agent, "session_id", None),
            "agent.session_id",
        )
        if requested_session_id != actual_session_id:
            raise AgentRuntimeError(
                "TurnRequest session_id does not match the configured AIAgent"
            )

    def _map_result(
        self,
        request: TurnRequest,
        task_id: str,
        raw_result: Mapping[str, object],
    ) -> TurnResult:
        status = _status_from_result(raw_result)
        response = _response_from_result(raw_result, status)
        session_id = _result_or_agent_text(
            raw_result,
            "session_id",
            self._agent,
        )
        provider = _result_or_agent_text(raw_result, "provider", self._agent)
        model = _result_or_agent_text(raw_result, "model", self._agent)

        return TurnResult(
            request_id=request.request_id,
            workflow_run_id=request.workflow_run_id,
            attempt=request.attempt,
            status=status,
            response=response,
            session_id=session_id,
            task_id=task_id,
            provider=provider,
            model=model,
            usage=_usage_from_result(raw_result),
            metadata=_diagnostic_metadata(raw_result),
        )

    def _interrupted_result(
        self,
        request: TurnRequest,
        task_id: str,
    ) -> TurnResult:
        return TurnResult(
            request_id=request.request_id,
            workflow_run_id=request.workflow_run_id,
            attempt=request.attempt,
            status=TurnStatus.INTERRUPTED,
            response=None,
            session_id=_optional_text(
                getattr(self._agent, "session_id", None),
                "agent.session_id",
            ),
            task_id=task_id,
            provider=_optional_text(
                getattr(self._agent, "provider", None),
                "agent.provider",
            ),
            model=_optional_text(
                getattr(self._agent, "model", None),
                "agent.model",
            ),
        )


def _status_from_result(result: Mapping[str, object]) -> TurnStatus:
    for name in ("interrupted", "failed", "completed"):
        if name in result and type(result[name]) is not bool:
            raise AgentRuntimeError(f"Hermes result {name} must be a bool")

    if result.get("interrupted") is True:
        return TurnStatus.INTERRUPTED
    if result.get("failed") is True:
        return TurnStatus.FAILED
    if "completed" not in result:
        raise AgentRuntimeError("Hermes result is missing completed")
    if result["completed"] is True:
        return TurnStatus.RETURNED
    return TurnStatus.FAILED


def _response_from_result(
    result: Mapping[str, object],
    status: TurnStatus,
) -> str | None:
    response = result.get("final_response")
    if response is None:
        if status is TurnStatus.RETURNED:
            raise AgentRuntimeError(
                "Hermes returned a completed turn without final_response"
            )
        return None
    if not isinstance(response, str):
        raise AgentRuntimeError("Hermes result final_response must be a string")
    if not response.strip():
        if status is TurnStatus.RETURNED:
            raise AgentRuntimeError(
                "Hermes returned a completed turn with an empty final_response"
            )
        return None
    return response


def _usage_from_result(result: Mapping[str, object]) -> RuntimeUsage | None:
    """Copy Hermes' public counters without adding billing/scope semantics."""

    values: dict[str, int | None] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = result.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise AgentRuntimeError(
                f"Hermes result {name} must be a non-negative integer"
            )
        values[name] = value  # type: ignore[assignment]
    if all(value is None for value in values.values()):
        return None
    return RuntimeUsage(**values)


def _diagnostic_metadata(result: Mapping[str, object]) -> Mapping[str, object]:
    metadata: dict[str, object] = {}
    for name in (
        "turn_exit_reason",
        "failure_reason",
        "error",
        "interrupt_message",
    ):
        value = result.get(name)
        if value is not None:
            metadata[name] = _required_text(value, f"Hermes result {name}")

    if "api_calls" in result:
        api_calls = result["api_calls"]
        if type(api_calls) is not int or api_calls < 0:
            raise AgentRuntimeError(
                "Hermes result api_calls must be a non-negative integer"
            )
        metadata["api_calls"] = api_calls
    if "partial" in result:
        partial = result["partial"]
        if type(partial) is not bool:
            raise AgentRuntimeError("Hermes result partial must be a bool")
        metadata["partial"] = partial
    return metadata


def _result_or_agent_text(
    result: Mapping[str, object],
    name: str,
    agent: _HermesAgent,
) -> str | None:
    value = result.get(name)
    if value is None:
        value = getattr(agent, name, None)
    return _optional_text(value, f"Hermes result {name}")


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentRuntimeError(f"{name} must be a non-empty string")
    return value
