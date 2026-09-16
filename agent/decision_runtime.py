"""Host-owned execution and privacy-safe telemetry for decision providers."""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, TypeVar

from agent.decision_provider import (
    BinaryQuestion,
    ChoiceQuestion,
    DecisionProvider,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionStatus,
    OrdinalQuestion,
    ProviderDecision,
    validate_provider_decision,
)
from agent.decision_registry import resolve_provider
from agent.secret_scope import (
    current_secret_scope,
    reset_secret_scope,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)
_VALID_MODES = frozenset({"active", "shadow", "replay"})
_TELEMETRY_LOCK = threading.Lock()
T = TypeVar("T")


class _ProviderInvocationError(Exception):
    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class DecisionRecorder(Protocol):
    def record(self, event: Mapping[str, Any]) -> None: ...


class JsonlDecisionRecorder:
    """Append privacy-minimal decision events below the owning profile home."""

    def __init__(self, path: Path):
        self.path = path

    def record(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, default=str)
        with _TELEMETRY_LOCK, self.path.open("a", encoding="utf-8") as handle:
            handle.write(row + "\n")


class DecisionRuntime:
    """Resolve, invoke, validate, and observe providers without touching agent prompt state."""

    def __init__(
        self, *, plugin_id: str, scope: str, recorder: Optional[DecisionRecorder] = None,
    ) -> None:
        self.plugin_id = plugin_id
        self.scope = scope
        self.recorder: DecisionRecorder = recorder or JsonlDecisionRecorder(
            Path(scope) / "logs" / "decisions.jsonl"
        )

    def evaluate(
        self,
        *,
        task: str,
        state: Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
        provider: Optional[str] = None,
        mode: str = "active",
        timeout: float = 30.0,
    ) -> DecisionResponse:
        """Evaluate one independent question batch.

        ``state`` is deep-copied into the request. No session, memory, prompt, tool, or plugin state
        is inherited. ``shadow`` changes only telemetry labeling; policy remains entirely with the caller.
        """
        request = self._request(task=task, state=state, questions=questions, mode=mode)
        secret_scope = current_secret_scope()
        selected = resolve_provider(
            provider,
            scope=self.scope,
            available=lambda candidate: self._in_profile_context(
                candidate.is_available, self.scope, secret_scope,
            ),
        )
        if selected is None:
            name = provider or ""
            return self._finish(DecisionResponse(
                task=request.task, provider=name, model="", version="", mode=request.mode,
                status=DecisionStatus.UNAVAILABLE,
                fallback_reason="provider unavailable" if name else "provider selection is ambiguous or unavailable",
            ))

        started = time.perf_counter()
        try:
            raw = self._invoke(selected, request, timeout, self.scope, secret_scope)
        except queue.Empty:
            response = DecisionResponse(
                task=request.task, provider=selected.name, model="", version="", mode=request.mode,
                status=DecisionStatus.TIMEOUT, latency_ms=(time.perf_counter() - started) * 1000,
                fallback_reason=f"provider timed out after {timeout:g}s",
            )
            return self._finish(response)
        except _ProviderInvocationError as exc:
            error = exc.error
            logger.warning("Decision provider %s failed for task %s: %s", selected.name, task, error)
            response = DecisionResponse(
                task=request.task, provider=selected.name, model="", version="", mode=request.mode,
                status=DecisionStatus.PROVIDER_ERROR,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_reason=f"provider raised {type(error).__name__}",
            )
            return self._finish(response)

        try:
            validated = validate_provider_decision(request, raw)
            status = DecisionStatus.ABSTAINED if validated.abstained or (
                validated.answers and all(answer.abstained for answer in validated.answers.values())
            ) else DecisionStatus.AVAILABLE
            response = DecisionResponse(
                task=request.task,
                provider=selected.name,
                model=validated.model,
                version=validated.version,
                mode=request.mode,
                status=status,
                answers=validated.answers,
                usage=validated.usage,
                latency_ms=(time.perf_counter() - started) * 1000,
                fallback_reason=(
                    validated.abstention_reason if status is DecisionStatus.ABSTAINED else None
                ),
            )
        except ValueError as exc:
            response = DecisionResponse(
                task=request.task, provider=selected.name, model="", version="", mode=request.mode,
                status=DecisionStatus.MALFORMED, latency_ms=(time.perf_counter() - started) * 1000,
                fallback_reason=str(exc),
            )
        return self._finish(response)

    @staticmethod
    def _request(
        *, task: str, state: Mapping[str, Any], questions: Mapping[str, DecisionQuestion], mode: str,
    ) -> DecisionRequest:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("decision task must be a non-empty string")
        if not isinstance(state, Mapping):
            raise TypeError("decision state must be a mapping")
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("decision questions must be a non-empty mapping")
        clean_questions: Dict[str, DecisionQuestion] = {}
        for key, question in questions.items():
            if not isinstance(key, str) or not key:
                raise ValueError("decision question names must be non-empty strings")
            if not isinstance(question, (BinaryQuestion, ChoiceQuestion, OrdinalQuestion)):
                raise TypeError(f"decision question {key!r} has an unsupported type")
            clean_questions[key] = question
        if mode not in _VALID_MODES:
            raise ValueError(f"decision mode must be one of {sorted(_VALID_MODES)}")
        return DecisionRequest(
            task=task.strip(), state=copy.deepcopy(dict(state)), questions=clean_questions, mode=mode,
        )

    @staticmethod
    def _in_profile_context(
        callback: Callable[[], T], scope: str, secret_scope: Optional[Mapping[str, str]],
    ) -> T:
        context = contextvars.Context()

        def run() -> T:
            home_token = set_hermes_home_override(scope)
            secret_token = set_secret_scope(secret_scope)
            try:
                return callback()
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)

        return context.run(run)

    @staticmethod
    def _invoke(
        provider: DecisionProvider,
        request: DecisionRequest,
        timeout: float,
        scope: str,
        secret_scope: Optional[Mapping[str, str]],
    ) -> ProviderDecision:
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision timeout must be a positive number") from exc
        if timeout <= 0:
            raise ValueError("decision timeout must be a positive number")
        outcome: queue.Queue[Any] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                outcome.put((True, DecisionRuntime._in_profile_context(
                    lambda: provider.evaluate(request), scope, secret_scope,
                )))
            except Exception as exc:  # provider failures are re-raised and normalized at the boundary
                outcome.put((False, exc))

        threading.Thread(
            target=run, name=f"decision:{provider.name}:{request.task}", daemon=True,
        ).start()
        ok, value = outcome.get(timeout=timeout)
        if not ok:
            raise _ProviderInvocationError(value)
        return value

    def _finish(self, response: DecisionResponse) -> DecisionResponse:
        event = {
            "task": response.task,
            "plugin": self.plugin_id,
            "provider": response.provider,
            "model": response.model,
            "version": response.version,
            "mode": response.mode,
            "status": response.status.value,
            "answers": {
                key: {
                    "selected": answer.selected,
                    "probabilities": dict(answer.probabilities),
                    "abstained": answer.abstained,
                }
                for key, answer in response.answers.items()
            },
            "latency_ms": response.latency_ms,
            "usage": dict(response.usage),
        }
        try:
            self.recorder.record(event)
        except Exception:
            logger.warning("Could not record decision telemetry for task %s", response.task, exc_info=True)
        return response
