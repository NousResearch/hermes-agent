"""MiniMax M3 adapter (Phase 4c REAL HTTP).

The default conversational LLM provider. Role: bulk reasoning, long
context, cheap worker, drafts, summaries, brainstorming.

Phase 4c turns this into a real HTTP adapter, structured around three
independent components:

* :class:`agent.provider_http.HttpTransport` — abstracts the HTTP client.
  The adapter accepts an injected transport (tests use ``FakeHttpTransport``;
  production uses :class:`HttpxTransport`). The adapter does NOT import
  ``httpx`` directly.

* :class:`agent.provider_errors.ProviderErrorClassifier` /
  :class:`MiniMaxErrorClassifier` — interprets the wire-format response
  into a :class:`ProviderErrorFact`. Does not decide disposition or health.

* :class:`agent.provider_errors.ProviderFailurePolicy` /
  :class:`DefaultProviderFailurePolicy` — maps a fact to a
  :class:`ProviderFailureDisposition`. Injected.

The adapter composes them: ``fact → policy.disposition(fact) → ProviderFailure → LLMExecutionResult``.
"""

from __future__ import annotations

import os
from typing import Any

from agent.provider_adapter import (
    LLMExecutionRequest,
    LLMExecutionResult,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderFailure,
    ProviderHealth,
    ProviderLimits,
    _now_iso_utc,
)
from agent.provider_errors import (
    DefaultProviderFailurePolicy,
    MiniMaxErrorClassifier,
    ProviderErrorClassifier,
    ProviderErrorFact,
    ProviderFailureDisposition,
    ProviderFailurePolicy,
    classify_missing_credentials,
)
from agent.provider_http import HttpTransport, HttpxTransport
from agent.providers.registry import register_provider


DEFAULT_BASE_URL = "https://api.minimaxi.chat/v1"
DEFAULT_MODEL = "minimax/M3"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_S = 30.0


class _ClassifyError:
    """Legacy classifier kept for backward compatibility.

    Delegates to :class:`MiniMaxErrorClassifier` for any exception that
    propagates through :meth:`MiniMaxAdapter.execute`.
    """

    @staticmethod
    def classify(exception: Exception) -> ProviderFailure:
        fact = MiniMaxErrorClassifier().classify_exception(exception)
        disposition = DefaultProviderFailurePolicy().disposition(fact)
        return _fact_to_provider_failure(fact, disposition)


def _fact_to_provider_failure(
    fact: ProviderErrorFact,
    disposition: ProviderFailureDisposition,
) -> ProviderFailure:
    """Convert a fact + disposition into a :class:`ProviderFailure`.

    Used by both the adapter's ``execute`` path and the legacy
    :class:`_ClassifyError` shim. Centralized so the conversion is
    explicit (the adapter never embeds a ProviderErrorFact in
    ``LLMExecutionResult.failure``).
    """
    retryable = disposition is ProviderFailureDisposition.RETRYABLE
    return ProviderFailure(
        error_code=fact.error_code.value,
        error_message=fact.reason,
        retryable=retryable,
        failed_at_utc=_now_iso_utc(),
    )


class MiniMaxAdapter(ProviderAdapter):
    """MiniMax M3 adapter (Phase 4c REAL HTTP).

    The adapter:

    * Loads the API key via ``os.getenv("MINIMAX_API_KEY")`` — the value
      is NEVER stored on the instance, NEVER printed, NEVER logged.
    * Builds the request body via :class:`HttpTransport`.
    * Delegates response/exception classification to the injected
      :class:`ProviderErrorClassifier`.
    * Delegates disposition to the injected :class:`ProviderFailurePolicy`.
    * Updates ``_health.last_error`` with the canonical error code so the
      :class:`agent.provider_health_monitor.ProviderHealthMonitor` can map
      it to :class:`ProviderHealthStatus`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        model: str = DEFAULT_MODEL,
        http_transport: HttpTransport | None = None,
        error_classifier: ProviderErrorClassifier | None = None,
        failure_policy: ProviderFailurePolicy | None = None,
    ) -> None:
        # Credential detection: only check existence, never store value.
        raw_key = api_key if api_key is not None else os.getenv("MINIMAX_API_KEY")
        self._has_credentials = bool(raw_key and raw_key.strip())
        self._base_url = base_url
        self._timeout_s = float(timeout_s)
        self._max_output_tokens = int(max_output_tokens)
        self._model = model

        # Components. Tests inject their own; production uses defaults.
        self._http_transport: HttpTransport = http_transport or HttpxTransport(
            timeout_s=timeout_s
        )
        self._error_classifier: ProviderErrorClassifier = (
            error_classifier or MiniMaxErrorClassifier()
        )
        self._failure_policy: ProviderFailurePolicy = (
            failure_policy or DefaultProviderFailurePolicy()
        )

        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=False,
            supports_long_context=True,
            max_input_tokens=128000,
            max_output_tokens=max_output_tokens,
            supported_models=(model,),
        )
        self._limits = ProviderLimits(
            default_cost_usd_micros=500,
            default_latency_ms=3000,
            max_retries=2,
            max_runtime_s=30,
            rate_limit_per_minute=120,
            max_concurrent_requests=8,
        )

        # Health snapshot (rolling). last_error stores the canonical
        # ProviderErrorCode.value so the monitor can map it.
        self._health_last_error: str | None = None
        self._consecutive_failures: int = 0

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    def health(self) -> ProviderHealth:
        # Phase 4c: snapshot is rolling. Missing credentials means we
        # are unavailable.
        is_available = self._has_credentials and (
            self._health_last_error is None or self._consecutive_failures <= 2
        )
        return ProviderHealth(
            is_available=is_available,
            last_checked_at_utc=_now_iso_utc(),
            consecutive_failures=self._consecutive_failures,
            last_error=self._health_last_error,
        )

    def execute(self, request: LLMExecutionRequest) -> LLMExecutionResult:
        started = _now_iso_utc()

        # Missing credentials: short-circuit with a MISSING_CREDENTIALS fact.
        if not self._has_credentials:
            fact = classify_missing_credentials()
            self._record_failure(fact)
            return self._build_result_from_fact(
                request=request,
                fact=fact,
                started_at_utc=started,
                completed_at_utc=_now_iso_utc(),
            )

        # Real HTTP call.
        url = f"{self._base_url}/text/chatcompletion_v2"
        # Authorization header carries the API key but never the value
        # is stored or logged by this adapter.
        api_key_value = os.getenv("MINIMAX_API_KEY") or ""
        headers = {
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": request.prompt},
            ],
            "max_tokens": self._max_output_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        try:
            response = self._http_transport.post(
                url,
                headers=headers,
                json=body,
                timeout=self._timeout_s,
            )
        except Exception as exc:
            fact = self._error_classifier.classify_exception(exc)
            self._record_failure(fact)
            return self._build_result_from_fact(
                request=request,
                fact=fact,
                started_at_utc=started,
                completed_at_utc=_now_iso_utc(),
            )

        # Parse body safely.
        try:
            body_dict: Any = response.json()
        except Exception:
            body_dict = None
        text = response.text or ""

        fact = self._error_classifier.classify_http_response(
            status_code=response.status_code,
            body=body_dict if isinstance(body_dict, dict) else None,
            text=text,
        )
        if fact is None:
            # 200 OK with parseable body → completed.
            self._reset_health()
            return self._build_completed_result(
                request=request,
                response=response,
                body=body_dict,
                started_at_utc=started,
                completed_at_utc=_now_iso_utc(),
            )
        self._record_failure(fact)
        return self._build_result_from_fact(
            request=request,
            fact=fact,
            started_at_utc=started,
            completed_at_utc=_now_iso_utc(),
        )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result_from_fact(
        self,
        *,
        request: LLMExecutionRequest,
        fact: ProviderErrorFact,
        started_at_utc: str,
        completed_at_utc: str,
    ) -> LLMExecutionResult:
        disposition = self._failure_policy.disposition(fact)
        # Translate disposition into LLMExecutionResult.status.
        if disposition is ProviderFailureDisposition.BLOCKED:
            status = "blocked"
        else:
            status = "failed"
        failure = _fact_to_provider_failure(fact, disposition)
        return LLMExecutionResult(
            request_id=request.request_id,
            decision_id=request.decision_id,
            provider=self.name,
            status=status,
            failure=failure,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
        )

    def _build_completed_result(
        self,
        *,
        request: LLMExecutionRequest,
        response: Any,
        body: Any,
        started_at_utc: str,
        completed_at_utc: str,
    ) -> LLMExecutionResult:
        # Best-effort extraction from OpenAI-compatible wire-format.
        output_text = ""
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                output_text = str(message.get("content", ""))
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        if usage is None:
            usage = {}
        return LLMExecutionResult(
            request_id=request.request_id,
            decision_id=request.decision_id,
            provider=self.name,
            status="completed",
            output_text=output_text,
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            cost_usd_micros=self._limits.default_cost_usd_micros,
            latency_ms=self._limits.default_latency_ms,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
        )

    # ------------------------------------------------------------------
    # Health snapshot updates
    # ------------------------------------------------------------------

    def _record_failure(self, fact: ProviderErrorFact) -> None:
        self._health_last_error = fact.error_code.value
        self._consecutive_failures += 1

    def _reset_health(self) -> None:
        self._health_last_error = None
        self._consecutive_failures = 0

    def normalize(self, raw_response: Any) -> LLMExecutionResult:
        if isinstance(raw_response, LLMExecutionResult):
            return raw_response
        raise ValueError(
            f"MiniMaxAdapter.normalize expected LLMExecutionResult, "
            f"got {type(raw_response).__name__}"
        )


# Register on import.
register_provider("minimax", MiniMaxAdapter)