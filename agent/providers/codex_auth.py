"""Codex Auth adapter (Codex real HTTP).

The critical-supervision LLM provider. Role: architecture review, code
review, approval of high-risk decisions, security review, mutation
review, final validation.

Phase 4c1 ships this as a REAL HTTP adapter that mirrors the
MiniMaxAdapter (Phase 4c) implementation:

* Uses :class:`HttpTransport` (no direct ``httpx`` import in core).
* Uses :class:`HttpErrorClassifier` (Template Method) via
  :class:`CodexErrorClassifier`.
* Uses :class:`DefaultProviderFailurePolicy` for disposition.
* Uses :class:`ProviderHealthMonitor` for health snapshot (no
  provider-specific logic in the monitor).
* Wired into :class:`LLMExecutor` via ``ExecutorRegistry`` with
  ``execution_mode="llm"``.

Credentials: read from ``CODEX_AUTH_TOKEN``. Never stored on the
instance, never printed, never persisted.

This file does NOT modify any other architectural component.
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
    CodexErrorClassifier,
    DefaultProviderFailurePolicy,
    ProviderErrorClassifier,
    ProviderErrorFact,
    ProviderFailureDisposition,
    ProviderFailurePolicy,
    classify_missing_credentials,
)
from agent.provider_http import HttpTransport, HttpxTransport
from agent.providers.registry import register_provider


DEFAULT_BASE_URL = "https://api.codex.example/v1"
DEFAULT_MODEL = "codex-auth-v1"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192


def _fact_to_provider_failure(
    fact: ProviderErrorFact,
    disposition: ProviderFailureDisposition,
) -> ProviderFailure:
    """Centralized fact → ProviderFailure conversion.

    Reused by both the adapter's ``execute`` path and the legacy
    shim below. The fact is NEVER exposed as ``LLMExecutionResult.failure``.
    """
    retryable = disposition is ProviderFailureDisposition.RETRYABLE
    return ProviderFailure(
        error_code=fact.error_code.value,
        error_message=fact.reason,
        retryable=retryable,
        failed_at_utc=_now_iso_utc(),
    )


class _ClassifyError:
    """Legacy shim kept for backward compatibility.

    Delegates to :class:`CodexErrorClassifier` + default policy.
    """

    @staticmethod
    def classify(exception: Exception) -> ProviderFailure:
        classifier = CodexErrorClassifier()
        policy = DefaultProviderFailurePolicy()
        fact = classifier.classify_exception(exception)
        disposition = policy.disposition(fact)
        return _fact_to_provider_failure(fact, disposition)


class CodexAuthAdapter(ProviderAdapter):
    """Codex Auth adapter (real HTTP).

    Mirrors :class:`MiniMaxAdapter` Phase 4c. The only provider-specific
    configuration lives here: env-var name, base URL, model, capabilities,
    limits, and the ``CodexErrorClassifier`` instance.
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
        raw_key = api_key if api_key is not None else os.getenv("CODEX_AUTH_TOKEN")
        self._has_credentials = bool(raw_key and raw_key.strip())
        self._base_url = base_url
        self._timeout_s = float(timeout_s)
        self._max_output_tokens = int(max_output_tokens)
        self._model = model

        self._http_transport: HttpTransport = http_transport or HttpxTransport(
            timeout_s=timeout_s
        )
        self._error_classifier: ProviderErrorClassifier = (
            error_classifier or CodexErrorClassifier()
        )
        self._failure_policy: ProviderFailurePolicy = (
            failure_policy or DefaultProviderFailurePolicy()
        )

        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=False,
            supports_vision=False,
            supports_long_context=True,
            max_input_tokens=200000,
            max_output_tokens=max_output_tokens,
            supported_models=(model,),
        )
        self._limits = ProviderLimits(
            default_cost_usd_micros=1500,
            default_latency_ms=8000,
            max_retries=1,
            max_runtime_s=60,
            rate_limit_per_minute=60,
            max_concurrent_requests=2,
        )

        # Rolling health snapshot — ProviderHealthMonitor interprets it.
        self._health_last_error: str | None = None
        self._consecutive_failures: int = 0

    @property
    def name(self) -> str:
        return "codex_auth"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    def health(self) -> ProviderHealth:
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

        if not self._has_credentials:
            fact = classify_missing_credentials()
            self._record_failure(fact)
            return self._build_result_from_fact(
                request=request,
                fact=fact,
                started_at_utc=started,
                completed_at_utc=_now_iso_utc(),
            )

        url = f"{self._base_url}/chat/completions"
        api_key_value = os.getenv("CODEX_AUTH_TOKEN") or ""
        headers = {
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": self._max_output_tokens,
            "temperature": float(request.temperature),
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
    # Result construction (mirrors MiniMaxAdapter)
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
            f"CodexAuthAdapter.normalize expected LLMExecutionResult, "
            f"got {type(raw_response).__name__}"
        )


# Register on import.
register_provider("codex_auth", CodexAuthAdapter)