"""FakeProviderAdapter for tests.

A configurable adapter that returns canned responses. The Phase 4b
tests register this under a known provider name (e.g., "minimax") and
verify the Engine's behavior using deterministic canned outputs.

The FakeAdapter is also the ONLY adapter used in Phase 4b's
integration tests. Real MiniMax / Codex adapters (in their own modules)
are stubs that return canned responses too — but the FakeAdapter is
the canonical "test double" with explicit configuration knobs.

Configuration (constructor params):
- `provider_name`: the name to register under (default "fake").
- `configured_response`: optional pre-canned LLMExecutionResult.
- `configured_delay_s`: optional delay to simulate slow adapters.
- `configured_health`: optional pre-canned ProviderHealth.
- `throw_exception`: if set, raise this exception instead of returning.
- `error_code`: if set, return a result with this error_code instead of "completed".
- `consecutive_failures_increment`: if True, increment consecutive_failures
  on each health() call (used to test health-driven fallback).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from agent.provider_adapter import (
    LLMExecutionRequest,
    LLMExecutionResult,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderHealth,
    ProviderLimits,
    _now_iso_utc,
)
from agent.providers.registry import register_provider


@dataclass
class FakeConfig:
    """Configuration for a single FakeProviderAdapter instance.

    Stored on the adapter; mutable from outside so tests can change
    behavior between calls.
    """

    configured_response: LLMExecutionResult | None = None
    configured_delay_s: float = 0.0
    configured_health: ProviderHealth | None = None
    throw_exception: Exception | None = None
    error_code: str | None = None  # if set, returns a failure result
    consecutive_failures_increment: bool = False

    call_count: int = 0
    last_request: LLMExecutionRequest | None = None
    call_history: list[LLMExecutionRequest] = field(default_factory=list)


class FakeProviderAdapter(ProviderAdapter):
    """A test-double ProviderAdapter. Returns canned responses."""

    DEFAULT_NAME = "fake"

    def __init__(
        self,
        provider_name: str | None = None,
        config: FakeConfig | None = None,
    ) -> None:
        self._name = provider_name or self.DEFAULT_NAME
        self.config = config or FakeConfig()
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=False,
            supports_long_context=True,
            max_input_tokens=128000,
            max_output_tokens=4096,
            supported_models=("fake-model",),
        )
        self._limits = ProviderLimits(
            default_cost_usd_micros=500,
            default_latency_ms=3000,
            max_retries=2,
            max_runtime_s=30,
            rate_limit_per_minute=120,
            max_concurrent_requests=8,
        )
        self._health = ProviderHealth(
            is_available=True,
            last_checked_at_utc=_now_iso_utc(),
            consecutive_failures=0,
            last_error=None,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def limits(self) -> ProviderLimits:
        return self._limits

    def health(self) -> ProviderHealth:
        if self.config.consecutive_failures_increment:
            self._health = replace(
                self._health,
                consecutive_failures=self._health.consecutive_failures + 1,
                last_checked_at_utc=_now_iso_utc(),
            )
            if self._health.consecutive_failures > 2:
                self._health = replace(self._health, is_available=False)
        elif self.config.configured_health is not None:
            self._health = self.config.configured_health
        else:
            self._health = replace(
                self._health,
                last_checked_at_utc=_now_iso_utc(),
            )
        return self._health

    def execute(self, request: LLMExecutionRequest) -> LLMExecutionResult:
        """Return canned response (or raise if configured)."""
        self.config.call_count += 1
        self.config.last_request = request
        self.config.call_history.append(request)

        if self.config.throw_exception is not None:
            raise self.config.throw_exception

        if self.config.configured_delay_s > 0:
            time.sleep(self.config.configured_delay_s)

        if self.config.error_code is not None:
            return LLMExecutionResult(
                request_id=request.request_id,
                decision_id=request.decision_id,
                provider=self.name,
                status=self.config.error_code,
                output_text="",
                started_at_utc=_now_iso_utc(),
                completed_at_utc=_now_iso_utc(),
            )

        if self.config.configured_response is not None:
            return self.config.configured_response

        # Default: a deterministic canned "completed" result.
        return LLMExecutionResult(
            request_id=request.request_id,
            decision_id=request.decision_id,
            provider=self.name,
            status="completed",
            output_text=f"[{self.name} stub] processed prompt of {len(request.prompt)} chars",
            output_tokens=10,
            input_tokens=max(1, len(request.prompt) // 4),
            cost_usd_micros=self._limits.default_cost_usd_micros,
            latency_ms=self._limits.default_latency_ms,
            started_at_utc=_now_iso_utc(),
            completed_at_utc=_now_iso_utc(),
        )

    def normalize(self, raw_response: Any) -> LLMExecutionResult:
        """Passthrough: if raw_response is already an LLMExecutionResult, return it.

        Otherwise raise ValueError (matches the ABC contract).
        """
        if isinstance(raw_response, LLMExecutionResult):
            return raw_response
        raise ValueError(
            f"FakeProviderAdapter.normalize expected LLMExecutionResult, "
            f"got {type(raw_response).__name__}"
        )


# Register under "fake" by default. Tests can register additional
# instances under different names by calling register_provider() directly.
register_provider(FakeProviderAdapter.DEFAULT_NAME, FakeProviderAdapter)