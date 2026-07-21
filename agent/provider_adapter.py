"""Provider Adapter ABC and supporting dataclasses.

Layer 4 of Hermes. Phase 4b NEW.

This module defines the contract that all LLM provider adapters must
implement. Adapters encapsulate provider-specific HTTP / SDK / wire
protocol. The Engine never sees provider-specific types.

The contract is intentionally minimal: an adapter exposes its identity,
capabilities, limits, health, and a single execute() method that
returns a normalized result.

Phase 4b ships only the ABC + dataclasses. Concrete adapters (MiniMax,
Codex Auth, Fake) live in agent/providers/*.py.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ── Closed enums (v1) ─────────────────────────────────────────────────────

ExecutionStatus = Literal[
    "completed",      # success
    "local",          # Engine returned local echo (no provider call)
    "blocked",        # Engine refused (decision was blocked)
    "failed",         # generic failure
    "timeout",        # adapter exceeded max_runtime_s
    "rate_limited",   # provider returned 429
    "auth_failed",    # credentials missing or invalid
    "policy_violation",  # provider rejected for policy reasons
    "model_not_found",   # requested model not supported
    "context_too_long",  # input exceeds max_input_tokens
]

VALID_EXECUTION_STATUSES = frozenset({
    "completed", "local", "blocked", "failed", "timeout",
    "rate_limited", "auth_failed", "policy_violation",
    "model_not_found", "context_too_long",
})


# ── Forbidden names (enforced by tests) ────────────────────────────────────

FORBIDDEN_METHOD_NAMES = frozenset({
    # Note: the Engine's primary method is `process()`, not `execute()`.
    # Adapters DO have `execute()` — that's their public contract.
    # The Engine does NOT have execute(); that's checked elsewhere.
    "run", "submit", "dispatch", "spawn",
})

FORBIDDEN_IMPORT_PATTERNS = (
    "openai",
    "anthropic",
    "litellm",
    "google.generativeai",
    "subprocess.run",
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.popen",
    "requests.",
    "urllib.",
    "httpx.",
    "aiohttp.",
)


# ── Dataclasses (frozen) ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static capabilities of a provider (immutable per provider).

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Do not extend or reorder fields without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_long_context: bool = False
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    supported_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderLimits:
    """Static limits of a provider (immutable per provider).

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Do not extend or reorder fields without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    default_cost_usd_micros: int = 0
    default_latency_ms: int = 0
    max_retries: int = 0
    max_runtime_s: int = 0
    rate_limit_per_minute: int = 0
    max_concurrent_requests: int = 1


@dataclass(frozen=True)
class ProviderHealth:
    """Current health of a provider.

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Do not extend or reorder fields without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    is_available: bool = False
    last_checked_at_utc: str = ""
    consecutive_failures: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class ProviderFailure:
    """Failure detail for a single provider call.

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Do not extend or reorder fields without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    error_code: str = ""           # one of 8 codes (see engine)
    error_message: str = ""
    retryable: bool = False
    failed_at_utc: str = ""


@dataclass(frozen=True)
class LLMExecutionRequest:
    """Input to a single adapter.execute() call.

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Do not extend or reorder fields without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    request_id: str
    decision_id: str
    prompt: str
    model: str = ""
    max_tokens: int = 0
    temperature: float = 0.0
    system_message: str = ""
    conversation_history: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LLMExecutionResult:
    """Output of a single adapter.execute() call. Engine-facing.

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Do not extend or reorder fields without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    request_id: str
    decision_id: str
    provider: str
    status: str
    output_text: str = ""
    output_tokens: int = 0
    input_tokens: int = 0
    cost_usd_micros: int = 0
    latency_ms: int = 0
    started_at_utc: str = ""
    completed_at_utc: str = ""
    failure: ProviderFailure | None = None
    fallback_triggered: bool = False
    fallback_reason: str = ""
    raw_response: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision_id": self.decision_id,
            "provider": self.provider,
            "status": self.status,
            "output_text": self.output_text,
            "output_tokens": self.output_tokens,
            "input_tokens": self.input_tokens,
            "cost_usd_micros": self.cost_usd_micros,
            "latency_ms": self.latency_ms,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "failure": (
                {
                    "error_code": self.failure.error_code,
                    "error_message": self.failure.error_message,
                    "retryable": self.failure.retryable,
                    "failed_at_utc": self.failure.failed_at_utc,
                }
                if self.failure is not None
                else None
            ),
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── ProviderAdapter ABC ─────────────────────────────────────────────────────


class ProviderAdapter(ABC):
    """Abstract base for all LLM provider adapters.

    Adapters encapsulate the provider-specific HTTP / SDK / wire
    protocol. The Engine never sees provider-specific types.

    Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
    Public surface: 6 methods/properties (name, capabilities, limits,
    health, execute, normalize). Adapters may add private helpers, but
    the public surface is fixed. Do not extend without bumping to v2.
    See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable, lowercase provider identifier.

        Must match one of the names used by the Execution Router
        (minimax, openai, codex, codex_auth, nemotron, anthropic,
        google, local_only, etc.)."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @property
    @abstractmethod
    def limits(self) -> ProviderLimits: ...

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    def execute(self, request: LLMExecutionRequest) -> LLMExecutionResult:
        """Execute the request. Returns a normalized result.

        MUST NOT raise: convert all exceptions to LLMExecutionResult
        with status='failed' and a ProviderFailure."""

    @abstractmethod
    def normalize(self, raw_response: Any) -> LLMExecutionResult:
        """Convert a provider-specific response object into
        LLMExecutionResult. Used internally by execute(); also exposed
        for adapter testing."""