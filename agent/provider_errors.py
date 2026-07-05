"""Provider error classification and disposition for Phase 4c.

Three independent components:

* :class:`ProviderErrorClassifier` / :class:`MiniMaxErrorClassifier` —
  interprets the provider's wire-format into a canonical
  :class:`ProviderErrorFact`. NO policy, NO health status, NO retryable.

* :class:`ProviderFailurePolicy` / :class:`DefaultProviderFailurePolicy` —
  maps a :class:`ProviderErrorFact` to a :class:`ProviderFailureDisposition`.

* :class:`ProviderErrorCode` — closed enum of error codes used across
  classifier, policy, and monitor.

:class:`ProviderHealthMonitor` (in ``agent.provider_health_monitor``)
remains the UNIQUE source of :class:`ProviderHealthStatus`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ProviderErrorCode(str, Enum):
    """Closed enum of provider error codes."""

    AUTH_ERROR = "auth_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    TRANSIENT_ERROR = "transient_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    MISSING_CREDENTIALS = "missing_credentials"


class ProviderFailureDisposition(str, Enum):
    """Caller-facing dispositions for provider failures.

    Only applies to errors. 200 OK does not produce a fact or a
    disposition; the adapter completes the ``LLMExecutionResult`` directly.
    """

    FAILED = "failed"
    BLOCKED = "blocked"
    RETRYABLE = "retryable"


@dataclass(frozen=True)
class ProviderErrorFact:
    """Provider-wire error observation.

    NOT the same as :class:`agent.provider_adapter.ProviderFailure`
    (the error object embedded in ``LLMExecutionResult.failure``).
    """

    error_code: ProviderErrorCode
    reason: str
    http_status: int | None = None
    schema_version: int = 1


@runtime_checkable
class ProviderErrorClassifier(Protocol):
    """Interprets the provider's wire-format into a fact."""

    def classify_http_response(
        self,
        *,
        status_code: int,
        body: Mapping[str, Any] | None,
        text: str = "",
    ) -> ProviderErrorFact | None: ...

    def classify_exception(
        self,
        exc: BaseException,
    ) -> ProviderErrorFact: ...


@runtime_checkable
class ProviderFailurePolicy(Protocol):
    """Maps a :class:`ProviderErrorFact` to a caller-facing disposition."""

    def disposition(self, fact: ProviderErrorFact) -> ProviderFailureDisposition: ...


# ---------------------------------------------------------------------------
# MiniMaxErrorClassifier
# ---------------------------------------------------------------------------


_QUOTA_HINTS = ("quota", "billing", "credits", "token_plan", "payment_required")
_CONTEXT_HINTS = (
    "context_length",
    "context_length_exceeded",
    "context_overflow",
    "max_tokens",
)
_AUTH_HINTS = (
    "auth_error",
    "unauthorized",
    "forbidden",
    "invalid_api_key",
    "authentication",
)
_TRANSIENT_HINTS = (
    "transient",
    "timeout",
    "service_unavailable",
    "bad_gateway",
    "internal_server_error",
)


def _looks_like(body_text: str, hints: tuple[str, ...]) -> bool:
    low = body_text.lower()
    return any(h in low for h in hints)


# ---------------------------------------------------------------------------
# HttpErrorClassifier (Template Method base)
# ---------------------------------------------------------------------------


class HttpErrorClassifier(ABC):
    """Template Method base for HTTP error classification.

    The base controls the algorithm; subclasses override
    :meth:`parse_provider_error` to add provider-specific knowledge.

    Common logic owned by the base (not overridable):
      * safe JSON parsing of ``text`` when ``body`` is missing.
      * Generic HTTP handling: 200 OK with empty body → INVALID_RESPONSE;
        5xx → TRANSIENT_ERROR; status unknown → fall through.
      * Transport exception handling: timeout / ConnectionError / OSError.

    Subclasses implement only:
      * :meth:`parse_provider_error` (pure, deterministic, no side effects).
    """

    # ----- final (Template Method) -----

    def classify_http_response(
        self,
        *,
        status_code: int,
        body: Mapping[str, Any] | None,
        text: str = "",
    ) -> ProviderErrorFact | None:
        if body is None:
            body = self.safe_json(text)
        generic = self.classify_generic_http(status_code=status_code, body=body)
        if generic is not None:
            return generic
        # 200 OK with parseable body is a success: no fact, no provider
        # interpretation. Provider-specific parsing only applies to
        # error status codes (4xx).
        if status_code == 200:
            return None
        return self.parse_provider_error(
            status_code=status_code, body=body, text=text
        )

    def classify_exception(self, exc: BaseException) -> ProviderErrorFact:
        name = type(exc).__name__
        low = name.lower()
        if "timeout" in low or "timedout" in low:
            return ProviderErrorFact(
                error_code=ProviderErrorCode.TIMEOUT,
                reason=f"{name}: timeout",
            )
        if isinstance(exc, (ConnectionError, OSError)):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.TIMEOUT,
                reason=f"{name}: connection error",
            )
        return ProviderErrorFact(
            error_code=ProviderErrorCode.INVALID_RESPONSE,
            reason=f"{name}: {str(exc)[:200]}",
        )

    # ----- common logic (subclasses MUST NOT override) -----

    @staticmethod
    def safe_json(text: str) -> Mapping[str, Any] | None:
        """Defensive JSON parse. Returns ``None`` on any failure."""
        if not text:
            return None
        import json

        try:
            data = json.loads(text)
        except Exception:
            return None
        if isinstance(data, Mapping):
            return data
        return None

    def classify_generic_http(
        self,
        *,
        status_code: int,
        body: Mapping[str, Any] | None,
    ) -> ProviderErrorFact | None:
        """Return a generic fact for cases the base handles itself.

        Returns ``None`` to fall through to provider-specific parsing.
        """
        if status_code == 200:
            if body is not None:
                return None  # 200 OK with parseable body → provider decides.
            return ProviderErrorFact(
                error_code=ProviderErrorCode.INVALID_RESPONSE,
                reason="200 OK with empty body",
                http_status=200,
            )
        if 500 <= status_code < 600:
            return ProviderErrorFact(
                error_code=ProviderErrorCode.TRANSIENT_ERROR,
                reason=f"HTTP {status_code} transient error",
                http_status=status_code,
            )
        return None

    # ----- provider extension point (only this is overridable) -----

    @abstractmethod
    def parse_provider_error(
        self,
        *,
        status_code: int,
        body: Mapping[str, Any] | None,
        text: str,
    ) -> ProviderErrorFact | None:
        """Return a fact for provider-specific errors; ``None`` otherwise."""
        ...


# ---------------------------------------------------------------------------
# MiniMaxErrorClassifier
# ---------------------------------------------------------------------------


class MiniMaxErrorClassifier(HttpErrorClassifier):
    """V1 classifier for MiniMax wire-format.

    Returns ``None`` when the HTTP response is 200 OK with a parseable
    body. Otherwise returns a :class:`ProviderErrorFact`.

    Does NOT decide disposition. Does NOT decide health status. Does NOT
    read secrets. Does NOT execute HTTP.

    Inherits :class:`HttpErrorClassifier` (Template Method); only
    overrides :meth:`parse_provider_error`.
    """

    _QUOTA_HINTS = ("quota", "billing", "credits", "token_plan", "payment_required")
    _CONTEXT_HINTS = (
        "context_length",
        "context_length_exceeded",
        "context_overflow",
        "max_tokens",
    )

    def parse_provider_error(
        self,
        *,
        status_code: int,
        body: Mapping[str, Any] | None,
        text: str,
    ) -> ProviderErrorFact | None:
        if status_code in (401, 403):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.AUTH_ERROR,
                reason=f"HTTP {status_code} auth error",
                http_status=status_code,
            )
        if status_code == 429:
            return ProviderErrorFact(
                error_code=ProviderErrorCode.RATE_LIMIT_EXCEEDED,
                reason="HTTP 429 rate limit exceeded",
                http_status=status_code,
            )
        if status_code == 402 or _looks_like(text or "", self._QUOTA_HINTS):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.QUOTA_EXCEEDED,
                reason="quota or billing limit reached",
                http_status=status_code if status_code != 200 else None,
            )
        if status_code in (400, 413) or _looks_like(text or "", self._CONTEXT_HINTS):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED,
                reason="context length exceeded",
                http_status=status_code if status_code != 200 else None,
            )
        return ProviderErrorFact(
            error_code=ProviderErrorCode.INVALID_RESPONSE,
            reason=f"unexpected HTTP {status_code}",
            http_status=status_code,
        )


# ---------------------------------------------------------------------------
# CodexErrorClassifier
# ---------------------------------------------------------------------------


class CodexErrorClassifier(HttpErrorClassifier):
    """V1 classifier for Codex wire-format.

    Mirrors :class:`MiniMaxErrorClassifier` against Codex's expected
    shape. Inherits :class:`HttpErrorClassifier` (Template Method);
    only overrides :meth:`parse_provider_error`.

    Codex is assumed to use an OpenAI-compatible error body shape:
        ``{"error": {"code": "...", "message": "...", "type": "..."}}``

    If Codex returns a different shape, only this class changes.
    """

    _QUOTA_HINTS = ("quota", "billing", "insufficient_quota", "credit", "payment")
    _CONTEXT_HINTS = (
        "context_length",
        "context_length_exceeded",
        "context_overflow",
        "max_tokens",
        "tokens_limit",
    )
    _AUTH_HINTS = (
        "auth",
        "unauthorized",
        "forbidden",
        "invalid_api_key",
        "auth_error",
    )

    def parse_provider_error(
        self,
        *,
        status_code: int,
        body: Mapping[str, Any] | None,
        text: str,
    ) -> ProviderErrorFact | None:
        if status_code in (401, 403):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.AUTH_ERROR,
                reason=f"HTTP {status_code} auth error",
                http_status=status_code,
            )
        if status_code == 429:
            return ProviderErrorFact(
                error_code=ProviderErrorCode.RATE_LIMIT_EXCEEDED,
                reason="HTTP 429 rate limit exceeded",
                http_status=status_code,
            )
        if status_code == 402 or _looks_like(text or "", self._QUOTA_HINTS):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.QUOTA_EXCEEDED,
                reason="quota or billing limit reached",
                http_status=status_code if status_code != 200 else None,
            )
        if status_code in (400, 413) or _looks_like(text or "", self._CONTEXT_HINTS):
            return ProviderErrorFact(
                error_code=ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED,
                reason="context length exceeded",
                http_status=status_code if status_code != 200 else None,
            )
        return ProviderErrorFact(
            error_code=ProviderErrorCode.INVALID_RESPONSE,
            reason=f"unexpected HTTP {status_code}",
            http_status=status_code,
        )


# ---------------------------------------------------------------------------
# DefaultProviderFailurePolicy
# ---------------------------------------------------------------------------


class DefaultProviderFailurePolicy:
    """V1 implementation. Maps ``error_code`` to ``disposition``.

    Injected into the adapter so future policy changes do not require
    changing :class:`MiniMaxErrorClassifier`.
    """

    _TABLE: Mapping[ProviderErrorCode, ProviderFailureDisposition] = {
        ProviderErrorCode.AUTH_ERROR: ProviderFailureDisposition.BLOCKED,
        ProviderErrorCode.QUOTA_EXCEEDED: ProviderFailureDisposition.BLOCKED,
        ProviderErrorCode.RATE_LIMIT_EXCEEDED: ProviderFailureDisposition.BLOCKED,
        ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED: ProviderFailureDisposition.BLOCKED,
        ProviderErrorCode.MISSING_CREDENTIALS: ProviderFailureDisposition.BLOCKED,
        ProviderErrorCode.TRANSIENT_ERROR: ProviderFailureDisposition.RETRYABLE,
        ProviderErrorCode.TIMEOUT: ProviderFailureDisposition.RETRYABLE,
        ProviderErrorCode.INVALID_RESPONSE: ProviderFailureDisposition.FAILED,
    }

    def disposition(self, fact: ProviderErrorFact) -> ProviderFailureDisposition:
        return self._TABLE[fact.error_code]


# ---------------------------------------------------------------------------
# Missing credentials helper
# ---------------------------------------------------------------------------


def classify_missing_credentials(reason: str = "MINIMAX_API_KEY not set") -> ProviderErrorFact:
    """Build a fact for the missing-credentials case.

    Kept here (not in :class:`MiniMaxErrorClassifier`) because missing
    credentials are detected at construction time, not from an HTTP
    response or exception.
    """
    return ProviderErrorFact(
        error_code=ProviderErrorCode.MISSING_CREDENTIALS,
        reason=reason,
        http_status=None,
    )