"""Structured provider error classification for Hermes Agent.

Centralises the error-classification logic that was previously scattered
as inline string-matching inside the retry loop of ``run_agent.py``.
Every provider/transport error is mapped to a :class:`ProviderError`
carrying a :class:`ProviderErrorReason` enum value and metadata (status
code, retryability, suggested action).

The public API is:

* :func:`classify_provider_error` — the main classifier
* :func:`is_retryable` — quick boolean check on a reason
* :func:`suggested_action` — human-readable recovery strategy
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Enum ─────────────────────────────────────────────────────────────────

class ProviderErrorReason(enum.Enum):
    """Canonical error categories for LLM provider failures."""

    AUTH = "auth"                              # 401/403 — may recover with credential refresh
    AUTH_PERMANENT = "auth_permanent"          # invalid key / bad credentials — won't recover
    RATE_LIMIT = "rate_limit"             # 429 / quota / too many requests
    OVERLOADED = "overloaded"             # 529 (Anthropic overloaded)
    BILLING = "billing"                   # 429 + extra-usage tier gate (not transient)
    MODEL_NOT_FOUND = "model_not_found"   # invalid/unknown model ID
    CONTEXT_OVERFLOW = "context_overflow" # context-length / token-limit exceeded
    PAYLOAD_TOO_LARGE = "payload_too_large"  # 413 / request entity too large
    FORMAT_ERROR = "format_error"         # ValueError/TypeError (local validation)
    TIMEOUT = "timeout"                   # request timeout
    SERVER_ERROR = "server_error"         # 5xx / generic transient server error
    STREAM_DROP = "stream_drop"           # connection lost/reset during streaming
    UNKNOWN = "unknown"                   # unclassified


# ── Dataclass ────────────────────────────────────────────────────────────

@dataclass
class ProviderError:
    """Structured representation of a provider error."""

    reason: ProviderErrorReason
    status_code: Optional[int] = None
    message: str = ""
    retryable: bool = True
    provider: Optional[str] = None
    original_error: Optional[Exception] = field(default=None, repr=False)


# ── Context-length keyword phrases (order doesn't matter) ───────────────

_CONTEXT_LENGTH_PHRASES = (
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "request entity too large",       # OpenRouter/Nous 413 safety-net
    "prompt is too long",             # Anthropic: "prompt is too long: N tokens > M maximum"
    "prompt exceeds max length",      # Z.AI / GLM: generic 400 overflow wording
)

_STREAM_DROP_PHRASES = (
    "connection lost",
    "connection reset",
    "connection closed",
    "network connection",
    "network error",
    "terminated",
)

_SERVER_DISCONNECT_ERROR_TYPES = frozenset({
    "ReadError",
    "RemoteProtocolError",
    "ServerDisconnectedError",
})


# ── Main classifier ─────────────────────────────────────────────────────

def classify_provider_error(
    error: Exception,
    error_msg: str = "",
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
    *,
    # Contextual hints the caller can provide for heuristic classification
    model: str = "",
    approx_tokens: int = 0,
    num_messages: int = 0,
    context_length: int = 200_000,
    error_body: Optional[dict] = None,
) -> ProviderError:
    """Classify a provider/transport exception into a :class:`ProviderError`.

    The function preserves the exact same classification priority and logic
    that was previously inline in ``run_agent.py``'s retry loop.

    Parameters
    ----------
    error:
        The caught exception.
    error_msg:
        ``str(error).lower()`` — pre-lowered for efficiency since the
        caller already computes this.
    status_code:
        HTTP status code (may be ``None`` for transport errors).
    provider:
        Provider name string (e.g. ``"anthropic"``, ``"openai"``).
    model:
        Model identifier (used for Sonnet long-context tier check).
    approx_tokens:
        Approximate token count of the current session.
    num_messages:
        Number of API messages in the current request.
    context_length:
        The compressor's current context_length (for heuristic checks).
    error_body:
        Parsed error body dict (``getattr(error, "body", None)``).
    """

    if not error_msg:
        error_msg = str(error).lower()
    error_type = type(error).__name__

    def _make(reason: ProviderErrorReason, retryable: bool, msg: str = "") -> ProviderError:
        return ProviderError(
            reason=reason,
            status_code=status_code,
            message=msg or error_msg,
            retryable=retryable,
            provider=provider,
            original_error=error,
        )

    # ── 1. Anthropic long-context tier gate (429 + extra usage + long context) ──
    if (
        status_code == 429
        and "extra usage" in error_msg
        and "long context" in error_msg
    ):
        return _make(ProviderErrorReason.BILLING, retryable=False,
                     msg="Anthropic long-context tier requires extra usage subscription")

    # ── 2. Rate limits (429 / quota phrases) ────────────────────────────
    if (
        status_code == 429
        or "rate limit" in error_msg
        or "too many requests" in error_msg
        or "rate_limit" in error_msg
        or "usage limit" in error_msg
        or "quota" in error_msg
    ):
        return _make(ProviderErrorReason.RATE_LIMIT, retryable=True)

    # ── 3. Payload too large (413 / phrases) ────────────────────────────
    if (
        status_code == 413
        or "request entity too large" in error_msg
        or "payload too large" in error_msg
        or "error code: 413" in error_msg
    ):
        return _make(ProviderErrorReason.PAYLOAD_TOO_LARGE, retryable=True)

    # ── 4. Context-length errors (explicit keyword phrases) ─────────────
    if any(phrase in error_msg for phrase in _CONTEXT_LENGTH_PHRASES):
        return _make(ProviderErrorReason.CONTEXT_OVERFLOW, retryable=True)

    # ── 5. Generic 400 + large session heuristic → probable context overflow
    if status_code == 400:
        is_large_session = approx_tokens > context_length * 0.4 or num_messages > 80
        is_generic_error = len(error_msg.strip()) < 30
        if is_large_session and is_generic_error:
            return _make(ProviderErrorReason.CONTEXT_OVERFLOW, retryable=True,
                         msg=f"Generic 400 with large session (~{approx_tokens:,} tokens, "
                             f"{num_messages} msgs) — probable context overflow")

    # ── 6. Server disconnect + large session heuristic → context overflow
    if not status_code:
        _is_server_disconnect = (
            "server disconnected" in error_msg
            or "peer closed connection" in error_msg
            or error_type in _SERVER_DISCONNECT_ERROR_TYPES
        )
        if _is_server_disconnect:
            _is_large = approx_tokens > context_length * 0.6 or num_messages > 200
            if _is_large:
                return _make(ProviderErrorReason.CONTEXT_OVERFLOW, retryable=True,
                             msg=f"Server disconnected with large session (~{approx_tokens:,} "
                                 f"tokens, {num_messages} msgs) — probable context overflow")

    # ── 7. Overloaded (529) ─────────────────────────────────────────────
    if status_code == 529:
        return _make(ProviderErrorReason.OVERLOADED, retryable=True)

    # ── 8. Model not found ──────────────────────────────────────────────
    if (
        "is not a valid model" in error_msg
        or "invalid model" in error_msg
        or "model not found" in error_msg
    ):
        return _make(ProviderErrorReason.MODEL_NOT_FOUND, retryable=False)

    # ── 9. Permanent auth failures (invalid key / authentication) ───────
    if (
        "invalid api key" in error_msg
        or "invalid_api_key" in error_msg
        or "authentication" in error_msg
    ):
        return _make(ProviderErrorReason.AUTH_PERMANENT, retryable=False)

    # ── 10. Auth errors (401/403) ───────────────────────────────────────
    if status_code in (401, 403):
        return _make(ProviderErrorReason.AUTH, retryable=False)

    # ── 11. Local validation errors (ValueError/TypeError, not UnicodeEncodeError)
    if isinstance(error, (ValueError, TypeError)) and not isinstance(error, UnicodeEncodeError):
        return _make(ProviderErrorReason.FORMAT_ERROR, retryable=False)

    # ── 12. Generic 400 from Anthropic OAuth (transient "Error" / empty body)
    if status_code == 400:
        _body = error_body if error_body is not None else (getattr(error, "body", None) or {})
        _err_message = (
            _body.get("error", {}).get("message", "")
            if isinstance(_body, dict)
            else ""
        )
        if _err_message.strip().lower() in ("error", ""):
            return _make(ProviderErrorReason.SERVER_ERROR, retryable=True,
                         msg="Generic 400 (probable transient Anthropic OAuth error)")

    # ── 13. Other 4xx client errors ─────────────────────────────────────
    # Also catch error-code phrases from error messages (e.g. "error code: 401")
    _has_client_error_phrase = any(phrase in error_msg for phrase in (
        "error code: 401", "error code: 403",
        "error code: 404", "error code: 422",
        "unauthorized", "forbidden", "not found",
    ))
    if isinstance(status_code, int) and 400 <= status_code < 500:
        # 413, 429, 529 already handled above; remaining 4xx are non-retryable
        return _make(ProviderErrorReason.AUTH_PERMANENT, retryable=False)
    if _has_client_error_phrase:
        return _make(ProviderErrorReason.AUTH_PERMANENT, retryable=False)

    # ── 14. Stream-drop / connection errors (no status code) ────────────
    if not status_code and any(p in error_msg for p in _STREAM_DROP_PHRASES):
        return _make(ProviderErrorReason.STREAM_DROP, retryable=True)

    # ── 15. Server errors (5xx) ─────────────────────────────────────────
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return _make(ProviderErrorReason.SERVER_ERROR, retryable=True)

    # ── 16. Server disconnect (not large session — still retryable) ─────
    if not status_code:
        _is_server_disconnect = (
            "server disconnected" in error_msg
            or "peer closed connection" in error_msg
            or error_type in _SERVER_DISCONNECT_ERROR_TYPES
        )
        if _is_server_disconnect:
            return _make(ProviderErrorReason.STREAM_DROP, retryable=True)

    # ── Fallback ────────────────────────────────────────────────────────
    return _make(ProviderErrorReason.UNKNOWN, retryable=True)


# ── Helpers ──────────────────────────────────────────────────────────────

def is_retryable(reason: ProviderErrorReason) -> bool:
    """Return whether the given error reason is retryable."""
    _NON_RETRYABLE = frozenset({
        ProviderErrorReason.AUTH_PERMANENT,
        ProviderErrorReason.BILLING,
        ProviderErrorReason.MODEL_NOT_FOUND,
        ProviderErrorReason.FORMAT_ERROR,
    })
    return reason not in _NON_RETRYABLE


def suggested_action(reason: ProviderErrorReason) -> str:
    """Return a suggested recovery action for the given error reason.

    Returns one of: ``'retry'``, ``'retry_with_backoff'``, ``'compress'``,
    ``'fallback'``, ``'refresh_credentials'``, ``'abort'``.
    """
    _ACTION_MAP = {
        ProviderErrorReason.AUTH: "refresh_credentials",
        ProviderErrorReason.AUTH_PERMANENT: "abort",
        ProviderErrorReason.RATE_LIMIT: "retry_with_backoff",
        ProviderErrorReason.OVERLOADED: "retry_with_backoff",
        ProviderErrorReason.BILLING: "abort",
        ProviderErrorReason.MODEL_NOT_FOUND: "fallback",
        ProviderErrorReason.CONTEXT_OVERFLOW: "compress",
        ProviderErrorReason.PAYLOAD_TOO_LARGE: "compress",
        ProviderErrorReason.FORMAT_ERROR: "abort",
        ProviderErrorReason.TIMEOUT: "retry",
        ProviderErrorReason.SERVER_ERROR: "retry",
        ProviderErrorReason.STREAM_DROP: "retry",
        ProviderErrorReason.UNKNOWN: "retry",
    }
    return _ACTION_MAP.get(reason, "retry")


# ── Rate-limit header tracking ────────────────────────────────────────────

@dataclass
class RateLimitState:
    """Tracks rate limit headers from provider responses."""

    remaining: Optional[int] = None   # X-RateLimit-Remaining
    limit: Optional[int] = None       # X-RateLimit-Limit
    reset_at: Optional[float] = None  # X-RateLimit-Reset (epoch time)
    updated_at: float = 0.0           # when last updated

    def should_throttle(self) -> bool:
        """Return True if remaining quota is below 10% of limit."""
        if self.remaining is not None and self.limit and self.limit > 0:
            return self.remaining / self.limit < 0.10
        return False

    def suggested_delay(self) -> float:
        """Return seconds to wait based on reset time, or 0."""
        if self.reset_at and self.reset_at > time.time():
            return min(self.reset_at - time.time(), 30.0)  # cap at 30s
        if self.should_throttle():
            return 2.0  # conservative 2s delay when low on quota
        return 0.0


def parse_rate_limit_headers(response) -> Optional[RateLimitState]:
    """Extract rate limit info from API response headers.

    Works with both OpenAI SDK responses (response.headers) and
    httpx responses. Handles common header names:
    - x-ratelimit-remaining / X-RateLimit-Remaining
    - x-ratelimit-limit / X-RateLimit-Limit
    - x-ratelimit-reset / X-RateLimit-Reset
    """
    # Try to get headers from response or response's http_response
    headers = None
    for attr in ("headers", "_headers"):
        h = getattr(response, attr, None)
        if h and hasattr(h, "get"):
            headers = h
            break
    if not headers:
        http_resp = getattr(response, "http_response", None) or getattr(
            response, "response", None
        )
        if http_resp:
            headers = getattr(http_resp, "headers", None)
    if not headers:
        return None

    remaining = _parse_int_header(
        headers, "x-ratelimit-remaining", "X-RateLimit-Remaining", "ratelimit-remaining"
    )
    limit = _parse_int_header(
        headers, "x-ratelimit-limit", "X-RateLimit-Limit", "ratelimit-limit"
    )
    reset_raw = _parse_header(
        headers, "x-ratelimit-reset", "X-RateLimit-Reset", "ratelimit-reset"
    )
    reset_at = None
    if reset_raw:
        try:
            val = float(reset_raw)
            # If val > 1e9 treat as epoch timestamp; otherwise relative seconds
            reset_at = val if val > 1e9 else time.time() + val
        except (TypeError, ValueError):
            pass

    if remaining is None and limit is None and reset_at is None:
        return None
    return RateLimitState(
        remaining=remaining, limit=limit, reset_at=reset_at, updated_at=time.time()
    )


def _parse_header(headers, *names) -> Optional[str]:
    """Return the first matching header value as a string, or None."""
    for name in names:
        val = headers.get(name)
        if val is not None:
            return str(val)
    return None


def _parse_int_header(headers, *names) -> Optional[int]:
    """Return the first matching header value as an int, or None."""
    raw = _parse_header(headers, *names)
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return None
