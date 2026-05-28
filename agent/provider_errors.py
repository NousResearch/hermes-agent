"""Typed provider/fallback failures and redacted gateway diagnostics."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


PROVIDER_FAILURE_ERROR_CLASS = "provider_config_or_fallback_failure"
CONTROLLED_AGENT_TURN_UNAVAILABLE_MESSAGE = (
    "Hermes received the message, but the agent turn path is unavailable. "
    "Gateway is online; agent runner/provider config failed. "
    "Check redacted gateway logs. "
    f"Error class: {PROVIDER_FAILURE_ERROR_CLASS}."
)


class HermesProviderConfigError(RuntimeError):
    """Provider config is missing, malformed, or cannot create a client."""

    error_class = PROVIDER_FAILURE_ERROR_CLASS


class HermesFallbackConfigError(RuntimeError):
    """Fallback config is missing, malformed, or unusable for failover."""

    error_class = PROVIDER_FAILURE_ERROR_CLASS


class HermesSessionStateError(RuntimeError):
    """Gateway/session state cannot safely be used for an agent turn."""

    error_class = PROVIDER_FAILURE_ERROR_CLASS


class HermesLocalProviderClientError(RuntimeError):
    """Provider/client failed locally before a normal HTTP response existed."""

    error_class = PROVIDER_FAILURE_ERROR_CLASS


_SENSITIVE_RE = re.compile(
    r"(?i)(authorization|bearer|cookie|token|api[_ -]?key|secret|sk-[a-z0-9_-]+|sk-proj-[a-z0-9_-]+)"
)


def redacted_url_class(base_url: Any) -> str:
    """Return only the endpoint host/class, never query strings or secrets."""
    raw = str(base_url or "").strip()
    if not raw:
        return "missing"
    try:
        parsed = urlparse(raw)
        host = parsed.netloc or parsed.path.split("/", 1)[0]
    except Exception:
        host = ""
    host = host.lower()
    if not host:
        return "invalid"
    if host in {"api.openai.com"}:
        return "api.openai.com"
    if host == "chatgpt.com":
        return "chatgpt.com"
    if host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith(".local"):
        return "local"
    return host


def sanitize_provider_error_text(error: BaseException | str) -> str:
    """Map unsafe raw exception text to a stable user/log diagnostic."""
    text = str(error or "")
    if "NoneType" in text and "not iterable" in text:
        return "local provider/client config failure before HTTP response"
    if _SENSITIVE_RE.search(text):
        return "provider/client failure with sensitive details redacted"
    if not text.strip():
        return "provider/client failure before HTTP response"
    return text[:240]


def is_provider_failure_exception(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            HermesProviderConfigError,
            HermesFallbackConfigError,
            HermesSessionStateError,
            HermesLocalProviderClientError,
        ),
    ):
        return True
    return isinstance(error, TypeError) and "NoneType" in str(error) and "not iterable" in str(error)


def controlled_agent_turn_unavailable_message(_error: BaseException | None = None) -> str:
    return CONTROLLED_AGENT_TURN_UNAVAILABLE_MESSAGE


def provider_failure_result(
    error: BaseException,
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    fallback_status: str,
) -> dict[str, Any]:
    """Build a run_conversation-style failed result without raw exception text."""
    return {
        "final_response": None,
        "completed": False,
        "failed": True,
        "error": sanitize_provider_error_text(error),
        "error_class": PROVIDER_FAILURE_ERROR_CLASS,
        "provider": str(provider or "").strip() or "unknown",
        "model": str(model or "").strip() or "unknown",
        "base_url_class": redacted_url_class(base_url),
        "fallback_status": fallback_status,
        "api_calls": 0,
        "messages": [],
    }
