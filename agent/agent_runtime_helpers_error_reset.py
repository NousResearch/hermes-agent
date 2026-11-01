"""Reset-window sources for :func:`agent.agent_runtime_helpers.extract_api_error_context`.

Split out of ``agent_runtime_helpers`` — a facade module already past its code-health line
target — so the provider-error reset ladder can gain a rung without growing it. The ladder is
ordered by trust: a body timestamp or ``Retry-After`` header first, then a vendor per-bucket
header, then the adapter's own parsed window, and finally the free-text grammar in
``agent.retry_utils``. Every rung records only the first window it finds.
"""

from __future__ import annotations
import re
import time
from typing import Any, Optional
from agent.credential_pool import _parse_absolute_timestamp
from agent.retry_utils import parse_retry_after_seconds


def _set_reset_from_retry_after(context: dict[str, Any], retry_after: Any) -> None:
    if "reset_at" in context:
        return
    seconds = parse_retry_after_seconds(retry_after)
    if seconds is not None:
        context["reset_at"] = time.time() + seconds


# OpenAI-style relative windows: "6m0s", "1.5s", "20ms", "1h2m3s" (also a bare number of seconds).
_DURATION_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
# Lowest-priority reset sources, after Retry-After and x-ratelimit-reset: OpenAI's per-bucket
# durations and Anthropic's per-bucket ISO-8601 timestamps. Plain OpenAI/Anthropic 429s often
# carry only these, and without them the retry status never names the reset window.
_VENDOR_RESET_HEADERS = (
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
    "anthropic-ratelimit-requests-reset", "anthropic-ratelimit-tokens-reset",
)


def _duration_string_seconds(text: str) -> Optional[float]:
    raw = text.strip().lower()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    parts = _DURATION_COMPONENT_RE.findall(raw)
    if not parts or "".join(n + u for n, u in parts) != raw:
        return None
    return sum(float(n) * _DURATION_UNIT_SECONDS[u] for n, u in parts)


def _set_reset_from_vendor_headers(context: dict[str, Any], headers: Any) -> None:
    for name in _VENDOR_RESET_HEADERS:
        value = headers.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        seconds = _duration_string_seconds(value)
        if seconds is None:
            absolute = _parse_absolute_timestamp(value)
            seconds = None if absolute is None else absolute - time.time()
        if seconds is not None and seconds > 0:
            context["reset_at"] = time.time() + seconds
            return


def _set_reset_from_error_retry_after(context: dict[str, Any], error: Exception) -> None:
    """Size the cooldown from the window the provider adapter already parsed onto the error.

    Gemini answers a free-tier 429 with ``Please retry in 24.9s`` prose — or a structured
    ``google.rpc.RetryInfo`` duration — and no ``Retry-After`` header. The adapter turns both into
    ``GeminiAPIError.retry_after``; the pool reads it here so a key throttled for 25 seconds is not
    benched for the one-hour 429 default. Read off the attribute, never injected into ``body``:
    a body-level ``retry_after`` also feeds the in-turn retry loop, which must keep rotating to
    another credential instead of sleeping through the window on the key that just failed.
    """
    _set_reset_from_retry_after(context, getattr(error, "retry_after", None))
