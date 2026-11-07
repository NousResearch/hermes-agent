"""Rate-limit headers on a provider refusal: the reset time it names and the headers worth keeping.

Split out of ``agent/agent_runtime_helpers.py`` (code-health size ratchet); ``extract_api_error_context``
there reads these.
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


# Rate-limit response headers worth keeping on a refusal. Prefix match: the unified
# subscription headers carry one family per window (anthropic-ratelimit-unified-5h-*,
# -7d-*, ...), and new windows must not need a code change to be recorded.
_RATELIMIT_HEADER_PREFIXES = ("anthropic-ratelimit-", "x-ratelimit-")
_RATELIMIT_HEADER_NAMES = ("retry-after",)
_RATELIMIT_VALUE_MAX = 120


def ratelimit_headers(headers: Any) -> dict[str, str]:
    """The rate-limit headers of a provider response, lower-cased and sorted, values capped."""
    out: dict[str, str] = {}
    items = getattr(headers, "items", None)
    if not callable(items):
        return out
    for name, value in items():
        key = str(name).strip().lower()
        if key in _RATELIMIT_HEADER_NAMES or key.startswith(_RATELIMIT_HEADER_PREFIXES):
            out[key] = str(value).strip()[:_RATELIMIT_VALUE_MAX]
    return dict(sorted(out.items()))
