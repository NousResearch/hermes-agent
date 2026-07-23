"""Parse optional client correlation headers for gateway observability.

Clients may send standard ``X-Request-Id`` and/or ``X-Stream-Token`` headers.
The gateway treats these as opaque strings — no client-specific naming or
semantics are assumed.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


# Maximum length for request_id / stream_token before truncation.
_MAX_CORR_VALUE_LEN = 128


def _sanitize_corr(value: Any) -> str:
    """Strip newlines, collapse whitespace, truncate to _MAX_CORR_VALUE_LEN."""
    if not value:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:_MAX_CORR_VALUE_LEN]


def parse_correlation_headers(headers: Mapping[str, Any]) -> Dict[str, str]:
    """Extract sanitized correlation ids from inbound HTTP headers."""
    corr: Dict[str, str] = {}
    request_id = _sanitize_corr(
        headers.get("X-Request-Id") or headers.get("X-Request-ID") or ""
    )
    if request_id:
        corr["request_id"] = request_id

    stream_token = _sanitize_corr(headers.get("X-Stream-Token", ""))
    if stream_token:
        corr["stream_token"] = stream_token
    return corr


def format_correlation_log_suffix(
    corr: Optional[Mapping[str, str]] = None,
    *,
    session_id: Optional[str] = None,
) -> str:
    """Build a space-separated suffix for structured log lines."""
    parts: list[str] = []
    if corr:
        rid = corr.get("request_id")
        if rid:
            parts.append(f"request_id={rid}")
        st = corr.get("stream_token")
        if st:
            parts.append(f"stream_token={st}")
    if session_id:
        parts.append(f"session={session_id}")
    return f" {' '.join(parts)}" if parts else ""
