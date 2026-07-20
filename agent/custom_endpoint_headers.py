"""Compatibility headers for generic OpenAI-compatible custom endpoints."""

from __future__ import annotations


_GENERIC_CUSTOM_ENDPOINT_USER_AGENT = "python-requests/2.32.3"


def generic_custom_endpoint_headers(base_url: str | None) -> dict[str, str]:
    """Return a neutral header set for generic custom endpoints.

    Some OpenAI-compatible relays block the OpenAI Python SDK's default
    User-Agent while allowing equivalent requests from simpler clients.
    Preserve explicit provider-specific headers elsewhere and only provide a
    fallback User-Agent for generic custom endpoints.
    """

    normalized = str(base_url or "").strip().lower()
    if not normalized:
        return {}
    if any(marker in normalized for marker in ("openrouter", "api.githubcopilot.com", "api.kimi.com")):
        return {}
    return {"User-Agent": _GENERIC_CUSTOM_ENDPOINT_USER_AGENT}
