"""OpenAI Responses API text verbosity helpers."""

from __future__ import annotations

from typing import Any


VALID_TEXT_VERBOSITIES = frozenset({"low", "medium", "high"})


def parse_text_verbosity(raw: Any) -> str | None:
    """Return a normalized text verbosity value, or None for the provider default."""
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    return value if value in VALID_TEXT_VERBOSITIES else None


def supports_openai_text_verbosity(
    model: Any,
    *,
    base_url_hostname: str = "",
    is_canonical_codex_route: bool = False,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
) -> bool:
    """Return whether the resolved GPT-5 Responses target supports this field."""
    if is_xai_responses or is_github_responses:
        return False
    model_id = str(model or "").strip().lower().rsplit("/", 1)[-1]
    is_gpt5 = (
        model_id == "gpt-5"
        or model_id.startswith("gpt-5.")
        or model_id.startswith("gpt-5-")
    )
    if not is_gpt5:
        return False
    return (
        is_canonical_codex_route
        or base_url_hostname.strip().lower() == "api.openai.com"
    )
