"""OpenAI Responses API text verbosity helpers."""

from __future__ import annotations

from typing import Any


VALID_TEXT_VERBOSITIES = {"low", "medium", "high"}


def parse_text_verbosity(raw: Any) -> str | None:
    """Return a normalized Responses API text verbosity value, or None."""
    value = str(raw or "").strip().lower()
    if not value:
        return None
    if value in VALID_TEXT_VERBOSITIES:
        return value
    return None


def supports_openai_text_verbosity(
    model: Any,
    *,
    base_url_hostname: str = "",
    is_codex_backend: bool = False,
) -> bool:
    """Return whether the resolved target supports GPT-5 text verbosity."""
    model_id = str(model or "").strip().lower().rsplit("/", 1)[-1]
    is_gpt5 = (
        model_id == "gpt-5"
        or model_id.startswith("gpt-5.")
        or model_id.startswith("gpt-5-")
    )
    if not is_gpt5:
        return False
    return is_codex_backend or base_url_hostname.strip().lower() == "api.openai.com"
