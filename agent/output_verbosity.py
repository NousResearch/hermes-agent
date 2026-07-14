"""OpenAI Responses API output verbosity helpers."""

from __future__ import annotations

from typing import Any


VALID_OUTPUT_VERBOSITIES = frozenset({"low", "medium", "high"})


def parse_output_verbosity(raw: Any) -> str | None:
    """Return a normalized output verbosity value, or None for the provider default."""
    value = str(raw or "").strip().lower()
    return value if value in VALID_OUTPUT_VERBOSITIES else None


def supports_openai_output_verbosity(
    model: Any,
    *,
    base_url_hostname: str = "",
    is_codex_backend: bool = False,
) -> bool:
    """Return whether the resolved GPT-5 Responses target supports this field."""
    model_id = str(model or "").strip().lower().rsplit("/", 1)[-1]
    is_gpt5 = (
        model_id == "gpt-5"
        or model_id.startswith("gpt-5.")
        or model_id.startswith("gpt-5-")
    )
    if not is_gpt5:
        return False
    return is_codex_backend or base_url_hostname.strip().lower() == "api.openai.com"
