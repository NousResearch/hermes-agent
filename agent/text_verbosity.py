"""OpenAI Responses API text verbosity helpers."""

from __future__ import annotations

from typing import Any, Mapping

from utils import base_url_hostname

VALID_TEXT_VERBOSITIES = ("low", "medium", "high")


def parse_text_verbosity(raw: Any) -> str | None:
    """Parse agent.text_verbosity into a Responses API verbosity value."""
    value = str(raw or "").strip().lower()
    if not value:
        return None
    if value in VALID_TEXT_VERBOSITIES:
        return value
    return None


def supports_openai_text_verbosity(
    *, provider: str | None, api_mode: str | None, base_url: str | None
) -> bool:
    """Return True when the runtime should receive Responses text.verbosity."""
    if api_mode != "codex_responses":
        return False
    provider_name = str(provider or "").strip().lower()
    if provider_name == "xai":
        return False
    hostname = base_url_hostname(str(base_url or ""))
    if hostname == "api.openai.com":
        return True
    return provider_name in {"openai", "openai-codex"} and hostname not in {"api.x.ai"}


def merge_text_verbosity_override(
    overrides: Mapping[str, Any] | None,
    text_verbosity: Any,
    *,
    provider: str | None,
    api_mode: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    """Merge text.verbosity into request overrides only for OpenAI Responses runtimes."""
    merged = dict(overrides or {})
    verbosity = parse_text_verbosity(text_verbosity)
    if not verbosity:
        return merged
    if not supports_openai_text_verbosity(
        provider=provider, api_mode=api_mode, base_url=base_url
    ):
        return merged

    text = merged.get("text")
    text_obj = dict(text) if isinstance(text, Mapping) else {}
    text_obj["verbosity"] = verbosity
    merged["text"] = text_obj
    return merged
