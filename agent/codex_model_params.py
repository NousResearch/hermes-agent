from __future__ import annotations

from typing import Any


_OPENAI_CODEX_RESPONSES_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "gpt-5.4": {
        "reasoning_summary": "auto",
        "text": {"verbosity": "low"},
    },
    "gpt-5.3-codex": {
        "reasoning_summary": "auto",
        "text": {"verbosity": "low"},
    },
}


def _normalize_model_slug(model: str | None) -> str:
    value = str(model or "").strip().lower()
    if "/" in value:
        value = value.split("/", 1)[1]
    return value.split(":", 1)[0]


def get_openai_codex_responses_model_defaults(
    provider: str | None,
    model: str | None,
) -> dict[str, Any] | None:
    if str(provider or "").strip().lower() != "openai-codex":
        return None
    return _OPENAI_CODEX_RESPONSES_MODEL_DEFAULTS.get(_normalize_model_slug(model))
