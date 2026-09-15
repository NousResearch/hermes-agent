"""OpenAI Responses API text verbosity helpers."""

from __future__ import annotations

from typing import Any


VALID_TEXT_VERBOSITIES = frozenset({"low", "medium", "high"})


def parse_text_verbosity(raw: Any) -> str | None:
    """Return a normalized verbosity, or None to keep the provider default."""
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    return value if value in VALID_TEXT_VERBOSITIES else None


def supports_openai_text_verbosity(
    model: Any,
    *,
    route_supported: bool,
) -> bool:
    """Return whether the effective model and resolved route accept the field."""
    if not route_supported:
        return False
    model_id = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return (
        model_id == "gpt-5"
        or model_id.startswith("gpt-5.")
        or model_id.startswith("gpt-5-")
    )


def finalize_text_verbosity_request(agent: Any, api_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Reapply the route/model boundary after mutable request middleware."""
    api_mode = str(getattr(agent, "api_mode", "") or "")
    if api_mode == "codex_responses":
        from agent.codex_responses_adapter import (
            supports_openai_text_verbosity_route,
        )
        from agent.transports.codex import _apply_text_verbosity

        _apply_text_verbosity(
            api_kwargs,
            configured=getattr(agent, "text_verbosity", None),
            route_supported=supports_openai_text_verbosity_route(agent),
        )
    elif api_mode == "chat_completions":
        from agent.transports.chat_completions import _strip_text_verbosity

        _strip_text_verbosity(api_kwargs)
    return api_kwargs
