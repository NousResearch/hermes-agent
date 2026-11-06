"""Reka provider profile.

Reka serves open-weight models behind an OpenAI-compatible API at
https://api.reka.ai/v1. This profile adds a curated model list with
per-model capabilities (context window, tool support, image input) and
maps Hermes' reasoning-effort setting onto Reka's top-level
``reasoning_effort`` field; with no effort configured nothing is sent.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

BASE_URL = "https://api.reka.ai/v1"

# Hermes' reasoning-effort levels, weakest first.
_EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# Recognized request values; live acceptance and behavior remain model-specific.
_REKA_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Models offered in the picker: id -> (context window in tokens, accepts image input).
# Capabilities confirmed via authenticated /v1/models on 2026-10-22. Catalog presence
# does not prove availability; a model is added only after it passes a live chat check
# through Hermes. Order sets the preferred picker order and fallback default.
_MODELS: dict[str, tuple[int, bool]] = {
    "qwen3.8-27b": (262_144, True),
    "glm5.3-flash": (1_048_576, True),
    "deepseek41-flash-20261017": (1_048_576, True),
    "deepseek4-flash": (262_144, False),
    "deepseek-v4-pro": (1_048_576, False),
    "qwen3.6-27b": (262_144, False),
}


def _reasoning_effort_for(reasoning_config: Any) -> str | None:
    """The ``reasoning_effort`` value to send, or None to leave the field out.

    Unset sends nothing. Turning reasoning off sends ``none``. A Hermes level Reka
    does not have (``ultra``) maps to the strongest one it does (``max``). A name
    that is not a Hermes level is never guessed onto the wire.
    """
    if not isinstance(reasoning_config, dict):
        return None
    requested = str(reasoning_config.get("effort") or "").strip().lower()
    if reasoning_config.get("enabled") is False or requested == "none":
        return "none"
    if requested in _REKA_EFFORTS:
        return requested
    if requested in _EFFORT_LADDER:
        rank = _EFFORT_LADDER.index(requested)
        return [level for level in _REKA_EFFORTS if _EFFORT_LADDER.index(level) < rank][-1]
    return None


class RekaProfile(ProviderProfile):
    """Reka's OpenAI-compatible endpoint with top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = _reasoning_effort_for(reasoning_config)
        return {}, ({"reasoning_effort": effort} if effort else {})

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if models is None:
            return None
        # Preserve the curated picker order and return each known, available model
        # exactly once; never invent an unavailable model here.
        available = {model for model in models if model in _MODELS}
        return [model for model in _MODELS if model in available]


reka = RekaProfile(
    name="reka",
    aliases=("reka-ai",),
    display_name="Reka",
    description="Reka - OpenAI-compatible API for open-weight models",
    signup_url="https://developer.reka.ai",
    env_vars=("REKA_API_KEY", "REKA_BASE_URL"),
    base_url=BASE_URL,
    auth_type="api_key",
    default_aux_model="qwen3.8-27b",
    fallback_models=tuple(_MODELS),
    supports_vision_tool_messages=False,
    model_capabilities={
        model_id: {
            "context_window": context_window,
            "supports_tools": True,
            "supports_reasoning": True,
    "supports_vision": vision,
        }
        for model_id, (context_window, vision) in _MODELS.items()
    },
)

register_provider(reka)
