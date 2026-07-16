"""Mistral AI provider profile.

Mistral's API is standard OpenAI-compatible.  Thinking-enabled models
(``mistral-small-2603+``, ``mistral-medium-2604+``, and the
``mistral-medium-3.5`` family) accept ``reasoning_effort`` as a top-level
kwarg.  Other models (codestral, mistral-large, pixtral, ministral) do not
support reasoning — ``build_api_kwargs_extras`` skips the parameter for
those models based on the model name.

Agentic models with tool-calling support:

- ``mistral-large-latest`` — flagship, strong reasoning + tool use (128K ctx)
- ``mistral-medium-latest`` — balanced, vision + reasoning-capable (256K ctx)
- ``mistral-small-latest`` — fast, cost-effective, vision + reasoning (128K ctx)
- ``codestral-latest`` — code generation (256K ctx)
- ``pixtral-12b-latest`` — multimodal vision specialist (128K ctx)
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


# Model families known to support reasoning_effort on Mistral's API.
# Based on API capabilities data as of July 2026.
_REASONING_PREFIXES = frozenset({
    "mistral-small-2603",
    "mistral-small-latest",
    "mistral-medium-2604",
    "mistral-medium-3",
    "mistral-medium-3.5",
    "mistral-medium-latest",
    "mistral-medium",  # resolves to latest reasoning-capable version
})


def _model_supports_reasoning(model: str | None) -> bool:
    """Check if ``model`` supports ``reasoning_effort``.

    Matches by prefix so that versioned model names
    (``mistral-small-2603-xyz``) are caught without an exhaustive list.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    return any(m.startswith(prefix) for prefix in _REASONING_PREFIXES)


class MistralProfile(ProviderProfile):
    """Mistral AI — standard OpenAI-compatible.

    ``reasoning_effort`` is passed through only for thinking-enabled models
    (matched by ``_model_supports_reasoning``).  Non-thinking models skip
    the parameter entirely, avoiding HTTP 400 from Mistral's API.
    """

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_reasoning(model):
            return extra_body, top_level

        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort:
                top_level["reasoning_effort"] = effort

        return extra_body, top_level


mistral = MistralProfile(
    name="mistral",
    aliases=("mistral-ai", "mistralai"),
    env_vars=("MISTRAL_API_KEY",),
    display_name="Mistral AI",
    description="Mistral AI — native Mistral API",
    signup_url="https://console.mistral.ai/",
    fallback_models=(
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "pixtral-12b-latest",
    ),
    base_url="https://api.mistral.ai/v1",
    supports_vision=True,
    default_aux_model="mistral-small-latest",
)

register_provider(mistral)
