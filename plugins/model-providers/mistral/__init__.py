"""Mistral AI provider profile.

Mistral's API is standard OpenAI-compatible with a ``reasoning_effort``
top-level kwarg for thinking-enabled models (``mistral-small-2603+``).
The ``reasoning_effort`` parameter is passed through transparently —
this profile adds no special handling beyond registering the provider.

Agentic models with tool-calling support:

- ``mistral-large-latest`` — flagship, strong reasoning + tool use (128K ctx)
- ``mistral-small-latest`` — fast, cost-effective, vision-capable (128K ctx)
- ``ministral-8b-latest`` — edge-optimised, small (128K ctx)
- ``ministral-14b-latest`` — edge-optimised, balanced (128K ctx)
- ``codestral-latest`` — code generation (256K ctx)
- ``pixtral-12b-latest`` — multimodal: text + image input (128K ctx)
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class MistralProfile(ProviderProfile):
    """Mistral AI — standard OpenAI-compatible, passes reasoning_effort through."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Mistral supports reasoning_effort as a top-level kwarg for thinking
        # models. Pass through the Hermes reasoning config transparently.
        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort:
                top_level["reasoning_effort"] = effort

        return extra_body, top_level


mistral = MistralProfile(
    name="mistral",
    aliases=("mistral-ai", "mistralai", "mixtral"),
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
