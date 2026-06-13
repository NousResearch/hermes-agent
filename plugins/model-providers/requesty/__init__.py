"""Requesty AI gateway provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class RequestyProfile(ProviderProfile):
    """Requesty — reasoning config passthrough."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = True,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if supports_reasoning and reasoning_config is not None:
            extra_body["reasoning"] = dict(reasoning_config)
        elif supports_reasoning:
            extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


requesty = RequestyProfile(
    name="requesty",
    aliases=("rq",),
    env_vars=("REQUESTY_API_KEY",),
    display_name="Requesty",
    description="Requesty — unified AI gateway for 300+ models",
    signup_url="https://app.requesty.ai/api-keys",
    base_url="https://router.requesty.ai/v1",
    models_url="https://router.requesty.ai/v1/models",
    default_aux_model="anthropic/claude-sonnet-4-6",
    fallback_models=(
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-7",
        "openai/gpt-5.5",
        "vertex/gemini-3.1-pro-preview:flex",
    ),
)

register_provider(requesty)
