"""EU Router provider profile.

EU Router (eurouter.ai) is an OpenAI/OpenRouter-API-compatible aggregator
whose distinguishing feature is EU data-residency guarantees: all inference
requests are processed entirely within the EU, and the request-time
``provider`` object accepts additional compliance fields
(``data_residency``, ``eu_owned``, ``max_retention_days``) on top of the
OpenRouter-shaped ``order``/``only``/``ignore``/``sort``/``data_collection``
fields Hermes already threads through ``provider_routing`` config.

See https://www.eurouter.ai/docs/concepts/routing for the ``provider``
object schema. The public model catalog at ``/api/v1/models`` requires no
authentication (verified live), matching OpenRouter's public-catalog shape.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class EuRouterProfile(ProviderProfile):
    """EU Router aggregator — provider routing + EU data-residency passthrough."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        prefs = self.filter_routing_preferences(context.get("provider_preferences"))
        if prefs:
            body["provider"] = prefs
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """EU Router passes the full reasoning_config dict as extra_body.reasoning,
        same shape as OpenRouter/Nous.
        """
        extra_body: dict[str, Any] = {}
        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


eurouter = EuRouterProfile(
    name="eurouter",
    aliases=("eu-router", "eur"),
    env_vars=("EUROUTER_API_KEY",),
    display_name="EU Router",
    description="EUrouter — EU-hosted, GDPR-compliant model routing",
    signup_url="https://www.eurouter.ai/",
    base_url="https://api.eurouter.ai/api/v1",
    models_url="https://api.eurouter.ai/api/v1/models",
    fallback_models=(
        "mistral-large-3",
        "deepseek-v3.2",
        "glm-5.2",
        "kimi-k2.6",
        "qwen3-235b-a22b-instruct",
    ),
    # Extends the shared OpenRouter-shaped baseline with EU Router's
    # compliance-routing fields. See
    # https://www.eurouter.ai/docs/concepts/routing
    routing_preference_keys=ProviderProfile.routing_preference_keys
    + ("data_residency", "eu_owned", "max_retention_days"),
)

register_provider(eurouter)
