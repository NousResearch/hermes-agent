"""ZAI / GLM provider profile."""

import logging
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

# Models that work via /chat/completions but are absent from /models catalog
_EXTRA_MODELS = {
    "glm-5.2",
    "glm-5",
    "glm-4-9b",
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.7",
    "glm-4.5-flash",
}


class ZAIProfile(ProviderProfile):
    """Z.AI / GLM provider with extended model catalog."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch live catalog and merge with known working models."""
        live_models = super().fetch_models(api_key=api_key, timeout=timeout)
        if live_models is None:
            # If live fetch fails, return our curated fallback list
            return list(_EXTRA_MODELS)
        # Merge: live models + extra models (deduped)
        merged = set(live_models) | _EXTRA_MODELS
        return sorted(merged)


zai = ZAIProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5-flash",
    ),
    base_url="https://api.z.ai/api/coding/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
