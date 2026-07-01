"""ZAI / GLM provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

# Universally-free models available to all Z.AI users (both international
# and China platforms), but not listed by the provider's /v1/models
# endpoint.  All have been verified with real API calls.
ZAI_FREE_MODELS = (
    "glm-4v-flash",
    "glm-4.6v-flash",
    "glm-4.1v-thinking-flash",
    "glm-4.5-flash",
    "glm-4-flash-250414",
)


def _fetch_models_with_free(self, *, api_key=None, timeout=8.0):
    """Fetch live model list and append universally-free models."""
    live = ProviderProfile.fetch_models(self, api_key=api_key, timeout=timeout)
    models = list(live) if live else []
    seen = {m.lower() for m in models}
    for m in ZAI_FREE_MODELS:
        if m.lower() not in seen:
            models.append(m)
            seen.add(m.lower())
    return models


zai = ProviderProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

# Replace fetch_models with our wrapper that appends free models.
zai.fetch_models = _fetch_models_with_free.__get__(zai)

register_provider(zai)
