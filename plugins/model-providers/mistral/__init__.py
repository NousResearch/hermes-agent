"""Mistral AI provider profile — OpenAI-compatible chat/completions at api.mistral.ai."""

from providers import register_provider
from providers.base import ProviderProfile

mistral = ProviderProfile(
    name="mistral",
    aliases=("mistralai", "mistral-ai"),
    display_name="Mistral AI",
    description="Mistral AI direct API (Mistral Large/Medium/Small, Codestral, Voxtral)",
    signup_url="https://console.mistral.ai/",
    env_vars=("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    base_url="https://api.mistral.ai/v1",
    auth_type="api_key",
    default_aux_model="mistral-small-latest",
    # Live catalog ({base_url}/models + models.dev, which is _MODELS_DEV_PREFERRED for
    # "mistral") also serves Z.ai and DeepSeek open models; these are the offline floor.
    fallback_models=(
        "mistral-large-latest",
        "zai-glm-5-3",
        "deepseek-v4-1-flash-1017",
        "mistral-medium-latest",
        "mistral-small-latest",
    ),
)

register_provider(mistral)
