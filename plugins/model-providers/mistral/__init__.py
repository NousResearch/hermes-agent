"""Mistral AI model-provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


mistral = ProviderProfile(
    name="mistral",
    aliases=("mistral-ai", "mistralai"),
    display_name="Mistral AI",
    description="Mistral AI direct API (Mistral, Codestral, and Devstral models)",
    signup_url="https://console.mistral.ai/",
    env_vars=("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    base_url="https://api.mistral.ai/v1",
    auth_type="api_key",
    api_mode="chat_completions",
    supports_vision=True,
    default_aux_model="mistral-small-latest",
    # Keep this as a deliberately small fallback set for offline picker/setup
    # paths. Live catalog/model metadata discovery supplies fresher entries.
    fallback_models=(
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest",
        "codestral-latest",
        "devstral-medium-latest",
    ),
)

register_provider(mistral)
