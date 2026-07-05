"""Mistral AI provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

mistral = ProviderProfile(
    name="mistral",
    aliases=("mistralai", "mistral-ai", "la-plateforme"),
    env_vars=("MISTRAL_API_KEY",),
    display_name="Mistral AI",
    description="Mistral AI — direct OpenAI-compatible API",
    signup_url="https://console.mistral.ai/api-keys/",
    base_url="https://api.mistral.ai/v1",
    default_aux_model="mistral-small-latest",
    fallback_models=(
        "mistral-medium-latest",
        "mistral-small-latest",
        "mistral-large-latest",
    ),
)

register_provider(mistral)
