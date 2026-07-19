"""Mistral AI provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

mistral = ProviderProfile(
    name="mistral",
    aliases=("mistral-ai",),
    display_name="Mistral AI",
    description="Mistral AI (Large, Small, Codestral, Pixtral; direct API)",
    env_vars=("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    base_url="https://api.mistral.ai/v1",
    signup_url="https://console.mistral.ai",
    supports_health_check=True,
    supports_vision=True,  # Pixtral models are vision-capable
    supports_vision_tool_messages=True,
    fallback_models=(
        "mistral-large-latest",
        "mistral-small-latest",
        "codestral-latest",
        "pixtral-large-latest",
        "open-mistral-nemo",
        "ministral-8b-latest",
    ),
)

register_provider(mistral)
