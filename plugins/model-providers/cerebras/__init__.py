"""Cerebras provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

cerebras = ProviderProfile(
    name="cerebras",
    aliases=("cerebras-ai",),
    display_name="Cerebras",
    description="Cerebras — fastest inference for open models",
    signup_url="https://cloud.cerebras.ai/api-keys",
    env_vars=("CEREBRAS_API_KEY",),
    base_url="https://api.cerebras.ai/v1",
    auth_type="api_key",
    default_aux_model="llama3.3-70b",
    fallback_models=(
        "llama3.3-70b",
        "llama3.1-70b",
        "llama3.1-8b",
    ),
)

register_provider(cerebras)