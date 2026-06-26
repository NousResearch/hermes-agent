"""Sakana AI Fugu provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

sakana = ProviderProfile(
    name="sakana",
    aliases=("fugu",),
    display_name="Sakana AI",
    description="Sakana Fugu — multi-agent reasoning system",
    signup_url="https://console.sakana.ai",
    env_vars=("SAKANA_API_KEY",),
    base_url="https://api.sakana.ai/v1",
    fallback_models=(
        "fugu",
        "fugu-ultra",
    ),
)

register_provider(sakana)
