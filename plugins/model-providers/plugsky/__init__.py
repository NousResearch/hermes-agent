"""Plugsky provider profile (OpenAI-compatible chat surface)."""

from providers import register_provider
from providers.base import ProviderProfile

plugsky = ProviderProfile(
    name="plugsky",
    aliases=("plugsky",),
    display_name="Plugsky",
    description="Plugsky — OpenAI-compatible API with a free Micro/Lite tier",
    signup_url="https://plugsky.com/",
    env_vars=("PLUGSKY_API_KEY", "PLUGSKY_BASE_URL"),
    base_url="https://api.plugsky.com/v1",
    auth_type="api_key",
    default_aux_model="plugsky-micro",
    fallback_models=("plugsky-lite", "plugsky-micro"),
)

register_provider(plugsky)
