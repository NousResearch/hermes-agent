"""Meta provider profile — Meta Model API (OpenAI-compatible)."""

from providers import register_provider
from providers.base import ProviderProfile

meta = ProviderProfile(
    name="meta",
    aliases=("meta-ai",),
    display_name="Meta",
    description="Meta — Model API (OpenAI-compatible)",
    signup_url="https://dev.meta.ai/",
    env_vars=("MODEL_API_KEY",),
    base_url="https://api.meta.ai/v1",
    auth_type="api_key",
    default_aux_model="muse-spark-1.1",
    fallback_models=("muse-spark-1.1",),
)

register_provider(meta)
