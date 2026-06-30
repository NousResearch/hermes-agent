"""LongCat provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


longcat = ProviderProfile(
    name="longcat",
    aliases=("long-cat", "meituan-longcat"),
    display_name="LongCat",
    description="LongCat — Meituan LongCat models via an OpenAI-compatible API",
    signup_url="https://longcat.chat/platform/",
    env_vars=("LONGCAT_API_KEY", "LONGCAT_BASE_URL"),
    base_url="https://api.longcat.chat/openai/v1",
    auth_type="api_key",
    default_aux_model="LongCat-2.0",
    fallback_models=("LongCat-2.0",),
)

register_provider(longcat)
