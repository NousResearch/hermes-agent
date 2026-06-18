"""Featherless provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

featherless = ProviderProfile(
    name="featherless",
    aliases=("featherless-ai", "featherlessai"),
    env_vars=("FEATHERLESS_API_KEY",),
    base_url="https://api.featherless.ai/v1",
)

register_provider(featherless)
