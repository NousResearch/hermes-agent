"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

xiaomi = ProviderProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY", "XIAOMI_BASE_URL"),
    base_url="https://token-plan-sgp.xiaomimimo.com/v1",
)

register_provider(xiaomi)
