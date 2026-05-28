"""Xiaomi MiMo Token Plan provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

xiaomi_token_plan = ProviderProfile(
    name="xiaomi-token-plan",
    aliases=(
        "xiaomi-token",
        "mimo-token",
        "token-plan",
    ),
    env_vars=("XIAOMI_TOKEN_PLAN_API_KEY",),
    base_url="https://token-plan-sgp.xiaomimimo.com/v1",
)

register_provider(xiaomi_token_plan)
