"""Hetzner Inference provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

hetzner = ProviderProfile(
    name="hetzner",
    aliases=("hetzner-inference",),
    display_name="Hetzner Inference",
    description="Hetzner Inference — OpenAI-compatible API on Hetzner infrastructure",
    signup_url="https://experiments.hetzner.com",
    env_vars=("HETZNER_API_KEY", "HETZNER_BASE_URL"),
    base_url="https://inference.hetzner.com/api/v1",
    auth_type="api_key",
    supports_vision=True,
    fallback_models=(
        "Qwen/Qwen3.6-35B-A3B-FP8",
    ),
)

register_provider(hetzner)
