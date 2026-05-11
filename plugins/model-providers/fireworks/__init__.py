"""Fireworks AI provider profile."""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

fireworks = ProviderProfile(
    name="fireworks",
    aliases=("fireworks-ai", "fw"),
    display_name="Fireworks AI",
    description="Fireworks AI — fast open model inference (DeepSeek, Kimi, GLM, MiniMax)",
    signup_url="https://fireworks.ai/",
    env_vars=("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL"),
    base_url="https://api.fireworks.ai/inference/v1",
    auth_type="api_key",
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    default_aux_model="accounts/fireworks/models/glm-5",
    fallback_models=(
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/kimi-k2p5",
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/glm-5",
        "accounts/fireworks/models/minimax-m2p7",
    ),
)

register_provider(fireworks)
