"""Ambient provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


ambient = ProviderProfile(
    name="ambient",
    display_name="Ambient",
    description="Ambient — verified AI inference with cryptographic proof of every call (GLM-5.1)",
    signup_url="https://app.ambient.xyz",
    env_vars=("AMBIENT_API_KEY", "AMBIENT_BASE_URL"),
    base_url="https://api.ambient.xyz/v1",
    auth_type="api_key",
    default_aux_model="zai-org/GLM-5.1-FP8",
    fallback_models=(
        "zai-org/GLM-5.1-FP8",
    ),
)

register_provider(ambient)
