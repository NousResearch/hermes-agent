"""Cerebras Inference provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

cerebras = ProviderProfile(
    name="cerebras",
    aliases=("cerebras-ai", "cerebras-cloud", "cerebras-inference"),
    env_vars=("CEREBRAS_API_KEY",),
    display_name="Cerebras",
    description="Cerebras Inference — high-speed OpenAI-compatible inference",
    signup_url="https://cloud.cerebras.ai/",
    base_url="https://api.cerebras.ai/v1",
    models_url="https://api.cerebras.ai/public/v1/models",
    default_aux_model="gpt-oss-120b",
    fallback_models=(
        "gpt-oss-120b",
        "gemma-4-31b",
        "zai-glm-4.7",
    ),
)

register_provider(cerebras)
