"""Cerebras Cloud provider profile.

Cerebras runs OpenAI-compatible inference on its wafer-scale chips —
typically several thousand tokens/sec on Llama and other open models.
"""

from providers import register_provider
from providers.base import ProviderProfile

cerebras = ProviderProfile(
    name="cerebras",
    aliases=("cerebras-cloud",),
    env_vars=("CEREBRAS_API_KEY",),
    display_name="Cerebras",
    description="Cerebras Cloud — wafer-scale inference (open models)",
    signup_url="https://cloud.cerebras.ai/",
    fallback_models=(
        "gpt-oss-120b",
        "qwen-3-235b-a22b-instruct-2507",
        "zai-glm-4.7",
        "llama3.1-8b",
    ),
    base_url="https://api.cerebras.ai/v1",
    default_aux_model="llama3.1-8b",
)

register_provider(cerebras)
