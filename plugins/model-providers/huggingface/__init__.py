"""Hugging Face provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

huggingface = ProviderProfile(
    name="huggingface",
    aliases=("hf", "hugging-face", "huggingface-hub"),
    env_vars=("HF_TOKEN",),
    display_name="HuggingFace",
    description="HuggingFace Inference API",
    signup_url="https://huggingface.co/settings/tokens",
    fallback_models=(
        "Qwen/Qwen3.5-122B-A10B",
        "deepseek-ai/DeepSeek-V3.2-Exp",
    ),
    base_url="https://router.huggingface.co/v1",
    supports_vision=True,
)

register_provider(huggingface)
