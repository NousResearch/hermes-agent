"""Chutes provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

chutes = ProviderProfile(
    name="chutes",
    aliases=("chutes-ai", "chutesai"),
    env_vars=("CHUTES_API_KEY", "CHUTES_BASE_URL"),
    display_name="Chutes",
    description="Chutes — decentralized, OpenAI-compatible inference (TEE confidential compute)",
    signup_url="https://chutes.ai/app/api",
    base_url="https://llm.chutes.ai/v1",
    default_aux_model="Qwen/Qwen3-32B-TEE",
    fallback_models=(
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "zai-org/GLM-5.1-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "MiniMaxAI/MiniMax-M2.5-TEE",
    ),
)

register_provider(chutes)
