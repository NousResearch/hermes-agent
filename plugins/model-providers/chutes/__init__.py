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
    # Cheap auxiliary model (compression, titles, vision). Must clear Hermes'
    # 64K MINIMUM_CONTEXT_LENGTH — so not the 40K Qwen3-32B; gemma-4-31B-turbo
    # is the cheapest tool-capable catalog model above that floor.
    default_aux_model="google/gemma-4-31B-turbo-TEE",
    fallback_models=(
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "zai-org/GLM-5.1-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "MiniMaxAI/MiniMax-M2.5-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ),
)

register_provider(chutes)
