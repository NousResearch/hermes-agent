"""ModelScope provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

modelscope = ProviderProfile(
    name="modelscope",
    aliases=("ms",),
    env_vars=("MODELSCOPE_API_KEY",),
    display_name="ModelScope",
    description="ModelScope Inference API",
    signup_url="https://modelscope.cn",
    base_url="https://api-inference.modelscope.cn/v1",
    auth_type="api_key",
    default_aux_model="Qwen/Qwen3.5-27B",
    fallback_models=(
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-397B-A17B",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-R1-0528",
        "ZhipuAI/GLM-5.1",
        "MiniMax/MiniMax-M2.7",
        "moonshotai/Kimi-K2.5",
    ),
)

register_provider(modelscope)
