"""Sference provider profile.

Sference — hosted LLM inference on bare-metal GPUs (B200/B300/AMD BW100).
OpenAI-compatible API at https://api.sference.com/v1, EMEA region.
"""

from providers import register_provider
from providers.base import ProviderProfile

sference = ProviderProfile(
    name="sference",
    aliases=("sference",),
    display_name="Sference",
    description="Sference — hosted LLM inference on bare-metal GPUs (EMEA)",
    signup_url="https://sference.com",
    env_vars=("SFERENCE_API_KEY", "SFERENCE_BASE_URL"),
    base_url="https://api.sference.com/v1",
    auth_type="api_key",
    fallback_models=(
        "zai-org/GLM-5.2",
        "deepseek-ai/DeepSeek-V4-Flash",
        "Qwen/Qwen3.6-35B-A3B",
        "bottlecapai/ThinkingCap-Qwen3.6-27B",
    ),
    default_aux_model="Qwen/Qwen3.6-35B-A3B",
)

register_provider(sference)
