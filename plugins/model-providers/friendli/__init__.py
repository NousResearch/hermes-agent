"""Friendli provider profile.

Friendli serves fast, cost-efficient inference for open models through an
OpenAI-compatible chat-completions endpoint.

Address models directly by their catalog ID, e.g.
``deepseek-ai/DeepSeek-V3.2`` or ``zai-org/GLM-5.2``.
"""

from providers import register_provider
from providers.base import ProviderProfile

friendli = ProviderProfile(
    name="friendli",
    aliases=("friendliai", "friendli-ai"),
    display_name="Friendli",
    description="Friendli — OpenAI-compatible serverless inference API",
    signup_url="https://friendli.ai/suite/~/setting/keys",
    env_vars=("FRIENDLI_API_KEY", "FRIENDLI_BASE_URL"),
    base_url="https://api.friendli.ai/serverless/v1",
    auth_type="api_key",
    default_aux_model="deepseek-ai/DeepSeek-V3.2",
    fallback_models=(
        "deepseek-ai/DeepSeek-V3.2",
        "zai-org/GLM-5.2",
        "MiniMaxAI/MiniMax-M2.5",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "google/gemma-4-31B-it",
    ),
)

register_provider(friendli)
