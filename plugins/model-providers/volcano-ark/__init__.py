"""Volcano Ark (火山方舟) provider profile.

Volcano Ark is ByteDance's ML platform. The Coding Plan exposes an
OpenAI-compatible chat completions endpoint at /api/coding/v3.

Common models available through the Coding Plan:
  - DeepSeek V4 Pro / Flash
  - Kimi K2.6 / K2.7 Code
  - GLM-5.2 / GLM-5.1
  - Qwen3.7 Max / Plus
  - MiniMax M3

The general Ark API at /api/v3 is a different surface and returns
"InvalidEndpointOrModel.NotFound" for Coding Plan models — always use
/api/coding/v3 for the Coding Plan.
"""

from providers import register_provider
from providers.base import ProviderProfile

volcano_ark = ProviderProfile(
    name="volcano-ark",
    aliases=("volcano", "ark", "volcengine", "火山", "火山方舟"),
    env_vars=("VOLCANO_API_KEY", "ARK_API_KEY"),
    display_name="Volcano Ark (火山方舟)",
    description="Volcano Ark Coding Plan — ByteDance ML platform (DeepSeek, Kimi, GLM, Qwen, MiniMax)",
    signup_url="https://www.volcengine.com/product/ark",
    base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
    default_aux_model="deepseek-v4-flash",
    fallback_models=(
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "glm-5.2",
        "glm-5.1",
    ),
)

register_provider(volcano_ark)
