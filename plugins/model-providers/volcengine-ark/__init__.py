"""Volcengine Ark (火山方舟) provider profile.

Supports both Agent Plan and API usage modes. Agent Plan uses a dedicated
endpoint (/api/plan/v3) with subscription-based billing (AFP credits).
"""

from providers import register_provider
from providers.base import ProviderProfile

volcengine_ark = ProviderProfile(
    name="volcengine-ark",
    aliases=("volc-ark", "ark", "huoshan-ark", "doubao"),
    display_name="Volcengine Ark (火山方舟)",
    description="Volcengine Ark - Agent Plan & API access (Doubao, DeepSeek, GLM, Kimi, MiniMax)",
    signup_url="https://www.volcengine.com/product/ark",
    env_vars=("VOLCENGINE_ARK_API_KEY", "ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
    auth_type="api_key",
)

register_provider(volcengine_ark)
