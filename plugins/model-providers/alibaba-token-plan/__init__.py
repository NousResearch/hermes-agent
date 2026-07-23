"""Alibaba Cloud Token Plan provider profile.

Separate from the standard `alibaba` and `alibaba-coding-plan` profiles
because it hits a different endpoint (token-plan.ap-southeast-1.maas.aliyuncs.com)
with a dedicated Token Plan API key (sk-sp- prefix).
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_token_plan = ProviderProfile(
    name="alibaba-token-plan",
    aliases=("alibaba_token", "alibaba-token", "dashscope-token"),
    display_name="Alibaba Cloud (New Token Plan)",
    description="Alibaba Cloud Token Plan Team Edition (Prepaid subscription)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

register_provider(alibaba_token_plan)
