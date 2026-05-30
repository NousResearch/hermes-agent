"""Alibaba Cloud Token Plan (Team Edition) provider profile.

Separate from the standard `alibaba` and `alibaba-coding-plan` profiles because
it hits a different endpoint (token-plan.cn-beijing.maas.aliyuncs.com) with
its own subscription-based billing (Credits).

Docs: https://help.aliyun.com/zh/model-studio/token-plan-overview
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_token_plan = ProviderProfile(
    name="alibaba-token-plan",
    aliases=("alibaba_token_plan", "aliyun-token-plan", "token-plan"),
    display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Token Plan — team subscription (Credits)",
    signup_url="https://www.aliyun.com/benefit/scene/tokenplan",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

register_provider(alibaba_token_plan)
