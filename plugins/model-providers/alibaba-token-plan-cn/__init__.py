"""Alibaba Cloud Token Plan (Team Edition), China / Beijing gateway, provider profile.

Separate from `alibaba-token-plan` (Global / Singapore) because the China
gateway is a distinct endpoint (token-plan.cn-beijing.maas.aliyuncs.com) with
its own console and its own API key. The model catalog matches Global.

Docs: https://help.aliyun.com/zh/model-studio/token-plan-overview
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_token_plan_cn = ProviderProfile(
    name="alibaba-token-plan-cn",
    aliases=("alibaba_token_plan_cn", "aliyun-token-plan-cn", "token-plan-cn"),
    display_name="Alibaba Cloud (Token Plan, China)",
    description="Alibaba Cloud Token Plan, China gateway — team subscription (Credits)",
    signup_url="https://www.aliyun.com/benefit/scene/tokenplan",
    env_vars=("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

register_provider(alibaba_token_plan_cn)
