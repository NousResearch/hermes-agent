"""Alibaba Cloud Token Plan (Model Studio flat-token tier) provider profiles.

Token Plan is a separate purchase tier from both standard DashScope and the
Coding Plan, with its own dedicated endpoints and API key
(``ALIBABA_TOKEN_PLAN_API_KEY``). Both regional endpoints speak the
OpenAI-compatible chat-completions protocol:

  - ``alibaba-token-plan``    → token-plan.ap-southeast-1.maas.aliyuncs.com (international)
  - ``alibaba-token-plan-cn`` → token-plan.cn-beijing.maas.aliyuncs.com (mainland China)

Profile names match the models.dev catalog keys exactly so model metadata
lines up and ``model.provider: alibaba-token-plan-cn`` resolves at runtime
instead of failing with "Unknown provider" (#73265).
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_token_plan = ProviderProfile(
    name="alibaba-token-plan",
    aliases=("dashscope-token-plan",),
    display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Model Studio Token Plan (flat-token tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

alibaba_token_plan_cn = ProviderProfile(
    name="alibaba-token-plan-cn",
    aliases=("dashscope-token-plan-cn",),
    display_name="Alibaba Cloud (Token Plan, China)",
    description="Alibaba Cloud Model Studio Token Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

register_provider(alibaba_token_plan)
register_provider(alibaba_token_plan_cn)
