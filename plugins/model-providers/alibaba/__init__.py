"""Alibaba Cloud DashScope provider profiles.

DashScope has region-split endpoints with the same key type:
  - ``alibaba``    → dashscope-intl.aliyuncs.com (international)
  - ``alibaba-cn`` → dashscope.aliyuncs.com (mainland China)

The Model Studio Token Plan (flat-token tier of the SAME vendor/service,
same OpenAI-compatible protocol, its own key + endpoints) registers here
too rather than as a new plugin directory — one module per vendor, matching
how the kimi module carries both of its endpoint variants:
  - ``alibaba-token-plan``    → token-plan.ap-southeast-1.maas.aliyuncs.com
  - ``alibaba-token-plan-cn`` → token-plan.cn-beijing.maas.aliyuncs.com

Profile names match the models.dev catalog keys exactly
(``alibaba`` / ``alibaba-cn``) so model metadata lines up and
``model.provider: alibaba-cn`` resolves at runtime (#73265).
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba = ProviderProfile(
    name="alibaba",
    aliases=("dashscope", "alibaba-cloud", "qwen-dashscope"),
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

alibaba_cn = ProviderProfile(
    name="alibaba-cn",
    aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)",
    description="Alibaba Cloud DashScope, mainland-China endpoint",
    env_vars=("DASHSCOPE_API_KEY", "DASHSCOPE_CN_BASE_URL"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

register_provider(alibaba)
register_provider(alibaba_cn)

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
    env_vars=("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

register_provider(alibaba_token_plan)
register_provider(alibaba_token_plan_cn)
