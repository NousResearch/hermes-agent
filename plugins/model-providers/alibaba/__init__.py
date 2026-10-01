"""Alibaba Cloud DashScope provider profiles (intl + CN, plus the Model Studio
Token Plan flat-token tier with its own key/endpoints — one module per vendor).

Profile names match models.dev catalog keys exactly so model metadata lines up
and ``model.provider: alibaba-cn`` resolves at runtime.
"""

from providers import register_provider
from providers.base import ProviderProfile
from providers.model_normalizers import MatchingPrefixProviderProfile


class AlibabaProfile(MatchingPrefixProviderProfile):
    model_prefix_exclusions = ("qwen-dashscope",)


alibaba = AlibabaProfile(
    name="alibaba", aliases=("dashscope", "alibaba-cloud", "qwen-dashscope", "aliyun"),
    display_name="Qwen Cloud", description="Qwen Cloud / DashScope (Qwen + multi-provider)",
    signup_url="https://modelstudio.console.alibabacloud.com/", env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", base_url_env_var="DASHSCOPE_BASE_URL",
)

alibaba_cn = ProviderProfile(
    name="alibaba-cn", aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)",
    description="Alibaba Cloud DashScope, mainland-China endpoint",
    signup_url="https://modelstudio.console.alibabacloud.com/", env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", base_url_env_var="DASHSCOPE_CN_BASE_URL",
)

alibaba_token_plan = ProviderProfile(
    name="alibaba-token-plan", aliases=("dashscope-token-plan",), display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Model Studio Token Plan (flat-token tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY",),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    base_url_env_var="ALIBABA_TOKEN_PLAN_BASE_URL", auth_type="api_key",
)

alibaba_token_plan_cn = ProviderProfile(
    name="alibaba-token-plan-cn", aliases=("dashscope-token-plan-cn",),
    display_name="Alibaba Cloud (Token Plan, China)",
    description="Alibaba Cloud Model Studio Token Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_API_KEY"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    base_url_env_var="ALIBABA_TOKEN_PLAN_CN_BASE_URL", auth_type="api_key",
)

register_provider(alibaba)
register_provider(alibaba_cn)
register_provider(alibaba_token_plan)
register_provider(alibaba_token_plan_cn)
