"""Alibaba Cloud Coding Plan provider profiles (intl + CN): a dedicated endpoint
and key tier separate from ``alibaba``. Names match models.dev catalog keys.

The CN profile checks its own key first and keeps the shared vars as ordered
fallbacks so existing CN users configured with the shared key keep working.
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_coding_plan = ProviderProfile(
    name="alibaba-coding-plan", aliases=("alibaba_coding", "alibaba-coding", "dashscope-coding"),
    display_name="Alibaba Cloud (Coding Plan)",
    description="Alibaba Cloud Coding Plan (Dedicated coding tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_BASE_URL"),
    base_url="https://coding-intl.dashscope.aliyuncs.com/v1", auth_type="api_key",
)

alibaba_coding_plan_cn = ProviderProfile(
    name="alibaba-coding-plan-cn", aliases=("alibaba-coding-cn", "dashscope-coding-cn"),
    display_name="Alibaba Cloud (Coding Plan, China)",
    description="Alibaba Cloud Coding Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_CN_API_KEY", "ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_CN_BASE_URL"),
    base_url="https://coding.dashscope.aliyuncs.com/v1", auth_type="api_key",
)

register_provider(alibaba_coding_plan)
register_provider(alibaba_coding_plan_cn)
