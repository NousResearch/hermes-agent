"""Alibaba Cloud Coding Plan provider profile.

Separate from the standard `alibaba` profile because it hits a different
endpoint (`coding.dashscope.aliyuncs.com` and `coding-intl.dashscope.aliyuncs.com`)
with dedicated API key tiers.
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_coding_plan = ProviderProfile(
    name="alibaba-coding-plan",
    aliases=("alibaba_coding", "alibaba-coding", "dashscope-coding"),
    display_name="Alibaba Cloud (Coding Plan)",
    description="Alibaba Cloud Coding Plan (Dedicated coding tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_BASE_URL"),
    base_url="https://coding.dashscope.aliyuncs.com/v1",
    auth_type="api_key",
)

alibaba_coding_plan_intl = ProviderProfile(
    name="alibaba-coding-plan-intl",
    aliases=("alibaba-coding-plan-international", "alibaba-coding-intl", "dashscope-coding-intl"),
    display_name="Alibaba Cloud (Coding Plan Intl)",
    description="Alibaba Cloud Coding Plan (International endpoint)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_INTL_BASE_URL", "ALIBABA_CODING_PLAN_BASE_URL"),
    base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
    auth_type="api_key",
)

register_provider(alibaba_coding_plan)
register_provider(alibaba_coding_plan_intl)
