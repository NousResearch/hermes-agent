"""Alibaba Cloud Token Plan (Personal Edition) provider profile.

Separate from `alibaba` (DashScope compatible-mode) and
`alibaba-coding-plan` (coding-intl) because Token Plan Personal Edition
hits a dedicated endpoint (token-plan.ap-southeast-1.maas.aliyuncs.com)
with its own key tier (`sk-sp-...`).

Endpoints (OpenAI-compatible):
  https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
Anthropic-compatible:
  https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic

The key also appears in the Qwen CLI as `BAILIAN_TOKEN_PLAN_API_KEY`.
"""

from providers import register_provider
from providers.base import ProviderProfile

alibaba_token_plan = ProviderProfile(
    name="alibaba-token-plan",
    aliases=("alibaba_token", "alibaba-token", "dashscope-token", "bailian-token"),
    display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Token Plan Personal Edition (dedicated token tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "BAILIAN_TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY"),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
)

register_provider(alibaba_token_plan)
