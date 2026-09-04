"""Alibaba Cloud Coding Plan provider profiles.

Separate from the standard `alibaba` profile because it hits a different
endpoint (coding-intl.dashscope.aliyuncs.com) with a dedicated API key tier.

Region split, mirroring the base DashScope pair (#73265):
  - ``alibaba-coding-plan``    → coding-intl.dashscope.aliyuncs.com (international)
  - ``alibaba-coding-plan-cn`` → coding.dashscope.aliyuncs.com (mainland China)

Profile names match the models.dev catalog keys exactly so model metadata
lines up and ``model.provider: alibaba-coding-plan-cn`` resolves at runtime.

The CN profile checks its own ``ALIBABA_CODING_PLAN_CN_API_KEY`` first (#101122,
mirroring kimi-coding-cn) and keeps the shared vars as ordered fallbacks so
existing CN users configured with the shared key keep working.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class AlibabaCodingPlanProfile(ProviderProfile):
    """Alibaba Cloud Coding Plan — top-level reasoning_effort passthrough.

    Qwen3 thinking models on DashScope support ``reasoning_effort`` as a
    top-level request parameter (xhigh / medium / low; server default:
    xhigh).  Without this override the field is silently omitted and the
    endpoint always applies its server-side default, making it impossible
    for users to lower thinking depth (#77818).
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        top_level: dict[str, Any] = {}

        if not isinstance(reasoning_config, dict):
            return {}, top_level

        effort = (reasoning_config.get("effort") or "").strip().lower()
        if not effort:
            return {}, top_level

        # Map Hermes effort levels to DashScope-supported values.
        # DashScope Qwen3 accepts: xhigh, medium, low.
        if effort in {"xhigh", "max", "ultra"}:
            top_level["reasoning_effort"] = "xhigh"
        elif effort in {"low", "medium", "high"}:
            top_level["reasoning_effort"] = effort
        # "none" / "minimal" → omit so the model applies its default
        # (users who set effort=none likely want thinking off, but
        # DashScope doesn't support that — omit to avoid 400 errors).

        return {}, top_level


alibaba_coding_plan = AlibabaCodingPlanProfile(
    name="alibaba-coding-plan",
    aliases=("alibaba_coding", "alibaba-coding", "dashscope-coding"),
    display_name="Alibaba Cloud (Coding Plan)",
    description="Alibaba Cloud Coding Plan (Dedicated coding tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_BASE_URL"),
    base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
    auth_type="api_key",
)

alibaba_coding_plan_cn = ProviderProfile(
    name="alibaba-coding-plan-cn",
    aliases=("alibaba-coding-cn", "dashscope-coding-cn"),
    display_name="Alibaba Cloud (Coding Plan, China)",
    description="Alibaba Cloud Coding Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_CODING_PLAN_CN_API_KEY", "ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_CN_BASE_URL"),
    base_url="https://coding.dashscope.aliyuncs.com/v1",
    auth_type="api_key",
)

register_provider(alibaba_coding_plan)
register_provider(alibaba_coding_plan_cn)
