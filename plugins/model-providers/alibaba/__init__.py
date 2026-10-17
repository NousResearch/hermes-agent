"""Alibaba Cloud DashScope provider profiles (intl + CN, plus the Model Studio
Token Plan flat-token tier with its own key/endpoints — one module per vendor).

Profile names match models.dev catalog keys exactly so model metadata lines up
and ``model.provider: alibaba-cn`` resolves at runtime.
"""

from typing import Any

from agent.reasoning_effort import DASHSCOPE_EFFORTS, clamp_effort, dashscope_preserve_thinking_extras
from providers import register_provider
from providers.base import ProviderProfile


class DashScopeProfile(ProviderProfile):
    """DashScope compatible-mode: translate the reasoning setting, keep replayed reasoning.

    The server default for Qwen thinking models is *full* thinking, so a profile that
    never translates the user's setting leaves ``--reasoning none`` and
    ``agent.reasoning_effort`` silent no-ops (#137174). Thinking rides on two fields
    that must agree — ``extra_body.enable_thinking`` plus a top-level
    ``reasoning_effort`` — otherwise DashScope answers HTTP 400 (``'reasoning_effort'
    must be 'none' when 'enable_thinking' is false``)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body = dashscope_preserve_thinking_extras(model)
        if not isinstance(reasoning_config, dict):
            return extra_body, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if reasoning_config.get("enabled", True) is False or effort in {"none", "false", "disabled"}:
            extra_body["enable_thinking"] = False
            return extra_body, {"reasoning_effort": "none"}
        clamped = clamp_effort(effort, DASHSCOPE_EFFORTS)
        if clamped in DASHSCOPE_EFFORTS:
            extra_body["enable_thinking"] = True
            return extra_body, {"reasoning_effort": clamped}
        return extra_body, {}


alibaba = DashScopeProfile(
    name="alibaba", aliases=("dashscope", "alibaba-cloud", "qwen-dashscope", "aliyun"), env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

alibaba_cn = DashScopeProfile(
    name="alibaba-cn", aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)",
    description="Alibaba Cloud DashScope, mainland-China endpoint",
    env_vars=("DASHSCOPE_API_KEY", "DASHSCOPE_CN_BASE_URL"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

alibaba_token_plan = DashScopeProfile(
    name="alibaba-token-plan", aliases=("dashscope-token-plan",), display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Model Studio Token Plan (flat-token tier)",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", auth_type="api_key",
)

alibaba_token_plan_cn = DashScopeProfile(
    name="alibaba-token-plan-cn", aliases=("dashscope-token-plan-cn",),
    display_name="Alibaba Cloud (Token Plan, China)",
    description="Alibaba Cloud Model Studio Token Plan, mainland-China endpoint",
    signup_url="https://help.aliyun.com/zh/model-studio/",
    env_vars=("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", auth_type="api_key",
)

register_provider(alibaba)
register_provider(alibaba_cn)
register_provider(alibaba_token_plan)
register_provider(alibaba_token_plan_cn)
