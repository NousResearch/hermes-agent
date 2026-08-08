"""Alibaba Cloud Coding Plan provider profile.

Separate from the standard `alibaba` profile because it hits a different
endpoint (coding-intl.dashscope.aliyuncs.com) with a dedicated API key tier.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class AlibabaCodingPlanProfile(ProviderProfile):
    """Alibaba Coding Plan — flat ``reasoning_effort`` wire shape.

    The generic fallback in ``_build_call_kwargs`` emits OpenRouter-shaped
    ``extra_body.reasoning = {"enabled": ..., "effort": ...}`` for profiles
    that don't handle reasoning. Aliyun's compatible-mode gateway (istio-envoy)
    does not reject that object — it silently blackholes the entire request
    (zero response bytes until client timeout), verified 2026-08-08 with
    byte-identical curl replays: ``reasoning: {...}`` → hang; flat
    ``reasoning_effort: "medium"`` → 200 with reasoning_tokens populated.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not reasoning_config or not isinstance(reasoning_config, dict):
            return {}, {}
        if reasoning_config.get("enabled") is False:
            # No documented "thinking off" field on the compatible-mode wire;
            # omitting keeps the server default instead of risking a blackhole.
            return {}, {}
        effort = reasoning_config.get("effort") or "medium"
        return {}, {"reasoning_effort": effort}


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

register_provider(alibaba_coding_plan)
