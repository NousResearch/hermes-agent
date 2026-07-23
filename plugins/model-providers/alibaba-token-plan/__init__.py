"""Alibaba Cloud Token Plan provider profile.

Separate from the standard `alibaba` profile because it hits the Token Plan's
exclusive MaaS endpoint (token-plan.ap-southeast-1.maas.aliyuncs.com) rather
than the public DashScope endpoint. The Token Plan exposes newer models
(qwen3.8-max-preview, qwen3.7-plus/max, qwen3.6-flash, glm-5.2, deepseek-v4-pro,
wan2.7-image[-pro]) and an Anthropic-protocol-compatible path under /apps/anthropic.

Auth: use ALIBABA_TOKEN_PLAN_API_KEY so regular DashScope and Coding Plan keys
are never probed against an incompatible billing-plan endpoint.
"""

from providers import register_provider
from providers.base import ProviderProfile


class AlibabaTokenPlanProfile(ProviderProfile):
    """Token Plan catalog restricted to agent-compatible chat models."""

    def fetch_models(self, **kwargs):
        models = super().fetch_models(**kwargs)
        if models is None:
            return None
        # The endpoint also advertises Wan image-generation models. They are
        # not valid agent chat models and must not enter the main model picker.
        return [model for model in models if not model.startswith("wan")]


alibaba_token_plan = AlibabaTokenPlanProfile(
    name="alibaba-token-plan",
    aliases=("alibaba-token", "token-plan", "alibaba-maas", "alibaba-tokenplan"),
    display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Token Plan — exclusive MaaS endpoint (Qwen 3.8/3.7, GLM 5.2, DeepSeek v4)",
    signup_url="https://www.alibabacloud.com/en/campaign/ai-landing-page-token",
    env_vars=(
        "ALIBABA_TOKEN_PLAN_API_KEY",
        "ALIBABA_TOKEN_PLAN_BASE_URL",
    ),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    auth_type="api_key",
    supports_health_check=True,
    supports_vision=False,
    default_aux_model="qwen3.6-flash",
    fallback_models=(
        "qwen3.8-max-preview",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-flash",
        "glm-5.2",
        "deepseek-v4-pro",
    ),
)

register_provider(alibaba_token_plan)
