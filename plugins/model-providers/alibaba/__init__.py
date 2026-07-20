"""Qwen Cloud PAYG provider profile via Alibaba Cloud DashScope."""

from typing import Any
from providers import register_provider
from providers.base import ProviderProfile


PAYG_FALLBACK_MODELS = (
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen3-coder-plus",
    "qwen3-coder-flash",
    "qwen3-coder-next",
)


class QwenCloudPaygProfile(ProviderProfile):
    """Qwen Cloud PAYG — DashScope hybrid-thinking request controls."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map explicit Hermes thinking toggles to Qwen's extra_body field.

        Qwen3.7, Qwen3.6, and Qwen3.5 use ``enable_thinking`` as a boolean
        inside ``extra_body``.  When Hermes has no reasoning configuration,
        omit the field so DashScope keeps its documented model default. Hermes
        effort levels intentionally do not become a made-up Qwen depth knob.
        Other PAYG models, including Qwen Coder, receive no hybrid-thinking
        field unless Qwen documents support for that exact family.
        """
        model_name = str(model or "").strip().lower()
        if not model_name.startswith(("qwen3.7-", "qwen3.6-", "qwen3.5-")):
            return {}, {}
        if not isinstance(reasoning_config, dict):
            return {}, {}

        enabled = reasoning_config.get("enabled")
        if isinstance(enabled, bool):
            return {"enable_thinking": enabled}, {}
        return {}, {}


alibaba = QwenCloudPaygProfile(
    name="alibaba",
    aliases=(
        "dashscope",
        "alibaba-cloud",
        "qwen-dashscope",
        "qwen-cloud",
        "qwencloud",
    ),
    display_name="Qwen Cloud (PAYG)",
    description="Qwen Cloud pay-as-you-go API via Alibaba Cloud DashScope",
    signup_url="https://modelstudio.console.alibabacloud.com/",
    env_vars=("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    fallback_models=PAYG_FALLBACK_MODELS,
    default_aux_model="qwen3.6-flash",
)

register_provider(alibaba)
