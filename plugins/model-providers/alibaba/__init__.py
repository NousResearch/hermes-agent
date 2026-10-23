"""Alibaba Cloud DashScope and Token Plan provider profiles."""

import hmac
import logging
import os
from typing import Any

from agent.reasoning_effort import dashscope_preserve_thinking_extras
from providers import register_provider
from providers.base import ProviderProfile
from providers.alibaba_token_plan_models import (
    PERSONAL_MODELS,
    TEAM_MODELS,
    UNLISTED_MODELS,
    DEFAULT_AUX_MODEL,
)

logger = logging.getLogger(__name__)
# Do not retain credentials as plaintext in discovery cache identities.
_AUX_CACHE_KEY = os.urandom(32)
_AUX_PREFERENCE = ("qwen3.6-flash", "deepseek-v4-flash-0731", "deepseek-v4-flash")
_VISION_AUX_PREFERENCE = ("qwen3.6-flash", "qwen3.6-plus", "qwen3.7-plus")
_ALWAYS_THINKING_MODELS = {"minimax-m2.5", "glm-5.3"}
_HYBRID_THINKING_MODELS = {
    model.lower() for model in TEAM_MODELS
} - _ALWAYS_THINKING_MODELS


class DashScopeProfile(ProviderProfile):
    """DashScope compatible-mode: ask the server to keep replayed prior-turn reasoning."""

    def build_api_kwargs_extras(
        self, *, model: str | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return dashscope_preserve_thinking_extras(model), {}


class QwenTokenPlanProfile(DashScopeProfile):
    """Shared catalogue and request behaviour for both Token Plan regions."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return entitled chat models in canonical order."""
        live = super().fetch_models(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        if live is None:
            return None
        live_ids = {str(model).strip().lower() for model in live}
        catalog = self.fallback_models if self.fallback_models else TEAM_MODELS
        unlisted_ids = {str(model).strip().lower() for model in UNLISTED_MODELS}
        return [
            model
            for model in catalog
            if model.lower() in live_ids or model.lower() in unlisted_ids
        ]

    def _resolve_credentials(self) -> tuple[str, str]:
        """Use the active profile's auth and configured discovery route."""
        from hermes_cli.auth import resolve_api_key_provider_credentials
        from hermes_cli.models import _configured_relay_base_url

        credentials = resolve_api_key_provider_credentials(self.name)
        base_url = _configured_relay_base_url(self.name) or credentials["base_url"]
        return credentials["api_key"], base_url

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """Return a live cheap model id for auxiliary tasks, or ""."""
        cache = getattr(self, "_aux_model_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_aux_model_cache", cache)
        resolved = ""
        cache_key = None
        try:
            api_key, base_url = self._resolve_credentials()
            if not api_key:
                return ""
            cache_key = (
                vision,
                base_url,
                hmac.digest(_AUX_CACHE_KEY, api_key.encode(), "sha256"),
            )
            if cache_key in cache:
                return cache[cache_key]
            live = self.fetch_models(api_key=api_key, base_url=base_url, timeout=4.0)
            if live:
                entitled = {model.lower() for model in live}
                preference = _VISION_AUX_PREFERENCE if vision else _AUX_PREFERENCE
                resolved = next(
                    (model for model in preference if model.lower() in entitled),
                    "",
                )
        except Exception:
            logger.debug("Token Plan auxiliary discovery failed", exc_info=True)
            # Never raise from a client-resolution path. An empty string is
            # the documented "no answer" value and keeps default_aux_model.
            if cache_key is None:
                return ""
        cache[cache_key] = resolved
        return resolved

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict[str, Any] | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate explicit Hermes reasoning controls for measured models."""
        model_name = str(model or "").strip().lower()
        config = reasoning_config if isinstance(reasoning_config, dict) else {}
        body, kwargs = super().build_api_kwargs_extras(model=model, **context)

        # qwen3.8-max (GA) accepts reasoning_effort low/medium/high/xhigh
        # (gateway acceptance enumerated 2026-08-05). Its retired preview id
        # carried the same contract while it lived.
        if model_name == "qwen3.8-max":
            effort = str(config.get("effort") or "").strip().lower()
            mapped_effort = {
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "xhigh",
                "xhigh": "xhigh",
                "max": "xhigh",
            }.get(effort)
            if mapped_effort:
                body["reasoning_effort"] = mapped_effort

        if model_name in _ALWAYS_THINKING_MODELS:
            if config.get("enabled") is True:
                body["enable_thinking"] = True
            return body, kwargs

        if model_name in _HYBRID_THINKING_MODELS and isinstance(
            config.get("enabled"), bool
        ):
            body["enable_thinking"] = config["enabled"]

        return body, kwargs


alibaba = DashScopeProfile(
    name="alibaba",
    aliases=("dashscope", "alibaba-cloud", "qwen-dashscope", "aliyun"),
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

alibaba_cn = DashScopeProfile(
    name="alibaba-cn",
    aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)",
    description="Alibaba Cloud DashScope, mainland-China endpoint",
    env_vars=("DASHSCOPE_API_KEY", "DASHSCOPE_CN_BASE_URL"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

alibaba_token_plan = QwenTokenPlanProfile(
    name="alibaba-token-plan",
    aliases=(
        "alibaba_token_plan",
        "aliyun-token-plan",
        "token-plan",
        "qwen-token-plan",
        "qwencloud-token-plan",
        "bailian-token-plan",
        "dashscope-token-plan",
    ),
    display_name="Token Plan Personal",
    description="Alibaba Token Plan Personal (Global)",
    signup_url="https://www.qwencloud.com/pricing/token-plan",
    env_vars=(
        "QWEN_TOKEN_PLAN_API_KEY",
        "BAILIAN_TOKEN_PLAN_API_KEY",
        "ALIBABA_TOKEN_PLAN_API_KEY",
        "ALIBABA_TOKEN_PLAN_PERSONAL_API_KEY",
        "ALIBABA_TOKEN_PLAN_BASE_URL",
    ),
    base_url="https://token-plan.maas.qwencloudapi.com/compatible-mode/v1",
    auth_type="api_key",
    supports_health_check=True,
    supports_vision=True,
    default_aux_model=DEFAULT_AUX_MODEL,
    fallback_models=PERSONAL_MODELS,
    supports_prompt_cache_key=True,
)

alibaba_token_plan_cn = QwenTokenPlanProfile(
    name="alibaba-token-plan-cn",
    aliases=(
        "alibaba_token_plan_cn",
        "aliyun-token-plan-cn",
        "token-plan-cn",
        "dashscope-token-plan-cn",
    ),
    display_name="Token Plan Personal (China)",
    description="Alibaba Token Plan Personal (China)",
    signup_url="https://www.aliyun.com/benefit/scene/tokenplan",
    env_vars=(
        "ALIBABA_TOKEN_PLAN_CN_API_KEY",
        "ALIBABA_TOKEN_PLAN_CN_PERSONAL_API_KEY",
        "ALIBABA_TOKEN_PLAN_API_KEY",
        "ALIBABA_TOKEN_PLAN_CN_BASE_URL",
    ),
    base_url="https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1",
    auth_type="api_key",
    supports_health_check=True,
    supports_vision=True,
    default_aux_model=DEFAULT_AUX_MODEL,
    fallback_models=PERSONAL_MODELS,
    supports_prompt_cache_key=True,
)

alibaba_token_plan_team = QwenTokenPlanProfile(
    name="alibaba-token-plan-team",
    aliases=(
        "alibaba_token_plan_team",
        "aliyun-token-plan-team",
        "token-plan-team",
        "qwen-token-plan-team",
    ),
    display_name="Token Plan Team",
    description="Alibaba Token Plan Team (Global)",
    signup_url="https://www.qwencloud.com/pricing/token-plan",
    env_vars=("ALIBABA_TOKEN_PLAN_TEAM_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.maas.qwencloudapi.com/compatible-mode/v1",
    auth_type="api_key",
    supports_health_check=True,
    supports_vision=True,
    default_aux_model=DEFAULT_AUX_MODEL,
    fallback_models=TEAM_MODELS,
    supports_prompt_cache_key=True,
)

alibaba_token_plan_cn_team = QwenTokenPlanProfile(
    name="alibaba-token-plan-cn-team",
    aliases=(
        "alibaba_token_plan_cn_team",
        "aliyun-token-plan-cn-team",
        "token-plan-cn-team",
    ),
    display_name="Token Plan Team (China)",
    description="Alibaba Token Plan Team (China)",
    signup_url="https://www.aliyun.com/benefit/scene/tokenplan",
    env_vars=("ALIBABA_TOKEN_PLAN_CN_TEAM_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1",
    auth_type="api_key",
    supports_health_check=True,
    supports_vision=True,
    default_aux_model=DEFAULT_AUX_MODEL,
    fallback_models=TEAM_MODELS,
    supports_prompt_cache_key=True,
)

register_provider(alibaba)
register_provider(alibaba_cn)
register_provider(alibaba_token_plan)
register_provider(alibaba_token_plan_team)
register_provider(alibaba_token_plan_cn)
register_provider(alibaba_token_plan_cn_team)
