"""Cerebras Inference provider profile."""

from typing import Any

from agent.reasoning_effort import clamp_effort
from providers import register_provider
from providers.base import ProviderProfile

_REASONING_EFFORTS = {
    "qwen-3.8-27b": ("none", "low", "medium", "high"),
    "gpt-oss-120b": ("low", "medium", "high"),
}


class CerebrasProfile(ProviderProfile):
    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        return _REASONING_EFFORTS.get(model or "", ())

    def build_api_kwargs_extras(
        self, *, model: str | None = None, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        supported = self.supported_reasoning_efforts(model)
        if not supported or not isinstance(reasoning_config, dict):
            return {}, {}

        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if reasoning_config.get("enabled") is False:
            effort = "none"
        # GPT-OSS always reasons. Leave its server default in place when "off" is requested.
        if effort == "none" and effort not in supported:
            return {}, {}

        effort = clamp_effort(effort, supported)
        return {}, ({"reasoning_effort": effort} if effort in supported else {})


register_provider(CerebrasProfile(
    name="cerebras",
    display_name="Cerebras",
    description="Cerebras Inference",
    signup_url="https://cloud.cerebras.ai/",
    env_vars=("CEREBRAS_API_KEY",),
    base_url="https://api.cerebras.ai/v1",
    auth_type="api_key",
    api_mode="chat_completions",
    default_aux_model="qwen-3.8-27b",
    fallback_models=("qwen-3.8-27b", "gpt-oss-120b"),
))
