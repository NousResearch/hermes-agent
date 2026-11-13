"""Verda Inference provider profile."""

from providers import register_provider
from providers.base import ProviderProfile
from typing import Any

from agent.reasoning_effort import (
    KIMI_K3_EFFORTS,
    clamp_effort,
    requested_effort,
)

class VerdaProfile(ProviderProfile):
    """Map reasoning controls to Verda's top-level reasoning_effort."""

    def supported_reasoning_efforts(
        self, model: str | None
    ) -> tuple[str, ...] | None:
        if model == "kimi-k3-1-pro":
            return KIMI_K3_EFFORTS
        return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        supported = self.supported_reasoning_efforts(model)
        if not supported:
            return {}, {}

        effort = requested_effort(reasoning_config)
        # Disabling reasoning is unconfirmed; leave the server default.
        if effort is None or effort == "none":
            return {}, {}

        clamped = clamp_effort(effort, supported)
        if clamped not in supported:
            return {}, {}

        return {}, {"reasoning_effort": clamped}

verda = VerdaProfile(
    name="verda",
    display_name="Verda Inference",
    description="Verda — OpenAI-compatible inference API",
    signup_url="https://docs.verda.com/inference/quickstart/",
    env_vars=("VERDA_INFERENCE_API_KEY",),
    base_url="https://inference.verda.com/v1",
    auth_type="api_key",
    default_aux_model="kimi-k3-1-pro",
    fallback_models=("kimi-k3-1-pro",),
    model_capabilities={
        "kimi-k3-1-pro": {
            "supports_reasoning": True,
            "supports_vision": True,
            "supports_tools": True,
            "context_window": 1048576,
        },
    },
)

register_provider(verda)