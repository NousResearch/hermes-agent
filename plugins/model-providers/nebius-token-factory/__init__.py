"""Nebius Token Factory provider profile."""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _flat_model_name(model: str | None) -> str:
    """Return a lowercase model id, tolerating vendor-prefixed IDs."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def _model_supports_reasoning_effort(model: str | None) -> bool:
    """Conservative allowlist for Nebius models that expose reasoning effort."""
    model_name = _flat_model_name(model)
    if not model_name:
        return False
    return any(
        marker in model_name
        for marker in (
            "deepseek-r1",
            "deepseek-v4",
            "deepseek-reasoner",
            "gpt-oss",
            "glm-5",
            "kimi-k2",
            "minimax-m2",
            "qwen3",
        )
    )


class NebiusTokenFactoryProfile(ProviderProfile):
    """Nebius Token Factory - top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not supports_reasoning and not _model_supports_reasoning_effort(model):
            return {}, {}

        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled", True)
            raw_effort = reasoning_config.get("effort", "medium")
        else:
            enabled = True
            raw_effort = "medium"

        effort = str(raw_effort or "medium").strip().lower()
        if enabled is False or effort in {"none", "off", "disabled"}:
            return {}, {}
        if effort in {"xhigh", "max"}:
            effort = "high"
        elif effort == "minimal":
            effort = "low"
        elif effort not in {"low", "medium", "high"}:
            effort = "medium"

        return {}, {"reasoning_effort": effort}


nebius_token_factory = NebiusTokenFactoryProfile(
    name="nebius-token-factory",
    aliases=(
        "nebius",
        "nebius-tokenfactory",
        "nebius-tf",
        "token-factory",
        "tokenfactory",
    ),
    display_name="Nebius Token Factory",
    description="Nebius Token Factory — OpenAI-compatible inference",
    signup_url="https://tokenfactory.nebius.com/",
    env_vars=(
        "NEBIUS_API_KEY",
        "NEBIUS_TOKEN_FACTORY_API_KEY",
        "NEBIUS_BASE_URL",
    ),
    base_url="https://api.tokenfactory.nebius.com/v1",
    models_url="https://api.tokenfactory.nebius.com/v1/models?verbose=true",
    auth_type="api_key",
    default_aux_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    fallback_models=(
        "Qwen/Qwen3.5-397B-A17B-fast",
        "deepseek-ai/DeepSeek-V4-Pro",
        "zai-org/GLM-5.1",
        "moonshotai/Kimi-K2.5-fast",
        "MiniMaxAI/MiniMax-M2.5-fast",
        "deepseek-ai/DeepSeek-V3.2-fast",
        "NousResearch/Hermes-4-70B",
        "openai/gpt-oss-120b-fast",
        "meta-llama/Llama-3.3-70B-Instruct",
    ),
)

register_provider(nebius_token_factory)
