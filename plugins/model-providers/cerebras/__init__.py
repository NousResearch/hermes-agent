"""Cerebras Inference provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CerebrasProfile(ProviderProfile):
    """Cerebras — OpenAI-compatible; reasoning models take top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # gpt-oss / GLM are reasoning models that accept a top-level
        # ``reasoning_effort``; other Cerebras models reject it. Cerebras is
        # not in _supports_reasoning_extra_body()'s allowlist, so this hook is
        # the only path that conveys reasoning effort for these models.
        ml = (model or "").lower()
        if "gpt-oss" not in ml and "glm" not in ml:
            return {}, {}
        if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
            return {}, {}
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if effort in ("xhigh", "max"):
            effort = "high"  # Cerebras tops out at "high"
        if effort not in ("low", "medium", "high"):
            effort = "medium"
        return {}, {"reasoning_effort": effort}


cerebras = CerebrasProfile(
    name="cerebras",
    env_vars=("CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"),
    display_name="Cerebras",
    description="Cerebras — ultra-fast wafer-scale inference (OpenAI-compatible)",
    signup_url="https://cloud.cerebras.ai/",
    base_url="https://api.cerebras.ai/v1",
    auth_type="api_key",
    default_aux_model="gpt-oss-120b",
    fallback_models=(
        "gpt-oss-120b",
        "zai-glm-4.7",
    ),
)

register_provider(cerebras)
